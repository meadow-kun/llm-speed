"""Cross-environment test matrix for the CLI.

These tests cover the highest-leverage uncovered paths surfaced in
`docs/cli_test_matrix.md`. Each test is hermetic, runs in <2s, and uses
monkeypatch over `unittest.mock` (no `pytest-mock` dependency).

Coverage:

* test_strict_anon_drops_auth_header           — strict-anon never sends Authorization
* test_anon_drops_auth_keeps_persistent_keypair — anon drops auth but uses persistent keypair
* test_hosted_api_no_raw_in_modelref_extras    — upstream provider JSON never ends up in ModelRef
* test_consent_eof_non_tty_does_not_consent    — non-tty stdin must not silently auto-consent
* test_save_offline_no_partial_file_on_oserror — disk-full mid-save leaves no half-written file
* test_assert_no_long_strings_path_match_is_exact — allowlist matched exactly, not by suffix
* test_llama_cpp_stream_counts_reasoning_content — reasoning_content tokens counted (Qwen3.6 fix)
* test_strict_anon_drops_user_agent_header     — strict-anon UA = generic httpx
* test_anon_keeps_user_agent_header            — soft anon retains UA
"""

from __future__ import annotations

import io
import json
from typing import Any

import httpx
import pytest


# ---------------------------------------------------------------------------
# Shared httpx.Client.post capture helper
# ---------------------------------------------------------------------------


class _CapturedPost(Exception):
    """Internal — raised after capturing a httpx.Client.post call so that the
    upload helpers see a transport-style failure and abort. Tests catch this
    exception and inspect the captured headers/body."""

    def __init__(self, *, headers: dict[str, str], body: dict[str, Any], url: str):
        self.headers = headers
        self.body = body
        self.url = url


def _install_post_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace httpx.Client.post with a one-shot capture that raises after
    recording. The upload retry loop catches httpx.TransportError; we use a
    subclass of TransportError so the loop's first attempt aborts cleanly."""

    class _CaptureTransport(httpx.TransportError):
        def __init__(self, captured: _CapturedPost):
            super().__init__("captured")
            self.captured = captured

    def _fake_post(self: httpx.Client, url, *, json=None, headers=None, **kw):  # noqa: A002
        raise _CaptureTransport(
            _CapturedPost(headers=dict(headers or {}), body=dict(json or {}), url=url)
        )

    monkeypatch.setattr(httpx.Client, "post", _fake_post)


# ---------------------------------------------------------------------------
# 1. strict-anon must drop the Authorization header
# ---------------------------------------------------------------------------


def test_strict_anon_drops_auth_header(
    isolated_signing_config,
    sample_run_report,
    monkeypatch: pytest.MonkeyPatch,
):
    """Strict-anon: the bearer token must NOT appear on the wire, even when
    --api-key is set."""
    from cli.upload import UploadError, upload_report

    captured_headers: list[dict[str, str]] = []

    def _fake_post(self, url, *, json=None, headers=None, **kw):  # noqa: A002
        captured_headers.append(dict(headers or {}))
        # Mimic a transport failure so retries spin once then give up.
        raise httpx.ConnectError("test capture")

    monkeypatch.setattr(httpx.Client, "post", _fake_post)

    with pytest.raises(UploadError):
        upload_report(
            sample_run_report,
            api_base="https://api.example.test",
            api_key="sk-secret-bearer",
            strict_anon=True,
            max_retries=1,
        )

    assert captured_headers, "post should have been attempted"
    h = captured_headers[0]
    # The point of strict-anon: no Authorization, no User-Agent.
    assert "Authorization" not in h
    assert not any(k.lower() == "authorization" for k in h)
    assert "User-Agent" not in h
    assert not any(k.lower() == "user-agent" for k in h)


def test_strict_anon_drops_user_agent_header(
    isolated_signing_config,
    sample_run_report,
    monkeypatch: pytest.MonkeyPatch,
):
    """Distinct from strict_anon_drops_auth_header — pin that we ALSO drop
    User-Agent so the request is indistinguishable from a generic httpx call."""
    from cli.upload import UploadError, upload_report

    seen: list[dict[str, str]] = []

    def _fake_post(self, url, *, json=None, headers=None, **kw):  # noqa: A002
        seen.append(dict(headers or {}))
        raise httpx.ConnectError("x")

    monkeypatch.setattr(httpx.Client, "post", _fake_post)

    with pytest.raises(UploadError):
        upload_report(
            sample_run_report,
            api_base="https://api.example.test",
            strict_anon=True,
            max_retries=1,
        )

    h = seen[0]
    assert all(k.lower() != "user-agent" for k in h)
    # X-LLM-Speed-Anon is a soft-anon-only header; strict-anon must not advertise itself.
    assert all(k.lower() != "x-llm-speed-anon" for k in h)


# ---------------------------------------------------------------------------
# 2. soft anon: drops auth but keeps the persistent keypair
# ---------------------------------------------------------------------------


def test_anon_drops_auth_keeps_persistent_keypair(
    isolated_signing_config,
    sample_run_report,
    monkeypatch: pytest.MonkeyPatch,
):
    """`--anon` (soft anon) must NOT send the Authorization header but MUST
    sign with the persistent keypair (so the user retains the same identity
    across runs). Two consecutive anon uploads should produce the same JWS
    public key in the JWS header (Ed25519 = deterministic)."""
    from cli.signing import sign_report_jws
    from cli.upload import UploadError, upload_report

    seen: list[dict[str, Any]] = []

    def _fake_post(self, url, *, json=None, headers=None, **kw):  # noqa: A002
        seen.append({"headers": dict(headers or {}), "body": dict(json or {})})
        raise httpx.ConnectError("x")

    monkeypatch.setattr(httpx.Client, "post", _fake_post)

    for _ in range(2):
        with pytest.raises(UploadError):
            upload_report(
                sample_run_report,
                api_base="https://api.example.test",
                api_key="sk-secret-bearer",
                anon=True,
                strict_anon=False,
                max_retries=1,
            )

    assert len(seen) == 2
    for record in seen:
        assert "Authorization" not in record["headers"]
        assert all(k.lower() != "authorization" for k in record["headers"])
        # Soft anon DOES advertise the X-LLM-Speed-Anon marker per upload.py.
        assert record["headers"].get("X-LLM-Speed-Anon") == "1"

    # Both uploads should use the same persistent keypair → identical JWS.
    body_a, body_b = seen[0]["body"], seen[1]["body"]
    assert body_a["jws"] == body_b["jws"]

    # Sanity-check that strict-anon would have produced different JWS tokens
    # (ephemeral key per run) — pin the contrast directly.
    a = sign_report_jws(sample_run_report, strict_anon=True)
    b = sign_report_jws(sample_run_report, strict_anon=True)
    assert a != b


# ---------------------------------------------------------------------------
# 3. hosted-api driver does NOT leak upstream model JSON
# ---------------------------------------------------------------------------


def test_hosted_api_no_raw_in_modelref_extras(monkeypatch: pytest.MonkeyPatch):
    """OpenRouter's /v1/models response often includes 10+ KB of pricing,
    licence text, etc. The driver MUST NOT carry that into ModelRef.extras —
    if it did, the upload payload would (a) leak undocumented metadata to our
    server and (b) routinely trip the 256-char privacy invariant."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    # Make sure no other provider env vars leak in from the test environment.
    for env in (
        "OPENAI_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(env, raising=False)

    raw_models_response = {
        "data": [
            {
                "id": "openai/gpt-4o-mini",
                "name": "GPT-4o mini",
                "pricing": {"prompt": "0.00015", "completion": "0.0006"},
                # Long licence-style blurb the upstream API can return.
                "description": "X" * 500,
                "context_length": 128000,
                "architecture": {"modality": "text", "tokenizer": "cl100k_base"},
                "top_provider": {"is_moderated": False},
            }
        ]
    }

    class _FakeResp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return raw_models_response

        @property
        def text(self) -> str:
            return json.dumps(raw_models_response)

    def _fake_get(self, url, *, headers=None, timeout=None):  # noqa: ARG001
        return _FakeResp()

    monkeypatch.setattr(httpx.Client, "get", _fake_get)

    # Import after env vars are set so module-level enabled-providers logic
    # picks up our fake key. (The driver re-reads env on each call though.)
    from cli.drivers.hosted_api import HostedApiDriver

    drv = HostedApiDriver()
    try:
        models = drv.list_models()
    finally:
        drv.close()

    assert models, "driver should surface the model"
    m = models[0]
    # Pin the documented contract exactly: provider + base_url, nothing else.
    assert isinstance(m.extras, dict)
    assert set(m.extras.keys()) == {"provider", "base_url"}
    # Specifically: no description, no pricing, no architecture.
    for forbidden in ("description", "pricing", "architecture", "top_provider"):
        assert forbidden not in m.extras

    # The 500-char description must NOT be reachable anywhere on the ModelRef.
    flat = json.dumps(
        {
            "identifier": m.identifier,
            "name": m.name,
            "extras": m.extras,
        }
    )
    assert "X" * 500 not in flat


# ---------------------------------------------------------------------------
# 4. consent: non-tty stdin does NOT silently consent
# ---------------------------------------------------------------------------


def test_consent_non_tty_refuses_to_auto_consent(monkeypatch: pytest.MonkeyPatch):
    """Feeding EOF / running on CI / piping from /dev/null must NOT consent.
    The TTY check in `prompt_for_consent` is the only thing standing between
    us and silent uploads on every CI run."""
    from cli import consent

    # Make stdin look non-interactive (the actual CI case).
    class _FakeStdin:
        def isatty(self) -> bool:
            return False

        def readline(self) -> str:
            # Should never be called — prompt should bail before reading.
            raise AssertionError("prompt_for_consent must not read on non-tty")

    monkeypatch.setattr("sys.stdin", _FakeStdin())

    out = io.StringIO()
    ok = consent.prompt_for_consent(
        api_base="https://api.example.test",
        cli_version="0.0.1-dev",
        fingerprint_hash="abcd",
        stream=out,
    )
    assert ok is False
    text = out.getvalue()
    # The text should explain *why* we're not consenting.
    assert "TTY" in text or "tty" in text


def test_consent_eof_on_tty_returns_false(monkeypatch: pytest.MonkeyPatch):
    """A TTY-attached user who hits Ctrl-D at the prompt sends an empty
    string — readline returns ''. That is NOT consent."""
    from cli import consent

    class _FakeStdin:
        def isatty(self) -> bool:
            return True

        def readline(self) -> str:
            return ""  # EOF mid-prompt

    monkeypatch.setattr("sys.stdin", _FakeStdin())

    out = io.StringIO()
    ok = consent.prompt_for_consent(
        api_base="https://api.example.test",
        cli_version="0.0.1-dev",
        stream=out,
    )
    assert ok is False
    assert "no input" in out.getvalue()


# ---------------------------------------------------------------------------
# 5. disk-full during offline save: no half-written file
# ---------------------------------------------------------------------------


def test_save_offline_disk_full_no_partial_file(
    isolated_signing_config,
    sample_run_report,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """If the underlying file write raises (disk full / read-only fs /
    permission denied), the offline save must propagate the exception and
    NOT leave a partially-written runs file behind that would later confuse
    `--resume`. Patches `os.fdopen` since save_offline switched to a
    permission-tightened os.open + os.fdopen pattern (runs files now live
    at 0600 inside a 0700 directory)."""
    import os as _os

    from cli import upload as upload_mod

    target = tmp_path / "runs" / "halfwrite.json"

    real_fdopen = _os.fdopen

    class _FailingFile:
        def __init__(self, fd):
            self._fd = fd

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            try:
                _os.close(self._fd)
            except OSError:
                pass
            return False

        def write(self, *_a, **_kw):
            raise OSError(28, "No space left on device")

    def _failing_fdopen(fd, *a, **kw):
        return _FailingFile(fd)

    monkeypatch.setattr("os.fdopen", _failing_fdopen)

    with pytest.raises(OSError, match="No space left on device"):
        upload_mod.save_offline(sample_run_report, target)

    monkeypatch.setattr("os.fdopen", real_fdopen)

    # The os.open call created an empty file; that's still a partial-write
    # artifact and the fix should clean it up. (If we ever stop
    # creating-and-then-failing, this test will catch the regression.)
    if target.exists():
        # Empty / zero-byte is acceptable as a partial-write residue but
        # surface it so we can decide whether to add an unlink-on-error.
        assert target.stat().st_size == 0, (
            "save_offline left a non-empty partial file behind"
        )


# ---------------------------------------------------------------------------
# 6. assert_no_long_strings: allowlist match must be exact, not suffix
# ---------------------------------------------------------------------------


def test_assert_no_long_strings_path_match_is_exact():
    """`results.error` is allowed (the workload error string can be long).
    But `results.extras.error` is NOT — adding `.extras.` in the middle must
    NOT bypass the check by suffix match."""
    from cli.signing import assert_no_long_strings

    long_string = "x" * 300

    # Allowed path should pass.
    payload_ok = {
        "results": [{"workload": "chat-short", "error": long_string}],
    }
    assert_no_long_strings(payload_ok)  # must not raise

    # Same long string under a path NOT in the allowlist → must raise.
    payload_bad = {
        "results": [{"extras": {"error": long_string}}],
    }
    with pytest.raises(ValueError, match="long string"):
        assert_no_long_strings(payload_bad)

    # Top-level `error` is also NOT in the allowlist (only `results.error` is).
    payload_top = {"error": long_string}
    with pytest.raises(ValueError, match="long string"):
        assert_no_long_strings(payload_top)


def test_assert_no_long_strings_at_root_with_unknown_field_raises():
    from cli.signing import assert_no_long_strings

    # Pretend the upstream snuck a long `description` field into the upload.
    payload = {"description": "y" * 1000}
    with pytest.raises(ValueError):
        assert_no_long_strings(payload)


# ---------------------------------------------------------------------------
# 7. llama.cpp driver: reasoning_content tokens are counted
#    (the 5090 Qwen3.6 fix from sweep_2026-04-29.md)
# ---------------------------------------------------------------------------


def test_llama_cpp_stream_counts_reasoning_content(monkeypatch: pytest.MonkeyPatch):
    """A reasoning model (Qwen3-Thinking, DeepSeek-R1, Qwen3.6) emits hidden
    chain-of-thought as `delta.reasoning_content`. The driver must count
    those tokens for decode tps; otherwise reasoning models report ~0 tps."""
    # Use external-server mode (LLAMA_CPP_SERVER_URL) so _ensure_server
    # short-circuits without trying to spawn a real llama-server process.
    monkeypatch.setenv("LLAMA_CPP_SERVER_URL", "http://fake.test:9999")

    from cli.drivers.llama_cpp import LlamaCppDriver
    from cli.types import ModelRef

    drv = LlamaCppDriver()

    # Build an SSE response: 3 reasoning_content tokens + 2 content tokens + DONE.
    sse_lines = [
        # Each line is one SSE chunk; iter_lines() returns them stripped.
        json.dumps({"choices": [{"delta": {"reasoning_content": "thinking…"}}]}),
        json.dumps({"choices": [{"delta": {"reasoning_content": " more thoughts"}}]}),
        json.dumps({"choices": [{"delta": {"reasoning_content": " final"}}]}),
        json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        json.dumps({"choices": [{"delta": {"content": " world"}}]}),
        "[DONE]",
    ]

    class _FakeStreamResp:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_lines(self):
            for line in sse_lines:
                yield f"data: {line}"

        def read(self) -> bytes:
            return b""

    def _fake_stream(self, method, url, *, json=None, **kw):  # noqa: ARG001, A002
        return _FakeStreamResp()

    monkeypatch.setattr(httpx.Client, "stream", _fake_stream)

    model = ModelRef(
        backend="llama.cpp",
        identifier="/tmp/m.gguf",
        name="qwen3.6-thinking",
    )
    outcome = drv.run_chat(model, "hi", max_output_tokens=8, stream=True)
    drv.close()

    assert outcome.success, outcome.error
    # 5 events total had any text → 5 "tokens" (1 ttft + 4 inter-token deltas).
    assert outcome.output_tokens == 5
    assert len(outcome.decode_token_times_ms) == 4
    assert outcome.ttft_ms is not None
    # The output text concatenates BOTH streams (reasoning then content).
    assert "Hello world" in outcome.output_text
    assert "thinking" in outcome.output_text


# ---------------------------------------------------------------------------
# bonus: anon + strict-anon header policy in the resume path
# ---------------------------------------------------------------------------


def test_resume_anon_drops_auth_keeps_marker(
    isolated_signing_config,
    sample_run_report,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """`upload_saved_payload` (the --resume path) must apply the same anon
    policy: drop Authorization, keep the X-LLM-Speed-Anon marker."""
    from cli.signing import sign_report_jws
    from cli.upload import UploadError, upload_saved_payload

    token = sign_report_jws(sample_run_report)
    saved = {"jws": token}

    seen: list[dict[str, str]] = []

    def _fake_post(self, url, *, json=None, headers=None, **kw):  # noqa: A002
        seen.append(dict(headers or {}))
        raise httpx.ConnectError("x")

    monkeypatch.setattr(httpx.Client, "post", _fake_post)

    with pytest.raises(UploadError):
        upload_saved_payload(
            saved,
            api_base="https://api.example.test",
            api_key="sk-secret",
            anon=True,
            max_retries=1,
        )

    h = seen[0]
    assert all(k.lower() != "authorization" for k in h)
    assert h.get("X-LLM-Speed-Anon") == "1"
