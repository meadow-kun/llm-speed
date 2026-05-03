"""Sign / verify round-trip across the JWS boundary.

`cli.signing.sign_report_jws` produces a compact JWS token; the same module's
`verify_jws_token` (and the API's `verify_jws_token`) must verify it. Tamper
detection works at the JWS layer — flipping any byte of the payload breaks
the signature.

Every test in this file uses `isolated_signing_config` so the Ed25519 keypair
is materialized under tmp_path, never under the user's real `~/.config/llm-speed`.
"""

from __future__ import annotations

import dataclasses

from cli.types import RunReport


def test_sign_report_jws_produces_compact_token(
    isolated_signing_config,
    sample_run_report: RunReport,
):
    from cli.signing import sign_report_jws

    token = sign_report_jws(sample_run_report)

    assert isinstance(token, str) and token
    assert token.count(".") == 2  # compact JWS = three b64url segments


def test_signed_token_verifies_locally(
    isolated_signing_config,
    sample_run_report: RunReport,
):
    from cli.signing import sign_report_jws, verify_jws_token

    token = sign_report_jws(sample_run_report)
    ok, reason, payload = verify_jws_token(token)

    assert ok is True, f"verify failed: {reason}"
    assert payload is not None
    assert payload["cli_version"] == "0.0.1-dev"
    assert payload["results"][0]["decode_tps"] == 187.4





# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_tampering_with_payload_segment_breaks_verification(
    isolated_signing_config,
    sample_run_report: RunReport,
):
    """Flip a byte in the JWS payload segment — signature must reject."""
    from cli.signing import sign_report_jws, verify_jws_token

    token = sign_report_jws(sample_run_report)
    header, payload, sig = token.split(".")

    # Mutate one base64 char in the payload segment.
    flip = "A" if payload[0] != "A" else "B"
    tampered = f"{header}.{flip}{payload[1:]}.{sig}"

    ok, _msg, _payload = verify_jws_token(tampered)
    assert ok is False


def test_tampering_with_signature_segment_breaks_verification(
    isolated_signing_config,
    sample_run_report: RunReport,
):
    from cli.signing import sign_report_jws, verify_jws_token

    token = sign_report_jws(sample_run_report)
    header, payload, sig = token.split(".")

    flip = "A" if sig[0] != "A" else "B"
    tampered = f"{header}.{payload}.{flip}{sig[1:]}"

    ok, _msg, _payload = verify_jws_token(tampered)
    assert ok is False


def test_malformed_token_returns_false(isolated_signing_config):
    from cli.signing import verify_jws_token

    ok, msg, _ = verify_jws_token("not.a.real-jws")
    assert ok is False
    assert msg


# ---------------------------------------------------------------------------
# Idempotency / determinism
# ---------------------------------------------------------------------------


def test_signing_twice_produces_identical_token_for_identical_payload(
    isolated_signing_config,
    sample_run_report: RunReport,
):
    """Ed25519 is deterministic. Signing the same canonical payload with the
    same persistent key MUST produce the same token byte-for-byte."""
    from cli.signing import sign_report_jws

    a = sign_report_jws(sample_run_report)
    b = sign_report_jws(sample_run_report)

    assert a == b


def test_changing_cli_version_changes_token(
    isolated_signing_config,
    sample_run_report: RunReport,
):
    """If any field in the canonical payload changes, the token must change too."""
    from cli.signing import sign_report_jws

    token_before = sign_report_jws(sample_run_report)

    mutated = dataclasses.replace(sample_run_report, cli_version="9.9.9-mutated")
    token_after = sign_report_jws(mutated)

    assert token_before != token_after


# ---------------------------------------------------------------------------
# Strict-anon: ephemeral keypair per run.
# ---------------------------------------------------------------------------


def test_strict_anon_uses_fresh_keypair_per_run(
    isolated_signing_config,
    sample_run_report: RunReport,
):
    """Two strict-anon signatures of the same payload must differ — the public
    key embedded in each header is a fresh ephemeral one."""
    from cli.signing import sign_report_jws, verify_jws_token

    a = sign_report_jws(sample_run_report, strict_anon=True)
    b = sign_report_jws(sample_run_report, strict_anon=True)

    assert a != b  # different ephemeral keys → different headers/sigs

    # Both still verify (each is self-contained via embedded jwk).
    ok_a, _, _ = verify_jws_token(a)
    ok_b, _, _ = verify_jws_token(b)
    assert ok_a and ok_b
