"""Tests for `cli.verify`.

Covers the pure helpers (sha256_file, parse_checksum) plus an end-to-end
verify_wheel that monkeypatches the network fetch so the test doesn't depend
on llm-speed.com being reachable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cli import verify

# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path: Path):
    blob = b"the bytes you are about to run"
    p = tmp_path / "file.bin"
    p.write_bytes(blob)
    assert verify.sha256_file(p) == hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# parse_checksum
# ---------------------------------------------------------------------------


def test_parse_checksum_accepts_bare_hex():
    digest = "a" * 64
    assert verify.parse_checksum(digest) == digest


def test_parse_checksum_accepts_sha256sum_format():
    digest = "f" * 64
    text = f"{digest}  llm_speed-0.0.1-py3-none-any.whl\n"
    assert verify.parse_checksum(text) == digest


def test_parse_checksum_returns_none_for_nonsense():
    assert verify.parse_checksum("") is None
    assert verify.parse_checksum("not a hash") is None


# ---------------------------------------------------------------------------
# verify_wheel: matching path
# ---------------------------------------------------------------------------


def test_verify_wheel_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    blob = b"wheel-bytes"
    wheel = tmp_path / "llm_speed-0.0.1-py3-none-any.whl"
    wheel.write_bytes(blob)
    expected = hashlib.sha256(blob).hexdigest()

    def _fake_fetch(url: str, **kw) -> str:
        # Server returns sha256sum-formatted output.
        return f"{expected}  {wheel.name}\n"

    monkeypatch.setattr(verify, "fetch_checksum", _fake_fetch)
    verdict = verify.verify_wheel(wheel, dist_base="https://llm-speed.com/dist")
    assert verdict.matched is True
    assert verdict.local_hash == expected
    assert verdict.expected_hash == expected
    assert verdict.sidecar_url.endswith(f"{wheel.name}.sha256")


def test_verify_wheel_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wheel = tmp_path / "llm_speed-0.0.1-py3-none-any.whl"
    wheel.write_bytes(b"local-bytes")
    other = "0" * 64

    monkeypatch.setattr(verify, "fetch_checksum", lambda url, **kw: other)
    verdict = verify.verify_wheel(wheel, dist_base="https://llm-speed.com/dist")
    assert verdict.matched is False
    assert verdict.expected_hash == other
    assert verdict.local_hash != other


def test_verify_wheel_missing_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wheel = tmp_path / "llm_speed-0.0.1-py3-none-any.whl"
    wheel.write_bytes(b"x")

    def _fake_fetch(url: str, **kw) -> str:
        return "(server returned no checksum)"

    monkeypatch.setattr(verify, "fetch_checksum", _fake_fetch)
    with pytest.raises(RuntimeError, match="does not contain a sha256 digest"):
        verify.verify_wheel(wheel, dist_base="https://llm-speed.com/dist")


def test_verify_wheel_path_not_found(tmp_path: Path):
    with pytest.raises(RuntimeError, match="wheel not found"):
        verify.verify_wheel(tmp_path / "missing.whl")


# ---------------------------------------------------------------------------
# cmd_verify: full path with mocked fetch
# ---------------------------------------------------------------------------


class _Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_cmd_verify_match_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    blob = b"x"
    wheel = tmp_path / "llm_speed-0.0.1-py3-none-any.whl"
    wheel.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    monkeypatch.setattr(
        verify, "fetch_checksum", lambda url, **kw: f"{digest}  {wheel.name}\n"
    )

    rc = verify.cmd_verify(
        _Args(wheel=str(wheel), dist_base="https://llm-speed.com/dist")
    )
    assert rc == 0


def test_cmd_verify_mismatch_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    wheel = tmp_path / "llm_speed-0.0.1-py3-none-any.whl"
    wheel.write_bytes(b"local")
    monkeypatch.setattr(verify, "fetch_checksum", lambda url, **kw: "0" * 64)
    rc = verify.cmd_verify(
        _Args(wheel=str(wheel), dist_base="https://llm-speed.com/dist")
    )
    assert rc == 1


def test_cmd_verify_missing_path_falls_back_to_remote_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """When no local wheel exists, verify downloads the published wheel and
    checks it against the sidecar — proves the CDN-served wheel matches its
    declared digest, even for pipx users without a local artifact."""
    monkeypatch.setattr(verify, "installed_wheel_path", lambda: None)
    fake_wheel = tmp_path / "llm_speed-fake.whl"
    fake_wheel.write_bytes(b"hello")
    # sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    expected_hash = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    monkeypatch.setattr(verify, "fetch_published_wheel_to_temp", lambda *a, **kw: fake_wheel)
    monkeypatch.setattr(verify, "fetch_checksum", lambda url: expected_hash)
    rc = verify.cmd_verify(_Args(wheel=None, dist_base=None))
    assert rc == 0


def test_cmd_verify_missing_path_remote_fetch_failure_returns_1(
    monkeypatch: pytest.MonkeyPatch,
):
    """When no local wheel AND the remote fetch fails, verify exits 1."""
    monkeypatch.setattr(verify, "installed_wheel_path", lambda: None)

    def _boom(*a, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(verify, "fetch_published_wheel_to_temp", _boom)
    rc = verify.cmd_verify(_Args(wheel=None, dist_base=None))
    assert rc == 1
