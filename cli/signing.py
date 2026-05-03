"""Ed25519 keypair persistence + JWS-based RunReport signing.

We sign uploads as JWS (RFC 7515) compact-serialization tokens with EdDSA over
Ed25519. The public key rides in the protected header as a JWK — verifiers
need nothing else. Standard envelope, library-implemented in Python (joserfc),
TypeScript (jose), and Rust (jsonwebtoken). No hand-rolled canonical JSON.

Default mode: long-lived keypair under `~/.config/llm-speed/keys/ed25519.key`
(mode 0600). The public key is derived from the private key on load — never
stored separately. Strict-anon mode generates an ephemeral keypair per run
that's never written to disk.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from joserfc import jws as joserfc_jws
from joserfc.jwk import OKPKey

from .config import CONFIG_DIR
from .fingerprint import to_uploadable_dict
from .types import RunReport

log = logging.getLogger("cli.signing")

KEY_DIR = CONFIG_DIR / "keys"
KEY_PATH = KEY_DIR / "ed25519.key"

ALG = "EdDSA"


# ---------------------------------------------------------------------------
# Keypair persistence (unchanged)
# ---------------------------------------------------------------------------


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    raw = path.read_bytes()
    if len(raw) == 32:
        return Ed25519PrivateKey.from_private_bytes(raw)
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except ValueError as exc:
        raise ValueError(
            f"unable to parse ed25519 private key at {path}: {exc}"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"key at {path} is not ed25519")
    return key


def _write_private_key(path: Path, key: Ed25519PrivateKey) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def get_or_create_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    if KEY_PATH.exists():
        try:
            priv = _load_private_key(KEY_PATH)
            return priv, priv.public_key()
        except ValueError as exc:
            log.warning("existing key unreadable (%s); regenerating", exc)
    priv = Ed25519PrivateKey.generate()
    _write_private_key(KEY_PATH, priv)
    log.debug("generated new ed25519 keypair at %s", KEY_PATH)
    return priv, priv.public_key()


def ephemeral_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


def _public_key_bytes(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _private_key_bytes(priv: Ed25519PrivateKey) -> bytes:
    return priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _ed25519_to_okp_jwk(priv: Ed25519PrivateKey, pub: Ed25519PublicKey) -> OKPKey:
    """Build a joserfc OKPKey holding both the private and public material."""
    return OKPKey.import_key(
        {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": _b64url(_public_key_bytes(pub)),
            "d": _b64url(_private_key_bytes(priv)),
        }
    )


def public_jwk(pub: Ed25519PublicKey) -> dict[str, str]:
    """Public-only JWK suitable for embedding in the JWS protected header."""
    return {"kty": "OKP", "crv": "Ed25519", "x": _b64url(_public_key_bytes(pub))}


# ---------------------------------------------------------------------------
# Privacy invariant — keep regardless of envelope format.
# Walks the dict tree and refuses upload if any non-allowlisted string > 256 chars.
# ---------------------------------------------------------------------------

# Fully-qualified paths only (matched EXACTLY against the dotted path emitted
# by `assert_no_long_strings`). The walker descends into lists without
# emitting an index segment, so `results.error` covers `results[i].error` for
# every i, but does NOT cover `results[i].extras.error` (that path is
# `results.extras.error`, which is not on the list).
#
# This is deliberately a closed allowlist: bare leaf names like `error`,
# `workload`, `backend` are TOP-LEVEL ONLY OR `results.*`-NESTED ONLY. Adding
# new long-string fields requires adding their full path here.
_LONG_STRING_ALLOWED = {
    # Top-level provenance fields
    "suite_version",
    "cli_version",
    # Fingerprint
    "fingerprint.accelerator_summary",
    "fingerprint.fingerprint_hash",
    # Per-workload-result fields (path inside results[*])
    "results.workload",
    "results.suite_version",
    "results.backend",
    "results.backend_version",
    "results.error",
    "results.model.identifier",
    "results.model.name",
    "results.model.size",
    "results.model.quant",
    "results.model.digest",
}

_MAX_STRING_LEN = 256


def _path_is_allowed(path: str, allow: set) -> bool:
    return path in allow


def assert_no_long_strings(
    payload: Any, *, allowed: set | None = None, _path: str = ""
) -> None:
    allow = allowed if allowed is not None else _LONG_STRING_ALLOWED
    if isinstance(payload, dict):
        for k, v in payload.items():
            child = f"{_path}.{k}" if _path else str(k)
            assert_no_long_strings(v, allowed=allow, _path=child)
    elif isinstance(payload, list):
        for item in payload:
            assert_no_long_strings(item, allowed=allow, _path=_path)
    elif isinstance(payload, str):
        if len(payload) > _MAX_STRING_LEN and not _path_is_allowed(_path, allow):
            raise ValueError(
                f"upload payload contains unexpected long string at {_path or '<root>'}; "
                f"refusing to upload (privacy invariant)"
            )


# ---------------------------------------------------------------------------
# Uploadable payload construction
# ---------------------------------------------------------------------------


def _trim_workload_result(
    result_dict: dict[str, Any], *, include_raw_timings: bool
) -> dict[str, Any]:
    """Strip per-result fields that would leak absolute filesystem paths or
    other host-local detail beyond what PRIVACY.md promises.

    Specifically (per the 2026-05-01 PII audit, F-1/F-2/F-3):
      - model.identifier may be a `str(Path.resolve())` from the llama.cpp /
        mlx / exllamav2 drivers — replace with the basename only so the
        username-bearing parent directories don't ship.
      - model.extras may carry absolute paths (`path`, `snapshot_path`,
        `served_by` LAN URLs) under keys outside the allowlist. Drop it
        entirely; it's driver-internal scratch, not part of PRIVACY.md §1.
      - results.error is allowlisted with no length cap and routinely
        contains exception strings like
        `RuntimeError: model not found: /Users/<user>/...`. Scrub
        username-bearing paths and cap to 256 chars.
    """
    out = dict(result_dict)
    if not include_raw_timings:
        out.pop("raw_timings_ms", None)

    # F-1: replace `model.identifier` with its basename. The Path basename
    # is path-shape-aware (handles both POSIX and Windows separators).
    model = out.get("model")
    if isinstance(model, dict):
        ident = model.get("identifier")
        if isinstance(ident, str):
            from pathlib import PurePath
            try:
                base = PurePath(ident).name or ident
                model = {**model, "identifier": base}
            except (ValueError, TypeError):
                pass
        # F-2: drop driver-internal extras entirely. Anything we want to
        # keep (e.g. backend version) is at the top level of the result,
        # not inside model.extras.
        if "extras" in model:
            model = {k: v for k, v in model.items() if k != "extras"}
        out["model"] = model

    # F-3: error strings — sanitize home-dir paths + cap length.
    err = out.get("error")
    if isinstance(err, str):
        out["error"] = _sanitize_error_string(err)

    return out


def _sanitize_error_string(s: str, *, max_len: int = 256) -> str:
    """Replace absolute home-dir paths with `~/` placeholders and cap length.
    Catches `/Users/<user>/...` (macOS), `/home/<user>/...` (Linux),
    `C:\\Users\\<user>\\...` (Windows). The username segment is the leak."""
    import re
    # macOS / Linux: /Users/<u>/... or /home/<u>/...
    s = re.sub(r"/(?:Users|home)/[^/\s]+", "~", s)
    # Windows: C:\Users\<u>\... (any drive letter)
    s = re.sub(r"[A-Za-z]:\\Users\\[^\\\s]+", "~", s)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def build_uploadable_report(
    report: RunReport,
    *,
    strict_anon: bool = False,
    include_raw_timings: bool = False,
) -> dict[str, Any]:
    """Privacy-trimmed dict that will be the JWS payload. No signature fields here."""
    fp_dict = to_uploadable_dict(report.fingerprint, strict_anon=strict_anon)
    results_out: list[dict[str, Any]] = []
    for r in report.results:
        rd = dataclasses.asdict(r)
        results_out.append(
            _trim_workload_result(rd, include_raw_timings=include_raw_timings)
        )
    return {
        "suite_version": report.suite_version,
        "cli_version": report.cli_version,
        "fingerprint": fp_dict,
        "results": results_out,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
    }


# ---------------------------------------------------------------------------
# JWS signing
# ---------------------------------------------------------------------------


def sign_payload_jws(
    payload_dict: dict[str, Any],
    *,
    strict_anon: bool = False,
) -> str:
    """Sign a payload dict, return the compact-serialized JWS token.

    Header: {alg: EdDSA, jwk: <pub>, typ: llm-speed+json}
    Payload: utf-8 JSON of payload_dict with separators (",", ":")
    """
    priv, pub = ephemeral_keypair() if strict_anon else get_or_create_keypair()
    okp = _ed25519_to_okp_jwk(priv, pub)
    header = {
        "alg": ALG,
        "typ": "llm-speed+json",
        "jwk": public_jwk(pub),
    }
    payload_bytes = json.dumps(payload_dict, separators=(",", ":")).encode("utf-8")
    # joserfc's default registry excludes EdDSA from "recommended" — pass it explicitly.
    token = joserfc_jws.serialize_compact(header, payload_bytes, okp, algorithms=[ALG])
    return token


def sign_report_jws(
    report: RunReport,
    *,
    strict_anon: bool = False,
    include_raw_timings: bool = False,
) -> str:
    """Build the privacy-trimmed payload from a RunReport, sign, return JWS token."""
    payload = build_uploadable_report(
        report,
        strict_anon=strict_anon,
        include_raw_timings=include_raw_timings,
    )
    assert_no_long_strings(payload)
    return sign_payload_jws(payload, strict_anon=strict_anon)


# ---------------------------------------------------------------------------
# Local verification (used by --resume to sanity-check before re-uploading)
# ---------------------------------------------------------------------------


def verify_jws_token(token: str) -> tuple[bool, str, dict[str, Any] | None]:
    """Verify a compact JWS produced by sign_payload_jws. Trusts the embedded jwk —
    the pubkey IS the identity here (anonymous-but-pinned-to-a-keypair model)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False, f"not a compact JWS (got {len(parts)} segments)", None
        header_raw = base64.urlsafe_b64decode(parts[0] + "=" * (-len(parts[0]) % 4))
        header = json.loads(header_raw)
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to parse header: {exc}", None

    if header.get("alg") != ALG:
        return False, f"unexpected alg: {header.get('alg')}", None
    pub_jwk = header.get("jwk")
    if (
        not isinstance(pub_jwk, dict)
        or pub_jwk.get("kty") != "OKP"
        or pub_jwk.get("crv") != "Ed25519"
    ):
        return False, "missing or wrong-shape jwk in header", None

    try:
        key = OKPKey.import_key(pub_jwk)
    except Exception as exc:  # noqa: BLE001
        return False, f"jwk invalid: {exc}", None

    try:
        obj = joserfc_jws.deserialize_compact(token, key, algorithms=[ALG])
    except Exception as exc:  # noqa: BLE001
        return False, f"verification failed: {exc}", None

    try:
        payload_dict = json.loads(obj.payload)
    except Exception as exc:  # noqa: BLE001
        return False, f"payload not valid JSON: {exc}", None

    return True, "ok", payload_dict


# ---------------------------------------------------------------------------
# Public-key fingerprint (unchanged contract for `python -m cli.signing`)
# ---------------------------------------------------------------------------


def public_key_fingerprint() -> str:
    """sha256(raw_pubkey)[:16] hex — stable identity for this machine's CLI."""
    _, pub = get_or_create_keypair()
    return hashlib.sha256(_public_key_bytes(pub)).hexdigest()[:16]


def _main() -> int:
    fp = public_key_fingerprint()
    sys.stdout.write(f"{fp}\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
