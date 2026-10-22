"""Phase 2: signed remote presets (off unless `remote_presets` is true).

The Cloudflare Worker signs each preset bundle with an Ed25519 key held only as a
Worker secret. The CLI pins a LIST of public keys; rotation = a CLI release.
Every check fails closed. Packaged presets (cli/presets/*.json) need no signature.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ..preset_schema import canonical_bytes
from .settings import agent_dir, atomic_write_json

# The Worker's preset signing key(s), pinned by a CLI release (never fetched and trusted
# at runtime). key_id -> raw 32-byte Ed25519 public key, base64url without padding.
# llm-speed-presets-2026-10: the api.llm-speed.com key (GET /v1/presets/keys, 2026-10-22).
# Rotation = add the new key here in a release, remove the old one a release later.
PINNED_PRESET_KEYS: dict[str, str] = {
    "llm-speed-presets-2026-10": "k3DZlxsk0y_7EhSk-5Jx8oLGklUlR_FceIP6-ZGbAho",
}


class PresetVerificationError(Exception):
    pass


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _candidate_keys(
    envelope: dict, keys: dict[str, str] | list[str] | None
) -> list[str]:
    """The pinned keys to try. With a key_id map and an envelope key_id, only that key
    (an unknown key_id fails); a bare list (tests) tries every key."""
    keys = PINNED_PRESET_KEYS if keys is None else keys
    if not keys:
        raise PresetVerificationError("no pinned preset keys in this CLI")
    if isinstance(keys, dict):
        kid = envelope.get("key_id") if isinstance(envelope, dict) else None
        if kid is not None:
            if kid not in keys:
                raise PresetVerificationError("signed with a key this CLI does not pin")
            return [keys[kid]]
        return list(keys.values())
    return list(keys)


def verify_bundle(
    envelope: dict, keys: dict[str, str] | list[str] | None = None
) -> dict:
    """Verify {bundle, sig[, key_id]} against the pinned key(s); return the bundle JSON.
    The bundle bytes must be canonical (sorted keys, compact, UTF-8) like the Worker's."""
    if not isinstance(envelope, dict):
        raise PresetVerificationError("malformed envelope")
    cands = _candidate_keys(envelope, keys)
    try:
        raw = _b64d(envelope["bundle"])
        sig = _b64d(envelope["sig"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PresetVerificationError("malformed envelope") from exc
    for k in cands:
        try:
            Ed25519PublicKey.from_public_bytes(_b64d(k)).verify(sig, raw)
        except (InvalidSignature, ValueError):
            continue
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise PresetVerificationError("bundle is not JSON") from exc
        if not isinstance(data, dict):
            raise PresetVerificationError("bundle is not an object")
        if canonical_bytes(data) != raw:
            raise PresetVerificationError("bundle bytes are not canonical")
        return data
    raise PresetVerificationError("signature does not match any pinned key")


def versions_path() -> Path:
    return agent_dir() / "preset_versions.json"


def check_and_record_version(name: str, version: int) -> None:
    """Monotonic per name: refuse anything older than the last version seen."""
    p = versions_path()
    seen = json.loads(p.read_text()) if p.exists() else {}
    last = int(seen.get(name, 0))
    if version < last:
        raise PresetVerificationError(
            f"{name} v{version} is older than v{last} already seen (rollback)"
        )
    seen[name] = version
    atomic_write_json(p, seen)


def is_revoked(revocations: dict, name: str, version: int | None) -> bool:
    for r in revocations.get("revoked", []):
        if r.get("preset") == name and (
            r.get("version") is None or r.get("version") == version
        ):
            return True
    return False


MAX_REVOCATION_AGE = dt.timedelta(days=7)


def check_revocations(
    client,
    name: str,
    version: int,
    keys: dict[str, str] | list[str] | None = None,
    now: dt.datetime | None = None,
) -> None:
    """Fetch + verify the signed revocation list; refuse a stale list (> 7 days) or a revoked preset."""
    try:
        revs = verify_bundle(client.get_revocations(), keys)
    except PresetVerificationError:
        raise
    except Exception as exc:  # network etc. => fail closed
        raise PresetVerificationError(
            f"could not fetch revocations: {type(exc).__name__}"
        ) from exc
    try:
        issued = dt.datetime.fromisoformat(
            str(revs.get("issued_at")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise PresetVerificationError("revocation list has no valid issued_at") from exc
    if (
        issued.tzinfo is None
        or (now or dt.datetime.now(dt.timezone.utc)) - issued > MAX_REVOCATION_AGE
    ):
        raise PresetVerificationError("revocation list is stale")
    if is_revoked(revs, name, version):
        raise PresetVerificationError(f"{name} v{version} is revoked")
    check_and_record_version(name, version)


def verify_remote_job(
    client,
    name: str,
    version: int,
    sha256: str,
    keys: dict[str, str] | list[str] | None = None,
) -> dict:
    """Phase 2: fetch + verify a published bundle for a job. Raises on any failure (fail closed).
    Running it needs the preset command to accept a bundle file (follow-up)."""
    check_revocations(client, name, version, keys)
    try:
        env = client.get_preset(name, version)
    except Exception as exc:  # noqa: BLE001
        raise PresetVerificationError(
            f"could not fetch the bundle: {type(exc).__name__}"
        ) from exc
    bundle = verify_bundle(env, keys)
    raw = _b64d(env["bundle"])
    if hashlib.sha256(raw).hexdigest() != sha256 or env.get("sha256") not in (
        None,
        sha256,
    ):
        raise PresetVerificationError("bundle sha256 does not match the job")
    if bundle.get("preset") != name or bundle.get("version") != version:
        raise PresetVerificationError("bundle name/version does not match the job")
    return bundle
