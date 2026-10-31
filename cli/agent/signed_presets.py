"""Signed remote presets (phase 2, ops/AGENT_QUEUE_API.md section 8).

The Cloudflare Worker signs each preset bundle with an Ed25519 key held only as a
Worker secret. The CLI pins a LIST of public keys; rotation = a CLI release.
Every check fails closed. Packaged presets (cli/presets/*.json) need no signature.

0.0.12: the agent runs a server-signed preset that isn't packaged
(`resolve_signed_preset`). Trust chain, in order, all against the job's pinned
(name, version, sha256):
  1. the signed revocation list from the enrolled api_base (pinned key, issued <= 7 days
     ago, not older than the newest list already accepted): the version is not revoked and
     not older than the highest version of that name this machine has verified (local
     high-water mark, preset_versions.json);
  2. the bundle from the local cache (re-verified) or GET {api_base}/v1/presets/NAME/VERSION;
  3. Ed25519 over the exact bundle bytes with a PINNED key (an unpinned key_id fails);
  4. bytes canonical, sha256(bytes) == the job's sha256, name/version == the job's;
  5. the same shared schema + cross-reference checks a packaged preset passes
     (preset_schema.validate_preset), and no staging placeholder (the release-guard rule).
Only then is the high-water mark raised. The preset reaches the bench child only as the
verified cache file, which the child re-verifies (`load_verified`).
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .. import preset_schema
from ..config import CACHE_DIR
from ..preset_schema import canonical_bytes
from .settings import agent_dir, atomic_write_json

# The Worker's preset signing key(s), pinned by a CLI release (never fetched and trusted
# at runtime). key_id -> raw 32-byte Ed25519 public key, base64url without padding.
# llm-speed-presets-2026-10: the api.llm-speed.com key (GET /v1/presets/keys, 2026-10-22).
# Rotation = add the new key here in a release, remove the old one a release later.
PINNED_PRESET_KEYS: dict[str, str] = {
    "llm-speed-presets-2026-10": "k3DZlxsk0y_7EhSk-5Jx8oLGklUlR_FceIP6-ZGbAho",
}

Keys = dict[str, str] | list[str] | None


class PresetVerificationError(Exception):
    """`code` is a short reason code (local status, release detail); never a secret."""

    def __init__(self, message: str, code: str = "unverified"):
        super().__init__(message)
        self.code = code


def _b64d(s: str) -> bytes:
    if not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9_\-]*", s):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _candidate_keys(envelope: dict, keys: Keys) -> list[str]:
    """The pinned keys to try. With a key_id map and an envelope key_id, only that key
    (an unknown key_id fails); a bare list (tests) tries every key."""
    keys = PINNED_PRESET_KEYS if keys is None else keys
    if not keys:
        raise PresetVerificationError(
            "no pinned preset keys in this CLI", "no_pinned_keys"
        )
    if isinstance(keys, dict):
        kid = envelope.get("key_id") if isinstance(envelope, dict) else None
        if kid is not None:
            if kid not in keys:
                raise PresetVerificationError(
                    "signed with a key this CLI does not pin", "unpinned_key_id"
                )
            return [keys[kid]]
        return list(keys.values())
    return list(keys)


def verify_bundle(envelope: dict, keys: Keys = None) -> dict:
    """Verify {bundle, sig[, key_id]} against the pinned key(s); return the bundle JSON.
    The bundle bytes must be canonical (sorted keys, compact, UTF-8) like the Worker's."""
    if not isinstance(envelope, dict):
        raise PresetVerificationError("malformed envelope", "malformed")
    cands = _candidate_keys(envelope, keys)
    try:
        raw = _b64d(envelope["bundle"])
        sig = _b64d(envelope["sig"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PresetVerificationError("malformed envelope", "malformed") from exc
    for k in cands:
        try:
            Ed25519PublicKey.from_public_bytes(_b64d(k)).verify(sig, raw)
        except (InvalidSignature, ValueError):
            continue
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise PresetVerificationError("bundle is not JSON", "malformed") from exc
        if not isinstance(data, dict):
            raise PresetVerificationError("bundle is not an object", "malformed")
        try:
            canon = canonical_bytes(data)
        except (ValueError, TypeError):  # NaN / Infinity / 1e400 (R-99)
            raise PresetVerificationError(
                "bundle has non-finite numbers", "malformed"
            ) from None
        if canon != raw:
            raise PresetVerificationError(
                "bundle bytes are not canonical", "not_canonical"
            )
        return data
    raise PresetVerificationError(
        "signature does not match any pinned key", "bad_signature"
    )


def versions_path() -> Path:
    return agent_dir() / "preset_versions.json"


def _seen_versions() -> dict:
    p = versions_path()
    try:
        seen = json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        # Unreadable high-water mark: fail closed rather than forget it.
        raise PresetVerificationError(
            "preset_versions.json is unreadable", "rollback_state_invalid"
        ) from None
    if not isinstance(seen, dict) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in seen.values()
    ):
        raise PresetVerificationError(  # junk high-water mark: fail closed (R-99)
            "preset_versions.json is malformed", "rollback_state_invalid"
        )
    return seen


def check_version(name: str, version: int) -> None:
    """Refuse anything older than the highest version of `name` already verified."""
    last = int(_seen_versions().get(name, 0))
    if version < last:
        raise PresetVerificationError(
            f"{name} v{version} is older than v{last} already seen (rollback)",
            "rollback",
        )


def record_version(name: str, version: int) -> None:
    seen = _seen_versions()
    if version > int(seen.get(name, 0)):
        seen[name] = version
        atomic_write_json(versions_path(), seen)


def check_and_record_version(name: str, version: int) -> None:
    """Monotonic per name: refuse anything older than the last version seen."""
    check_version(name, version)
    record_version(name, version)


def is_revoked(revocations: dict, name: str, version: int | None) -> bool:
    for r in revocations.get("revoked", []):
        if r.get("preset") == name and (
            r.get("version") is None or r.get("version") == version
        ):
            return True
    return False


MAX_REVOCATION_AGE = dt.timedelta(days=7)
MAX_REVOCATION_SKEW = dt.timedelta(minutes=10)


def revocations_seen_path() -> Path:
    return agent_dir() / "revocations_seen.json"


def _check_revocations_not_replayed(issued: dt.datetime) -> None:
    """The revocation list carries no sequence number, so its signed issued_at is the
    sequence: refuse one older than the newest this machine has accepted (a replayed
    older list could hide a later revocation), then remember the newest."""
    p = revocations_seen_path()
    last = None
    if p.exists():
        try:
            last = dt.datetime.fromisoformat(
                str(json.loads(p.read_text()).get("issued_at")).replace("Z", "+00:00")
            )
        except (OSError, ValueError, AttributeError):
            raise PresetVerificationError(
                "revocations_seen.json is unreadable", "rollback_state_invalid"
            ) from None
    if (
        last is not None
        and last.tzinfo is not None
        and issued < last - MAX_REVOCATION_SKEW
    ):
        raise PresetVerificationError(
            "revocation list is older than one already seen (replay)",
            "revocations_replayed",
        )
    if last is None or issued > last:
        atomic_write_json(p, {"issued_at": issued.isoformat()})


def check_revocations(
    client,
    name: str,
    version: int,
    keys: Keys = None,
    now: dt.datetime | None = None,
    *,
    record: bool = True,
) -> None:
    """Fetch + verify the signed revocation list; refuse a missing, stale (> 7 days),
    future-dated or malformed list, a revoked preset, or a rollback."""
    try:
        revs = verify_bundle(client.get_revocations(), keys)
    except PresetVerificationError:
        raise
    except Exception as exc:  # network etc. => fail closed
        raise PresetVerificationError(
            f"could not fetch revocations: {type(exc).__name__}",
            "revocations_unavailable",
        ) from exc
    try:
        issued = dt.datetime.fromisoformat(
            str(revs.get("issued_at")).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise PresetVerificationError(
            "revocation list has no valid issued_at", "revocations_invalid"
        ) from exc
    now = now or dt.datetime.now(dt.timezone.utc)
    if issued.tzinfo is None or now - issued > MAX_REVOCATION_AGE:
        raise PresetVerificationError("revocation list is stale", "revocations_stale")
    if issued - now > MAX_REVOCATION_SKEW:
        raise PresetVerificationError(
            "revocation list is dated in the future", "revocations_invalid"
        )
    _check_revocations_not_replayed(issued)
    if not isinstance(revs.get("revoked"), list):
        raise PresetVerificationError(
            "revocation list is malformed", "revocations_invalid"
        )
    if is_revoked(revs, name, version):
        raise PresetVerificationError(f"{name} v{version} is revoked", "revoked")
    if record:
        check_and_record_version(name, version)
    else:
        check_version(name, version)


def verify_remote_job(
    client, name: str, version: int, sha256: str, keys: Keys = None
) -> dict:
    """Fetch + verify a published bundle for a job (signature, canonical bytes, sha256,
    name/version). `resolve_signed_preset` adds the schema checks and the cache."""
    check_revocations(client, name, version, keys)
    try:
        env = client.get_preset(name, version)
    except Exception as exc:  # noqa: BLE001
        raise PresetVerificationError(
            f"could not fetch the bundle: {type(exc).__name__}", "fetch_failed"
        ) from exc
    bundle = verify_bundle(env, keys)
    raw = _b64d(env["bundle"])
    if hashlib.sha256(raw).hexdigest() != sha256 or env.get("sha256") not in (
        None,
        sha256,
    ):
        raise PresetVerificationError(
            "bundle sha256 does not match the job", "sha256_mismatch"
        )
    if bundle.get("preset") != name or bundle.get("version") != version:
        raise PresetVerificationError(
            "bundle name/version does not match the job", "name_mismatch"
        )
    return bundle


# ------------------------------------------------------------------ 0.0.12: run signed presets

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]{0,62}$")
MAX_BUNDLE_BYTES = 64 * 1024  # the Worker's publish cap


def cache_dir() -> Path:
    """~/.cache/llm-speed/presets/_signed. `_` can't start a preset name, so this never
    collides with a preset's results dir (~/.cache/llm-speed/presets/NAME)."""
    return CACHE_DIR / "presets" / "_signed"


def _cache_path(sha256: str) -> Path:
    if not isinstance(sha256, str) or not SHA_RE.fullmatch(sha256):
        raise PresetVerificationError("bad sha256", "malformed")
    return cache_dir() / f"{sha256}.json"


def _safe_cache_dir(create: bool) -> Path | None:
    """The cache dir as a real 0700 directory owned by us (never a symlink), or None."""
    d = cache_dir()
    if create:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        st = os.lstat(d)
    except OSError:
        return None
    if not stat.S_ISDIR(st.st_mode) or (
        hasattr(os, "getuid") and st.st_uid != os.getuid()
    ):
        return None
    if stat.S_IMODE(st.st_mode) != 0o700:
        try:
            os.chmod(d, 0o700)
        except OSError:
            return None
    return d


# Names of presets packaged with the CLI (allowlist.REVIEWED_PRESETS): a signed bundle may
# not take one (Security Desk R-96), so a packaged name always means the packaged bytes.
PACKAGED_NAMES = ("flint-40b", "selftest")
_GPU_RX_ATOM = re.compile(r"[A-Za-z0-9 _\-]")
_GPU_RX_COUNT = re.compile(r"\{\d{1,2}(,\d{1,2})?\}")


def safe_gpu_regex(p: Any) -> bool:
    r"""R-97: `requires_gpu_regex` from signed data runs in Python `re` only if it is in a
    bounded subset with no nested quantifiers: literals [A-Za-z0-9 _-], `\d`, `\.`, `\b`,
    groups and `|`, and `+ ? {n} {n,m}` (n, m <= 99) only right after a single literal or
    `\d` (never after a group), at most 3 variable-length repeats in the whole pattern
    (so matching a GPU name is at worst cubic in its ~40 characters, never exponential)."""
    if not isinstance(p, str) or not 0 < len(p) <= 200:
        return False
    i, atom, depth, ranged = 0, False, 0, 0
    while i < len(p):
        c = p[i]
        if c == "\\":
            if i + 1 >= len(p) or p[i + 1] not in "db.":
                return False
            atom, i = p[i + 1] != "b", i + 2
        elif c in "+?":
            if not atom:
                return False
            ranged += c == "+"
            atom, i = False, i + 1
        elif c == "{":
            m = _GPU_RX_COUNT.match(p, i)
            if not m or not atom:
                return False
            ranged += m.group(1) is not None
            atom, i = False, m.end()
        elif c == "(":
            depth, atom, i = depth + 1, False, i + 1
        elif c == ")":
            depth, atom, i = depth - 1, False, i + 1
            if depth < 0:
                return False
        elif c == "|":
            atom, i = False, i + 1
        elif _GPU_RX_ATOM.fullmatch(c):
            atom, i = True, i + 1
        else:
            return False
    if depth or ranged > 3:
        return False
    try:
        re.compile(p)
    except re.error:
        return False
    return True


def check_document(bundle: Any) -> None:
    """The checks a packaged preset gets (shared schema + cross-refs) plus the release
    guard's placeholder rule. A signed `selftest` is refused: it is built in."""
    if not isinstance(bundle, dict) or bundle.get("schema") != "llm-speed-preset/1":
        raise PresetVerificationError(
            "not an llm-speed-preset/1 document", "schema_invalid"
        )
    if bundle.get("preset") in PACKAGED_NAMES:
        raise PresetVerificationError(
            "a signed preset may not reuse a packaged preset's name", "name_reserved"
        )
    try:
        preset_schema.validate_preset(bundle)
    except preset_schema.PresetInvalid as exc:
        raise PresetVerificationError(
            f"schema: {exc}"[:300], "schema_invalid"
        ) from None
    marks = preset_schema.placeholder_markers(bundle)
    if marks:
        raise PresetVerificationError(f"placeholder: {marks[0]}", "placeholder")
    for m in bundle["models"].values():
        rx = m.get("requires_gpu_regex")
        if rx is not None and not safe_gpu_regex(rx):
            raise PresetVerificationError(
                "requires_gpu_regex is outside the safe subset", "unsafe_gpu_regex"
            )


def verify_envelope_for(
    env: Any, name: str, version: int, sha256: str, keys: Keys = None
) -> dict:
    """Signature (pinned key), canonical bytes, sha256, name/version, schema. Returns the preset."""
    if not isinstance(env, dict) or not isinstance(env.get("bundle"), str):
        raise PresetVerificationError("malformed envelope", "malformed")
    if len(env["bundle"]) > MAX_BUNDLE_BYTES * 2:
        raise PresetVerificationError("bundle too large", "malformed")
    bundle = verify_bundle(env, keys)
    raw = _b64d(env["bundle"])
    if hashlib.sha256(raw).hexdigest() != sha256 or env.get("sha256") not in (
        None,
        sha256,
    ):
        raise PresetVerificationError(
            "bundle sha256 does not match the job", "sha256_mismatch"
        )
    if bundle.get("preset") != name or bundle.get("version") != version:
        raise PresetVerificationError(
            "bundle name/version does not match the job", "name_mismatch"
        )
    check_document(bundle)
    return bundle


def _write_cache(env: dict, sha256: str) -> None:
    d = _safe_cache_dir(create=True)
    if d is None:
        raise PresetVerificationError(
            "the signed preset cache is not a private directory", "cache_unsafe"
        )
    keep = {k: env[k] for k in ("bundle", "sig", "key_id") if k in env}
    keep["sha256"] = sha256
    p = _cache_path(sha256)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(keep, f, sort_keys=True)
        os.replace(tmp, p)  # replaces a symlink at p itself, never follows it
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _read_cache(sha256: str) -> dict | None:
    p = _cache_path(sha256)
    if _safe_cache_dir(create=False) is None:
        return None
    try:
        fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None  # missing, or a symlink (ELOOP): not trusted
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BUNDLE_BYTES * 3:
            return None
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            fd = -1
            env = json.loads(f.read())
    except (OSError, ValueError):
        return None
    finally:
        if fd != -1:
            os.close(fd)
    return env if isinstance(env, dict) else None


def load_verified(sha256: str, name: str | None = None, keys: Keys = None) -> dict:
    """A cached signed preset, re-verified on every read (signature, canonical bytes,
    sha256, schema). Used by the agent and by `bench --preset NAME --preset-sha256 SHA`."""
    env = _read_cache(sha256)
    if env is None:
        raise PresetVerificationError("no cached signed preset", "not_cached")
    bundle = verify_bundle(env, keys)
    if hashlib.sha256(_b64d(env["bundle"])).hexdigest() != sha256:
        raise PresetVerificationError(
            "cached bundle sha256 mismatch", "sha256_mismatch"
        )
    if name is not None and bundle.get("preset") != name:
        raise PresetVerificationError(
            "cached bundle is another preset", "name_mismatch"
        )
    check_document(bundle)
    return bundle


def resolve_signed_preset(
    client, name: str, version: int, sha256: str, keys: Keys = None
) -> dict:
    """The preset a job pins, verified end to end; raises PresetVerificationError(code).
    `client` is the agent's QueueClient: it only ever talks to the enrolled api_base
    (no redirects followed)."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise PresetVerificationError("bad preset name", "malformed")
    if not isinstance(sha256, str) or not SHA_RE.fullmatch(sha256):
        raise PresetVerificationError("bad preset sha256", "malformed")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise PresetVerificationError("bad preset version", "malformed")
    # Online, every job, fail closed. The high-water mark is raised only after the
    # bundle verified, so an unverifiable job can't poison it.
    check_revocations(client, name, version, keys, record=False)
    bundle = None
    cached = _read_cache(sha256)
    if cached is not None:
        try:
            bundle = verify_envelope_for(cached, name, version, sha256, keys)
        except PresetVerificationError:
            _cache_path(sha256).unlink(missing_ok=True)  # refetch below; never trusted
    if bundle is None:
        try:
            env = client.get_preset(name, version)
        except Exception as exc:  # noqa: BLE001  network / 404 => fail closed
            too_big = getattr(exc, "code", None) == "response_too_large"
            raise PresetVerificationError(
                f"could not fetch the bundle: {type(exc).__name__}",
                "too_large" if too_big else "fetch_failed",
            ) from exc
        bundle = verify_envelope_for(env, name, version, sha256, keys)
        _write_cache(env, sha256)
    record_version(name, version)
    return bundle
