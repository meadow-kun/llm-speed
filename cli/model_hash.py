"""Full-file sha256 of a model file, for `bench --lab-session` pins only (R-195).

The ordinary `model.digest` stays the 16-hex prefix of the first 256 KiB
(`cli/drivers/llama_cpp.py:_digest_prefix`); nothing here changes it for a non-lab
run. A lab run is pinned at mint to the full-file sha256 of each model
(`lab_sessions.model_sha256s[].sha256`), and the Worker's `pinsMatch` compares every
`results[].model.digest` to it, so a lab upload carries the full 64-hex digest there.

The file is streamed in 8 MiB chunks, never loaded into memory. The result is cached
in `CACHE_DIR/model_sha256.json`, keyed by (absolute path, size, st_mtime_ns). An
entry holds only those three, the sha256 and the hash duration in seconds: no token,
no run data. A change of size or mtime is a miss. A file whose size or mtime changes
while it is being hashed is refused and not cached.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CACHE_NAME = "model_sha256.json"
CACHE_VERSION = 1
MAX_ENTRIES = 64
CHUNK_BYTES = 8 * 1024 * 1024
_SHA_RE = re.compile(r"[0-9a-f]{64}", re.ASCII)
_ENTRY_KEYS = ["duration_s", "mtime_ns", "path", "sha256", "size"]


class ModelHashError(ValueError):
    """The full-file hash could not be produced."""


@dataclasses.dataclass(frozen=True)
class FileHash:
    path: str  # absolute path (local cache key only; never uploaded)
    size: int
    mtime_ns: int
    sha256: str
    duration_s: float  # how long the hash took when it was computed
    cached: bool  # True when served from the cache


def cache_path() -> Path:
    from . import config

    return Path(config.CACHE_DIR) / CACHE_NAME


def _valid_entry(e: Any) -> bool:
    return (
        isinstance(e, dict)
        and sorted(e) == _ENTRY_KEYS
        and isinstance(e["path"], str)
        and all(
            isinstance(e[k], int) and not isinstance(e[k], bool)
            for k in ("size", "mtime_ns")
        )
        and isinstance(e["sha256"], str)
        and _SHA_RE.fullmatch(e["sha256"]) is not None
        and isinstance(e["duration_s"], (int, float))
        and not isinstance(e["duration_s"], bool)
    )


def _load(path: Path) -> list[dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(obj, dict) or obj.get("version") != CACHE_VERSION:
        return []
    entries = obj.get("entries")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if _valid_entry(e)]


def _store(path: Path, entries: list[dict[str, Any]]) -> None:
    """Atomic 0600 write inside a 0700 directory. Best effort: a failed write only
    means the next run hashes again."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {"version": CACHE_VERSION, "entries": entries[-MAX_ENTRIES:]},
                f,
                indent=1,
                sort_keys=True,
            )
        os.replace(tmp, path)
    except OSError as exc:
        log.debug("could not write the model hash cache: %s", exc)


def _stream_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def full_sha256(path: str | os.PathLike[str]) -> FileHash:
    """The full-file sha256 of `path`, from the cache when (path, size, mtime) match."""
    p = Path(path).expanduser()
    try:
        p = p.resolve(strict=True)
        st = os.stat(p)
    except OSError as exc:
        raise ModelHashError(
            f"cannot read model file {p.name}: {exc.strerror}"
        ) from None
    if not p.is_file():
        raise ModelHashError(f"model {p.name} is not a regular file")
    key = (str(p), st.st_size, st.st_mtime_ns)
    cpath = cache_path()
    entries = _load(cpath)
    for e in entries:
        if (e["path"], e["size"], e["mtime_ns"]) == key:
            log.info(
                "model sha256 for %s: cache hit (hashed earlier in %.2f s)",
                p.name,
                e["duration_s"],
            )
            return FileHash(*key, e["sha256"], float(e["duration_s"]), True)
    t0 = time.monotonic()
    try:
        digest = _stream_sha256(p)
        after = os.stat(p)
    except OSError as exc:
        raise ModelHashError(
            f"cannot read model file {p.name}: {exc.strerror}"
        ) from None
    duration = round(time.monotonic() - t0, 3)
    if (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        raise ModelHashError(
            f"model file {p.name} changed while it was being hashed; not pinning it"
        )
    log.info(
        "model sha256 for %s: hashed %d bytes in %.2f s", p.name, st.st_size, duration
    )
    # Drop stale entries for the same path (a changed file is a new key).
    kept = [e for e in entries if e["path"] != key[0]]
    kept.append(
        {
            "path": key[0],
            "size": key[1],
            "mtime_ns": key[2],
            "sha256": digest,
            "duration_s": duration,
        }
    )
    _store(cpath, kept)
    return FileHash(*key, digest, duration, False)


__all__ = ["FileHash", "ModelHashError", "cache_path", "full_sha256"]
