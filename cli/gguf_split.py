"""Split / multi-part GGUF files (llama.cpp ``<name>-00001-of-00003.gguf``).

llama.cpp treats that set as one model: ``-m`` is the first shard, and it opens
the rest from the same directory using the name. Ollama is not a GGUF-path
backend (it serves an installed tag), so this applies to llama.cpp only:
``llm-speed bench --model`` and preset / fit-sweep entries.

A preset entry may list the shards in ``files`` or name any one shard in
``file``. ``bytes`` is the total of every shard. Each shard in ``files``
carries its own ``sha256`` (``{"file", "sha256"}``); a supplied hash that
does not match the bytes fails closed before llama-server starts. Nothing
here adds an upload field: the shard basenames ride the existing
``extras.model_file`` string.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path, PurePath

# `$` matches before a trailing newline under Python re and JavaScript. The
# lookahead requires the string to end on the 64th hex digit (R-153).
_SHA_RE = re.compile(r"^[0-9a-f]{64}(?![\s\S])")

# llama.cpp common/common.cpp: exactly five digits, count >= 2, index in 1..count.
_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<idx>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$",
    re.IGNORECASE,
)

# fit_attempts.model_file is 200 characters; the privacy walker refuses any
# non-allowlisted string over 256. Stay inside the tighter of the two.
MODEL_FILE_MAX = 200


class SplitSpecError(ValueError):
    """``files`` is not one complete llama.cpp shard set."""


class MissingShardError(RuntimeError):
    """A shard named by the set is not on disk next to the others."""

    def __init__(self, missing: list[Path]) -> None:
        self.missing = tuple(missing)
        names = [p.name for p in missing]
        listed = ", ".join(names)
        msg = f"split GGUF is missing {len(names)} shard(s): {listed}"
        if len(msg) > 180:
            msg = msg[:179] + "…"
        super().__init__(
            msg
            + ". Every part must sit beside the first shard before llama.cpp can load the set."
        )


class ShardSet:
    """Every shard of one split, in index order, whether or not each file exists."""

    def __init__(self, directory: Path, prefix: str, count: int, suffix: str) -> None:
        self.directory = directory
        self.prefix = prefix
        self.count = count
        self.suffix = suffix
        self.shards = tuple(
            directory / f"{prefix}-{i:05d}-of-{count:05d}{suffix}"
            for i in range(1, count + 1)
        )

    @property
    def first(self) -> Path:
        return self.shards[0]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.shards)

    def missing(self) -> tuple[Path, ...]:
        return tuple(p for p in self.shards if not p.is_file())

    def complete(self) -> bool:
        return not self.missing()

    def total_bytes(self) -> int:
        """Sum of the shards that exist. Callers that need the whole model
        check ``complete()`` first so a partial set is not treated as the model."""
        total = 0
        for p in self.shards:
            if p.is_file():
                total += p.stat().st_size
        return total


def parse_split_name(name: str) -> tuple[str, int, int] | None:
    """``(prefix, index, count)`` or None. ``count`` is at least 2 and the index
    is inside ``1..count``. The prefix keeps the filename's own casing."""
    base = PurePath(str(name).replace("\\", "/")).name
    m = _SPLIT_RE.fullmatch(base)
    if not m:
        return None
    idx = int(m.group("idx"))
    count = int(m.group("count"))
    if count < 2 or idx < 1 or idx > count:
        return None
    return m.group("prefix"), idx, count


def _suffix(name: str) -> str:
    i = name.lower().rfind(".gguf")
    return name[i:] if i >= 0 else ".gguf"


def shard_set_for(path: Path) -> ShardSet | None:
    """The set implied by ``path``'s filename, in that file's directory. None
    when ``path`` is an ordinary single-file GGUF."""
    parsed = parse_split_name(path.name)
    if parsed is None:
        return None
    prefix, _idx, count = parsed
    return ShardSet(path.parent, prefix, count, _suffix(path.name))


def require_complete(path: Path) -> ShardSet | None:
    """The set, if ``path`` is a split shard and every part is on disk.
    None for a single-file GGUF. Raises ``MissingShardError`` naming each
    absent shard (basenames only)."""
    ss = shard_set_for(path)
    if ss is None:
        return None
    missing = ss.missing()
    if missing:
        raise MissingShardError(list(missing))
    return ss


def keep_gguf_filename(path: Path) -> Path:
    """Absolute path that still ends in the ``.gguf`` shard name.

    Hugging Face stores the bytes as a symlink into ``blobs/<hash>``. Resolving
    that link drops the ``-00001-of-00003.gguf`` name, and llama.cpp then will
    not open the other shards.
    """
    try:
        path = path.expanduser().absolute()
    except OSError:
        return path
    if path.is_symlink():
        return path
    try:
        resolved = path.resolve()
    except OSError:
        return path
    if resolved.name.lower().endswith(".gguf"):
        return resolved
    return path


def llama_model_path(path: Path) -> Path:
    """Path to pass as llama-server ``-m``: the first shard, filename intact."""
    ss = shard_set_for(path)
    target = ss.first if ss is not None else path
    return keep_gguf_filename(target)


def shard_list_label(names: list[str] | tuple[str, ...]) -> str:
    """Exact shard basenames for ``extras.model_file``. The joined list when it
    fits in the existing field, otherwise the first shard (its
    ``-00001-of-000NN`` name identifies the rest). Never a path."""
    items = list(names)
    if not items:
        return ""
    joined = ",".join(items)
    if len(joined) <= MODEL_FILE_MAX:
        return joined
    return items[0]


def _basename(value: object) -> str | None:
    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        return None
    return PurePath(value).name


def exact_sha256(value: object) -> str | None:
    """The digest when it is exactly 64 lowercase hex and nothing else.

    A trailing LF or CRLF is not a digest. ``None`` means the value is not a
    digest; callers that required one must raise, not treat this as a match.
    """
    if isinstance(value, str) and _SHA_RE.fullmatch(value):
        return value
    return None


def _shard_item(item: object) -> tuple[str, str | None]:
    """``(basename, sha256 or None)`` from a ``files`` entry.

    A string is a name with no hash. An object carries ``file`` and, when
    present, ``sha256``. A sha256 that is not exactly 64 lowercase hex raises.
    """
    if isinstance(item, str):
        name = _basename(item)
        if name is None:
            raise SplitSpecError(
                "split GGUF `files` must be GGUF basenames, one shard set"
            )
        return name, None
    if isinstance(item, dict):
        name = (
            _basename(item.get("file")) if isinstance(item.get("file"), str) else None
        )
        if name is None:
            raise SplitSpecError("split GGUF `files` entries need a GGUF file basename")
        if "sha256" not in item:
            return name, None
        digest = exact_sha256(item.get("sha256"))
        if digest is None:
            raise SplitSpecError(f"{name}: sha256 must be 64 lowercase hex")
        return name, digest
    raise SplitSpecError("split GGUF `files` must be GGUF basenames, one shard set")


def shard_names_for_entry(entry: Mapping) -> list[str] | None:
    """Basenames of a split set declared by a preset / plan entry, in index
    order. None when the entry is one ordinary GGUF. Raises ``SplitSpecError``
    when ``files`` is present and is not one complete set.

    A lone ``file`` whose name is ``<prefix>-NNNNN-of-MMMMM.gguf`` is a split
    even without ``files``: the other names are implied.
    """
    raw = entry.get("files")
    declared: list[str] | None = None
    if isinstance(raw, list) and raw:
        declared = [_shard_item(item)[0] for item in raw]
    file_name = (
        _basename(entry.get("file")) if isinstance(entry.get("file"), str) else None
    )
    if isinstance(entry.get("file"), str) and entry.get("file") and file_name is None:
        raise SplitSpecError("`file` must be a GGUF basename")

    if (
        declared is not None
        and len(declared) == 1
        and parse_split_name(declared[0]) is None
    ):
        if file_name and file_name != declared[0]:
            raise SplitSpecError("`file` does not match `files`")
        return None
    if declared is not None:
        names = _complete_declared(declared)
        if file_name and file_name not in names:
            raise SplitSpecError("`file` is not one of the split shards")
        return names
    parsed = parse_split_name(file_name) if file_name else None
    if file_name and parsed:
        prefix, _idx, count = parsed
        return [
            f"{prefix}-{i:05d}-of-{count:05d}{_suffix(file_name)}"
            for i in range(1, count + 1)
        ]
    return None


def _complete_declared(declared: list[str]) -> list[str]:
    seen: set[str] = set()
    for name in declared:
        if name in seen:
            raise SplitSpecError(f"duplicate shard {name}")
        seen.add(name)
        if parse_split_name(name) is None:
            raise SplitSpecError(
                f"{name} is not a llama.cpp split shard (<name>-00001-of-000NN.gguf)"
            )
    first = parse_split_name(declared[0])
    assert first is not None
    prefix, _idx, count = first
    suffix = _suffix(declared[0])
    expected = [f"{prefix}-{i:05d}-of-{count:05d}{suffix}" for i in range(1, count + 1)]
    for name in declared:
        if name not in expected:
            raise SplitSpecError(f"extra shard {name}")
    if set(declared) != set(expected):
        raise SplitSpecError(
            f"split GGUF `files` is not a complete set of {count} shards"
        )
    return expected


def shard_hashes(entry: Mapping) -> dict[str, str | None] | None:
    """Per-shard sha256 for a split, in shard order. None when the entry is
    one ordinary GGUF. A shard listed as a bare name, or implied by ``file``
    and not listed in ``files``, has ``None`` (the hash was not supplied)."""
    names = shard_names_for_entry(entry)
    if names is None:
        return None
    found: dict[str, str | None] = {}
    raw = entry.get("files")
    if isinstance(raw, list):
        for item in raw:
            name, digest = _shard_item(item)
            found[name] = digest
    return {name: found.get(name) for name in names}


def missing_hashes(entry: Mapping) -> list[str]:
    """Shard basenames with no sha256. Empty for a single-file entry (that
    case is the entry's own ``sha256`` string) and for a split whose every
    shard carries one."""
    hashes = shard_hashes(entry)
    if not hashes:
        return []
    return [name for name, digest in hashes.items() if not digest]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _single_file_item(entry: Mapping) -> tuple[str, str | None] | None:
    """A one-item ``files`` list that names an ordinary GGUF, or None.

    That digest is a pin. It is not ignored in favour of a top-level
    ``sha256`` (R-154). A disagreement between the two raises.
    """
    raw = entry.get("files")
    if not isinstance(raw, list) or len(raw) != 1:
        return None
    name, digest = _shard_item(raw[0])
    if parse_split_name(name) is not None:
        return None
    return name, digest


def pin_errors(entry: Mapping) -> list[str]:
    """Shape problems for a llama.cpp GGUF pin. Empty when the pins that are
    present are exact 64-hex digests and agree. Does not read the files."""
    if entry.get("backend") not in (None, "llama.cpp"):
        return []
    if "file" not in entry and "files" not in entry:
        return []
    try:
        single = _single_file_item(entry)
        names = shard_names_for_entry(entry)
    except SplitSpecError as exc:
        return [str(exc)]
    if names is not None:
        try:
            hashes = shard_hashes(entry) or {}
        except SplitSpecError as exc:
            return [str(exc)]
        missing = [name for name, digest in hashes.items() if not digest]
        if missing:
            return [f"no sha256 for {missing[0]}"]
        return _top_level_sha_error(entry)
    top_err = _top_level_sha_error(entry)
    if top_err:
        return top_err
    top = exact_sha256(entry.get("sha256")) if "sha256" in entry else None
    if single is not None:
        name, digest = single
        if digest is None and top is None:
            return [f"no sha256 for {name}"]
        if digest is not None and top is not None and digest != top:
            return ["sha256 conflict: files[0] does not match sha256"]
        return []
    if "file" in entry and top is None and "sha256" not in entry:
        return ["has no sha256"]
    return []


def _top_level_sha_error(entry: Mapping) -> list[str]:
    if "sha256" not in entry:
        return []
    if exact_sha256(entry.get("sha256")) is None:
        return ["sha256 must be 64 lowercase hex"]
    return []


def verify_pins(entry: Mapping, model_path: Path) -> list[str]:
    """Basenames whose bytes matched their pin.

    Raises ``SplitSpecError`` on an invalid pin, a missing required pin, a
    conflict, or a byte mismatch. Returns an empty list only for a single
    file that declares no pin at all (that launch is not hash-verified).
    A returned name means the digest comparison succeeded.
    """
    problems = pin_errors(entry)
    if problems:
        raise SplitSpecError(problems[0])
    names = shard_names_for_entry(entry)
    if names is not None:
        compared: list[str] = []
        hashes = shard_hashes(entry) or {}
        for name in names:
            digest = hashes.get(name)
            path = model_path.parent / name
            if not isinstance(digest, str) or not path.is_file():
                raise SplitSpecError(f"no sha256 for {name}")
            if _sha256_file(path) != digest:
                raise SplitSpecError(f"sha256 mismatch: {name}")
            compared.append(name)
        return compared
    pin = _effective_single_pin(entry)
    if pin is None:
        return []
    path = model_path if model_path.is_file() else None
    label = path.name if path is not None else _single_name(entry)
    if path is None or _sha256_file(path) != pin:
        raise SplitSpecError(f"sha256 mismatch: {label}")
    return [label]


def _single_name(entry: Mapping) -> str:
    single = _single_file_item(entry)
    if single is not None:
        return single[0]
    file_name = (
        _basename(entry.get("file")) if isinstance(entry.get("file"), str) else None
    )
    return file_name or "model.gguf"


def _effective_single_pin(entry: Mapping) -> str | None:
    """The one digest for a non-split file, after ``pin_errors`` has passed."""
    single = _single_file_item(entry)
    top = exact_sha256(entry.get("sha256")) if "sha256" in entry else None
    if single is not None and single[1] is not None:
        return single[1]
    return top


def is_split_entry(entry: Mapping) -> bool:
    """True when the entry is a multi-part GGUF (including a malformed ``files``
    list, which is still not a single-file GGUF)."""
    try:
        return shard_names_for_entry(entry) is not None
    except SplitSpecError:
        return True


def primary_basename(entry: Mapping) -> str:
    """The shard llama.cpp is given: ``-00001-of-…``, or the single file."""
    try:
        names = shard_names_for_entry(entry)
    except SplitSpecError:
        names = None
    if names:
        return names[0]
    file_name = (
        _basename(entry.get("file")) if isinstance(entry.get("file"), str) else None
    )
    if file_name:
        return file_name
    raw = entry.get("files")
    if isinstance(raw, list) and raw:
        try:
            return _shard_item(raw[0])[0]
        except SplitSpecError:
            pass
    file = entry.get("file")
    if isinstance(file, str) and file:
        return PurePath(str(file).replace("\\", "/")).name
    raise KeyError("file")


def sum_tree_bytes(names: list[str], tree: list[dict]) -> int | None:
    """Total size of every named shard in a Hugging Face tree listing, or None
    if any shard is absent. Matches on the basename so a repo subdirectory still
    counts."""
    found: dict[str, int] = {}
    wanted = set(names)
    for row in tree:
        if not isinstance(row, dict):
            continue
        base = str(row.get("path") or "").rsplit("/", 1)[-1]
        if base not in wanted or base in found:
            continue
        size = row.get("size")
        if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
            found[base] = size
    if len(found) != len(names):
        return None
    return sum(found[n] for n in names)


def on_disk_bytes(entry: Mapping, path: Path) -> int:
    """Bytes of this entry already on disk next to ``path`` (one file, or the
    shards that exist). A missing file is 0. Does not follow the 'complete set'
    rule: disk accounting wants the partial sum."""
    try:
        names = shard_names_for_entry(entry)
    except SplitSpecError:
        names = None
    if names:
        total = 0
        for name in names:
            part = path.parent / name
            if part.is_file():
                total += part.stat().st_size
        return total
    if path.is_file():
        return path.stat().st_size
    return 0


def weight_bytes(entry: Mapping, path: Path | None) -> int | None:
    """Size the fit / VRAM check should use.

    A complete local shard set contributes the sum of every shard (never one
    part). A single file on disk contributes its size. The entry's authored
    ``bytes`` is the whole model when the files are not all here yet. When both
    are known, the larger one wins so a per-shard ``bytes`` cannot sneak a
    bigger set past the check. None when nothing is known.
    """
    local: int | None = None
    if path is not None:
        try:
            names = shard_names_for_entry(entry)
        except SplitSpecError:
            names = None
        if names:
            parts = [path.parent / n for n in names]
            if all(p.is_file() for p in parts):
                local = sum(p.stat().st_size for p in parts)
        elif path.is_file():
            local = path.stat().st_size
    authored = entry.get("bytes")
    authored_n = (
        int(authored)
        if isinstance(authored, int) and not isinstance(authored, bool)
        else None
    )
    if local is not None and authored_n is not None:
        return max(local, authored_n)
    return local if local is not None else authored_n
