"""Validate presets against the ONE shared schema (cli/schemas/llm-speed-preset-1.schema.json, also
imported by the Worker) and compute preset identity (name, version, sha256).

Stdlib only: a small JSON Schema subset validator covering exactly the keywords the
shared schema uses (tests cross-check it against the `jsonschema` library). A schema
edit that uses a new keyword fails loudly here instead of being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).parent / "schemas" / "llm-speed-preset-1.schema.json"
PRESETS_DIR = Path(__file__).parent / "presets"

# Exactly the keyword subset the shared schema declares in its $comment (the Worker's
# json-schema-lite.ts supports the same set), so both validators enforce the same rules.
_SUPPORTED = {
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "$defs",
    "type",
    "const",
    "enum",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "properties",
    "required",
    "additionalProperties",
    "patternProperties",
    "propertyNames",
    "anyOf",
    "$ref",
}
_DECIMAL_RE = re.compile(r"^-?\d+\.\d{1,6}$")


class PresetInvalid(ValueError):
    pass


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def canonical_bytes(obj: Any) -> bytes:
    """Canonical serialization for the preset sha256 (same rule the Worker uses)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def preset_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _type_ok(v: Any, t: str) -> bool:
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "string":
        return isinstance(v, str)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "null":
        return v is None
    raise PresetInvalid(f"schema uses unsupported type {t!r}")


def _resolve(ref: str, root: dict) -> dict:
    if not ref.startswith("#/$defs/"):
        raise PresetInvalid(f"unsupported $ref {ref!r}")
    return root["$defs"][ref[len("#/$defs/") :]]


def _errors(v: Any, s: dict, root: dict, path: str) -> list[str]:
    unknown = set(s) - _SUPPORTED
    if unknown:
        raise PresetInvalid(
            f"schema keyword(s) not supported by the CLI validator: {sorted(unknown)}"
        )
    if "$ref" in s:
        return _errors(v, _resolve(s["$ref"], root), root, path)
    errs: list[str] = []
    if "type" in s and not _type_ok(v, s["type"]):
        return [f"{path}: expected {s['type']}"]
    if "const" in s and v != s["const"]:
        errs.append(f"{path}: must be {s['const']!r}")
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: must be one of {s['enum']}")
    if "anyOf" in s and not any(not _errors(v, sub, root, path) for sub in s["anyOf"]):
        errs.append(f"{path}: matches none of the allowed alternatives")
    if isinstance(v, str):
        if len(v) < s.get("minLength", 0):
            errs.append(f"{path}: too short")
        if "maxLength" in s and len(v) > s["maxLength"]:
            errs.append(f"{path}: too long")
        if "pattern" in s and not re.search(s["pattern"], v):
            errs.append(f"{path}: does not match {s['pattern']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            errs.append(f"{path}: below {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            errs.append(f"{path}: above {s['maximum']}")
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0):
            errs.append(f"{path}: needs at least {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            errs.append(f"{path}: at most {s['maxItems']} items")
        if s.get("uniqueItems") and len(
            {json.dumps(x, sort_keys=True) for x in v}
        ) != len(v):
            errs.append(f"{path}: items must be unique")
        if "items" in s:
            for i, item in enumerate(v):
                errs += _errors(item, s["items"], root, f"{path}[{i}]")
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}: missing {k!r}")
        if len(v) < s.get("minProperties", 0):
            errs.append(f"{path}: needs at least {s['minProperties']} keys")
        props, pats = s.get("properties", {}), s.get("patternProperties", {})
        for k, item in v.items():
            if "propertyNames" in s:
                errs += _errors(k, s["propertyNames"], root, f"{path}.<key {k}>")
            matched = False
            if k in props:
                matched = True
                errs += _errors(item, props[k], root, f"{path}.{k}")
            for pat, sub in pats.items():
                if re.search(pat, k):
                    matched = True
                    errs += _errors(item, sub, root, f"{path}.{k}")
            if not matched:
                ap = s.get("additionalProperties", True)
                if ap is False:
                    errs.append(f"{path}: unexpected key {k!r}")
                elif isinstance(ap, dict):
                    errs += _errors(item, ap, root, f"{path}.{k}")
    return errs


def _semantic_errors(p: dict) -> list[str]:
    errs: list[str] = []
    models = p.get("models", {})
    for mid, m in models.items():
        if m.get("backend") == "llama.cpp" and "file" not in m:
            errs.append(f"$.models.{mid}: llama.cpp entries need 'file'")
        if m.get("backend") == "mlx" and "dir" not in m:
            errs.append(f"$.models.{mid}: mlx entries need 'dir'")
    for prof, d in (p.get("profiles") or {}).items():
        for key in ("entries", "fallback", "control"):
            for mid in d.get(key, []):
                if mid not in models:
                    errs.append(f"$.profiles.{prof}.{key}: unknown model id {mid!r}")
        for mid in d.get("contexts") or {}:
            if mid not in models:
                errs.append(f"$.profiles.{prof}.contexts: unknown model id {mid!r}")
        probe = d.get("offload_probe")
        if probe and probe.get("entry") not in models:
            errs.append(
                f"$.profiles.{prof}.offload_probe: unknown model id {probe.get('entry')!r}"
            )

    def walk(v: Any, path: str) -> None:
        if isinstance(v, float) and (v.is_integer() or not _DECIMAL_RE.match(repr(v))):
            errs.append(
                f"{path}: write integral numbers without '.0'; decimals need <= 6 fractional digits"
            )
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")

    walk(p, "$")
    return errs


def validate_preset(p: Any) -> None:
    root = load_schema()
    errs = _errors(p, root, root, "$")
    if not errs and isinstance(p, dict):
        errs = _semantic_errors(p)
    if errs:
        raise PresetInvalid("; ".join(errs[:8]))


PLACEHOLDER_KEY = "_placeholder"
# A hash that is one character repeated (000…, 111…, ccc…) is a staging value, never a real
# git revision (40) or sha256 (64).
_FAKE_HEX_RE = re.compile(r"([0-9a-fA-F])\1{39}|([0-9a-fA-F])\2{63}")
PLACEHOLDER_MAX_BYTES = 1024  # no real model file is this small
PLACEHOLDER_LLAMA_BUILD = 999999  # sentinel "build not known yet"


def placeholder_markers(p: Any) -> list[str]:
    """Why a preset is still a staging placeholder (empty = real). Markers: a top-level
    `_placeholder` note, the word DUMMY in any string, a revision/sha256 made of one
    repeated character, a single-file llama.cpp entry without sha256, a model of
    <= 1024 bytes, or min_llama_build >= 999999.
    Used by the agent (never advertised or run), `bench --preset` (refused) and
    tools/release_guard.py (release refused), so dummy values can't ship or upload."""
    out: list[str] = []
    if isinstance(p, dict) and PLACEHOLDER_KEY in p:
        out.append(f"has a {PLACEHOLDER_KEY} note")

    def walk(v: Any, path: str) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{path}.{k}")
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
        elif isinstance(v, str):
            if "DUMMY" in v.upper():
                out.append(f"{path} says DUMMY")
            elif _FAKE_HEX_RE.fullmatch(v):
                out.append(f"{path} is a repeated-character placeholder hash")

    walk(p, "$")
    if isinstance(p, dict):
        build = p.get("min_llama_build")
        if (
            isinstance(build, int)
            and not isinstance(build, bool)
            and build >= PLACEHOLDER_LLAMA_BUILD
        ):
            out.append(f"$.min_llama_build is the {PLACEHOLDER_LLAMA_BUILD} sentinel")
        models = p.get("models")
        for k, m in (models if isinstance(models, dict) else {}).items():
            if not isinstance(m, dict):
                continue
            if m.get("backend") == "llama.cpp" and "file" in m and not m.get("sha256"):
                out.append(
                    f"$.models.{k} has no sha256"
                )  # every single-file GGUF must be pinned
            size = m.get("bytes")
            if (
                isinstance(size, int)
                and not isinstance(size, bool)
                and size <= PLACEHOLDER_MAX_BYTES
            ):
                out.append(f"$.models.{k}.bytes is a placeholder size ({size})")
    return out


def identity(p: dict) -> dict[str, Any]:
    return {"name": p["preset"], "version": p["version"], "sha256": preset_sha256(p)}


def packaged_preset(name: str) -> dict | None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._\-]{0,79}", name or ""):
        return None
    f = PRESETS_DIR / f"{name}.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def packaged_identities(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Identities of installed presets that validate (invalid ones are left out)."""
    out = []
    files = sorted(PRESETS_DIR.glob("*.json")) if PRESETS_DIR.is_dir() else []
    for f in files:
        if names is not None and f.stem not in names:
            continue
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
            validate_preset(p)
        except (ValueError, OSError):
            continue
        if p.get("preset") == f.stem:
            out.append(identity(p))
    return out
