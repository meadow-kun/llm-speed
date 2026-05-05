"""
Aggregation: collapse the /v1/results listing into one row per
(model_slug, hardware_slug) — the "best decode_tps cell" view used by
every tool except `compare`.

The listing endpoint returns each run's headline only (top_decode_tps,
top_model_name). That's enough for the MCP tools, since they all surface
a single best-tok/s number per cell. We do NOT pull /v1/results/{id} for
every cell — that would be N+1 against the API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .slugs import (
    canonical_hardware_slug,
    canonical_model_slug,
    clean_model_display_name,
)

# Strip C0 control bytes (except \t and \n), bidi-format characters, and
# zero-width characters before any string flows into an LLM client. Mirrors
# `CTRL_RE` / `PHISH_CAP` in `web/lib/seo.ts` + `web/lib/distribution.ts`.
# Today the worker validates inputs at write time, but our display strings
# come from the API verbatim — if a future API regression lets a hostile
# string land in `top_model_name` / `accelerator_summary`, this strips the
# prompt-injection payload before the LLM client ever sees it.
_UNSAFE_DISPLAY_RE = re.compile(
    "["
    "\x00-\x08"  # C0 controls (kept: \t \n)
    "\x0b\x0c"
    "\x0e-\x1f"
    "‪-‮"  # bidi formatting
    "⁦-⁩"  # bidi isolates
    "​-‍"  # zero-width space / ZWNJ / ZWJ
    "﻿"        # BOM / zero-width nbsp
    "]"
)
_DISPLAY_MAX_LEN = 256


def _safe_display(s: str | None, *, max_len: int = _DISPLAY_MAX_LEN) -> str:
    """Sanitize a free-text string before returning it to an LLM client.

    LLM clients have no raw-string-vs-attribute distinction, so any control
    or bidi character we pass through becomes a possible prompt-injection
    surface. Strip them; clamp length so a hostile 1MB string can't
    monopolise the context window.
    """
    if not s:
        return ""
    cleaned = _UNSAFE_DISPLAY_RE.sub("", s)
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1] + "…"
    return cleaned

# Keys defined in the API response.
_DECODE = "top_decode_tps"


# Hosted/proprietary models we surface in /vs and the leaderboard. Anything
# whose display name starts with one of these org prefixes is considered
# hosted ("not runnable on your laptop"). Mirrors the website's framing.
_HOSTED_PREFIXES = ("anthropic/", "openai/", "google/", "deepseek/", "qwen/")


@dataclass(frozen=True)
class Cell:
    """One (model, hardware) row, anchored to the run that backs it."""

    model_name: str
    model_slug: str
    model_display: str
    hardware_label: str
    hardware_slug: str
    backend: str
    decode_tps: float
    workload: str | None
    run_id: str
    received_at: str
    locality: str  # "local" | "hosted"


def _is_hosted(model_name: str) -> bool:
    lower = (model_name or "").lower()
    return any(lower.startswith(p) for p in _HOSTED_PREFIXES)


def _model_size_b(model_name: str | None) -> float | None:
    """Best-effort parse of model parameter count in billions from the
    model_name. Returns None if we can't tell (which is most of the time
    for hosted models — that's fine, recommend() filters those out).
    """
    if not model_name:
        return None
    # Match patterns like "70B", "405b", "0.5B", "32B-Instruct".
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model_name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def best_cells(summaries: Iterable[dict[str, Any]]) -> list[Cell]:
    """Collapse a listing into one Cell per (model_slug, hardware_slug),
    keeping the row with the highest decode_tps. Sorted desc by tok/s.
    """
    best: dict[tuple[str, str], Cell] = {}
    for s in summaries:
        decode = s.get(_DECODE)
        model = s.get("top_model_name")
        hardware = s.get("accelerator_summary")
        if decode is None or not model or not hardware:
            continue
        m_slug = canonical_model_slug(model)
        h_slug = canonical_hardware_slug(hardware)
        if not m_slug or not h_slug:
            continue
        key = (m_slug, h_slug)
        prev = best.get(key)
        if prev is None or decode > prev.decode_tps:
            best[key] = Cell(
                model_name=model,
                model_slug=m_slug,
                model_display=clean_model_display_name(model) or model,
                hardware_label=hardware,
                hardware_slug=h_slug,
                backend=s.get("top_backend") or "unknown",
                decode_tps=float(decode),
                workload=s.get("top_workload"),
                run_id=str(s["id"]),
                received_at=str(s.get("received_at", "")),
                locality="hosted" if _is_hosted(model) else "local",
            )
    return sorted(best.values(), key=lambda c: c.decode_tps, reverse=True)


def cell_to_dict(c: Cell) -> dict[str, Any]:
    """Serialize a Cell for an MCP tool response.

    Free-text fields (`model`, `model_canonical`, `hardware`, `backend`,
    `workload`) come from the API verbatim. They're sanitised through
    `_safe_display` before reaching the LLM client so that a future
    worker input-validation bypass cannot use this surface as a
    prompt-injection vector.
    """
    return {
        "model": _safe_display(c.model_display),
        "model_canonical": _safe_display(c.model_name),
        "model_slug": _safe_display(c.model_slug, max_len=128),
        "hardware": _safe_display(c.hardware_label),
        "hardware_slug": _safe_display(c.hardware_slug, max_len=128),
        "backend": _safe_display(c.backend, max_len=64),
        "decode_tps": round(c.decode_tps, 2),
        "workload": _safe_display(c.workload, max_len=64),
        "run_id": _safe_display(c.run_id, max_len=64),
        "received_at": _safe_display(c.received_at, max_len=64),
        "locality": _safe_display(c.locality, max_len=16),
        "model_size_b": _model_size_b(c.model_name),
    }


def model_size_b(model_name: str | None) -> float | None:
    """Public re-export of the size parser for recommend()."""
    return _model_size_b(model_name)
