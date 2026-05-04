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
    """Serialize a Cell for an MCP tool response."""
    return {
        "model": c.model_display,
        "model_canonical": c.model_name,
        "model_slug": c.model_slug,
        "hardware": c.hardware_label,
        "hardware_slug": c.hardware_slug,
        "backend": c.backend,
        "decode_tps": round(c.decode_tps, 2),
        "workload": c.workload,
        "run_id": c.run_id,
        "received_at": c.received_at,
        "locality": c.locality,
        "model_size_b": _model_size_b(c.model_name),
    }


def model_size_b(model_name: str | None) -> float | None:
    """Public re-export of the size parser for recommend()."""
    return _model_size_b(model_name)
