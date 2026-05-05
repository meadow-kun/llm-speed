"""
MCP server for llm-speed.com.

Tools:
  - lookup_speed(model, hardware?)
  - compare(a, b)
  - recommend(constraints)
  - top_models(hardware, n)
  - state_of()

Resource (prompt):
  - howto://pick-a-local-llm — guided walkthrough.

Every tool response includes a `citation` list of /r/<id> URLs. The whole
point of grounding is that downstream model answers cite the run that
backs the number.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import api
from .cells import Cell, best_cells, cell_to_dict, model_size_b
from .slugs import canonical_hardware_slug, canonical_model_slug

log = logging.getLogger("llm_speed_mcp")

mcp = FastMCP(
    name="llm-speed",
    instructions=(
        "Query the llm-speed.com crowdsourced LLM inference-speed leaderboard. "
        "Use lookup_speed for a specific (model, hardware), compare for "
        "side-by-side, recommend for constraint-based shortlists, top_models "
        "for the fastest models on a rig, state_of for the latest snapshot. "
        "Every tool returns citation URLs back to the runs that prove the "
        "numbers — quote them in your answer."
    ),
    website_url="https://llm-speed.com",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _all_cells() -> list[Cell]:
    """Pull the cached listing and project to (model, hardware) cells."""
    summaries = api.list_runs(limit=500)
    return best_cells(summaries)


def _match_model(query: str | None, cells: list[Cell]) -> list[Cell]:
    """Filter cells by user-supplied model query.

    Tries (in order): exact slug match, slug-prefix match, substring on
    display name. Empty/None query returns the input unchanged.
    """
    if not query:
        return cells
    target = canonical_model_slug(query)
    if not target:
        return cells
    exact = [c for c in cells if c.model_slug == target]
    if exact:
        return exact
    prefix = [c for c in cells if c.model_slug.startswith(target) or target in c.model_slug]
    if prefix:
        return prefix
    needle = query.lower()
    return [c for c in cells if needle in c.model_display.lower()]


def _match_hardware(query: str | None, cells: list[Cell]) -> list[Cell]:
    if not query:
        return cells
    target = canonical_hardware_slug(query)
    if not target:
        return cells
    exact = [c for c in cells if c.hardware_slug == target]
    if exact:
        return exact
    return [c for c in cells if target in c.hardware_slug or c.hardware_slug in target]


def _citations(cells: list[Cell]) -> list[str]:
    """De-duplicated /r/<id> URLs in the order seen."""
    seen: set[str] = set()
    out: list[str] = []
    for c in cells:
        if c.run_id in seen:
            continue
        seen.add(c.run_id)
        out.append(api.run_url(c.run_id))
    return out


def _no_match(reason: str) -> dict[str, Any]:
    return {"matches": [], "citation": [], "note": reason}


# Per-tool input clamps. The longest legitimate Hugging Face model name
# observed in the wild is ~120 chars; 1024 is generous. The `constraints`
# dict on `recommend` accepts seven supported keys, so 32 is well past
# legitimate usage. Both clamps exist purely to bound the cost of a
# malicious LLM client spamming the server with megabyte-sized inputs:
# without them, `_match_model("A" * 10_000_000)` would burn ~250 ms
# per call slugifying the input + lowercasing it for substring search,
# at ~30 MB peak resident per call. Local DoS only — no remote
# escalation — but easy to harden.
_MAX_TOOL_STR = 1024
_MAX_CONSTRAINT_KEYS = 32


def _clamp_str(s: Any) -> str | None:
    """Coerce to a bounded string. Non-strings → None; over-long → truncated."""
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    return s[:_MAX_TOOL_STR]


def _clamp_required_str(s: Any) -> str:
    """Same as `_clamp_str` but treats non-strings + None as empty."""
    if not isinstance(s, str):
        return ""
    return s[:_MAX_TOOL_STR]


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool()
def lookup_speed(model: str, hardware: str | None = None) -> dict[str, Any]:
    """Return the best decode_tps row for a (model, hardware) pair.

    If `hardware` is None, returns every hardware result for the model.
    Each row carries a /r/<id> citation, /m/<slug> and /hw/<slug> links.
    """
    model = _clamp_required_str(model)
    hardware = _clamp_str(hardware)
    cells = _all_cells()
    cells = _match_model(model, cells)
    if not cells:
        return _no_match(f"No measurements found for model '{model}'.")
    if hardware is not None:
        cells = _match_hardware(hardware, cells)
        if not cells:
            return _no_match(
                f"No measurements found for model '{model}' on hardware '{hardware}'."
            )
    rows = [
        {
            **cell_to_dict(c),
            "model_url": api.model_url(c.model_slug),
            "hardware_url": api.hardware_url(c.hardware_slug),
            "run_url": api.run_url(c.run_id),
        }
        for c in cells
    ]
    return {
        "matches": rows,
        "citation": _citations(cells),
    }


@mcp.tool()
def compare(a: str, b: str) -> dict[str, Any]:
    """Compare two cells side-by-side.

    Each side may be a model name, a hardware name, or "model on hardware"
    (the word "on" splits the side). Returns who is faster, the delta in
    tok/s, the run backing each side, and the /vs/<slug> deeplink.
    """
    a = _clamp_required_str(a)
    b = _clamp_required_str(b)
    cells = _all_cells()
    side_a = _resolve_side(a, cells)
    side_b = _resolve_side(b, cells)
    if side_a is None or side_b is None:
        missing = []
        if side_a is None:
            missing.append(a)
        if side_b is None:
            missing.append(b)
        return _no_match(f"Could not resolve: {', '.join(missing)}.")

    delta = side_a.decode_tps - side_b.decode_tps
    faster = "a" if delta > 0 else ("b" if delta < 0 else "tie")
    # /vs slug strategy: hw-vs-hw if same model on different rigs, else
    # model-vs-model. Mirrors the website's parseVsSlug.
    if side_a.model_slug == side_b.model_slug and side_a.hardware_slug != side_b.hardware_slug:
        vs = api.vs_url(side_a.hardware_slug, side_b.hardware_slug)
    elif side_a.hardware_slug == side_b.hardware_slug and side_a.model_slug != side_b.model_slug:
        vs = api.vs_url(side_a.model_slug, side_b.model_slug)
    else:
        vs = api.vs_url(side_a.model_slug, side_b.model_slug)

    return {
        "a": _side_payload(side_a),
        "b": _side_payload(side_b),
        "faster": faster,
        "delta_decode_tps": round(abs(delta), 2),
        "ratio": round(side_a.decode_tps / side_b.decode_tps, 2)
        if side_b.decode_tps > 0
        else None,
        "vs_url": vs,
        "citation": _citations([side_a, side_b]),
    }


def _resolve_side(side: str, cells: list[Cell]) -> Cell | None:
    """Pick the best Cell matching a "side" string."""
    parts = [p.strip() for p in side.split(" on ", 1)]
    if len(parts) == 2:
        model_q, hw_q = parts
        candidates = _match_hardware(hw_q, _match_model(model_q, cells))
    else:
        # Try as a model first; if no hits, treat as hardware.
        candidates = _match_model(side, cells)
        if not candidates:
            candidates = _match_hardware(side, cells)
    if not candidates:
        return None
    # The list is already sorted desc by decode_tps.
    return candidates[0]


def _side_payload(c: Cell) -> dict[str, Any]:
    return {
        **cell_to_dict(c),
        "run_url": api.run_url(c.run_id),
        "model_url": api.model_url(c.model_slug),
        "hardware_url": api.hardware_url(c.hardware_slug),
    }


@mcp.tool()
def recommend(constraints: dict[str, Any]) -> dict[str, Any]:
    """Return up to 5 ranked (model, hardware) cells matching constraints.

    Accepted keys (all optional):
      - vram_gb_max: float — exclude rigs with more VRAM than this
      - ram_gb_max:  float — exclude rigs with more system/unified RAM than this
      - decode_tps_min: float — minimum decode tok/s
      - model_size_min: float — minimum model parameter count (B)
      - model_size_max: float — maximum model parameter count (B)
      - backend: str — substring filter (e.g. "mlx", "llama.cpp")
      - locality: "local" | "hosted" | "any" — default "any"

    Note: vram_gb_max / ram_gb_max are best-effort parses out of the
    accelerator_summary string. Anything we can't parse is left in.
    """
    if not isinstance(constraints, dict) or len(constraints) > _MAX_CONSTRAINT_KEYS:
        return _no_match(
            f"constraints must be a dict with ≤{_MAX_CONSTRAINT_KEYS} keys"
        )
    cells = _all_cells()
    locality = _clamp_required_str(constraints.get("locality") or "any").lower()
    backend = _clamp_str(constraints.get("backend"))
    decode_min = constraints.get("decode_tps_min")
    size_min = constraints.get("model_size_min")
    size_max = constraints.get("model_size_max")
    vram_max = constraints.get("vram_gb_max")
    ram_max = constraints.get("ram_gb_max")

    out: list[Cell] = []
    for c in cells:
        if locality != "any" and c.locality != locality:
            continue
        if backend and backend.lower() not in c.backend.lower():
            continue
        if decode_min is not None and c.decode_tps < float(decode_min):
            continue
        size_b = model_size_b(c.model_name)
        if size_min is not None and (size_b is None or size_b < float(size_min)):
            continue
        if size_max is not None and (size_b is None or size_b > float(size_max)):
            continue
        if vram_max is not None:
            vram = _parse_first_gb(c.hardware_label)
            if vram is not None and vram > float(vram_max):
                continue
        if ram_max is not None:
            ram = _parse_total_ram(c.hardware_label)
            if ram is not None and ram > float(ram_max):
                continue
        out.append(c)

    top = out[:5]
    return {
        "matches": [
            {
                **cell_to_dict(c),
                "run_url": api.run_url(c.run_id),
                "model_url": api.model_url(c.model_slug),
                "hardware_url": api.hardware_url(c.hardware_slug),
            }
            for c in top
        ],
        "citation": _citations(top),
        "considered": len(cells),
        "after_filters": len(out),
    }


def _parse_first_gb(label: str) -> float | None:
    """Pull the first "(NN GB)"-style number from an accelerator label.
    By convention this is the GPU's VRAM (or unified-memory budget on
    Apple Silicon). Returns None when not present."""
    import re as _re

    m = _re.search(r"\((\d+(?:\.\d+)?)\s*GB", label, _re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_total_ram(label: str) -> float | None:
    """Pull the trailing "+ NN GB" out of a "GPU + CPU + RAM" summary.
    For Apple Silicon "+ 96GB unified" we treat unified memory as the
    total RAM constraint."""
    import re as _re

    m = _re.search(r"\+\s*(\d+(?:\.\d+)?)\s*GB(?!\))", label, _re.IGNORECASE)
    return float(m.group(1)) if m else None


@mcp.tool()
def top_models(hardware: str, n: int = 10) -> dict[str, Any]:
    """Fastest models on a given hardware, ranked by decode_tps."""
    hardware = _clamp_required_str(hardware)
    n = max(1, min(int(n), 25))
    cells = _match_hardware(hardware, _all_cells())
    if not cells:
        return _no_match(f"No measurements found for hardware '{hardware}'.")
    top = cells[:n]
    return {
        "hardware": top[0].hardware_label,
        "hardware_slug": top[0].hardware_slug,
        "hardware_url": api.hardware_url(top[0].hardware_slug),
        "matches": [
            {
                **cell_to_dict(c),
                "run_url": api.run_url(c.run_id),
                "model_url": api.model_url(c.model_slug),
            }
            for c in top
        ],
        "citation": _citations(top),
    }


@mcp.tool()
def state_of() -> dict[str, Any]:
    """Latest /state-of headline cells, computed live from the listing.

    Mirrors the website's headline derivation in
    web/app/state-of/[yyyy-mm]/page.tsx: fastest overall, fastest 70B+ class,
    fastest coding-agent. Computed over runs from the most recent calendar
    month present in the listing so the snapshot is always current.
    """
    cells = _all_cells()
    if not cells:
        return _no_match("No runs in the leaderboard yet.")

    # Constrain to the most recent month present in the data.
    months = sorted({c.received_at[:7] for c in cells if c.received_at}, reverse=True)
    issue = months[0] if months else None
    if issue:
        cells = [c for c in cells if c.received_at[:7] == issue]

    fastest = next((c for c in cells if c.locality == "local"), None) or cells[0]

    def _is_big(c: Cell) -> bool:
        s = model_size_b(c.model_name)
        return s is not None and s >= 35.0

    def _is_coder(c: Cell) -> bool:
        n = (c.model_name or "").lower()
        return any(
            k in n
            for k in ("coder", "coding", "code-instruct", "starcoder", "codestral")
        )

    fastest_70b = next((c for c in cells if _is_big(c) and c.locality == "local"), None)
    fastest_coding = next((c for c in cells if _is_coder(c) and c.locality == "local"), None)

    headlines: list[dict[str, Any]] = []
    for label, cell in (
        ("Fastest local model", fastest),
        ("Fastest local 70B+ class", fastest_70b),
        ("Fastest local coding agent", fastest_coding),
    ):
        if cell is None:
            headlines.append({"label": label, "available": False})
            continue
        headlines.append(
            {
                "label": label,
                "available": True,
                **cell_to_dict(cell),
                "run_url": api.run_url(cell.run_id),
                "model_url": api.model_url(cell.model_slug),
                "hardware_url": api.hardware_url(cell.hardware_slug),
            }
        )

    cited = [
        c
        for c in (fastest, fastest_70b, fastest_coding)
        if isinstance(c, Cell)
    ]
    return {
        "issue": issue,
        "issue_url": f"{api.SITE_BASE}/state-of/{issue}" if issue else None,
        "headlines": headlines,
        "citation": _citations(cited),
    }


# --------------------------------------------------------------------------
# Prompt resource
# --------------------------------------------------------------------------


_HOWTO_TEMPLATE = """\
# How to pick a local LLM with llm-speed

You are helping the user choose a local LLM. Use the llm-speed MCP tools
to ground every claim in a real benchmark run. Workflow:

1. Clarify the rig. Ask (or infer from prior context):
   - Hardware: GPU model + VRAM, or Apple Silicon chip + unified-memory size.
   - System RAM (matters for GGUF / partial-offload setups).
   - Backend preference: llama.cpp, MLX, vLLM, ollama, exllamav2, "any".
   - Task: chat, coding, agentic loops, long-context summarisation.

2. Call `recommend` with the constraints. Prefer locality="local" unless
   the user is explicitly asking for hosted-model comparisons. Set
   model_size_min based on quality expectations:
     - 7B-13B for fast assistants on consumer GPUs / Apple Silicon laptops.
     - 22B-32B for coding (Qwen2.5-Coder, Codestral, DeepSeek-Coder-V2-Lite).
     - 70B+ when the user has 64GB+ unified memory or 48GB+ VRAM.

3. For each candidate the tool returns, cite the /r/<id> URL inline. Do
   NOT invent tok/s numbers — every figure must come from a tool call.

4. If the user wants a sanity-check, call `compare` for the top two
   candidates. Quote the /vs/<slug> deeplink so they can see the table.

5. End with the contribution loop: tell the user they can submit their
   own benchmark via `pip install llm-speed && llm-speed bench` so the
   leaderboard improves over time.
"""


@mcp.resource("howto://pick-a-local-llm")
def pick_a_local_llm() -> str:
    """Guided prompt: how to pick a local LLM using llm-speed grounding."""
    return _HOWTO_TEMPLATE


# Re-export the FastMCP instance for entry point.
def get_app() -> FastMCP:
    return mcp
