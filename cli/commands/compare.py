"""`llm-speed compare RUN1.json RUN2.json [...]` — local diff of saved runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

log = logging.getLogger("cli.commands.compare")


_METRICS = (
    ("decode_tps", "decode tps", 1),
    ("prefill_tps", "prefill tps", 1),
    ("ttft_ms", "ttft ms", 1),
    ("decode_p50_latency_ms", "p50 ms", 2),
    ("decode_p95_latency_ms", "p95 ms", 2),
    ("wall_ms", "wall ms", 0),
)


def _label_for(report: dict, fallback: str) -> str:
    fp = report.get("fingerprint") or {}
    accel = fp.get("accelerator_summary") or fallback
    results = report.get("results") or []
    backend = ""
    if results:
        first = results[0]
        b = first.get("backend") or "?"
        bv = first.get("backend_version") or "?"
        backend = f" [{b}@{bv}]"
    return f"{accel}{backend}"


def _fmt(v, ndigits: int) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(v)


def cmd_compare(args) -> int:
    console = Console()
    paths: list[Path] = [Path(p) for p in args.runs]
    if len(paths) < 2:
        console.print("[red]compare needs at least two run files[/red]")
        return 2

    reports = []
    for p in paths:
        if not p.exists():
            console.print(f"[red]not found:[/red] {p}")
            return 2
        try:
            reports.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            console.print(f"[red]bad json in {p}:[/red] {exc}")
            return 2

    labels = [_label_for(r, paths[i].name) for i, r in enumerate(reports)]

    # Collect the union of workload names per report, preserving file order.
    all_workloads: list[str] = []
    seen: set[str] = set()
    for rep in reports:
        for r in rep.get("results", []):
            wn = r.get("workload")
            if wn and wn not in seen:
                seen.add(wn)
                all_workloads.append(wn)

    for wname in all_workloads:
        table = Table(title=f"workload: {wname}", show_header=True, header_style="bold")
        table.add_column("metric")
        for label in labels:
            table.add_column(label, justify="right")

        for key, display, ndigits in _METRICS:
            row = [display]
            for rep in reports:
                match = next(
                    (r for r in rep.get("results", []) if r.get("workload") == wname),
                    None,
                )
                row.append(_fmt(match.get(key) if match else None, ndigits))
            table.add_row(*row)
        console.print(table)
        console.print()

    return 0
