"""Pretty-printing for run summaries (rich-based)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .types import RunReport

_console = Console()


def get_console() -> Console:
    return _console


def _fmt(v, ndigits: int = 1, suffix: str = "") -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{ndigits}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def print_run_summary(
    report: RunReport,
    *,
    result_url: str | None = None,
    offline_path: Path | None = None,
) -> None:
    """Headline + per-workload table + footer."""
    fp = report.fingerprint
    accel = fp.accelerator_summary or "(unknown hardware)"

    # Pull backend / model from the first non-error result, else first result.
    primary = next((r for r in report.results if not r.error), None) or (
        report.results[0] if report.results else None
    )
    backend_str = "(no results)"
    model_str = ""
    if primary is not None:
        bv = primary.backend_version or "?"
        backend_str = f"{primary.backend}@{bv}"
        m = primary.model
        bits = [m.name or m.identifier or "?"]
        meta = []
        if m.size:
            meta.append(m.size)
        if m.quant:
            meta.append(m.quant)
        if meta:
            bits.append(f"({', '.join(meta)})")
        model_str = " ".join(bits)

    _console.rule(f"[bold]{accel}[/bold]")
    _console.print(f"[cyan]{backend_str}[/cyan]   [magenta]{model_str}[/magenta]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("workload", style="bold")
    table.add_column("decode tps", justify="right")
    table.add_column("prefill tps", justify="right")
    table.add_column("ttft ms", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("error?", style="red")

    for r in report.results:
        table.add_row(
            r.workload,
            _fmt(r.decode_tps),
            _fmt(r.prefill_tps),
            _fmt(r.ttft_ms),
            _fmt(r.decode_p50_latency_ms),
            _fmt(r.decode_p95_latency_ms),
            (r.error[:60] if r.error else ""),
        )

    _console.print(table)

    if result_url:
        _console.print(f"[green]Submitted:[/green] {result_url}")
    elif offline_path:
        _console.print(f"[yellow]Saved locally:[/yellow] {offline_path}")
        _console.print(
            f"  Re-upload later with: [bold]llm-speed bench --resume {offline_path}[/bold]"
        )


def print_error(msg: str) -> None:
    _console.print(f"[red]error:[/red] {msg}")


def print_warning(msg: str) -> None:
    _console.print(f"[yellow]warn:[/yellow] {msg}")


def print_info(msg: str) -> None:
    _console.print(msg)
