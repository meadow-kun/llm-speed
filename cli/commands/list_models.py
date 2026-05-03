"""`llm-speed list-models` — list models per available backend."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.table import Table

from ..registry import all_driver_names, get_driver

log = logging.getLogger("cli.commands.list_models")


def cmd_list_models(args) -> int:
    console = Console()
    backend = getattr(args, "backend", None)

    candidates = [backend] if backend else all_driver_names()
    if not candidates:
        console.print("[yellow]no drivers registered[/yellow]")
        return 0

    table = Table(show_header=True, header_style="bold")
    table.add_column("backend")
    table.add_column("identifier")
    table.add_column("name")
    table.add_column("size")
    table.add_column("quant")

    rows = 0
    for name in candidates:
        try:
            drv = get_driver(name)
        except KeyError as exc:
            console.print(f"[red]error:[/red] {exc}")
            return 2
        try:
            det = drv.detect()
        except Exception as exc:
            log.debug("detect %s failed: %s", name, exc)
            det = None

        if det is not None and not det.available and not backend:
            # Skip unavailable backends unless user asked for it explicitly.
            continue

        try:
            models = drv.list_models()
        except Exception as exc:
            log.debug("list_models %s failed: %s", name, exc)
            console.print(f"[yellow]{name}: list_models failed: {exc}[/yellow]")
            continue

        for m in models:
            table.add_row(
                m.backend,
                m.identifier,
                m.name or "-",
                m.size or "-",
                m.quant or "-",
            )
            rows += 1

    if rows == 0:
        console.print("[dim]no models found[/dim]")
    else:
        console.print(table)
    return 0
