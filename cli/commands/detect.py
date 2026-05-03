"""`llm-speed detect` — print fingerprint + available backends + workloads."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.table import Table

from ..fingerprint import detect_fingerprint
from ..registry import all_driver_names, all_workload_names, get_driver

log = logging.getLogger("cli.commands.detect")


def cmd_detect(args) -> int:
    console = Console()
    try:
        fp = detect_fingerprint()
    except Exception as exc:
        log.exception("fingerprint failed")
        console.print(f"[red]fingerprint failed:[/red] {exc}")
        return 1

    console.rule("[bold]Hardware fingerprint[/bold]")
    console.print(f"  OS:         {fp.os_name} {fp.os_version}")
    console.print(f"  CPU:        {fp.cpu_model}  ({fp.cpu_cores} cores)")
    console.print(f"  RAM:        {fp.ram_gb} GB")
    if fp.gpus:
        for g in fp.gpus:
            mem = f"{g.memory_gb}GB" if g.memory_gb else "?"
            drv = f", driver {g.driver_version}" if g.driver_version else ""
            console.print(f"  GPU:        {g.name} ({g.kind}, {mem}{drv})")
    else:
        console.print("  GPU:        none detected")
    console.print(f"  Summary:    {fp.accelerator_summary}")
    console.print(f"  Hash:       {fp.fingerprint_hash}")

    console.rule("[bold]Available backends[/bold]")
    driver_names = all_driver_names()
    if not driver_names:
        console.print("  [dim]no drivers registered[/dim]")
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("backend")
        table.add_column("available")
        table.add_column("version")
        table.add_column("notes")
        for name in driver_names:
            try:
                drv = get_driver(name)
                det = drv.detect()
                avail = "[green]yes[/green]" if det.available else "[red]no[/red]"
                table.add_row(name, avail, det.version or "-", det.notes or "")
            except Exception as exc:
                log.debug("driver %s detect failed: %s", name, exc)
                table.add_row(name, "[red]error[/red]", "-", str(exc)[:80])
        console.print(table)

    console.rule("[bold]Registered workloads[/bold]")
    workload_names = all_workload_names()
    if not workload_names:
        console.print("  [dim]no workloads registered[/dim]")
    else:
        for name in workload_names:
            console.print(f"  - {name}")

    return 0
