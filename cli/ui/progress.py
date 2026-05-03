"""Rich-based progress + status helpers.

Two main building blocks:

* :class:`StepReporter` - sequential `[..] step name ... [ok]` lines, used by
  the bench flow when streaming workload-by-workload progress.
* :func:`spinner` - context manager that wraps a long-running call in a
  transient spinner; auto-disabled when stdout is not a tty so CI logs stay
  flat.

Designed to be safe to call without any prior setup - they fall back to plain
prints when rich's live rendering can't render (no tty, dumb term).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console

from .theme import (
    C_BRAND,
    C_ERR,
    C_HINT,
    C_MUTED,
    C_OK,
    C_WARN,
    SYM_ERR,
    SYM_OK,
    SYM_RUN,
    SYM_WARN,
    get_console,
)


def _tty() -> bool:
    return sys.stdout.isatty()


class StepReporter:
    """Tracks a sequence of named steps and prints status transitions.

    Usage::

        sr = StepReporter(["fingerprint", "pick backend", "run workloads"])
        sr.start("fingerprint")
        ...
        sr.ok("fingerprint", detail="M3 Pro / 36GB")

    When the program completes, :meth:`summary` renders the final per-step
    status table.
    """

    def __init__(self, steps: list[str], *, console: Console | None = None) -> None:
        self.console = console or get_console()
        self.steps = list(steps)
        self.statuses: dict[str, str] = {s: "pending" for s in steps}
        self.details: dict[str, str] = {s: "" for s in steps}

    def start(self, step: str, detail: str = "") -> None:
        if step not in self.statuses:
            self.steps.append(step)
            self.statuses[step] = "pending"
            self.details[step] = ""
        self.statuses[step] = "running"
        self.details[step] = detail
        suffix = f" [dim]{detail}[/]" if detail else ""
        self.console.print(f"[{C_HINT}]{SYM_RUN}[/] {step}{suffix}")

    def ok(self, step: str, detail: str = "") -> None:
        self.statuses[step] = "ok"
        self.details[step] = detail
        suffix = f" [dim]({detail})[/]" if detail else ""
        self.console.print(f"  [{C_OK}]{SYM_OK}[/] {step}{suffix}")

    def warn(self, step: str, detail: str = "") -> None:
        self.statuses[step] = "warn"
        self.details[step] = detail
        suffix = f" [dim]({detail})[/]" if detail else ""
        self.console.print(f"  [{C_WARN}]{SYM_WARN}[/] {step}{suffix}")

    def fail(self, step: str, detail: str = "") -> None:
        self.statuses[step] = "fail"
        self.details[step] = detail
        suffix = f" [dim]{detail}[/]" if detail else ""
        self.console.print(f"  [{C_ERR}]{SYM_ERR}[/] {step}{suffix}")

    def summary(self) -> None:
        self.console.print()
        self.console.print(f"[{C_MUTED}]Run summary:[/]")
        for s in self.steps:
            sym = {
                "ok": f"[{C_OK}]{SYM_OK}[/]",
                "warn": f"[{C_WARN}]{SYM_WARN}[/]",
                "fail": f"[{C_ERR}]{SYM_ERR}[/]",
                "running": f"[{C_HINT}]{SYM_RUN}[/]",
                "pending": f"[{C_MUTED}]....[/]",
            }.get(self.statuses[s], "?")
            detail = self.details.get(s) or ""
            tail = f"  [dim]{detail}[/]" if detail else ""
            self.console.print(f"  {sym} {s}{tail}")


@contextmanager
def spinner(label: str, *, console: Console | None = None) -> Iterator[None]:
    """Show a transient spinner for the duration of the block.

    Falls back to a single plain line on non-tty stdout (CI). The contract is
    "this block prints one progress notice and clears it on exit".
    """
    cons = console or get_console()
    if not _tty():
        cons.print(f"[{C_HINT}]{SYM_RUN}[/] {label} ...")
        yield
        return

    with cons.status(f"[{C_BRAND}]{label}[/]", spinner="dots"):
        yield


def announce(msg: str, *, console: Console | None = None) -> None:
    """Print a brand-coloured top-level header line."""
    cons = console or get_console()
    cons.print(f"[{C_BRAND}]>>[/] {msg}")


def info(msg: str, *, console: Console | None = None) -> None:
    cons = console or get_console()
    cons.print(f"  [dim]{msg}[/]")
