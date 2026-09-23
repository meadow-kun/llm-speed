"""`llm-speed doctor` — interactive dependency setup + non-happy-path guide.

Runs the :mod:`cli.preflight` checks and then, on a TTY, walks the user through
fixing whatever is missing: it shows the exact command, asks for an explicit
y/N, runs the install only on yes (license-gated by ``InstallPlan.auto_installable``),
re-runs the checks, and loops until the required gate passes or the user bows out.

On a non-TTY (CI / pipe / `ssh box llm-speed doctor`) it NEVER prompts — it
prints the same guidance (including the exact commands to run by hand) and
returns a non-zero exit code iff a *required* check failed. This mirrors the
consent.py non-TTY discipline (KFM-007), so scripts and the Docker test matrix
get a deterministic signal.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from rich.console import Console

from ..auto_install import InstallPlan, confirm_and_install
from ..preflight import FAIL, OK, WARN, Check, required_failed, run_checks
from ..ui.progress import announce, info
from ..ui.theme import (
    C_ERR,
    C_HINT,
    C_OK,
    C_WARN,
    SYM_ERR,
    SYM_OK,
    SYM_RUN,
    SYM_WARN,
    get_console,
)
from ..ui.welcome import _default_confirm

log = logging.getLogger("cli.commands.doctor")

# Safety bound: each accepted install can legitimately unlock a new sub-state
# (install → start daemon → pull model), so we allow several rounds, but never
# loop forever if a fix "succeeds" without changing the symptom.
_MAX_ROUNDS = 6

_SYM = {
    OK: (SYM_OK, C_OK),
    WARN: (SYM_WARN, C_WARN),
    FAIL: (SYM_ERR, C_ERR),
}


def _render(checks: list[Check], *, console: Console) -> None:
    console.print()
    for c in checks:
        sym, color = _SYM.get(c.status, (SYM_RUN, C_HINT))
        detail = f" [dim]{c.detail}[/]" if c.detail else ""
        console.print(f"  [{color}]{sym:>4}[/] {c.title}{detail}")


def _print_guidance(checks: list[Check], *, console: Console) -> None:
    """Non-interactive: list every non-ok check with its manual fix command."""
    issues = [c for c in checks if c.status != OK and c.id != "backend.any"]
    if not issues:
        return
    console.print()
    console.print(f"[{C_HINT}]To fix:[/]")
    for c in issues:
        console.print(f"  - {c.title}: {c.detail}")
        if c.fix is not None:
            console.print(f"      run: [bold]{c.fix.command}[/]")
        if c.remediation:
            console.print(f"      {c.remediation}")
    # Surface the summary remediation (e.g. backend.any) last.
    summary = next((c for c in checks if c.id == "backend.any" and c.remediation), None)
    if summary and summary.status != OK:
        console.print(f"  - {summary.remediation}")


def _final_line(checks: list[Check], *, console: Console) -> None:
    console.print()
    if required_failed(checks):
        console.print(
            f"[{C_ERR}]not ready[/] - resolve the items above, then re-run "
            "[bold]llm-speed doctor[/]."
        )
    else:
        console.print(
            f"[{C_OK}]ready[/] - run [bold]llm-speed bench --quick[/] to benchmark."
        )


def run_doctor(
    *,
    api_base: str | None = None,
    interactive: bool | None = None,
    confirm: Callable[[str], bool] | None = None,
    runner: Callable[[InstallPlan], int] | None = None,
    console: Console | None = None,
    run_checks_fn: Callable[[], list[Check]] | None = None,
) -> int:
    """Run the doctor flow. Returns an exit code (0 ready, 1 not ready)."""
    cons = console or get_console()
    do_checks = run_checks_fn or (lambda: run_checks(api_base=api_base))
    confirm_fn = confirm or _default_confirm
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()

    announce("dependency check")
    checks = do_checks()
    _render(checks, console=cons)

    if not interactive:
        _print_guidance(checks, console=cons)
        _final_line(checks, console=cons)
        return 1 if required_failed(checks) else 0

    # Interactive remediation loop. ``attempted`` stops us re-offering a fix
    # that already ran (and either succeeded-but-didn't-help or was the only
    # thing left) so the loop always terminates.
    attempted: set[str] = set()
    rounds = 0
    while required_failed(checks) and rounds < _MAX_ROUNDS:
        rounds += 1
        progressed = False
        for chk in checks:
            if chk.id == "backend.any" or chk.status == OK:
                continue
            if chk.fix is not None and chk.fix.command not in attempted:
                attempted.add(chk.fix.command)
                if confirm_and_install(chk.fix, confirm=confirm_fn, runner=runner):
                    progressed = True
                    break  # re-run all checks from a clean state
                if chk.remediation:
                    info(chk.remediation, console=cons)
            elif chk.fix is None and chk.remediation:
                info(chk.remediation, console=cons)
        if not progressed:
            break
        cons.print()
        announce("re-checking")
        checks = do_checks()
        _render(checks, console=cons)

    _final_line(checks, console=cons)
    return 1 if required_failed(checks) else 0


def cmd_doctor(args) -> int:
    """argparse entry point for `llm-speed doctor`."""
    # The global --auto flag forces the non-interactive, scriptable path.
    interactive = False if getattr(args, "auto", False) else None
    return run_doctor(api_base=getattr(args, "api_base", None), interactive=interactive)
