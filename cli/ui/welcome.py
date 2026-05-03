"""First-run welcome wizard.

Flow:

1. Splash screen with the wordmark + a one-liner ("benchmark any LLM, share
   the result").
2. Detect installed backends; print a one-line status table.
3. If at least one backend is available, offer to run the auto-mode quick
   bench right now.
4. If no backend is available, offer to install the easiest one for this OS.
   Every install requires explicit y/n confirmation - we surface the exact
   shell command first.
5. Run the quick bench. Consent prompt fires inside the existing flow.
6. After the run, print the share affordances panel.

The wizard is the entry point for ``llm-speed`` invoked with no args on a
TTY. It refuses to run on a non-tty stdin (CI / piped) and instead falls
through to ``--help``-equivalent behaviour. Each step is small so individual
chunks can be unit-tested by mocking the console.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..auto_install import (
    InstallPlan,
    confirm_and_install,
    detect_missing,
    is_llama_cpp_installed,
    is_mlx_installed,
    is_ollama_installed,
    plan_for,
)
from .auto import run_auto
from .progress import announce, info
from .theme import (
    C_BRAND,
    C_HINT,
    C_MUTED,
    C_OK,
    C_WARN,
    PRIVACY_URL,
    REPO_URL,
    banner,
    get_console,
    hint_bar,
)

log = logging.getLogger("cli.ui.welcome")


# ---------------------------------------------------------------------------
# Detection summary
# ---------------------------------------------------------------------------


def _backend_status() -> list[tuple[str, bool, str]]:
    """Return `(name, installed, hint)` rows for the detection panel."""
    rows: list[tuple[str, bool, str]] = []
    rows.append(("llama.cpp", is_llama_cpp_installed(), "GGUF, llama-server"))
    rows.append(("ollama", is_ollama_installed(), "daemon at :11434"))
    rows.append(("mlx", is_mlx_installed(), "Apple Silicon only"))
    return rows


def render_splash(*, console: Console | None = None) -> None:
    cons = console or get_console()
    cons.print()
    cons.print(banner())
    cons.print(
        f"  [dim]welcome - this is a one-time setup. "
        f"every byte the cli sends is documented at {PRIVACY_URL}[/]"
    )
    cons.print()


def render_backends(*, console: Console | None = None) -> None:
    cons = console or get_console()
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("backend")
    table.add_column("status")
    table.add_column("notes", style="dim")
    for name, installed, hint in _backend_status():
        if installed:
            table.add_row(name, f"[{C_OK}]installed[/]", hint)
        else:
            table.add_row(name, f"[{C_MUTED}]not found[/]", hint)
    cons.print(table)
    cons.print()


# ---------------------------------------------------------------------------
# Confirmation helpers
# ---------------------------------------------------------------------------


def _default_confirm(prompt: str) -> bool:
    """Print ``prompt`` and read a y/n answer from stdin."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        line = sys.stdin.readline()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return False
    if not line:
        return False
    return line.strip().lower() in ("y", "yes")


def offer_install(
    backend: str,
    *,
    confirm: Callable[[str], bool] | None = None,
    console: Console | None = None,
) -> bool:
    """Offer the install plan for ``backend``. Returns True iff installed.

    The caller-supplied ``confirm`` callable takes the proposed prompt string
    (which already contains the literal command we will run) and must return
    True only when the user explicitly typed yes.
    """
    cons = console or get_console()
    plan: InstallPlan = plan_for(backend)
    if not plan.available:
        cons.print(f"[{C_WARN}]skip {backend}[/]: {plan.skip_reason}")
        return False
    cons.print(f"[{C_HINT}]install offer:[/] {plan.description}")
    return confirm_and_install(plan, confirm=confirm or _default_confirm)


# ---------------------------------------------------------------------------
# Wizard entry point
# ---------------------------------------------------------------------------


def _is_tty() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def run_wizard(
    *,
    api_base: str | None = None,
    confirm: Callable[[str], bool] | None = None,
    console: Console | None = None,
) -> int:
    """Top-level first-run wizard. Returns an exit code."""
    cons = console or get_console()
    confirm_fn = confirm or _default_confirm

    if not _is_tty():
        cons.print(banner())
        cons.print()
        cons.print(
            f"[{C_MUTED}]non-interactive shell detected - falling back to auto mode "
            f"(see `llm-speed --help` for explicit commands).[/]"
        )
        return run_auto(api_base=api_base)

    render_splash(console=cons)
    render_backends(console=cons)

    missing = detect_missing()
    has_any = any(installed for _, installed, _ in _backend_status())

    if not has_any:
        cons.print(
            f"[{C_WARN}]no benchmark backend installed yet.[/] "
            f"{C_BRAND}llm-speed[/] needs one of: llama.cpp, ollama, mlx."
        )
        # Offer the *first* missing backend's install plan; user can install more later.
        for candidate in missing:
            installed = offer_install(candidate, confirm=confirm_fn, console=cons)
            if installed:
                has_any = True
                break

    if not has_any:
        cons.print()
        cons.print(
            Panel(
                f"No backend is set up yet. After installing one (see {REPO_URL} for the\n"
                f"full list), re-run [bold]llm-speed[/] and we'll pick up from here.",
                border_style=C_WARN,
                expand=False,
            )
        )
        return 0

    # ------------------------------------------------------------------ smoke run
    cons.print()
    announce("ready to benchmark")
    info("we'll run the QUICK workload (about 30s) on the first available backend")
    if not confirm_fn("Run the quick benchmark now? [Y/n] "):
        cons.print(f"[{C_MUTED}]skipped. run anytime with[/] llm-speed bench --quick")
        return 0

    rc = run_auto(api_base=api_base)
    cons.print()
    cons.print(
        hint_bar(
            ("b", "benchmark another model"),
            ("v", "verify the binary"),
            ("q", "quit"),
        )
    )
    return rc
