"""Non-interactive auto mode.

``llm-speed --auto`` (or implicitly: ``llm-speed`` invoked with no args from a
non-tty stdin) picks sensible defaults and runs the quick workload set:

1. Fingerprint the machine.
2. Pick the highest-priority *available* backend.
3. Pick the smallest reasonable model exposed by that backend.
4. Run the QUICK workload set so first-time runs return in under a minute.
5. Save locally; upload only if consent is already granted.

Auto mode never prompts. If consent isn't on file, the result is saved
offline and a one-line hint tells the user how to upload it.

Implemented as a thin wrapper that constructs the same argparse-style
namespace the existing ``cmd_bench`` consumes, so we don't fork the bench
flow.
"""

from __future__ import annotations

import argparse
import logging

from ..config import DEFAULT_API_BASE
from .progress import announce, info
from .theme import get_console

log = logging.getLogger("cli.ui.auto")


def build_auto_args(
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    no_upload: bool = False,
) -> argparse.Namespace:
    """Build the args namespace ``cmd_bench`` expects for an auto run."""
    ns = argparse.Namespace()
    ns.command = "bench"
    ns.backend = None
    ns.model = None
    ns.workload = None
    ns.quick = True
    ns.no_upload = no_upload
    ns.json = None
    ns.api_key = api_key
    ns.anon = False
    ns.strict_anon = False
    ns.dry_run = False
    ns.print_payload = False
    ns.resume = None
    ns.api_base = api_base or DEFAULT_API_BASE
    ns.verbose = False
    return ns


def run_auto(*, no_upload: bool = False, api_base: str | None = None) -> int:
    """Run the bench in auto mode and return its exit code."""
    cons = get_console()
    cons.print()
    announce("auto mode: quick workload, sensible defaults")
    info("running the QUICK workload set against the first available backend")
    info("press Ctrl-C any time to stop")
    cons.print()

    # Imported lazily to keep `python -m cli.ui.auto` cheap and to avoid
    # circular imports during package init.
    from ..commands.bench import cmd_bench

    args = build_auto_args(api_base=api_base, no_upload=no_upload)
    return cmd_bench(args)
