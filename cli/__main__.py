"""llm-speed CLI entry point.

Argparse dispatcher across `bench`, `detect`, `list-models`, `compare`,
`login`, `self-update`, plus the new `verify`, `about` subcommands and the
default no-args interactive wizard.

Side-effect-imports `cli.drivers` and `cli.workloads` to populate the
registries before dispatch.

Exit codes:
  0  success
  1  runtime error (bench failure, network, etc.)
  2  bad command-line args / unknown command
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-speed",
        description=(
            "Benchmark any LLM on any hardware. Submit reproducible numbers to llm-speed.com.\n"
            "Run `llm-speed` with no arguments to enter the interactive wizard."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"llm-speed {__version__}"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable verbose logging + tracebacks"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="override API base URL (default: $LLM_SPEED_API)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="non-interactive auto mode: pick a backend + model + run the quick suite",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = False  # we handle missing-command ourselves so help is friendlier

    # bench --------------------------------------------------------------
    p_bench = sub.add_parser("bench", help="run the benchmark suite")
    p_bench.add_argument(
        "--backend", default=None, help="explicit backend name (e.g. ollama, llama.cpp)"
    )
    p_bench.add_argument(
        "--model", default=None, help="explicit model identifier (driver-specific)"
    )
    p_bench.add_argument(
        "--workload", default=None, help="comma-separated list of workloads to run"
    )
    p_bench.add_argument(
        "--quick", action="store_true", help="run a smoke-test subset (< 30s)"
    )
    p_bench.add_argument(
        "--no-upload", action="store_true", help="don't upload; save locally only"
    )
    p_bench.add_argument(
        "--json",
        default=None,
        help="write result JSON to this exact path (implies --no-upload)",
    )
    p_bench.add_argument("--api-key", default=None, help="bearer token for upload auth")
    p_bench.add_argument(
        "--anon",
        action="store_true",
        help="anonymous upload: drop the Authorization bearer token; persistent keypair still ships",
    )
    p_bench.add_argument(
        "--strict-anon",
        action="store_true",
        help="strongest privacy: rotate keypair per run, drop fingerprint_hash, no User-Agent",
    )
    p_bench.add_argument(
        "--dry-run",
        action="store_true",
        help="run workloads but skip upload AND skip local save; print would-be payload",
    )
    p_bench.add_argument(
        "--print-payload",
        action="store_true",
        help="print the exact JSON that would be uploaded (works with or without --dry-run)",
    )
    p_bench.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="upload a previously-saved offline run from PATH (no fresh benchmark; verifies signature first)",
    )

    # detect -------------------------------------------------------------
    sub.add_parser("detect", help="print fingerprint + available backends/workloads")

    # list-models --------------------------------------------------------
    p_list = sub.add_parser("list-models", help="list models per available backend")
    p_list.add_argument(
        "--backend", default=None, help="restrict listing to one backend"
    )

    # compare ------------------------------------------------------------
    p_cmp = sub.add_parser("compare", help="diff two or more saved runs")
    p_cmp.add_argument("runs", nargs="+", help="paths to run JSON files")

    # login --------------------------------------------------------------
    sub.add_parser("login", help="claim a contributor profile via GitHub OAuth (stub)")

    # self-update --------------------------------------------------------
    sub.add_parser("self-update", help="check for a newer suite manifest (stub)")

    # verify -------------------------------------------------------------
    p_verify = sub.add_parser(
        "verify", help="prove the local wheel matches the publicly-served one"
    )
    p_verify.add_argument(
        "wheel",
        nargs="?",
        default=None,
        help="path to the local wheel (default: ./dist/<the wheel>)",
    )
    p_verify.add_argument(
        "--dist-base",
        default=None,
        help="override the base URL where the published wheel + sidecar live",
    )
    p_verify.add_argument(
        "--no-cross-check",
        action="store_true",
        help=(
            "skip the GitHub-Releases cross-check (offline / corp firewall). "
            "Degrades the trust chain to a single distribution domain — only "
            "use this if you genuinely cannot reach github.com."
        ),
    )

    # about --------------------------------------------------------------
    sub.add_parser("about", help="show what llm-speed is + links")

    return parser


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _cmd_about(args) -> int:
    """5-line `about` screen."""
    from .ui.theme import (
        C_HINT,
        C_MUTED,
        PRIVACY_URL,
        REPO_URL,
        SITE_URL,
        TAGLINE,
        banner,
        get_console,
    )

    cons = get_console()
    cons.print()
    cons.print(banner())
    cons.print(f"  [{C_MUTED}]{TAGLINE}[/]")
    cons.print(f"  [{C_HINT}]website   [/] {SITE_URL}")
    cons.print(f"  [{C_HINT}]source    [/] {REPO_URL}")
    cons.print(f"  [{C_HINT}]privacy   [/] {PRIVACY_URL}")
    cons.print()
    return 0


def _dispatch(args, parser: argparse.ArgumentParser) -> int:
    cmd = args.command

    # Side-effect-import to populate registries.
    try:
        from . import drivers as _drivers  # noqa: F401
        from . import workloads as _workloads  # noqa: F401
    except Exception as exc:
        # Drivers/workloads packages should never raise (they self-isolate),
        # but guard anyway so the CLI doesn't die on a bad install.
        if args.verbose:
            traceback.print_exc()
        print(f"warning: failed to load drivers/workloads: {exc}", file=sys.stderr)

    # ---- new TUI surface --------------------------------------------------
    if cmd is None:
        # No subcommand: enter the wizard if we have a tty, otherwise auto.
        if getattr(args, "auto", False):
            from .ui.auto import run_auto

            return run_auto(api_base=getattr(args, "api_base", None))
        from .ui.welcome import run_wizard

        return run_wizard(api_base=getattr(args, "api_base", None))

    if cmd == "bench":
        # Honour --auto on `llm-speed --auto bench` as a no-op (bench already
        # has explicit knobs); the flag matters mostly when paired with no
        # subcommand.
        from .commands.bench import cmd_bench

        return cmd_bench(args)
    if cmd == "detect":
        from .commands.detect import cmd_detect

        return cmd_detect(args)
    if cmd == "list-models":
        from .commands.list_models import cmd_list_models

        return cmd_list_models(args)
    if cmd == "compare":
        from .commands.compare import cmd_compare

        return cmd_compare(args)
    if cmd == "login":
        from .commands.login import cmd_login

        return cmd_login(args)
    if cmd == "self-update":
        from .commands.self_update import cmd_self_update

        return cmd_self_update(args)
    if cmd == "verify":
        from .verify import cmd_verify

        return cmd_verify(args)
    if cmd == "about":
        return _cmd_about(args)

    parser.error(f"unknown command: {cmd}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    try:
        return _dispatch(args, parser)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 1
    except Exception as exc:
        if getattr(args, "verbose", False):
            traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
