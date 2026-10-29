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
        help=(
            "override API base URL (default: $LLM_SPEED_API, else "
            "https://api.llm-speed.com). Any other host gets a warning naming it; "
            "plain http:// is refused except to localhost / 127.0.0.1 / ::1"
        ),
    )
    parser.add_argument(
        "--allow-token-to-host",
        default=None,
        metavar="HOST",
        help=(
            "send your --api-key / bearer token to this non-default API host (exact "
            "host[:port], e.g. queue.example.org or 127.0.0.1:8787). Without it, or "
            "$LLM_SPEED_ALLOW_TOKEN_HOST, uploads to any host other than "
            "api.llm-speed.com go without the key"
        ),
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
        "--credit",
        default=None,
        metavar="HANDLE",
        help=(
            "credit this run's upload to your X (@name) or Reddit (u/name) handle, shown "
            "publicly as unverified; overrides `llm-speed config credit` for this run. "
            "Not allowed with --strict-anon or --anon"
        ),
    )
    p_bench.add_argument(
        "--no-credit",
        action="store_true",
        help="upload this run without your saved credit handle",
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
        "--llama-args",
        action="append",
        default=None,
        metavar="ARGS",
        help=(
            "extra llama-server args, e.g. --llama-args='-c 4096 -ngl 999 --n-cpu-moe 20'. "
            "Applied when llm-speed starts llama-server; with LLAMA_CPP_SERVER_URL they are "
            "only recorded (declare what the external server runs with). Also "
            "$LLM_SPEED_LLAMA_ARGS. Allowlisted flags and values are uploaded as a label; "
            "anything else is redacted."
        ),
    )
    # Sampling sent with every request (0.0.10). Validated in cli/sampling.py.
    p_bench.add_argument(
        "--temperature",
        default=None,
        metavar="T",
        help="sampling temperature sent with every request (0-5, e.g. 0 for greedy)",
    )
    p_bench.add_argument(
        "--seed",
        default=None,
        metavar="N",
        help="sampling seed sent with every request (0-4294967295)",
    )
    p_bench.add_argument(
        "--top-k",
        dest="top_k",
        default=None,
        metavar="K",
        help="top-k sent with every request (0-1000)",
    )
    p_bench.add_argument(
        "--top-p",
        dest="top_p",
        default=None,
        metavar="P",
        help="top-p sent with every request (0-1)",
    )
    p_bench.add_argument(
        "--min-p",
        dest="min_p",
        default=None,
        metavar="P",
        help="min-p sent with every request (0-1)",
    )
    p_bench.add_argument(
        "--upload-failures",
        action="store_true",
        help=(
            "also upload a run where the model failed to load (out of memory, unsupported "
            "architecture, load timeout). Off by default; consent rules still apply."
        ),
    )
    p_bench.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        help=(
            "guided launch run, e.g. --preset flint-40b: detects the GPU, picks quants that "
            "fit, downloads (resumable), runs 4K/16K/32K, records failures, then uploads "
            "through the normal consent flow. Rerun to resume."
        ),
    )
    p_bench.add_argument(
        "--with-control",
        action="store_true",
        help="--preset: also run the preset's control model",
    )
    p_bench.add_argument(
        "--models-dir",
        default=None,
        metavar="DIR",
        help="--preset: where to look for / download model files (default ~/.cache/llm-speed/models)",
    )
    p_bench.add_argument(
        "--llama-server",
        default=None,
        metavar="PATH",
        help="--preset: llama-server binary to use",
    )
    p_bench.add_argument(
        "--no-download",
        action="store_true",
        help="--preset: never download; use files already on disk",
    )
    p_bench.add_argument(
        "--keep-models",
        action="store_true",
        help="--preset: never delete a model this preset downloaded, even when disk is tight",
    )
    p_bench.add_argument(
        "--yes",
        action="store_true",
        help="--preset: answer yes to the llama.cpp install prompt (and never ask the one-time credit question)",
    )
    p_bench.add_argument(
        "--force",
        action="store_true",
        help="--preset: run even if the llama.cpp build number can't be confirmed",
    )
    p_bench.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help="upload a previously-saved offline run from PATH (no fresh benchmark; verifies signature first)",
    )

    # preset -------------------------------------------------------------
    p_preset = sub.add_parser(
        "preset", help="inspect launch presets (hash for review-then-publish)"
    )
    p_preset.add_argument("preset_action", choices=["hash", "show", "list"])
    p_preset.add_argument(
        "name", nargs="?", default=None, help="preset name, e.g. flint-40b"
    )

    # config -------------------------------------------------------------
    p_config = sub.add_parser(
        "config", help="show or change settings (credit: your public handle on uploads)"
    )
    p_config.add_argument("config_key", choices=["credit"], metavar="credit")
    p_config.add_argument(
        "value",
        nargs="?",
        default=None,
        metavar="HANDLE",
        help="@name (X) or u/name (Reddit); omit to show the current setting",
    )
    p_config.add_argument(
        "--clear",
        action="store_true",
        help="remove the handle; uploads become anonymous",
    )

    # detect -------------------------------------------------------------
    sub.add_parser("detect", help="print fingerprint + available backends/workloads")

    # doctor -------------------------------------------------------------
    # Preflight every dependency and, on a TTY, walk the user through
    # installing whatever is missing (license-gated). On a non-TTY it prints
    # guidance and exits non-zero if a required dependency is absent.
    sub.add_parser(
        "doctor",
        help="check dependencies + interactively install/guide what's missing",
    )

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
        help=(
            "path to the local wheel (default: ./dist/<the wheel> or the pipx cache, "
            "else the published wheel is downloaded from the GitHub release / PyPI)"
        ),
    )
    p_verify.add_argument(
        "--no-cross-check",
        action="store_true",
        help=(
            "skip the GitHub release (offline / corp firewall): check against "
            "PyPI's digest only. Not allowed with --sigstore."
        ),
    )
    p_verify.add_argument(
        "--sigstore",
        action="store_true",
        help=(
            "strict: also require the Sigstore bundle "
            "llm_speed-V-py3-none-any.whl.sigstore.json from the public GitHub "
            "release vV, verified against meadow-kun/llm-speed-web "
            "release-sign.yml@refs/tags/vV, and require the GitHub release sha256 "
            "and PyPI's sha256 to agree with the wheel. Needs `sigstore` installed "
            "(pipx inject llm-speed sigstore). Anything missing, unreachable or "
            "failing prints NOT VERIFIED and exits 1."
        ),
    )

    # about --------------------------------------------------------------
    from .agent.command import add_parser as _add_agent_parser

    _add_agent_parser(sub)
    sub.add_parser("about", help="show what llm-speed is + links")

    # mcp ---------------------------------------------------------------
    # Boots a Model Context Protocol stdio server so AI assistants
    # (Claude Desktop, Cline, Cursor, etc.) can query the leaderboard
    # inline. Requires the optional [mcp] extra:
    #   pipx install 'llm-speed[mcp]'
    # or on an existing install:
    #   pipx inject llm-speed mcp cachetools
    sub.add_parser(
        "mcp",
        help="boot the Model Context Protocol server (stdio transport)",
    )

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
    if cmd == "preset":
        from .commands.preset import cmd_preset_tool

        return cmd_preset_tool(args)
    if cmd == "config":
        from .credit import cmd_config

        return cmd_config(args)
    if cmd == "detect":
        from .commands.detect import cmd_detect

        return cmd_detect(args)
    if cmd == "doctor":
        from .commands.doctor import cmd_doctor

        return cmd_doctor(args)
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
    if cmd == "mcp":
        return _cmd_mcp(args)
    if cmd == "agent":
        from .agent.command import dispatch as _agent_dispatch

        return _agent_dispatch(args)

    parser.error(f"unknown command: {cmd}")
    return 2


def _cmd_mcp(args) -> int:
    """Boot the MCP server over stdio. Requires the [mcp] extra."""
    try:
        from .mcp import api as mcp_api
        from .mcp.server import mcp as mcp_app
    except ImportError as exc:
        # The optional `mcp` SDK or `cachetools` aren't installed.
        # Tell the user the exact install command instead of a stack trace.
        print(
            "error: the MCP server requires the optional [mcp] extra.\n"
            "Install with:\n"
            "  pipx install 'llm-speed[mcp]'\n"
            "or, on an existing install:\n"
            "  pipx inject llm-speed mcp cachetools\n"
            f"\n(underlying import failure: {exc})",
            file=sys.stderr,
        )
        return 1
    # Warn (not block) when LLM_SPEED_API_BASE / LLM_SPEED_SITE_BASE is
    # set to anything other than the canonical hosts. Mirrors the
    # symmetric warning in cli.api_host.warn_if_noncanonical. No
    # bearer token leaks here (the MCP server is read-only / no auth),
    # but a hostile env var would feed poisoned rows to the user's LLM
    # client — which compounds with the I-1 sanitisation defence.
    for var, current, default in (
        ("LLM_SPEED_API_BASE", mcp_api.API_BASE, mcp_api.DEFAULT_API_BASE),
        ("LLM_SPEED_SITE_BASE", mcp_api.SITE_BASE, mcp_api.DEFAULT_SITE_BASE),
    ):
        if current != default:
            print(
                f"warning: MCP server is using non-default {var}='{current}' "
                f"(default: '{default}'). If you did not set this yourself, "
                "abort and inspect your environment for a malicious override "
                "— the LLM client gets fed whatever this server returns.",
                file=sys.stderr,
            )
    mcp_app.run()
    return 0


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
