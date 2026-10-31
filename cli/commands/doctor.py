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


def agent_token_checks() -> list[Check]:
    """If this machine is enrolled as an agent: which store holds its token, and
    whether it can be read (never the token). Not required for `bench`."""
    try:
        from ..agent.settings import Settings
        from ..agent.tokens import describe_store, load_token

        s = Settings.load()
    except (OSError, ValueError, TypeError):  # unreadable agent config: nothing to say
        return []
    if not s.machine_id:
        return []
    where = describe_store(s.token_store)
    if load_token(s.machine_id, s.token_store):
        return [Check(id="agent.token", title="agent token", status=OK, detail=where)]
    return [
        Check(
            id="agent.token",
            title="agent token",
            status=WARN,
            detail=f"enrolled as {s.machine_id} but no token found ({where})",
            remediation="ask your queue admin to revoke this machine, then run "
            "`llm-speed agent install --force` to pair again",
        )
    ]


def _on_windows_drive(p: str) -> bool:
    import os

    return os.path.realpath(os.path.expanduser(p)).startswith("/mnt/")


def wsl_checks(*, runner=None, force: bool | None = None) -> list[Check]:
    """Under WSL2 only: where models live, CUDA, the Windows nvidia-smi.exe the busy check
    reads, VM memory vs the host, a Linux NVIDIA driver installed by mistake, and a
    CUDA llama.cpp build. Read-only; never changes WSL or Windows settings."""
    import os
    from pathlib import Path

    from ..agent import gates, wsl
    from ..config import CACHE_DIR

    if not (wsl.is_wsl() if force is None else force):
        return []
    checks: list[Check] = []
    try:
        from ..agent.settings import Settings

        s = Settings.load()
    except (OSError, ValueError, TypeError):
        s = None
    dirs = {
        "llm-speed cache": str(CACHE_DIR),
        "Hugging Face cache": os.environ.get("HF_HOME") or "~/.cache/huggingface",
    }
    if s and s.models_dir:
        dirs["agent models_dir"] = s.models_dir
    bad = [f"{k} {v}" for k, v in dirs.items() if _on_windows_drive(v)]
    checks.append(
        Check(
            id="wsl.models",
            title="WSL: models on the distro disk",
            status=WARN if bad else OK,
            detail=("on a Windows drive (slow 9P reads): " + "; ".join(bad))
            if bad
            else "cache and models dirs are inside the distro",
            remediation="keep models under the distro's home (unset HF_HOME / "
            "`llm-speed agent config models_dir auto`)"
            if bad
            else None,
        )
    )
    cuda = Path(wsl.WSL_LIB) / "libcuda.so.1"
    checks.append(
        Check(
            id="wsl.cuda",
            title="WSL: CUDA from the Windows driver",
            status=OK if cuda.exists() else WARN,
            detail=str(cuda) if cuda.exists() else f"{cuda} missing",
            remediation=None
            if cuda.exists()
            else "install or update the NVIDIA driver on Windows (no Linux driver inside "
            "WSL); if it is still missing, WSL itself may be too old, but `wsl --update` "
            "stops every running distro, so only run it when nothing else in WSL needs "
            "to keep running (llm-speed never runs it)",
        )
    )
    run = runner or gates._run
    smi, util, notes, _present = gates.read_nvidia_util(
        s or Settings(), runner=run, wsl=True
    )
    exe_ok = bool(smi and smi.endswith(".exe"))
    checks.append(
        Check(
            id="wsl.nvidia_smi",
            title="WSL: Windows nvidia-smi.exe (busy check)",
            status=OK if exe_ok else WARN,
            detail=f"{smi}: utilization {util}%"
            if exe_ok
            else (
                f"not readable ({'; '.join(notes) or 'not found'}); "
                + (
                    f"using {smi}, which can't see Windows apps"
                    if smi
                    else "the agent will treat the GPU as busy"
                )
            ),
            remediation=None
            if exe_ok
            else "check [interop] enabled=true in /etc/wsl.conf, or pin it: "
            "`llm-speed agent config nvidia_smi /mnt/c/Windows/System32/nvidia-smi.exe`",
        )
    )
    vm, host = wsl.vm_ram_gb(), wsl.host_ram_gb()
    checks.append(
        Check(
            id="wsl.memory",
            title="WSL: memory",
            status=OK,
            detail=(
                f"the VM sees {vm} GB" + (f" of the host's {host} GB" if host else "")
            )
            + "; WSL's default cap is 50% of host RAM (`memory=` in .wslconfig, a global "
            "setting llm-speed never changes); CPU offload is limited by the VM's share",
        )
    )
    pkgs, libs = wsl.linux_nvidia_driver_packages(), wsl.stray_libcuda()
    checks.append(
        Check(
            id="wsl.linux_driver",
            title="WSL: no Linux NVIDIA driver inside WSL",
            status=WARN if (pkgs or libs) else OK,
            detail=("found " + ", ".join(pkgs + libs)) if (pkgs or libs) else "none",
            remediation=(
                "NVIDIA: never install a Linux display driver inside WSL; it shadows the "
                "Windows driver's /usr/lib/wsl/lib/libcuda.so. Remove it: "
                f"sudo apt remove {' '.join(pkgs) or '<the nvidia driver package>'}"
            )
            if (pkgs or libs)
            else None,
        )
    )
    checks.append(_wsl_llama_check())
    return checks


def _wsl_llama_check() -> Check:
    from pathlib import Path

    from . import preset as pr

    path, info, _ = pr.find_llama_server(None, None)
    if not path:
        return Check(
            id="wsl.llama_cpp",
            title="WSL: llama.cpp CUDA build",
            status=WARN,
            detail="no llama-server found",
            remediation="`llm-speed bench --preset <name>` offers the official CUDA 12.8+ "
            f"Linux build (unpacked under {pr.prebuilt_root()}); RTX 50xx needs CUDA 12.8+",
        )
    cuda = (
        "CUDA" in ((info or {}).get("flags") or [])
        or (Path(path).parent / "libggml-cuda.so").exists()
    )
    return Check(
        id="wsl.llama_cpp",
        title="WSL: llama.cpp CUDA build",
        status=OK if cuda else WARN,
        detail=f"{Path(path).name} {(info or {}).get('build') or '?'}"
        + ("" if cuda else " (no CUDA backend found next to it)"),
        remediation=None
        if cuda
        else "use a CUDA 12.8+ build: `llm-speed bench --preset <name>` offers the "
        "official one",
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
    do_checks = run_checks_fn or (
        lambda: run_checks(api_base=api_base) + agent_token_checks() + wsl_checks()
    )
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
