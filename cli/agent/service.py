"""launchd (macOS) / systemd --user (Linux) service files for `llm-speed agent run`.

User-level only: no sudo, no system daemons. The service runs this exact Python
(`sys.executable`, i.e. pipx's venv) so `pipx upgrade llm-speed` + restart updates it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from . import tokens, wsl
from .pyflags import isolation_flags
from .settings import agent_dir, logs_dir

LABEL = "com.llm-speed.agent"
UNIT = "llm-speed-agent.service"


# The package this module belongs to ("cli"), so the service runs `python -m <it>` even
# if the top-level package is ever renamed; sys.executable is the venv's interpreter
# (pipx / uv tool), kept unresolved so the venv stays active.
PACKAGE = __name__.split(".")[0]

WINDOWS_MESSAGE = (
    "`llm-speed agent install` doesn't run on Windows itself. Run the agent inside WSL2 "
    "instead: a dedicated Ubuntu distro, `uv tool install llm-speed`, `llm-speed doctor`, then "
    "`llm-speed agent install` from that distro (it registers a Task Scheduler task that "
    "starts it at logon). Step by step: docs/CLI.md, section 'Windows (WSL2)' "
    "(https://github.com/meadow-kun/llm-speed-web/blob/main/docs/CLI.md#windows-wsl2)."
)


def program_args() -> list[str]:
    # R-71 / R-74: -I in a venv, else -P (3.11+), so a stray `cli` package in the working
    # directory can't stand in for the agent (the service cwd is pinned to / as well).
    return [
        os.path.abspath(sys.executable),
        *isolation_flags(),
        "-m",
        PACKAGE,
        "agent",
        "run",
    ]


def platform_kind() -> str:
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux"):
        return "wsl-task" if wsl.is_wsl() else "systemd"
    if sys.platform in ("win32", "cygwin"):
        return "windows"
    return "unsupported"


def _check_program(prog: list[str]) -> str | None:
    """None if `python <flags> -m cli --version` runs with the interpreter and flags the
    service will use (from /, so a checkout in the installer's cwd isn't what gets
    checked). Catches e.g. an install that -I can't see."""
    head = prog[: prog.index("-m")] if "-m" in prog else prog[:1]
    shown = " ".join(head)
    try:
        r = subprocess.run(
            [*head, "-m", PACKAGE, "--version"],
            cwd="/",
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{shown} -m {PACKAGE} --version failed: {e}"
    if r.returncode == 0:
        return None
    return (
        f"{shown} -m {PACKAGE} --version exited {r.returncode}: the service could not "
        "start the agent with this Python. Install llm-speed with `uv tool install "
        "llm-speed` or `pipx install llm-speed` and run this again."
    )


def _writable_dir(d: Path) -> str | None:
    probe = d / ".preflight"
    try:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(
            probe,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.close(fd)
        probe.unlink()
    except OSError as e:
        return f"{d} is not writable ({e.strerror or e})"
    return None


def _manager_probe(kind: str) -> str | None:
    """Is the user service manager the install will talk to reachable from here?"""
    if kind == "systemd":
        argv = ["systemctl", "--user", "show-environment"]
        hint = (
            "no systemd user manager in this session (ssh without lingering, or a "
            "container?); log in normally or run `loginctl enable-linger`"
        )
    elif kind == "launchd":
        argv = ["launchctl", "print", f"gui/{os.getuid()}"]
        hint = "no GUI launchd domain (run this in the logged-in desktop session)"
    else:
        return None
    try:
        r = subprocess.run(
            argv, check=False, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"`{' '.join(argv)}` failed ({e}): {hint}"
    if r.returncode != 0:
        return f"`{' '.join(argv)}` exited {r.returncode}: {hint}"
    return None


def preflight(*, dry_run: bool = False, token_store: str | None = None) -> str | None:
    """A reason the service can't be installed on this machine, checked before pairing
    (a single-use enrolment token is spent by pairing; R-74). A dry run runs and writes
    nothing, so it only gets the checks that need neither."""
    kind = platform_kind()
    if kind not in ("wsl-task", "systemd", "launchd"):
        return None
    if kind == "wsl-task":
        bad = wsl.preflight(program_args(), check_windows_user=False)
        if bad:
            return bad
    if dry_run:
        return None
    for d in (agent_dir(), logs_dir()):
        bad = _writable_dir(d)
        if bad:
            return bad
    bad = tokens.probe_store(token_store)
    if bad:
        return bad
    if kind == "wsl-task":
        bad = wsl.preflight(
            program_args(), check_windows_user=True, workdir=agent_dir()
        )
    else:
        bad = _manager_probe(kind)
    return bad or _check_program(program_args())


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def unit_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user" / UNIT


def render_plist() -> str:
    args = "\n".join(f"    <string>{escape(a)}</string>" for a in program_args())
    out = escape(str(logs_dir() / "service.out.log"))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>WorkingDirectory</key><string>/</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>Nice</key><integer>5</integer>
  <key>StandardOutPath</key><string>{out}</string>
  <key>StandardErrorPath</key><string>{out}</string>
</dict>
</plist>
"""


def _sd_quote(a: str) -> str:
    return (
        '"' + a.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if any(c in a for c in ' "\\')
        else a
    )


def render_unit() -> str:
    cmd = " ".join(_sd_quote(a) for a in program_args())
    return f"""[Unit]
Description=llm-speed agent (runs allowlisted benchmark presets from the llm-speed queue)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/
ExecStart={cmd}
Restart=on-failure
RestartSec=60
Nice=5
IOSchedulingClass=idle
KillMode=mixed
TimeoutStopSec=90

[Install]
WantedBy=default.target
"""


def _run(cmd: list[str], dry: bool, out: list[str]) -> int:
    out.append("$ " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, check=False, capture_output=True, text=True).returncode


def install(*, dry_run: bool = False) -> tuple[int, list[str]]:
    kind = platform_kind()
    steps: list[str] = []
    if not dry_run:
        logs_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
    if kind == "wsl-task":
        prog = program_args()
        warn = wsl.shared_distro_warning(wsl.distro_name(), wsl.own_distro_listeners())
        if warn:
            steps.append(warn)
        bad = None if dry_run else _check_program(prog)
        if bad:
            return 1, steps + [bad]
        rc, more = wsl.install(prog, dry_run=dry_run, workdir=agent_dir())
        return rc, steps + more
    if kind == "launchd":
        p = plist_path()
        steps.append(f"write {p}")
        if not dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(render_plist(), encoding="utf-8")
        dom = f"gui/{os.getuid()}"
        _run(
            ["launchctl", "bootout", f"{dom}/{LABEL}"], dry_run, steps
        )  # ok if not loaded
        rc = _run(["launchctl", "bootstrap", dom, str(p)], dry_run, steps)
        return rc, steps
    if kind == "systemd":
        p = unit_path()
        steps.append(f"write {p}")
        if not dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(render_unit(), encoding="utf-8")
        _run(["systemctl", "--user", "daemon-reload"], dry_run, steps)
        rc = _run(["systemctl", "--user", "enable", "--now", UNIT], dry_run, steps)
        steps.append(
            "note: to keep it running without a login session: sudo loginctl enable-linger $USER"
        )
        return rc, steps
    if kind == "windows":
        return 1, [WINDOWS_MESSAGE]
    return 1, ["unsupported platform"]


def uninstall(*, dry_run: bool = False) -> tuple[int, list[str]]:
    kind = platform_kind()
    steps: list[str] = []
    if kind == "wsl-task":
        return wsl.uninstall(dry_run=dry_run, workdir=agent_dir())
    if kind == "launchd":
        _run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], dry_run, steps)
        p = plist_path()
        steps.append(f"remove {p}")
        if not dry_run:
            p.unlink(missing_ok=True)
        return 0, steps
    if kind == "systemd":
        _run(["systemctl", "--user", "disable", "--now", UNIT], dry_run, steps)
        p = unit_path()
        steps.append(f"remove {p}")
        if not dry_run:
            p.unlink(missing_ok=True)
        _run(["systemctl", "--user", "daemon-reload"], dry_run, steps)
        return 0, steps
    if kind == "windows":
        return 1, [WINDOWS_MESSAGE]
    return 1, ["unsupported platform"]


def restart() -> int:
    kind = platform_kind()
    if kind == "wsl-task":
        return wsl.restart()
    if kind == "launchd":
        return subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{LABEL}"], check=False
        ).returncode
    if kind == "systemd":
        return subprocess.run(
            ["systemctl", "--user", "restart", UNIT], check=False
        ).returncode
    return 1


def service_state() -> str:
    kind = platform_kind()
    try:
        if kind == "wsl-task":
            return wsl.task_state()
        if kind == "launchd":
            if not plist_path().exists():
                return "not installed"
            r = subprocess.run(
                ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"],
                check=False,
                capture_output=True,
                text=True,
            )
            return "loaded" if r.returncode == 0 else "installed, not loaded"
        if kind == "systemd":
            if not unit_path().exists():
                return "not installed"
            r = subprocess.run(
                ["systemctl", "--user", "is-active", UNIT],
                check=False,
                capture_output=True,
                text=True,
            )
            return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"
    return "unsupported"
