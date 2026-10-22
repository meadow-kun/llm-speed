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

from .settings import logs_dir

LABEL = "com.llm-speed.agent"
UNIT = "llm-speed-agent.service"


def program_args() -> list[str]:
    return [sys.executable, "-m", "cli", "agent", "run"]


def platform_kind() -> str:
    if sys.platform == "darwin":
        return "launchd"
    if sys.platform.startswith("linux"):
        return "systemd"
    return "unsupported"


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
    return 1, ["unsupported platform (Windows support comes later)"]


def uninstall(*, dry_run: bool = False) -> tuple[int, list[str]]:
    kind = platform_kind()
    steps: list[str] = []
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
    return 1, ["unsupported platform"]


def restart() -> int:
    kind = platform_kind()
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
