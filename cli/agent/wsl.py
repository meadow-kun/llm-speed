"""WSL2 support for `llm-speed agent` (design: docs/cli_wsl2_agent_design.md).

Inside a WSL2 distro the agent is the Linux agent, started by a per-user Windows Task
Scheduler task at logon (`wsl.exe -d <this distro> -u <user> --exec <python> -m cli agent run`):
systemd services don't keep a WSL instance alive, a process launched by wsl.exe does.

Rules this module enforces (tests pin them):
- never touches %UserProfile%\\.wslconfig and never runs `wsl --shutdown` (it would stop
  every distro, including other people's services);
- the only distro it manages is the one it runs in; `wsl --terminate` only for that one;
- never stops or reconfigures other services; a distro that looks shared gets a warning.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

RECOMMENDED_DISTRO = "llm-speed-agent"
SYSTEM32 = "/mnt/c/Windows/System32"
SCHTASKS = f"{SYSTEM32}/schtasks.exe"
WSL_EXE = f"{SYSTEM32}/wsl.exe"
WHOAMI = f"{SYSTEM32}/whoami.exe"
WIN_SYSTEM32 = "C:\\Windows\\System32"
# Task name prefix in the ROOT Task Scheduler folder. A standard (non-elevated) user may
# register a task for themself in the root folder, but creating a new task folder such as
# \llm-speed\ needs elevation, so the task never goes into a folder of its own.
TASK_PREFIX = "llm-speed-agent-"
WSL_LIB = "/usr/lib/wsl/lib"


class WslRefused(RuntimeError):
    """A WSL / Windows command this module will never run."""


def is_wsl(osrelease: Path = Path("/proc/sys/kernel/osrelease")) -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in osrelease.read_text(encoding="utf-8").lower()
    except OSError:
        return False


# R-72: the only distro names this module works with. Anything else (quotes, spaces,
# tabs, ...) is refused rather than sanitised, so two distros can't map to one task.
_DISTRO_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def distro_name() -> str | None:
    """The distro we are running in (WSL sets WSL_DISTRO_NAME for every process it
    starts). None when unset or not a plain [A-Za-z0-9._-]{1,64} name."""
    name = (os.environ.get("WSL_DISTRO_NAME") or "").strip()
    return name if _DISTRO_RE.fullmatch(name) else None


def task_name(distro: str) -> str:
    if not _DISTRO_RE.fullmatch(distro or ""):
        raise WslRefused("unexpected distro name")
    return f"\\{TASK_PREFIX}{distro}"


def _run(argv: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, check=False, capture_output=True, text=True, timeout=timeout
    )


# ----------------------------------------------------------------- the wsl.exe guard


def wsl_exe_argv(args: list[str], own: str) -> list[str]:
    """The only way this module builds a wsl.exe command line. Refuses --shutdown and any
    distro other than `own`, and anything that changes WSL-wide settings."""
    low = [a.lower() for a in args]
    banned = {
        "--shutdown",
        "--set-default",
        "-s",
        "--set-version",
        "--set-default-version",
        "--unregister",
        "--update",
        "--install",
        "--manage",
        "--export",
        "--import",
        "--mount",
        "--unmount",
    }
    if banned & set(low):
        raise WslRefused(
            f"refusing to run `wsl.exe {' '.join(args)}` (never shuts down or reconfigures WSL)"
        )
    for flag in ("-d", "--distribution", "-t", "--terminate", "--distribution-id"):
        if flag in low:
            i = low.index(flag)
            if i + 1 >= len(args) or args[i + 1] != own:
                raise WslRefused(
                    f"refusing to manage a distro other than this one ({own})"
                )
    if not any(f in low for f in ("-d", "--distribution", "-t", "--terminate")):
        raise WslRefused("every wsl.exe call must name this distro explicitly")
    return [WSL_EXE, *args]


def terminate_own_distro(own: str) -> list[str]:
    """`wsl --terminate <own>` (stops only this distro). Not called by install/uninstall;
    provided so nothing ever needs `wsl --shutdown`."""
    return wsl_exe_argv(["--terminate", own], own)


# ----------------------------------------------------------------- Windows side


# Built-in service identities a task must never run as (whoami.exe prints these when the
# caller is itself a service, e.g. "nt authority\\system").
_SERVICE_DOMAINS = (
    "nt authority\\",
    "nt service\\",
    "font driver host\\",
    "window manager\\",
)


def windows_user(run=None) -> str | None:
    """DOMAIN\\user of the Windows account (for the logon trigger; stays on this machine).
    None for a service identity (SYSTEM, LocalService, ...): the task runs as the person
    logged on, never as a service account."""
    run = run or _run
    try:
        r = run([WHOAMI])
    except (OSError, subprocess.TimeoutExpired):
        return None
    u = (r.stdout or "").strip().splitlines()
    if not (r.returncode == 0 and u and "\\" in u[0]):
        return None
    user = u[0].strip()
    return None if user.lower().startswith(_SERVICE_DOMAINS) else user


NO_DISTRO = (
    "WSL detected but WSL_DISTRO_NAME is not set or is not a plain distro name "
    "(letters, digits, '.', '_', '-'; at most 64)"
)
NO_WINDOWS_USER = (
    f"could not get the logged-on Windows user from {WHOAMI} (is WSL interop "
    "enabled in /etc/wsl.conf? the task is never created for a service account)"
)


def _is_root() -> bool:
    return hasattr(os, "getuid") and os.getuid() == 0


def linux_user() -> str | None:
    """The distro user installing the agent. The task passes it to wsl.exe (-u) so the
    agent runs as this user, not as the distro's default user (which may be root)."""
    import pwd

    try:
        name = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        return None
    return name if re.fullmatch(r"[a-z_][a-z0-9_-]*[$]?", name or "") else None


def to_windows_path(path: Path, run=None) -> str | None:
    run = run or _run
    try:
        r = run(["wslpath", "-w", str(path)])
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (r.stdout or "").strip()
    return out if r.returncode == 0 and out else None


def task_command(
    distro: str, program: list[str], linux_user: str | None = None
) -> tuple[str, str]:
    """(Command, Arguments) of the task action. conhost --headless keeps the console
    window hidden; wsl.exe --exec runs the agent directly (no shell, no profile);
    `-u USER` pins the distro user that installed it."""
    if not _DISTRO_RE.fullmatch(distro or ""):
        raise WslRefused("unexpected distro name")
    for a in [distro, *program, linux_user or ""]:
        if any(c in a for c in '"\r\n\t'):
            raise WslRefused("unexpected character in the agent command line")
    user_args = ["-u", linux_user] if linux_user else []
    # --cd /: a fixed cwd instead of Task Scheduler's default (System32); R-74
    wsl_args = wsl_exe_argv(
        ["-d", distro, *user_args, "--cd", "/", "--exec", *program], distro
    )[1:]
    args = ["--headless", f"{WIN_SYSTEM32}\\wsl.exe", *wsl_args]
    quoted = subprocess.list2cmdline(args)  # MS C runtime quoting rules
    return f"{WIN_SYSTEM32}\\conhost.exe", quoted


def render_task_xml(
    distro: str, user: str, program: list[str], linux_user: str | None = None
) -> str:
    """Logon task of `user` only: InteractiveToken (runs only while that user is logged
    on, with their own non-elevated token; no stored password) and LeastPrivilege (never
    "HighestAvailable"). No SYSTEM / service principal, no group principal."""
    cmd, args = task_command(distro, program, linux_user)
    u = escape(user)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>llm-speed agent in WSL distro {escape(distro)} (runs allowlisted benchmark presets from the llm-speed queue). Remove with: llm-speed agent uninstall</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>{u}</UserId>
      <Delay>PT1M</Delay>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{u}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings><StopOnIdleEnd>false</StopOnIdleEnd><RestartOnIdle>false</RestartOnIdle></IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure><Interval>PT5M</Interval><Count>288</Count></RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(cmd)}</Command>
      <Arguments>{escape(args)}</Arguments>
    </Exec>
  </Actions>
</Task>
"""


def schtasks_create(name: str, xml_winpath: str) -> list[str]:
    return [SCHTASKS, "/Create", "/TN", name, "/XML", xml_winpath, "/F"]


def schtasks_run(name: str) -> list[str]:
    return [SCHTASKS, "/Run", "/TN", name]


def schtasks_end(name: str) -> list[str]:
    return [SCHTASKS, "/End", "/TN", name]


def schtasks_delete(name: str) -> list[str]:
    return [SCHTASKS, "/Delete", "/TN", name, "/F"]


def schtasks_query(name: str) -> list[str]:
    return [SCHTASKS, "/Query", "/TN", name, "/FO", "CSV", "/NH"]


def parse_query_status(out: str) -> str | None:
    """`schtasks /Query /FO CSV /NH`: "TaskName","Next Run Time","Status" (localized)."""
    for line in (out or "").replace("\r", "\n").split("\n"):
        cells = re.findall(r'"([^"]*)"', line)
        if len(cells) >= 3:
            return cells[2] or None
    return None


# ----------------------------------------------------------------- shared-distro check


def _listen_inodes(proc: Path) -> dict[str, int]:
    """socket inode -> port for TCP listeners (state 0A) visible here."""
    out: dict[str, int] = {}
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc / "net" / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) > 9 and parts[3] == "0A":
                out[parts[9]] = int(parts[1].rsplit(":", 1)[1], 16)
    return out


def own_distro_listeners(
    proc: Path = Path("/proc"), self_pid: int | None = None
) -> list[tuple[int, str]]:
    """(port, process name) for TCP listeners owned by processes of THIS distro.
    WSL2 distros share one network namespace, so /proc/net/tcp also lists other
    distros' sockets; only a socket held by a process we can see here is ours."""
    inodes = _listen_inodes(proc)
    if not inodes:
        return []
    found: dict[int, str] = {}
    me = str(self_pid or os.getpid())
    for d in proc.iterdir():
        if not d.name.isdigit() or d.name == me:
            continue
        try:
            fds = list((d / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            m = re.fullmatch(r"socket:\[(\d+)\]", target)
            if m and m.group(1) in inodes:
                try:
                    comm = (d / "comm").read_text().strip()
                except OSError:
                    comm = "?"
                found.setdefault(inodes[m.group(1)], comm)
    return sorted(found.items())


def shared_distro_warning(
    distro: str | None, listeners: list[tuple[int, str]]
) -> str | None:
    if not listeners and distro == RECOMMENDED_DISTRO:
        return None
    parts = []
    if listeners:
        svc = ", ".join(f":{p} ({c})" for p, c in listeners[:5])
        parts.append(f"this distro also runs long-lived services ({svc})")
    if distro != RECOMMENDED_DISTRO:
        parts.append(
            f"it is '{distro or '?'}', not a dedicated '{RECOMMENDED_DISTRO}' distro"
        )
    return (
        "Warning: "
        + "; ".join(parts)
        + ". Benchmarks here share its RAM, disk and GPU with those services. Recommended: "
        f"a separate distro (`wsl --install Ubuntu-24.04 --name {RECOMMENDED_DISTRO}` from "
        "Windows, see docs/CLI.md 'Windows (WSL2)'). Nothing else in this distro is "
        "stopped or changed."
    )


# ----------------------------------------------------------------- install / uninstall


def preflight(
    program: list[str],
    *,
    check_windows_user: bool = False,
    run=None,
    workdir: Path | None = None,
) -> str | None:
    """Why the logon task can't be installed from here, checked before pairing (a
    single-use enrolment token must not be spent on a machine that then can't run the
    agent). None when it can. `check_windows_user` also runs whoami.exe, `wslpath -w
    workdir` and `schtasks.exe /Query` (not in a dry run, which runs nothing)."""
    if not distro_name():
        return NO_DISTRO + "; run this from a WSL shell"
    for p in program[:1]:
        if p.startswith("/mnt/"):
            return (
                f"the agent's Python is on a Windows drive ({p}); install llm-speed inside "
                "the distro (`uv tool install llm-speed`) and run this again"
            )
    if _is_root():
        return (
            "refusing to install the agent as root: run `llm-speed agent install` as your "
            "normal user in this distro (the one `wsl --install` created)"
        )
    if not linux_user():
        return "could not determine the Linux user name of this process"
    if not check_windows_user:
        return None
    # The Windows side, before pairing too (R-74; not in a dry run, which runs nothing):
    # whoami.exe, wslpath -w for the task file's directory, and schtasks.exe itself.
    run = run or _run
    if not windows_user(run):
        return NO_WINDOWS_USER
    if workdir is not None and not to_windows_path(workdir, run):
        return (
            f"`wslpath -w {workdir}` failed, so Task Scheduler could not be given the task "
            "file"
        )
    try:
        r = run(schtasks_query(task_name(distro_name() or "")))
    except (OSError, subprocess.TimeoutExpired) as e:
        return (
            f"could not run {SCHTASKS} ({e}); Windows interop may be off in this distro "
            "(/etc/wsl.conf [interop] enabled=false)"
        )
    if r.returncode not in (0, 1):  # 1 = no such task yet
        return f"{SCHTASKS} /Query exited {r.returncode}"
    return None


def _write_private(path: Path, data: bytes) -> None:
    """R-73: 0600, never through a symlink, replacing any older copy."""
    fd = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, data)
    finally:
        os.close(fd)


def install(
    program: list[str], *, dry_run: bool, run=None, workdir: Path
) -> tuple[int, list[str]]:
    """Register (or replace) the logon task for THIS distro and start it once."""
    run = run or _run
    steps: list[str] = []
    bad = preflight(program)
    if bad:
        return 1, [bad]
    distro = distro_name() or ""
    lx_user = linux_user()
    name = task_name(distro)
    user = "DOMAIN\\user" if dry_run else windows_user(run)
    if not user:
        return 1, [NO_WINDOWS_USER]
    xml = render_task_xml(distro, user, program, lx_user)
    xml_path = workdir / "agent-task.xml"
    steps.append(
        f"write {xml_path} (Task Scheduler definition, logon of the current Windows user)"
    )
    if not dry_run:
        workdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_private(xml_path, xml.encode("utf-16"))
    winpath = (
        "\\\\wsl.localhost\\<distro>\\..."
        if dry_run
        else to_windows_path(xml_path, run)
    )
    if not winpath:
        xml_path.unlink(missing_ok=True)
        return 1, steps + ["could not convert the task file path with wslpath -w"]
    for argv in (schtasks_create(name, winpath), schtasks_run(name)):
        steps.append("$ " + " ".join(argv))
        if dry_run:
            continue
        try:
            r = run(argv)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 1, steps + [f"failed: {e}"]
        finally:
            if argv[1] == "/Create":
                # R-73: schtasks keeps its own copy; the file isn't needed after /Create
                xml_path.unlink(missing_ok=True)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "").strip().splitlines()
            return r.returncode, steps + [f"failed: {msg[-1] if msg else r.returncode}"]
    steps.append(
        f"Windows task {name} starts the agent at every logon of this Windows user "
        f"(wsl.exe -d {distro} -u {lx_user} --exec ...). No .wslconfig change, no "
        "wsl --shutdown, no admin rights, no listening port."
    )
    return 0, steps


def uninstall(
    *, dry_run: bool, run=None, workdir: Path | None = None
) -> tuple[int, list[str]]:
    run = run or _run
    if workdir is not None and not dry_run:
        (workdir / "agent-task.xml").unlink(
            missing_ok=True
        )  # R-73 (left by 0.0.11 pre-builds)
    distro = distro_name()
    if not distro:
        return 1, [NO_DISTRO]
    name = task_name(distro)
    steps: list[str] = []
    rc = 0
    for argv, must in ((schtasks_end(name), False), (schtasks_delete(name), True)):
        steps.append("$ " + " ".join(argv))
        if dry_run:
            continue
        try:
            r = run(argv)
        except (OSError, subprocess.TimeoutExpired) as e:
            steps.append(f"failed: {e}")
            rc = rc or (1 if must else 0)
            continue
        if (
            must
            and r.returncode != 0
            and "cannot find" not in (r.stderr or r.stdout or "").lower()
        ):
            rc = r.returncode
    steps.append(
        "If `llm-speed agent status` still shows the agent working, run "
        "`llm-speed agent pause --now`; nothing else in this distro was touched."
    )
    return rc, steps


def task_state(run=None) -> str:
    run = run or _run
    distro = distro_name()
    if not distro:
        return "unknown (WSL_DISTRO_NAME not set)"
    try:
        r = run(schtasks_query(task_name(distro)))
    except (OSError, subprocess.TimeoutExpired):
        return "unknown (schtasks.exe not reachable)"
    if r.returncode != 0:
        return "not installed"
    st = parse_query_status(r.stdout)
    return f"Windows task {task_name(distro)}: {st or 'installed'}"


def restart(run=None) -> int:
    run = run or _run
    distro = distro_name()
    if not distro:
        return 1
    name = task_name(distro)
    for argv in (schtasks_end(name), schtasks_run(name)):
        try:
            r = run(argv)
        except (OSError, subprocess.TimeoutExpired):
            return 1
    return r.returncode


# ----------------------------------------------------------------- doctor helpers

POWERSHELL = f"{SYSTEM32}/WindowsPowerShell/v1.0/powershell.exe"
HOST_RAM_ARGV = [
    POWERSHELL,
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
]
LINUX_DRIVER_PKG = re.compile(
    r"^(nvidia-driver-|nvidia-dkms-|cuda-drivers|libnvidia-compute-|nvidia-kernel-)"
)


def vm_ram_gb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_ram_gb(run=None) -> float | None:
    run = run or _run
    try:
        r = run(HOST_RAM_ARGV, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (r.stdout or "").strip()
    return round(int(out) / 1024**3, 1) if r.returncode == 0 and out.isdigit() else None


def linux_nvidia_driver_packages(
    status: Path = Path("/var/lib/dpkg/status"),
) -> list[str]:
    """Installed Debian/Ubuntu packages that put a Linux NVIDIA driver inside WSL."""
    try:
        text = status.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for block in text.split("\n\n"):
        pkg = re.search(r"^Package:\s*(\S+)", block, re.M)
        st = re.search(r"^Status:\s*(.+)$", block, re.M)
        if (
            pkg
            and st
            and "installed" in st.group(1).split()
            and LINUX_DRIVER_PKG.match(pkg.group(1))
        ):
            out.append(pkg.group(1))
    return sorted(out)


def stray_libcuda(lib_dirs=("/usr/lib/x86_64-linux-gnu", "/usr/lib64")) -> list[str]:
    """libcuda.so.1 copies outside /usr/lib/wsl/lib that don't point back into it (they
    shadow the Windows driver's WSL stub)."""
    out = []
    for d in lib_dirs:
        p = Path(d) / "libcuda.so.1"
        if p.exists() and not os.path.realpath(p).startswith(WSL_LIB):
            out.append(str(p))
    return out
