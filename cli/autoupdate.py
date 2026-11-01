"""Self-update for the llm-speed CLI (0.0.14).

On by default, like Claude Code's updater. At most once a day a command starts a
detached background check of https://pypi.org/pypi/llm-speed/json; if a newer release
exists it is installed with the same installer that installed this copy (pipx, uv tool,
Homebrew, or pip inside a writable virtual environment), and the next run prints
``Updated llm-speed A -> B`` once (stderr). The current command never waits for it.

Opt out: ``LLM_SPEED_DISABLE_AUTOUPDATE=1`` or ``llm-speed config set auto_update
false``. Also off when a CI variable is set or stdout is not a terminal, except for
``llm-speed agent run``, which updates between jobs (never during a claim or a run) and
then restarts itself (``restart_agent``).

Safety (docs/CLI.md "Auto-update"): only the package ``llm-speed``; the target version
comes from PyPI JSON and is validated and compared; never a downgrade; pre-releases and
yanked releases skipped unless ``auto_update_prereleases`` is true; https only, no
redirects, 10 s timeout, 4 MiB response cap; index / find-links / constraint settings
from the environment and pip/uv config files are removed from the installer's
environment; every subprocess is an argv list from absolute paths, never a shell; one
update at a time (lockfile). Nothing is sent to llm-speed servers.

Test-only hook (installer gate): honoured only when BOTH ``LLM_SPEED_UPDATE_TEST=1`` and
``LLM_SPEED_UPDATE_TEST_WHEEL=file:///abs/path/llm_speed-<ver>-py3-none-any.whl`` (an
existing local file) are set. The version then comes from the wheel's file name instead of
PyPI, and the installer gets ``--find-links <that wheel's dir>`` so the same pinned spec
``llm-speed==<ver>`` resolves to it. Anything else (http URLs, other names, a missing file,
only one of the two variables) is ignored and PyPI is used.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from . import __version__

log = logging.getLogger("cli.autoupdate")

PACKAGE = "llm-speed"
DIST = "llm_speed"
BREW_FORMULA = "meadow-kun/tap/llm-speed"
PYPI_JSON_URL = "https://pypi.org/pypi/llm-speed/json"
PYPI_SIMPLE = "https://pypi.org/simple"
FETCH_TIMEOUT_S = 10.0
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
CHECK_INTERVAL_S = 24 * 3600
INSTALL_TIMEOUT_S = 15 * 60
WINDOWS_PARENT_WAIT_S = 15 * 60

DISABLE_ENV = "LLM_SPEED_DISABLE_AUTOUPDATE"
TEST_ENV = "LLM_SPEED_UPDATE_TEST"
TEST_WHEEL_ENV = "LLM_SPEED_UPDATE_TEST_WHEEL"
# Local, untracked tap the installer gate creates with `brew tap-new --no-git`.
GATE_TAP = "llm-speed-gate/local"
CONFIG_KEY = "auto_update"
PRERELEASE_KEY = "auto_update_prereleases"
CACHE_NAME = "update-check.json"
LOCK_NAME = "update.lock"
INUSE_NAME = "in-use"  # R-136: one <pid> file per running bench / job / command
BUSY_RETRY_S = 3600
INUSE_TRIES = 20  # R-140: marker create/lock attempts, INUSE_RETRY_S apart
INUSE_RETRY_S = 0.05

# Exit code of `agent run` when an update was installed but the in-place re-exec could not
# happen: non-zero, so systemd (Restart=on-failure), launchd (KeepAlive SuccessfulExit=false)
# and the WSL2 task (RestartOnFailure) start the new version. EX_TEMPFAIL.
AGENT_RESTART_EXIT = 75

# Installer settings that could point the install somewhere other than PyPI, or add a
# package spec, are dropped from the installer's environment.
_SCRUB_PREFIXES = ("PIP_", "UV_", "PIPX_DEFAULT_")
_KEEP_ENV = {
    # where the tool lives (needed to find this install), caches, and the interpreter
    "UV_TOOL_DIR",
    "UV_TOOL_BIN_DIR",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "UV_PYTHON_DOWNLOADS",
    "UV_NATIVE_TLS",
    "PIP_CACHE_DIR",
    "PIP_CERT",
    "PIP_CLIENT_CERT",
    "PIP_PROXY",
    "PIP_TIMEOUT",
    "PIP_RETRIES",
}

_VER_RE = re.compile(
    r"^(\d+(?:\.\d+){0,3})(?:(a|b|rc)(\d+))?(?:\.post(\d+))?(?:\.dev(\d+))?$"
)

# ------------------------------------------------------------------ versions


def parse_version(v: Any):
    """A comparable key, or None if `v` is not a plain release version. packaging.version
    when installed, else a strict regex (N[.N]{0,3}[aN|bN|rcN][.postN][.devN])."""
    if not isinstance(v, str) or len(v) > 40 or not v.strip() or v != v.strip():
        return None
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:  # pragma: no cover - packaging is usually present
        Version = None  # type: ignore[assignment]
    if Version is not None:
        try:
            pv = Version(v)
        except InvalidVersion:
            return None
        if pv.local or pv.epoch:
            return None
        return pv
    return _regex_version(v)


@dataclass(frozen=True, order=True)
class _RegexVersion:
    key: tuple
    pre: bool
    text: str = field(default="", compare=False)

    @property
    def is_prerelease(self) -> bool:
        return self.pre

    def __str__(self) -> str:  # the canonical version, like packaging's Version
        return self.text


def _regex_version(v: str):
    m = _VER_RE.match(v)
    if not m:
        return None
    rel = tuple(int(x) for x in m.group(1).split("."))
    rel = rel + (0,) * (4 - len(rel))
    pre_rank = {"a": 0, "b": 1, "rc": 2, None: 3}[m.group(2)]
    pre_n = int(m.group(3) or 0)
    post = int(m.group(4)) if m.group(4) is not None else -1
    dev = int(m.group(5)) if m.group(5) is not None else None
    if dev is not None and pre_rank == 3 and post < 0:
        pre_rank = -1  # 1.0.dev1 < 1.0a1 < 1.0
    key = (rel, pre_rank, pre_n, post, 10**9 if dev is None else dev)
    text = ".".join(str(int(x)) for x in m.group(1).split("."))
    if m.group(2):
        text += f"{m.group(2)}{pre_n}"
    if post >= 0:
        text += f".post{post}"
    if dev is not None:
        text += f".dev{dev}"
    return _RegexVersion(key, m.group(2) is not None or dev is not None, text)


def is_prerelease(pv) -> bool:
    return bool(getattr(pv, "is_prerelease", False))


def newer(candidate: str, current: str = __version__) -> bool:
    a, b = parse_version(candidate), parse_version(current)
    return a is not None and b is not None and a > b


# ------------------------------------------------------------------ config + cache


def _config_dir() -> Path:
    from . import config as _cfg  # read at call time so tests can redirect it

    return Path(_cfg.CONFIG_DIR)


def _load_config() -> dict[str, Any]:
    from .credit import load_config

    return load_config()


def config_auto_update() -> bool:
    """Saved setting; default on. Off for JSON false or a false-looking string."""
    v = _load_config().get(CONFIG_KEY, True)
    if v is False:
        return False
    if isinstance(v, str) and v.strip().lower() in ("false", "off", "0", "no"):
        return False
    return True


def config_prereleases() -> bool:
    """Only JSON true counts (R-103 style)."""
    return _load_config().get(PRERELEASE_KEY) is True


def cache_path() -> Path:
    return _config_dir() / CACHE_NAME


def load_cache() -> dict[str, Any]:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(data: dict[str, Any]) -> None:
    p = cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except OSError as exc:
        log.debug("update cache not written: %s", exc)


def update_cache(**kw: Any) -> dict[str, Any]:
    data = load_cache()
    for k, v in kw.items():
        if v is None:
            data.pop(k, None)
        else:
            data[k] = v
    save_cache(data)
    return data


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def check_due(now: float | None = None, cache: dict | None = None) -> bool:
    now = time.time() if now is None else now
    last = _num((load_cache() if cache is None else cache).get("checked_at"))
    if last > now + 3600:  # clock went backwards: don't stay throttled for days
        return True
    return now - last >= CHECK_INTERVAL_S


# ------------------------------------------------------------------ opt-outs


def _truthy(v: str | None) -> bool:
    return v is not None and v.strip().lower() not in ("", "0", "false", "no", "off")


def _ci() -> bool:
    from .credit import ci_detected

    return ci_detected()


def _stdout_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def disabled_reason(*, agent: bool = False) -> str | None:
    """Why auto-update is off right now, or None if it may run. `agent` is
    `llm-speed agent run`: it has no terminal, so the non-TTY rule doesn't apply."""
    if _truthy(os.environ.get(DISABLE_ENV)):
        return f"{DISABLE_ENV} is set"
    if not config_auto_update():
        return "auto_update is off (llm-speed config set auto_update true)"
    if _ci():
        return "CI environment"
    if not agent:
        from .agent_job import FD_ENV

        if os.environ.get(FD_ENV):
            return "inside an agent job"
        if not _stdout_tty():
            return "stdout is not a terminal"
    return None


# ------------------------------------------------------------------ PyPI


class FetchError(Exception):
    pass


def _test_wheel() -> Path | None:
    """The test-only wheel (see module docstring), or None."""
    if os.environ.get(TEST_ENV) != "1":
        return None
    raw = os.environ.get(TEST_WHEEL_ENV) or ""
    u = urlparse(raw)
    if u.scheme != "file" or u.netloc not in ("", "localhost") or not u.path:
        return None
    path = unquote(u.path)
    # no UNC / network forms: file:////server/share, //server, \\server
    if path.startswith(("//", "\\")) or "\\" in path:
        return None
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    p = Path(path)
    if not p.is_absolute():
        return None
    try:  # every component resolved: no symlinked file or parent directory
        real = p.resolve(strict=True)
        if os.path.normcase(str(real)) != os.path.normcase(str(p)) or not p.is_file():
            return None
    except (OSError, RuntimeError):
        return None
    if _wheel_version(p.name) is None:
        return None
    return p


def _wheel_version(name: str) -> str | None:
    m = re.match(rf"^{DIST}-([^-]+)-py3-none-any\.whl$", name)
    if not m or parse_version(m.group(1)) is None:
        return None
    return m.group(1)


def _http_get_json(url: str) -> Any:
    if not url.startswith("https://"):
        raise FetchError("https only")
    import httpx

    try:
        with httpx.Client(
            timeout=httpx.Timeout(FETCH_TIMEOUT_S, connect=5.0),
            follow_redirects=False,
            headers={"Accept": "application/json"},
        ) as c:
            with c.stream("GET", url) as r:
                if r.status_code != 200:
                    raise FetchError(f"HTTP {r.status_code}")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > MAX_RESPONSE_BYTES:
                        raise FetchError("response too large")
    except httpx.HTTPError as exc:
        raise FetchError(type(exc).__name__) from exc
    try:
        return json.loads(bytes(buf).decode("utf-8"))
    except ValueError as exc:
        raise FetchError("not JSON") from exc


def pick_version(data: Any, *, prereleases: bool) -> str | None:
    """The newest acceptable version in PyPI's JSON: `info.version`, unless it is yanked
    or a pre-release (then the newest release that isn't), or a newer pre-release when
    they are opted in. Only versions with at least one non-yanked file count."""
    if not isinstance(data, dict):
        return None
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    if info.get("name") not in (None, PACKAGE, DIST, "llm_speed"):
        return None
    releases = data.get("releases") if isinstance(data.get("releases"), dict) else {}
    ok: list[tuple[Any, str]] = []
    for ver, files in releases.items():
        pv = parse_version(ver)
        if pv is None or not isinstance(files, list):
            continue
        if not any(isinstance(f, dict) and not f.get("yanked") for f in files):
            continue
        if is_prerelease(pv) and not prereleases:
            continue
        canon = str(pv)  # R-138: canonical form to the installer
        if parse_version(canon) is None:  # never hand on anything we can't re-read
            continue
        ok.append((pv, canon))
    head = info.get("version")
    if not ok:
        # no releases block: fall back to info.version alone (not yanked, not pre)
        pv = parse_version(head)
        if pv is None or info.get("yanked") or (is_prerelease(pv) and not prereleases):
            return None
        canon = str(pv)
        return canon if parse_version(canon) is not None else None
    ok.sort()
    return ok[-1][1]


def latest_version(*, prereleases: bool | None = None) -> str | None:
    """Raises FetchError when PyPI can't be read."""
    tw = _test_wheel()
    if tw is not None:
        return _wheel_version(tw.name)
    pre = config_prereleases() if prereleases is None else prereleases
    return pick_version(_http_get_json(PYPI_JSON_URL), prereleases=pre)


# ------------------------------------------------------------------ installer detection


@dataclass
class Install:
    method: str  # pipx | uv | brew | pip | unknown
    argv: list[str] | None  # None: can't auto-update here
    manual: str  # the command to print instead
    reason: str | None = None  # why argv is None
    python: str | None = None  # interpreter of the install (to read its version)


def _abs_tool(
    name: str, which: Callable[[str], str | None] = shutil.which
) -> str | None:
    p = which(name)
    if not p or not os.path.isabs(p):
        return None
    return os.path.realpath(p)


def _writable(p: Path) -> bool:
    return p.exists() and os.access(p, os.W_OK)


def _site_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _externally_managed() -> bool:
    import sysconfig

    try:
        return (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").exists()
    except (KeyError, TypeError, OSError):
        return False


def detect(
    target: str,
    *,
    prefix: str | None = None,
    base_prefix: str | None = None,
    executable: str | None = None,
    site_dir: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    externally_managed: bool | None = None,
) -> Install:
    prefix = prefix or sys.prefix
    base_prefix = base_prefix or sys.base_prefix
    executable = os.path.abspath(executable or sys.executable)
    site_dir = site_dir or _site_dir()
    rp = Path(os.path.realpath(prefix))
    spec = f"{PACKAGE}=={target}"
    if site_dir.name not in ("site-packages", "dist-packages"):
        # a source checkout / editable install: never replace it with a wheel
        return Install(
            "source",
            None,
            "git pull (this llm-speed runs from a source checkout)",
            "running from a source checkout",
        )

    # Homebrew: <brew prefix>/Cellar/llm-speed/<ver>/libexec
    parts = rp.parts
    if "Cellar" in parts:
        i = parts.index("Cellar")
        if i + 1 < len(parts) and parts[i + 1] == PACKAGE:
            brew_prefix = Path(*parts[:i])
            brew = brew_prefix / "bin" / "brew"
            manual = f"brew upgrade {BREW_FORMULA}"
            if not brew.is_file():
                return Install(
                    "brew", None, manual, "brew not found next to the Cellar"
                )
            if not _writable(brew_prefix / "Cellar" / PACKAGE):
                return Install("brew", None, manual, "Homebrew Cellar is not writable")
            formula = _brew_formula(Path(*parts[: i + 3]))
            if formula is None:
                return Install(
                    "brew",
                    None,
                    manual,
                    "not installed from the meadow-kun/tap formula",
                )
            return Install(
                "brew",
                [str(brew), "upgrade", formula],
                manual,
                python=str(
                    brew_prefix / "opt" / PACKAGE / "libexec" / "bin" / "python"
                ),
            )

    # pipx: <PIPX_HOME>/venvs/llm-speed with pipx_metadata.json
    if (
        rp.name == PACKAGE
        and rp.parent.name == "venvs"
        and (rp / "pipx_metadata.json").is_file()
    ):
        manual = f"pipx upgrade {PACKAGE}"
        pipx = _abs_tool("pipx", which)
        if pipx is None:
            return Install("pipx", None, manual, "pipx not found on PATH")
        if not _writable(rp):
            return Install("pipx", None, manual, "pipx venv is not writable")
        return Install("pipx", [pipx, "upgrade", PACKAGE], manual, python=executable)

    # uv tool: <UV_TOOL_DIR>/llm-speed with uv-receipt.toml
    if rp.name == PACKAGE and (rp / "uv-receipt.toml").is_file():
        manual = f"uv tool upgrade {PACKAGE}"
        uv = _abs_tool("uv", which)
        if uv is None:
            return Install("uv", None, manual, "uv not found on PATH")
        if not _writable(rp):
            return Install("uv", None, manual, "uv tool environment is not writable")
        return Install(
            "uv", [uv, "tool", "upgrade", PACKAGE], manual, python=executable
        )

    # pip, only inside a virtual environment this user can write to
    manual = f"{executable} -m pip install -U {spec}"
    if os.path.realpath(prefix) == os.path.realpath(base_prefix):
        em = _externally_managed() if externally_managed is None else externally_managed
        why = "externally managed Python" if em else "not in a virtual environment"
        hint = f"pipx upgrade {PACKAGE}" if em else manual
        return Install("unknown", None, hint, why)
    if not _writable(site_dir):
        return Install("pip", None, manual, "site-packages is not writable")
    return Install(
        "pip",
        [
            executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-input",
            "--disable-pip-version-check",
            spec,
        ],
        manual,
        python=executable,
    )


def _brew_formula(keg: Path) -> str | None:
    """The formula to upgrade, from the keg's INSTALL_RECEIPT.json: meadow-kun/tap only.
    Installer-gate exception (test only): the local tap GATE_TAP, and only while
    LLM_SPEED_UPDATE_TEST=1 is set."""
    try:
        receipt = json.loads((keg / "INSTALL_RECEIPT.json").read_text(encoding="utf-8"))
        tap = (receipt.get("source") or {}).get("tap")
    except (OSError, ValueError, AttributeError):
        return None
    if tap == "meadow-kun/tap":
        return BREW_FORMULA
    if tap == GATE_TAP and os.environ.get(TEST_ENV) == "1":
        return f"{GATE_TAP}/{PACKAGE}"
    return None


def install_argv(inst: Install, wheel: Path | None) -> list[str] | None:
    """The installer argv, with the test-only --find-links when a test wheel is set."""
    if inst.argv is None:
        return None
    argv = list(inst.argv)
    if wheel is None:
        return argv
    d = str(wheel.parent)
    if inst.method == "pip":
        return argv[:-1] + ["--find-links", d, argv[-1]]
    if inst.method == "pipx":
        return argv + [f"--pip-args=--find-links={d}"]
    if inst.method == "uv":
        return argv + ["--find-links", d]
    # brew installs only from its tap; the test wheel only names the target version
    return argv


def installer_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    for k in list(env):
        if k.startswith(_SCRUB_PREFIXES) and k not in _KEEP_ENV:
            del env[k]
    env.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_INDEX_URL": PYPI_SIMPLE,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "UV_NO_CONFIG": "1",
            "UV_DEFAULT_INDEX": PYPI_SIMPLE,
            "HOMEBREW_NO_ENV_HINTS": "1",
        }
    )
    env[DISABLE_ENV] = "1"  # the installer's own Python runs never recurse into us
    return env


# ------------------------------------------------------------------ lock


class Lock:
    """Exclusive, non-blocking lock on <config>/update.lock (fcntl / msvcrt). Released when
    the process exits, so a crash never leaves it held."""

    def __init__(self, path: Path | None = None):
        self.path = path or (_config_dir() / LOCK_NAME)
        self.f = None

    def acquire(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.path, "a+")  # noqa: SIM115
        except OSError:
            return False
        try:
            if os.name == "nt":  # pragma: no cover - Windows
                import msvcrt

                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            self.f = None
            return False
        return True

    def release(self) -> None:
        if self.f is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover
                import msvcrt

                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.f, fcntl.LOCK_UN)
        except OSError:
            pass
        self.f.close()
        self.f = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


# ------------------------------------------------------------------ in use (R-136)


def _try_lock(f) -> bool:
    try:
        if os.name == "nt":  # pragma: no cover - Windows
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _same_file(f, p: Path) -> bool:
    """The open file `f` is still the file at `p` (not removed or replaced)."""
    try:
        return os.path.samestat(os.fstat(f.fileno()), os.stat(p, follow_symlinks=False))
    except OSError:
        return False


class InUse:
    """Marks this process as running llm-speed code (a command, a bench, an agent job):
    <config>/in-use/<pid>, held open and locked for as long as the process runs. The
    updater never replaces the install while another marker is locked. The kernel drops
    the lock when the process exits, so a marker left by a dead (or since reused) PID is
    unlocked and counts as stale."""

    def __init__(self, directory: Path | None = None) -> None:
        self.dir = directory
        self.path: Path | None = None
        self.f = None

    def __enter__(self) -> InUse:
        """Create and lock the marker. R-140: the file is kept only if, once locked, it
        is still the one at the path (a cleaner may have removed it in between); a file
        with our name that someone else has locked is a stale one from a reused PID that
        a cleaner is looking at. Retries for about a second, then runs unmarked."""
        try:
            d = self.dir or (_config_dir() / INUSE_NAME)
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.debug("in-use marker not written: %s", exc)
            return self
        p = d / str(os.getpid())
        for attempt in range(INUSE_TRIES):
            try:
                fd = os.open(
                    p, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
                )
            except OSError as exc:
                log.debug("in-use marker open failed: %s", exc)
                time.sleep(INUSE_RETRY_S)
                continue
            f = os.fdopen(fd, "r+")
            if _try_lock(f):
                if _same_file(f, p):
                    try:
                        f.seek(0)
                        f.truncate()
                        f.write(str(os.getpid()))
                        f.flush()
                    except OSError:
                        pass
                    self.path, self.f = p, f
                    return self
            elif attempt >= 1:
                # still locked by someone else: no live process can own our PID's
                # marker, so it is a stale file a cleaner holds. Replace it; the
                # cleaner re-checks the path before it unlinks and leaves ours alone.
                try:
                    if _same_file(f, p):
                        p.unlink()
                except OSError:
                    pass
            f.close()
            time.sleep(INUSE_RETRY_S)
        log.debug(
            "in-use marker not held after %d tries; running unmarked", INUSE_TRIES
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        p, self.path = self.path, None
        removed = False
        if p is not None:
            try:
                p.unlink()
                removed = True
            except OSError:
                pass
        if self.f is not None:
            self.f.close()  # releases the lock
            self.f = None
        if p is not None and not removed:  # Windows: an open file can't be removed
            try:
                p.unlink()
            except OSError:
                pass


def busy_pids(ignore: tuple[int, ...] = (), directory: Path | None = None) -> list[int]:
    """Live llm-speed processes other than us (and `ignore`). A marker whose lock can be
    taken belongs to a process that has exited: it is removed."""
    out: list[int] = []
    d = directory or (_config_dir() / INUSE_NAME)
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        if not n.isdigit():
            continue
        pid = int(n)
        if pid == os.getpid() or pid in ignore:
            continue
        p = d / n
        try:
            fd = os.open(p, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            continue
        except OSError:  # can't open it (Windows: held open): fall back to the PID
            if _pid_alive(pid):
                out.append(pid)
            continue
        with os.fdopen(fd, "r+") as f:
            if not _try_lock(f):
                out.append(pid)  # its owner still runs
                continue
            try:  # stale: the owner exited. Unlink only while locked, and only the
                # file we locked (R-140): it may have been replaced meanwhile.
                if _same_file(f, p):
                    p.unlink()
            except OSError:
                pass
    return out


# ------------------------------------------------------------------ the update itself


@dataclass
class Result:
    status: (
        str  # updated | current | manual | offline | locked | busy | failed | invalid
    )
    current: str = __version__
    latest: str | None = None
    method: str | None = None
    manual: str | None = None
    detail: str | None = None


def installed_version(python: str | None, runner=subprocess.run) -> str | None:
    """The llm-speed version a fresh interpreter of the install sees now."""
    if not python or not os.path.isabs(python) or not os.path.exists(python):
        return None
    try:
        r = runner(
            [
                python,
                "-I",
                "-c",
                f"import importlib.metadata as m; print(m.version('{PACKAGE}'))",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    v = (r.stdout or "").strip()
    return v if r.returncode == 0 and parse_version(v) is not None else None


def run_update(
    *,
    check_only: bool = False,
    runner=None,
    fetch: Callable[[], str | None] | None = None,
    inst: Install | None = None,
    lock: Lock | None = None,
) -> Result:
    """Check PyPI and, unless `check_only`, install the newer version. Never raises."""
    runner = runner or subprocess.run
    now = time.time()
    try:
        latest = (fetch or latest_version)()
    except FetchError as exc:
        update_cache(checked_at=now, last_result="offline")
        return Result("offline", detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - never break the caller
        update_cache(checked_at=now, last_result="offline")
        return Result("offline", detail=type(exc).__name__)
    if latest is None:
        update_cache(checked_at=now, last_result="invalid")
        return Result("invalid", detail="no usable version on PyPI")
    if not newer(latest):
        update_cache(checked_at=now, latest=latest, last_result="current")
        return Result("current", latest=latest)
    inst = inst or detect(latest)
    if check_only:
        return Result(
            "manual" if inst.argv is None else "available",
            latest=latest,
            method=inst.method,
            manual=inst.manual,
            detail=inst.reason,
        )
    wheel = _test_wheel()
    tmp: str | None = None
    if wheel is not None and inst.method in ("pip", "pipx", "uv"):
        # R-139: find-links gets a fresh private dir holding only a copy of the wheel
        try:
            tmp = tempfile.mkdtemp(prefix="llm-speed-update-")  # 0700
            wheel = Path(shutil.copyfile(wheel, Path(tmp) / wheel.name))
        except OSError:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
            return Result(
                "failed", latest=latest, method=inst.method, detail="test wheel copy"
            )
    try:
        return _install(inst, latest, wheel, now, runner, lock)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _install(inst, latest, wheel, now, runner, lock) -> Result:
    argv = install_argv(inst, wheel)
    if argv is None:
        update_cache(
            checked_at=now, latest=latest, last_result="manual", manual_hint=inst.manual
        )
        return Result(
            "manual",
            latest=latest,
            method=inst.method,
            manual=inst.manual,
            detail=inst.reason,
        )
    try:  # loaded now: `brew upgrade` + cleanup may delete this keg's files
        from .agent import service as _service  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    lk = lock or Lock()
    if not lk.acquire():
        return Result(
            "locked",
            latest=latest,
            method=inst.method,
            detail="another update is running",
        )
    try:
        others = busy_pids()
        if others:
            # R-136: never replace the code under a running bench, agent job or command
            update_cache(
                checked_at=now - CHECK_INTERVAL_S + BUSY_RETRY_S,
                latest=latest,
                last_result="busy",
            )
            return Result(
                "busy",
                latest=latest,
                method=inst.method,
                detail=f"llm-speed is running (pid {others[0]})",
            )
        try:
            r = runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT_S,
                env=installer_env(),
                cwd=str(_config_dir()) if _config_dir().is_dir() else None,
                stdin=subprocess.DEVNULL,
            )
            rc = r.returncode
        except (OSError, subprocess.SubprocessError):
            rc = -1
        got = installed_version(inst.python, runner=runner) if rc == 0 else None
        if rc != 0 or got is None or not newer(got):
            update_cache(
                checked_at=now,
                latest=latest,
                last_result="failed",
                manual_hint=inst.manual,
            )
            return Result(
                "failed",
                latest=latest,
                method=inst.method,
                manual=inst.manual,
                detail=f"installer exit {rc}" + (f"; now {got}" if got else ""),
            )
        update_cache(
            checked_at=now,
            latest=latest,
            last_result="updated",
            notice={"from": __version__, "to": got},
            manual_hint=None,
        )
        _refresh_service()
        return Result("updated", latest=got, method=inst.method)
    finally:
        lk.release()


def _refresh_service() -> None:
    """After an update: an agent service still registered with a Homebrew Cellar path
    (0.0.13 or older) is rewritten to the stable opt path. Never raises."""
    try:
        from .agent import service

        for line in service.refresh_registration():
            log.info("%s", line)
    except Exception as exc:  # noqa: BLE001
        log.debug("service refresh skipped: %s", exc)


# ------------------------------------------------------------------ CLI hooks


def print_notice(stream=None) -> None:
    """At most once: the `Updated llm-speed A -> B` line, and the daily manual hint."""
    stream = stream or sys.stderr
    try:
        c = load_cache()
        n = c.get("notice")
        if isinstance(n, dict):
            if n.get("to") == __version__ and parse_version(n.get("from")) is not None:
                stream.write(f"Updated llm-speed {n['from']} -> {n['to']}\n")
            update_cache(notice=None)
            return
        hint, latest = c.get("manual_hint"), c.get("latest")
        if isinstance(hint, str) and isinstance(latest, str) and newer(latest):
            if time.time() - _num(c.get("hint_shown_at")) >= CHECK_INTERVAL_S:
                stream.write(
                    f"llm-speed {latest} is available (you have {__version__}). "
                    f"Upgrade: {hint}\n"
                )
                update_cache(hint_shown_at=time.time())
    except Exception as exc:  # noqa: BLE001
        log.debug("notice failed: %s", exc)


def _worker_argv(parent_pid: int) -> list[str]:
    from .agent.pyflags import isolation_flags

    return [
        os.path.abspath(sys.executable),
        *isolation_flags(),
        "-m",
        f"{__package__}.autoupdate",
        "--background",
        str(parent_pid),
        str(os.getppid()),
    ]


def maybe_start_background(command: str | None, *, popen=subprocess.Popen) -> bool:
    """Called at CLI start. Prints a pending notice, then, if due and allowed, starts the
    detached worker and returns at once. Never raises, never waits."""
    try:
        if command in ("update", "self-update", "mcp", "agent"):
            return False
        if disabled_reason() is not None:
            return False
        print_notice()
        if not check_due():
            return False
        # record the attempt first: a worker that dies still counts for today
        stamp = time.time()
        update_cache(checked_at=stamp, last_result="started")
        if _num(load_cache().get("checked_at")) != stamp:
            return False  # R-137: cache not writable, so no daily throttle: don't run
        kw: dict[str, Any] = dict(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            cwd=str(_config_dir()),
        )
        if os.name == "nt":  # pragma: no cover - Windows
            kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000
        else:
            kw["start_new_session"] = True
        popen(_worker_argv(os.getpid()), **kw)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("background update not started: %s", exc)
        return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def background_main(*pids: int, sleep=time.sleep) -> int:
    """The detached worker. It first waits (up to 15 minutes) for the CLI that started it
    (and its llm-speed.exe launcher, the parent's parent) to exit: on every OS so the
    code isn't replaced under a running command, and on Windows also because a running
    .exe can't be replaced (KFM-036)."""
    # R-136: wait for the command that started us on every OS (not only Windows); if it
    # is still running after the wait (a long bench), run_update() sees it as busy.
    end = time.monotonic() + WINDOWS_PARENT_WAIT_S
    while any(_pid_alive(p) for p in pids) and time.monotonic() < end:
        sleep(2.0)
    run_update()
    return 0


# ------------------------------------------------------------------ agent


_agent_next_check = 0.0  # time.monotonic(); R-137: works without a writable cache


def _disk_version() -> str | None:
    """The installed version on disk now, only if that install is the code running here
    (a source checkout next to an installed copy never counts)."""
    try:
        from importlib import metadata

        dist = metadata.distribution(PACKAGE)
        here = Path(__file__).resolve().parent / "__init__.py"
        if Path(str(dist.locate_file("cli/__init__.py"))).resolve() != here:
            return None
        return dist.version
    except Exception:  # noqa: BLE001
        return None


def agent_between_jobs(
    *, run=run_update, clock=time.monotonic, disk=_disk_version
) -> str | None:
    """`agent run` only, between jobs. Returns the new version if one was installed (by
    this agent, or by another llm-speed while the agent sat idle)."""
    global _agent_next_check
    if disabled_reason(agent=True) is not None:
        return None
    on_disk = disk()
    if on_disk and newer(on_disk):
        return on_disk  # the files changed under us: restart before the next job
    now = clock()
    if now < _agent_next_check or not check_due():
        return None
    _agent_next_check = now + CHECK_INTERVAL_S
    r = run()
    if r.status == "busy":
        _agent_next_check = now + BUSY_RETRY_S
    if r.status == "updated":
        return r.latest
    if r.status in ("manual", "failed"):
        log.warning("llm-speed %s is available; upgrade with: %s", r.latest, r.manual)
    return None


def _stable_python() -> str:
    """The interpreter to re-exec: Homebrew's versioned Cellar path is mapped to the
    stable `opt/llm-speed` link (cli/agent/pyflags.py); other paths are unchanged."""
    from .agent.pyflags import stable_python

    return stable_python()


def restart_argv(argv: list[str] | None = None) -> list[str]:
    from .agent.pyflags import isolation_flags

    rest = list((sys.argv if argv is None else argv)[1:])
    return [_stable_python(), *isolation_flags(), "-m", __package__ or "cli", *rest]


def restart_agent(new_version: str, *, execv=os.execv, runner=subprocess.run) -> int:
    """Replace this `agent run` process with the new version (same PID, so systemd and
    launchd keep tracking it, and it works without a supervisor). The run lock is
    close-on-exec, so the new process takes it again. If the new install can't be
    started, return AGENT_RESTART_EXIT so the service manager restarts it."""
    argv = restart_argv()
    try:
        r = runner(
            [argv[0], *_probe_args(argv)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        ok = r.returncode == 0 and new_version in (r.stdout or "")
    except (OSError, subprocess.SubprocessError):
        ok = False
    if not ok:
        log.warning(
            "updated to %s but could not start it; exiting %d for a restart",
            new_version,
            AGENT_RESTART_EXIT,
        )
        return AGENT_RESTART_EXIT
    log.warning(
        "updated llm-speed %s -> %s; restarting the agent", __version__, new_version
    )
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:  # noqa: BLE001
            pass
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        execv(argv[0], argv)
    except OSError as exc:
        log.warning(
            "re-exec failed (%s); exiting %d for a restart", exc, AGENT_RESTART_EXIT
        )
    return AGENT_RESTART_EXIT


def _probe_args(argv: list[str]) -> list[str]:
    """`<flags> -m cli --version` for the restart target."""
    i = argv.index("-m")
    return [*argv[1:i], "-m", argv[i + 1], "--version"]


# ------------------------------------------------------------------ `llm-speed update`


def cmd_update(args) -> int:
    check = bool(getattr(args, "check", False))
    print(f"llm-speed {__version__}")
    r = run_update(check_only=check)
    if r.status == "offline":
        print(f"Could not reach PyPI ({r.detail}). Nothing changed.", file=sys.stderr)
        return 1
    if r.status == "invalid":
        print("PyPI returned no usable version. Nothing changed.", file=sys.stderr)
        return 1
    if r.status == "current":
        print(f"Up to date (latest on PyPI: {r.latest}).")
        return 0
    if r.status == "available":
        print(f"Update available: {r.latest} (installed with {r.method}).")
        print(f"Run `llm-speed update` to install it, or: {r.manual}")
        return 0
    if r.status == "manual":
        print(f"Update available: {r.latest}.")
        print(
            f"Can't update this install automatically ({r.detail}). Run:\n  {r.manual}"
        )
        return 0 if check else 1
    if r.status == "busy":
        print(
            f"Not updating while another llm-speed is running ({r.detail}); "
            "try again when it has finished.",
            file=sys.stderr,
        )
        return 1
    if r.status == "locked":
        print(
            "Another llm-speed update is running; try again in a minute.",
            file=sys.stderr,
        )
        return 1
    if r.status == "failed":
        print(
            f"Update to {r.latest} failed ({r.detail}). Run:\n  {r.manual}",
            file=sys.stderr,
        )
        return 1
    update_cache(notice=None)
    print(f"Updated llm-speed {__version__} -> {r.latest}")
    return 0


# ------------------------------------------------------------------ `llm-speed config set`

SETTABLE = {
    CONFIG_KEY: "automatic updates (default true)",
    PRERELEASE_KEY: "follow pre-releases when updating (default false)",
}


def cmd_config_set(args) -> int:
    """`llm-speed config set KEY [true|false]`, same style as `config storage-info`."""
    from .credit import load_config, save_config

    key = getattr(args, "value", None)
    raw = getattr(args, "setting_value", None)
    if getattr(args, "clear", False) or key not in SETTABLE:
        keys = "|".join(SETTABLE)
        print(f"usage: llm-speed config set {keys} true|false", file=sys.stderr)
        return 2
    if raw is None:
        on = config_auto_update() if key == CONFIG_KEY else config_prereleases()
        print(f"{key}: {'true' if on else 'false'} ({SETTABLE[key]})")
        return 0
    v = raw.strip().lower()
    if v not in ("true", "false", "on", "off"):
        print("error: use true or false", file=sys.stderr)
        return 2
    on = v in ("true", "on")
    data = load_config()
    data[key] = on
    save_config(data)
    if key == CONFIG_KEY:
        print(
            "Automatic updates on: llm-speed checks pypi.org once a day and updates itself."
            if on
            else "Automatic updates off. Update by hand with: llm-speed update"
        )
    else:
        print(f"{key}: {'true' if on else 'false'}")
    return 0


def _main(argv: list[str]) -> int:
    if (
        len(argv) in (2, 3)
        and argv[0] == "--background"
        and all(a.isdigit() for a in argv[1:])
    ):
        try:
            return background_main(*(int(a) for a in argv[1:]))
        except Exception:  # noqa: BLE001 - detached: nothing to report to
            return 0
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
