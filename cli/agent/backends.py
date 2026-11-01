"""Which benchmark backends a job needs and whether this machine has them, looked up
the way the agent service looks (0.0.13). Local only: nothing here is uploaded or sent
to the queue except the result reason codes `backend_missing` /
`backend_install_failed`."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

from . import service
from .settings import Settings

# PATH a service starts with when its unit / plist / task sets none (they don't): launchd
# gives `/usr/bin:/bin:/usr/sbin:/sbin`; a systemd user manager its own environment
# (`systemctl --user show-environment`), usually the distro default below; the WSL2
# logon task (`wsl.exe --exec`) the distro default.
LAUNCHD_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _systemd_user_path() -> str | None:
    import subprocess

    try:
        r = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (r.stdout or "").splitlines():
        if line.startswith("PATH="):
            return line[5:]
    return None


def service_env(kind: str | None = None) -> dict[str, str]:
    """The environment the agent service would look up llama-server in: its PATH, and
    no $LLAMA_SERVER (a shell export doesn't reach the service)."""
    kind = kind or service.platform_kind()
    if kind == "launchd":
        path = LAUNCHD_PATH
    elif kind == "systemd":
        path = _systemd_user_path() or DEFAULT_PATH
    else:
        path = DEFAULT_PATH
    return {"PATH": path}


def needed(
    preset: Mapping[str, Any], kind: str | None, overrides: Mapping | None = None
) -> set[str]:
    """Backends the job's profile for this machine kind runs ("llama.cpp", "mlx").
    The control model counts only when the job includes it."""
    prof = ((preset.get("profiles") or {}).get(kind or "") or {}) if kind else {}
    ids = list(prof.get("entries") or []) + list(prof.get("fallback") or [])
    if (overrides or {}).get("include_control"):
        ids += list(prof.get("control") or [])
    models = preset.get("models") or {}
    return {
        str((models.get(i) or {}).get("backend") or "llama.cpp")
        for i in ids
        if i in models
    }


def probe(
    settings: Settings,
    min_build: Any = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """{"llama.cpp": {"path", "build", "ok"} | None, "mlx": bool}. `env` None = this
    process's environment (what the running service sees)."""
    from ..commands import preset as pr

    path, info, _ = pr.find_llama_server(settings.llama_server, min_build, env)
    llama = None
    if path:
        llama = {
            "path": path,
            "build": (info or {}).get("build")
            or (
                f"b{(info or {}).get('build_number')}"
                if (info or {}).get("build_number")
                else None
            ),
            "ok": pr.llama_build_ok(path, info, min_build),
        }
    return {"llama.cpp": llama, "mlx": importlib.util.find_spec("mlx_lm") is not None}


def missing(need: set[str], found: Mapping[str, Any]) -> list[str]:
    out = []
    if "llama.cpp" in need and not (found.get("llama.cpp") or {}).get("ok"):
        out.append("llama.cpp")
    if "mlx" in need and not found.get("mlx"):
        out.append("mlx")
    return out


def blocking(need: set[str], miss: list[str]) -> bool:
    """The job can't run at all: llama.cpp is missing, or every backend it needs is.
    (MLX steps alone missing next to llama.cpp ones are skipped, as before.)"""
    return bool(miss) and ("llama.cpp" in miss or set(miss) >= need)


def describe(found: Mapping[str, Any]) -> str:
    ll = found.get("llama.cpp")
    llama = (
        f"llama-server {ll['path']} ({ll.get('build') or 'build unknown'})"
        if ll
        else "llama-server not found"
    )
    return f"{llama}; mlx-lm {'installed' if found.get('mlx') else 'not installed'}"


# What the runner calls before launching a job (tests replace it).
job_probe = probe


def status_probe(settings: Settings) -> dict[str, Any]:
    """For `agent status` / `doctor` (typed in a shell): look the way the service will,
    with its PATH and without the shell's $LLAMA_SERVER (tests replace it)."""
    return probe(settings, None, env=service_env())
