"""Agent paths and local policy (agent/config.json). Policy is local-only: the
server can never change thresholds, paths, binaries or the API base."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..config import CACHE_DIR, CONFIG_DIR


def agent_dir() -> Path:
    return CONFIG_DIR / "agent"


def state_dir() -> Path:
    return CACHE_DIR / "agent"


def pause_path() -> Path:
    return agent_dir() / "PAUSE"


def config_path() -> Path:
    return agent_dir() / "config.json"


def logs_dir() -> Path:
    return state_dir() / "logs"


def outbox_dir() -> Path:
    return state_dir() / "outbox"


def ensure_dirs() -> None:
    for d in (agent_dir(), state_dir(), logs_dir(), outbox_dir()):
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass


def atomic_write_json(path: Path, data: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    os.chmod(path, mode)


@dataclass
class Settings:
    machine_id: str | None = None
    api_base: str | None = None  # queue that issued the token (set by enroll)
    label: str | None = None
    consent: dict | None = (
        None  # {"at", "cli_version", "text_sha256"}; None = no consent
    )
    allow_presets: list[str] | None = (
        None  # None = every reviewed preset in the allowlist
    )
    manual_approve: bool = False
    approve_timeout_min: int = 120
    quiet_hours: str | None = None  # "23:00-08:00" local time
    gpu_busy_mib: int = 4096  # other processes' VRAM above this = busy
    gpu_busy_util_pct: int = 30
    min_free_mem_gb: float = 24.0  # Apple unified memory
    min_free_disk_gb: float = 30.0
    models_dir: str | None = None  # passed to the preset as --models-dir
    llama_server: str | None = None  # passed as --llama-server
    upload: bool = True
    remote_presets: bool = False  # phase 2 (signed presets) flag
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> Settings:
        p = config_path()
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        atomic_write_json(config_path(), asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)


# Keys `agent config KEY VALUE` may set, with their parser.
def _bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "on"):
        return True
    if v.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {v!r}")


def _opt_str(v: str) -> str | None:
    return None if v.lower() in ("", "none", "null", "-") else v


def _presets(v: str) -> list[str] | None:
    return (
        None
        if v.lower() in ("", "all", "none")
        else [x.strip() for x in v.split(",") if x.strip()]
    )


SETTABLE = {
    "label": _opt_str,
    "manual_approve": _bool,
    "approve_timeout_min": int,
    "quiet_hours": _opt_str,
    "gpu_busy_mib": int,
    "gpu_busy_util_pct": int,
    "min_free_mem_gb": float,
    "min_free_disk_gb": float,
    "models_dir": _opt_str,
    "llama_server": _opt_str,
    "upload": _bool,
    "remote_presets": _bool,
    "allow_presets": _presets,
}
