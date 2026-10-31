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
    # Busy = the GPU is working, not merely holding memory: utilization sampled
    # `busy_samples` times over `busy_window_s`; busy when the mean, or more than half of
    # the samples, is above `busy_util_pct`. Resident VRAM of an idle server is not busy.
    busy_util_pct: int = 15
    busy_window_s: float = 15.0
    busy_samples: int = 5
    # Resource needs are derived from the claimed job's preset (largest model file +
    # the preset's memory headroom; files still to download + the preset's disk margin).
    # The *_margin_gb keys replace the preset's margins; need_free_*_gb replace the whole
    # derived need. None = derive from the preset.
    vram_margin_gb: float | None = None  # NVIDIA: added to the largest model file
    mem_margin_gb: float | None = None  # Apple unified memory: same
    disk_margin_gb: float | None = None  # added to the bytes still to download
    need_free_vram_gb: float | None = None
    need_free_mem_gb: float | None = None
    need_free_disk_gb: float | None = None
    models_dir: str | None = None  # passed to the preset as --models-dir
    llama_server: str | None = None  # passed as --llama-server
    # nvidia-smi used for the busy check: None = auto (under WSL the Windows
    # nvidia-smi.exe first), an absolute path to pin one, "off" = no NVIDIA sampling.
    nvidia_smi: str | None = None
    upload: bool = True
    remote_presets: bool = (
        False  # also check packaged presets against the revocation list
    )
    # 0.0.12: run presets the server signed with a key pinned in this CLI (verified,
    # schema-checked, never new executable surface). Off by default (Security Desk R-93):
    # opt in per machine with `llm-speed agent signed-presets on` (asks to confirm).
    # Off = packaged presets only; turning it off again is the kill switch.
    signed_presets: bool = False
    # Which store holds the machine token: "keychain" | "file" (set by enroll; the
    # token itself is never written here). None = enrolled before 0.0.10.
    token_store: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> Settings:
        p = config_path()
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        # 0.0.9 keys (gpu_busy_mib, gpu_busy_util_pct, min_free_mem_gb, min_free_disk_gb)
        # were fixed thresholds written with their defaults into every config; they are
        # dropped here (see LEGACY_KEYS) so an old default never acts as an override.
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


def _opt_float(v: str) -> float | None:
    if v.lower() in ("", "none", "null", "-", "auto"):
        return None
    f = float(v)
    if not 0 <= f < 100_000:
        raise ValueError(f"out of range: {v!r}")
    return f


def _pct(v: str) -> int:
    n = int(v)
    if not 0 <= n <= 100:
        raise ValueError(f"must be 0-100, got {v!r}")
    return n


def _window(v: str) -> float:
    f = float(v)
    if not 0 <= f <= 300:
        raise ValueError(f"must be 0-300 seconds, got {v!r}")
    return f


def _samples(v: str) -> int:
    n = int(v)
    if not 1 <= n <= 60:
        raise ValueError(f"must be 1-60, got {v!r}")
    return n


def _smi_path(v: str) -> str | None:
    v = v.strip()
    if _opt_str(v) is None or v.lower() == "auto":
        return None
    if v.lower() == "off":
        return "off"
    if not os.path.isabs(v):
        raise ValueError(f"must be an absolute path, 'off' or 'auto', got {v!r}")
    return v


def _presets(v: str) -> list[str] | None:
    return (
        None
        if v.lower() in ("", "all", "none")
        else [x.strip() for x in v.split(",") if x.strip()]
    )


def _label(v: str) -> str | None:
    from .label import LabelError, validate_label

    if _opt_str(v) is None:
        return None
    try:
        return validate_label(v)
    except LabelError as e:
        raise ValueError(str(e)) from None


SETTABLE = {
    "label": _label,
    "manual_approve": _bool,
    "approve_timeout_min": int,
    "quiet_hours": _opt_str,
    "busy_util_pct": _pct,
    "busy_window_s": _window,
    "busy_samples": _samples,
    "vram_margin_gb": _opt_float,
    "mem_margin_gb": _opt_float,
    "disk_margin_gb": _opt_float,
    "need_free_vram_gb": _opt_float,
    "need_free_mem_gb": _opt_float,
    "need_free_disk_gb": _opt_float,
    "models_dir": _opt_str,
    "llama_server": _opt_str,
    "nvidia_smi": _smi_path,
    "upload": _bool,
    "remote_presets": _bool,
    "allow_presets": _presets,
}

# 0.0.9 keys that no longer exist, with what replaced them (`agent config` explains).
LEGACY_KEYS = {
    "gpu_busy_mib": "busy is now GPU utilization (busy_util_pct, busy_window_s, "
    "busy_samples); resident VRAM is compared with what the claimed job needs "
    "(vram_margin_gb, need_free_vram_gb)",
    "gpu_busy_util_pct": "renamed busy_util_pct (sampled over busy_window_s)",
    "min_free_mem_gb": "derived from the claimed job's preset; override with "
    "mem_margin_gb or need_free_mem_gb",
    "min_free_disk_gb": "derived from the claimed job's preset; override with "
    "disk_margin_gb or need_free_disk_gb",
}
