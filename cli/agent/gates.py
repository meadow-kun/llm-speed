"""Gates checked before claiming a job: pause, quiet hours, busy GPU, free memory, disk.

Read-only: gates only look (nvidia-smi queries, vm_stat, disk usage); they never
signal or touch any process.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CACHE_DIR
from .settings import Settings, pause_path


@dataclass
class GateResult:
    open: bool
    reasons: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)


def parse_quiet_hours(spec: str) -> tuple[dt.time, dt.time]:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", spec or "")
    if not m:
        raise ValueError(f"quiet_hours must look like 23:00-08:00, got {spec!r}")
    h1, m1, h2, m2 = map(int, m.groups())
    if not (h1 < 24 and h2 < 24 and m1 < 60 and m2 < 60):
        raise ValueError(f"bad time in quiet_hours {spec!r}")
    return dt.time(h1, m1), dt.time(h2, m2)


def in_quiet_hours(spec: str | None, now: dt.datetime | None = None) -> bool:
    if not spec:
        return False
    start, end = parse_quiet_hours(spec)
    t = (now or dt.datetime.now()).time()
    if start == end:
        return False
    if start < end:
        return start <= t < end
    return t >= start or t < end  # wraps midnight


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def parse_compute_apps(out: str) -> list[tuple[int, int]]:
    """`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`."""
    apps = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                apps.append((int(parts[0]), int(float(parts[1]))))
            except ValueError:
                continue
    return apps


def parse_gpu_util(out: str) -> int | None:
    vals = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
    return max(vals) if vals else None


def nvidia_busy(
    settings: Settings, own_pids: set[int], *, runner=_run
) -> tuple[bool, dict]:
    apps_out = runner(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if apps_out is None:
        return False, {"nvidia": False}
    apps = parse_compute_apps(apps_out)
    other = sum(mem for pid, mem in apps if pid not in own_pids)
    util = parse_gpu_util(
        runner(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        or ""
    )
    facts = {"nvidia": True, "other_vram_mib": other, "gpu_util_pct": util}
    busy = other > settings.gpu_busy_mib or (
        util is not None and util > settings.gpu_busy_util_pct
    )
    return busy, facts


def parse_vm_stat_available_gb(out: str) -> float | None:
    m = re.search(r"page size of (\d+) bytes", out)
    page = int(m.group(1)) if m else 16384
    total = 0
    found = False
    for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        mm = re.search(rf"{key}:\s+(\d+)", out)
        if mm:
            total += int(mm.group(1))
            found = True
    return round(total * page / 1024**3, 1) if found else None


def apple_mem_low(settings: Settings, *, runner=_run) -> tuple[bool, dict]:
    if sys.platform != "darwin":
        return False, {}
    avail = parse_vm_stat_available_gb(runner(["vm_stat"]) or "")
    return (avail is not None and avail < settings.min_free_mem_gb), {
        "available_mem_gb": avail
    }


def disk_low(settings: Settings) -> tuple[bool, dict]:
    target = (
        Path(os.path.expanduser(settings.models_dir))
        if settings.models_dir
        else CACHE_DIR
    )
    while not target.exists() and target.parent != target:
        target = target.parent
    free_gb = round(shutil.disk_usage(target).free / 1e9, 1)
    return free_gb < settings.min_free_disk_gb, {"free_disk_gb": free_gb}


def check(
    settings: Settings,
    *,
    own_pids: set[int] | None = None,
    now: dt.datetime | None = None,
    runner=_run,
) -> GateResult:
    reasons: list[str] = []
    facts: dict = {}
    if pause_path().exists():
        reasons.append("paused (local pause file)")
    if in_quiet_hours(settings.quiet_hours, now):
        reasons.append(f"quiet hours {settings.quiet_hours}")
    busy, f = nvidia_busy(settings, own_pids or set(), runner=runner)
    facts.update(f)
    if busy:
        reasons.append(
            f"GPU busy (other processes {f.get('other_vram_mib')} MiB > {settings.gpu_busy_mib} MiB"
            f" or util {f.get('gpu_util_pct')}% > {settings.gpu_busy_util_pct}%)"
        )
    low, f = apple_mem_low(settings, runner=runner)
    facts.update(f)
    if low:
        reasons.append(
            f"free memory {f.get('available_mem_gb')} GB < {settings.min_free_mem_gb} GB"
        )
    low, f = disk_low(settings)
    facts.update(f)
    if low:
        reasons.append(
            f"free disk {f.get('free_disk_gb')} GB < {settings.min_free_disk_gb} GB"
        )
    return GateResult(open=not reasons, reasons=reasons, facts=facts)
