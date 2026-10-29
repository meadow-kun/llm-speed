"""Gates: pause, quiet hours and a busy GPU before claiming a job; after a claim, whether
this machine has the free VRAM / unified memory / disk that the claimed job's preset needs.

Busy means the GPU is *working*: utilization sampled over a short window (nvidia-smi
`utilization.gpu`; on Apple Silicon the IOAccelerator "Device Utilization %"). Memory that
an idle process keeps resident (a local coding server, ComfyUI) is not "busy"; it only
matters when it leaves less free memory than the job needs, and then the job is skipped
with a reason that says so (docs/cli_known_failure_modes.md KFM-023).

Read-only: gates only look (nvidia-smi / ioreg / vm_stat queries, file sizes, disk usage);
they never signal or touch any process.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import fit_sweep
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


MIB = 1024**2
GIB = 1024**3
# Never ask for the whole device: the driver / display keep some memory for themselves.
CAPACITY_RESERVE_MIB = 1024
DEFAULT_HEADROOM_MIB = {"nvidia": 1536, "apple": 4096}
DEFAULT_DISK_MARGIN_GB = 3.0


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
    """`--query-gpu=utilization.gpu`: the busiest GPU's percentage, one line per GPU
    (LF or CRLF: the Windows nvidia-smi.exe prints CRLF). None when the output is empty
    or any GPU's line isn't a plain 0-100 number (`[N/A]`, `[Not Supported]`, an error):
    one unreadable GPU makes the whole reading unreadable, never a silent 0."""
    lines = [x.strip() for x in (out or "").replace("\r", "\n").split("\n")]
    lines = [x for x in lines if x]
    if not lines or not all(x.isdigit() and int(x) <= 100 for x in lines):
        return None
    return max(int(x) for x in lines)


def parse_gpu_memory(out: str) -> tuple[int, int] | None:
    """`--query-gpu=memory.total,memory.free --format=csv,noheader,nounits`: GPU 0's
    (total MiB, free MiB)."""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                return int(float(parts[0])), int(float(parts[1]))
            except ValueError:
                return None
    return None


def parse_ioreg_util(out: str) -> int | None:
    """`ioreg -r -d 1 -c IOAccelerator`: Apple GPU "Device Utilization %" (no root needed)."""
    vals = [int(v) for v in re.findall(r'"Device Utilization %"\s*=\s*(\d+)', out)]
    return max(vals) if vals else None


UTIL_ARGS = ["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
NVIDIA_UTIL = ["nvidia-smi", *UTIL_ARGS]
# Under WSL2 the Linux-side NVML does not support the utilization query (NVIDIA's CUDA
# on WSL guide, "Features not yet supported"), and only the Windows driver sees what
# Windows apps (games, ComfyUI) are doing, so the Windows nvidia-smi.exe goes first.
WIN_SMI = "/mnt/c/Windows/System32/nvidia-smi.exe"
WIN_SMI_WINPATH = "C:\\Windows\\System32\\nvidia-smi.exe"
WSL_SMI = "/usr/lib/wsl/lib/nvidia-smi"
NVIDIA_MEM = [
    "nvidia-smi",
    "--query-gpu=memory.total,memory.free",
    "--format=csv,noheader,nounits",
]
NVIDIA_APPS = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory",
    "--format=csv,noheader,nounits",
]
APPLE_UTIL = ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"]


def _locate(path: str) -> str | None:
    """An executable at `path` (absolute) or on PATH (bare name); tests replace this."""
    if os.path.isabs(path):
        return path if os.path.isfile(path) and os.access(path, os.X_OK) else None
    return shutil.which(path)


def is_wsl(osrelease: Path = Path("/proc/sys/kernel/osrelease")) -> bool:
    """Running inside WSL (any distro): WSL_DISTRO_NAME, or a 'microsoft' kernel."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in osrelease.read_text(encoding="utf-8").lower()
    except OSError:
        return False


def nvidia_smi_candidates(
    settings: Settings, *, runner=_run, wsl: bool | None = None
) -> list[str]:
    """nvidia-smi binaries to read utilization from, best first. A pinned
    `nvidia_smi` path is the only candidate; "off" means none (no NVIDIA sampling)."""
    pinned = getattr(settings, "nvidia_smi", None)
    if pinned:
        return [] if pinned.lower() == "off" else [pinned]
    if wsl is None:
        wsl = is_wsl()
    out: list[str] = []
    if wsl:
        out.append(WIN_SMI)
        # C: mounted somewhere else (wsl.conf automount root) or Windows PATH appended
        conv = (runner(["wslpath", "-u", WIN_SMI_WINPATH]) or "").strip()
        if conv:
            out.append(conv)
        exe = shutil.which("nvidia-smi.exe")
        if exe:
            out.append(exe)
        out.append(WSL_SMI)
    out.append("nvidia-smi")
    seen: list[str] = []
    for c in out:
        if c not in seen:
            seen.append(c)
    return seen


def _short(out: str | None) -> str:
    if out is None:
        return "no output (error, timeout or not installed)"
    text = " ".join((out or "").split())
    return repr(text[:40]) if text else "empty output"


def read_nvidia_util(
    settings: Settings, *, runner=_run, wsl: bool | None = None
) -> tuple[str | None, int | None, list[str], bool]:
    """(binary, util, notes, present). The first candidate that gives a readable
    utilization wins. `present` = some candidate exists or answered, i.e. there is an
    NVIDIA GPU to protect even if no reading worked."""
    notes: list[str] = []
    present = False
    for smi in nvidia_smi_candidates(settings, runner=runner, wsl=wsl):
        located = _locate(smi) is not None
        out = runner([smi, *UTIL_ARGS])
        present = present or located or out is not None
        if out is None and not located:
            continue
        util = parse_gpu_util(out or "")
        if util is not None:
            return smi, util, notes, True
        notes.append(f"{smi}: {_short(out)}")
    return None, None, notes, present


def sample_util(read, samples: int, window_s: float, sleep=time.sleep) -> list[int]:
    """`samples` readings spread evenly over `window_s` seconds (first one immediately).
    A reading that fails is skipped; if the first fails there is no GPU signal at all."""
    n = max(1, int(samples))
    gap = float(window_s) / (n - 1) if n > 1 else 0.0
    out: list[int] = []
    for i in range(n):
        if i and gap > 0:
            sleep(gap)
        v = read()
        if v is None:
            if i == 0:
                return []
            continue
        out.append(int(v))
    return out


def util_busy(samples: list[int], pct: int) -> bool:
    """Busy when the mean, or more than half of the samples, is above `pct`."""
    if not samples:
        return False
    above = sum(1 for x in samples if x > pct)
    return sum(samples) / len(samples) > pct or above * 2 > len(samples)


def gpu_busy(
    settings: Settings,
    own_pids: set[int] | None = None,
    *,
    runner=_run,
    sleep=time.sleep,
    wsl: bool | None = None,
) -> tuple[bool, dict]:
    """(busy, facts). NVIDIA via nvidia-smi (under WSL the Windows nvidia-smi.exe first);
    Apple Silicon via ioreg; neither = not busy. An NVIDIA GPU whose utilization can't
    be read counts as busy (facts["gpu_util_unreadable"] says why): the agent must not
    take the card while it can't tell whether someone is using it (KFM-027)."""
    smi, first, notes, present = read_nvidia_util(settings, runner=runner, wsl=wsl)
    if smi is not None:
        vendor = "nvidia"
        pending = [first]

        def read():
            if pending:
                return pending.pop()
            return parse_gpu_util(runner([smi, *UTIL_ARGS]) or "")

    elif present:
        return True, {
            "gpu_vendor": "nvidia",
            "gpu_util": None,
            "gpu_util_unreadable": "; ".join(notes) or "no reading",
            "busy_util_pct": settings.busy_util_pct,
        }
    elif (
        sys.platform == "darwin"
        and parse_ioreg_util(runner(APPLE_UTIL) or "") is not None
    ):
        vendor = "apple"

        def read():
            return parse_ioreg_util(runner(APPLE_UTIL) or "")

    else:
        return False, {"gpu_util": None}
    samples = sample_util(read, settings.busy_samples, settings.busy_window_s, sleep)
    facts: dict = {
        "gpu_vendor": vendor,
        "gpu_util_samples": samples,
        "gpu_util_mean": round(sum(samples) / len(samples), 1) if samples else None,
        "busy_util_pct": settings.busy_util_pct,
    }
    if vendor == "nvidia":
        facts["gpu_util_source"] = smi
        apps = parse_compute_apps(runner([smi, *NVIDIA_APPS[1:]]) or "")
        # Diagnostic only: resident memory of other processes is not "busy" by itself.
        facts["other_vram_mib"] = sum(
            m for pid, m in apps if pid not in (own_pids or set())
        )
    return util_busy(samples, settings.busy_util_pct), facts


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


# ------------------------------------------------------------------ what a job needs


@dataclass
class Needs:
    """Free resources the claimed job's preset needs on this machine. None = no check."""

    preset: str
    kind: str | None = None  # "nvidia" | "apple" | None
    memory_mib: float | None = None  # free VRAM (nvidia) or free unified memory (apple)
    disk_bytes: int | None = None
    largest_entry: str | None = None
    largest_mib: float | None = None
    headroom_mib: float | None = None
    capacity_mib: float | None = None
    download_bytes: int = 0
    disk_margin_bytes: int = 0
    self_gated: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "preset": self.preset,
            "kind": self.kind,
            "need_memory_gib": _gib(self.memory_mib),
            "need_disk_gb": _gb(self.disk_bytes),
            "largest_entry": self.largest_entry,
            "download_gb": _gb(self.download_bytes),
        }
        return {k: v for k, v in d.items() if v is not None}


def _gib(mib: float | None) -> float | None:
    return None if mib is None else round(mib / 1024, 1)


def _gb(b: int | None) -> float | None:
    return None if b is None else round(b / 1e9, 1)


def models_dirs(settings: Settings) -> tuple[Path, Path]:
    """Where the preset child keeps model files (same rule as `llm-speed bench --preset`)."""
    d = Path(
        os.path.expanduser(
            settings.models_dir
            or os.environ.get("LLM_SPEED_MODELS_DIR")
            or str(CACHE_DIR / "models")
        )
    )
    return d, d / "mlx"


def present_bytes(entry: dict, models_dir: Path, mlx_dir: Path) -> int:
    """Bytes of this entry already on disk (models dir or a complete Hugging Face cache copy)."""
    exp = int(entry.get("bytes") or 0)
    if fit_sweep.hf_cache_hit(entry) is not None:
        return exp
    if entry.get("backend") == "mlx":
        p = mlx_dir / entry.get("dir", str(entry.get("repo", "")).split("/")[-1])
    else:
        p = models_dir / str(entry.get("file", ""))
    have = fit_sweep.path_bytes(p) if p.exists() else 0
    return min(have, exp) if exp else have


def preset_needs(
    preset: dict, overrides: dict | None, settings: Settings, facts: dict
) -> Needs:
    """Derive the job's needs from the preset itself (no fixed thresholds):

    - memory: the largest model file this machine would load (the preset's own plan for
      this GPU, narrowed by the job's overrides) plus the preset's memory headroom
      (`memory_headroom_mib`, or vram_margin_gb / mem_margin_gb), never more than the
      device holds minus 1 GiB. Models load one at a time, so this is a max, not a sum.
    - disk: the bytes of those files still to download (files already in the models dir
      or the HF cache count as present) plus the preset's `disk_margin_gb` (or
      disk_margin_gb). Entries with their own `min_free_disk_gb` gate themselves in the
      preset (recorded as skipped_disk without downloading) and are not counted here.
    need_free_vram_gb / need_free_mem_gb / need_free_disk_gb replace a derived value.
    """
    from ..commands.preset import build_plan

    name = str(preset.get("preset") or "?")
    ov = overrides or {}
    kind = facts.get("kind") if facts.get("kind") in ("nvidia", "apple") else None
    needs = Needs(preset=name, kind=kind)
    if name == "selftest" or not preset.get("models"):
        return needs
    mdir, mlx = models_dirs(settings)
    try:
        plan, _ = build_plan(
            preset,
            facts,
            with_control=bool(ov.get("include_control")),
            models_dir=mdir,
            mlx_dir=mlx,
            llama_server=None,
        )
    except SystemExit as e:  # no profile for this machine: the preset run reports it
        needs.notes.append(str(e))
        return needs
    only = set(ov.get("entries") or [])
    live = [
        e
        for e in plan["entries"]
        if not e.get("skip_status")
        and not e.get("optional")
        and (not only or e["id"] in only)
    ]
    if not live:
        needs.notes.append("nothing in this preset runs on this machine")
        return needs

    # memory
    if kind:
        largest = max(live, key=lambda e: int(e.get("bytes") or 0))
        needs.largest_entry = largest["id"]
        needs.largest_mib = int(largest.get("bytes") or 0) / MIB
        margin = settings.vram_margin_gb if kind == "nvidia" else settings.mem_margin_gb
        head = (preset.get("memory_headroom_mib") or {}).get(
            kind, DEFAULT_HEADROOM_MIB[kind]
        )
        needs.headroom_mib = float(margin) * 1024 if margin is not None else float(head)
        if kind == "nvidia" and facts.get("gpu_memory_total_mib"):
            needs.capacity_mib = float(facts["gpu_memory_total_mib"])
        elif kind == "apple" and facts.get("ram_gb"):
            wired = facts.get("iogpu_wired_limit_mb") or 0
            needs.capacity_mib = float(wired) or float(facts["ram_gb"]) * 1024 * float(
                preset.get("apple_gpu_memory_fraction", 0.75)
            )
        need = needs.largest_mib + needs.headroom_mib
        if needs.capacity_mib:
            need = min(need, needs.capacity_mib - CAPACITY_RESERVE_MIB)
        needs.memory_mib = round(need, 1)
    override = (
        settings.need_free_vram_gb if kind == "nvidia" else settings.need_free_mem_gb
    )
    if kind and override is not None:
        needs.memory_mib = float(override) * 1024
        needs.notes.append("memory need set by the machine's config")

    # disk
    missing = 0
    for e in live:
        if e.get("min_free_disk_gb"):
            needs.self_gated.append(e["id"])
            continue
        missing += max(0, int(e.get("bytes") or 0) - present_bytes(e, mdir, mlx))
    margin_gb = (
        settings.disk_margin_gb
        if settings.disk_margin_gb is not None
        else float(preset.get("disk_margin_gb", DEFAULT_DISK_MARGIN_GB))
    )
    needs.download_bytes = missing
    needs.disk_margin_bytes = int(margin_gb * 1e9)
    needs.disk_bytes = missing + needs.disk_margin_bytes
    if settings.need_free_disk_gb is not None:
        needs.disk_bytes = int(float(settings.need_free_disk_gb) * 1e9)
        needs.notes.append("disk need set by the machine's config")
    return needs


# ------------------------------------------------------------------ what is free


def disk_target(settings: Settings) -> Path:
    target = models_dirs(settings)[0]
    while not target.exists() and target.parent != target:
        target = target.parent
    return target


def free_disk_gb(settings: Settings) -> float:
    return round(shutil.disk_usage(disk_target(settings)).free / 1e9, 1)


def free_memory_mib(kind: str | None, *, runner=_run) -> tuple[float | None, dict]:
    """Free VRAM on GPU 0 (nvidia) or available unified memory (apple), in MiB."""
    if kind == "nvidia":
        mem = parse_gpu_memory(runner(NVIDIA_MEM) or "")
        if mem is None:
            return None, {}
        return float(mem[1]), {"vram_total_mib": mem[0], "vram_free_mib": mem[1]}
    if kind == "apple" and sys.platform == "darwin":
        avail = parse_vm_stat_available_gb(runner(["vm_stat"]) or "")
        if avail is None:
            return None, {}
        return avail * 1024, {"available_mem_gb": avail}
    return None, {}


def resource_reasons(
    needs: Needs, settings: Settings, *, runner=_run
) -> tuple[list[str], dict]:
    reasons: list[str] = []
    facts: dict = {"needs": needs.to_dict()}
    if needs.memory_mib is not None:
        free, f = free_memory_mib(needs.kind, runner=runner)
        facts.update(f)
        if free is not None and free < needs.memory_mib:
            what = "VRAM" if needs.kind == "nvidia" else "unified memory"
            how = (
                f"{needs.largest_entry} {needs.largest_mib / 1024:.1f} GiB + "
                f"{needs.headroom_mib / 1024:.1f} GiB headroom"
                if needs.largest_entry and needs.largest_mib is not None
                else "machine config"
            )
            used = ""
            if needs.kind == "nvidia" and f.get("vram_total_mib"):
                used = f"; other processes hold {(f['vram_total_mib'] - free) / 1024:.1f} GiB"
            reasons.append(
                f"not enough free {what} for {needs.preset}: {free / 1024:.1f} GiB free, "
                f"needs {needs.memory_mib / 1024:.1f} GiB ({how}){used}"
            )
    if needs.disk_bytes is not None:
        free_b = shutil.disk_usage(disk_target(settings)).free
        facts["free_disk_gb"] = round(free_b / 1e9, 1)
        if free_b < needs.disk_bytes:
            reasons.append(
                f"not enough free disk for {needs.preset}: {free_b / 1e9:.1f} GB free, needs "
                f"{needs.disk_bytes / 1e9:.1f} GB ({needs.download_bytes / 1e9:.1f} GB still to "
                f"download + {needs.disk_margin_bytes / 1e9:.1f} GB margin)"
            )
    return reasons, facts


def check(
    settings: Settings,
    *,
    own_pids: set[int] | None = None,
    now: dt.datetime | None = None,
    runner=_run,
    sleep=time.sleep,
    needs: Needs | None = None,
    sample_busy: bool = True,
) -> GateResult:
    """Before a claim: pause, quiet hours, busy GPU. With `needs` (after a claim): also
    free VRAM / unified memory / disk for that job."""
    reasons: list[str] = []
    facts: dict = {}
    if pause_path().exists():
        reasons.append("paused (local pause file)")
    if in_quiet_hours(settings.quiet_hours, now):
        reasons.append(f"quiet hours {settings.quiet_hours}")
    if sample_busy and not reasons:  # no point sampling for 15 s when already closed
        busy, f = gpu_busy(settings, own_pids, runner=runner, sleep=sleep)
        facts.update(f)
        if busy and f.get("gpu_util_unreadable"):
            reasons.append(
                "GPU utilization unreadable; treating as busy "
                f"({f['gpu_util_unreadable']}). Pin a working binary with "
                "`llm-speed agent config nvidia_smi PATH`"
            )
        elif busy:
            reasons.append(
                f"GPU busy (utilization {f.get('gpu_util_samples')}%, mean "
                f"{f.get('gpu_util_mean')}% over {settings.busy_window_s:g} s; "
                f"busy above {settings.busy_util_pct}%)"
            )
    if needs is not None and not reasons:
        r, f = resource_reasons(needs, settings, runner=runner)
        facts.update(f)
        reasons += r
    return GateResult(open=not reasons, reasons=reasons, facts=facts)
