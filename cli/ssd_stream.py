"""SSD-streamed llama.cpp entries (0.0.13, backlog row 61): a preset model entry with
`"ssd_stream": true` keeps most MoE expert weights on disk (mmap, `--n-cpu-moe N`) and
streams them while it runs, so its file may be larger than VRAM (up to 512 GiB).

Everything a run with it uploads is listed in UPLOADED_EXTRAS_KEYS (plain booleans,
integers and allowlisted names only: never paths, device or mount names)."""

from __future__ import annotations

import os
import re
import shutil
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_MAX_BYTES = 137438953472  # 128 GiB: any preset entry
SSD_STREAM_MAX_BYTES = 549755813888  # 512 GiB: only with "ssd_stream": true
VRAM_NEED_GIB_MAX = 256

# Tensors an ssd_stream entry may pin to CPU memory (`-ot ^NAME$=CPU`). Exact names only;
# there is no free-form -ot. SECURITY: these names are uploaded as extras.cpu_tensors, so
# adding a name here is a new upload value and needs Security Desk review first (keep
# worker/src/ssd-stream.ts CPU_TENSOR_ALLOWLIST identical; a parity test checks it).
CPU_TENSOR_ALLOWLIST: tuple[str, ...] = ("per_layer_token_embd.weight",)

# llama-server flags an ssd_stream entry must not use: they defeat streaming from disk.
FORBIDDEN_SERVER_ARGS = frozenset({"--no-mmap", "--mlock"})

# Mid-run floors (abort cleanly, upload nothing from the aborted step).
DISK_FLOOR_GB_DEFAULT = 5.0  # free space on the models disk
RAM_FLOOR_GB_DEFAULT = 2.0  # available system memory
POLL_S = 2.0

# Reason codes (fit_sweep status, preset exit code, agent result reason).
DISK_BELOW_FLOOR = "disk_below_floor"
RAM_BELOW_FLOOR = "ram_below_floor"
FLOOR_RC = {DISK_BELOW_FLOOR: 14, RAM_BELOW_FLOOR: 15}

# Uploaded with the run (WorkloadResult.extras) for ssd_stream entries only.
UPLOADED_EXTRAS_KEYS: tuple[str, ...] = (
    "ssd_stream",
    "cpu_tensors",
    "peak_rss_bytes",
    "disk_read_bytes",
)

# How fit_sweep tells its `llm-speed bench` child (external-server mode) about the entry.
ENV_SSD_STREAM = "LLM_SPEED_SSD_STREAM"
ENV_CPU_TENSORS = "LLM_SPEED_CPU_TENSORS"
ENV_SERVER_PID = "LLM_SPEED_SERVER_PID"


def is_ssd_stream(entry: Mapping[str, Any] | None) -> bool:
    return bool(entry) and entry.get("ssd_stream") is True


def valid_cpu_tensors(value: Any) -> list[str] | None:
    """The list if every name is allowlisted, unique, and there are no more names than
    the allowlist holds; else None (the whole field is dropped, never redacted)."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if len(value) > len(CPU_TENSOR_ALLOWLIST) or len(set(value)) != len(value):
        return None
    if not all(isinstance(v, str) and v in CPU_TENSOR_ALLOWLIST for v in value):
        return None
    return list(value)


def cpu_tensor_args(names: Iterable[str]) -> list[str]:
    """`-ot ^<escaped name>$=CPU` per allowlisted name (anything else: nothing)."""
    ok = valid_cpu_tensors(list(names))
    out: list[str] = []
    for n in ok or []:
        out += ["-ot", f"^{re.escape(n)}$=CPU"]
    return out


def semantic_errors(path: str, m: Mapping[str, Any], preset_args: list) -> list[str]:
    """Rules the JSON schema can't express (shared with worker/src/ssd-stream.ts)."""
    errs: list[str] = []
    ssd = m.get("ssd_stream")
    b = m.get("bytes")
    if isinstance(b, int) and b > DEFAULT_MAX_BYTES and ssd is not True:
        errs.append(f"{path}.bytes: above {DEFAULT_MAX_BYTES} (needs ssd_stream)")
    if ssd is not True:
        for k in ("n_cpu_moe", "vram_need_gib", "cpu_tensors"):
            if k in m:
                errs.append(f"{path}.{k}: only allowed with ssd_stream")
        return errs
    if m.get("backend") != "llama.cpp":
        errs.append(f"{path}.ssd_stream: llama.cpp entries only")
    for k in ("n_cpu_moe", "min_free_disk_gb", "vram_need_gib"):
        if k not in m:
            errs.append(f"{path}: ssd_stream needs {k!r}")
    v = m.get("vram_need_gib")
    if isinstance(v, (int, float)) and not (0 < v <= VRAM_NEED_GIB_MAX):
        errs.append(f"{path}.vram_need_gib: must be > 0 and <= {VRAM_NEED_GIB_MAX}")
    bad = [
        a
        for a in list(m.get("server_args") or []) + list(preset_args or [])
        if a in FORBIDDEN_SERVER_ARGS
    ]
    if bad:
        errs.append(f"{path}: ssd_stream forbids {', '.join(sorted(set(bad)))}")
    if "cpu_tensors" in m and valid_cpu_tensors(m.get("cpu_tensors")) is None:
        errs.append(
            f"{path}.cpu_tensors: only {', '.join(CPU_TENSOR_ALLOWLIST)} (no repeats)"
        )
    return errs


def bench_env(entry: Mapping[str, Any], server_pid: int | None) -> dict[str, str]:
    # always set all three, so a value left in the caller's environment never labels a
    # run that isn't ssd_stream
    env = {ENV_SSD_STREAM: "0", ENV_CPU_TENSORS: "", ENV_SERVER_PID: ""}
    if not is_ssd_stream(entry):
        return env
    env[ENV_SSD_STREAM] = "1"
    cpu = valid_cpu_tensors(entry.get("cpu_tensors"))
    if cpu:
        env[ENV_CPU_TENSORS] = ",".join(cpu)
    if server_pid:
        env[ENV_SERVER_PID] = str(int(server_pid))
    return env


def upload_extras(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """extras for a run of an ssd_stream entry: {"ssd_stream": true} plus the
    allowlisted cpu_tensors (dropped whole if any name isn't on the list)."""
    e = os.environ if env is None else env
    if e.get(ENV_SSD_STREAM) != "1":
        return {}
    out: dict[str, Any] = {"ssd_stream": True}
    raw = e.get(ENV_CPU_TENSORS, "")
    cpu = valid_cpu_tensors([x for x in raw.split(",") if x]) if raw else None
    if cpu:
        out["cpu_tensors"] = cpu
    return out


class ProcSampler:
    """Peak RSS and disk bytes read by the llama-server process during one workload
    (DR-005). Integers only; disk_read_bytes is absent where the OS doesn't report it
    (macOS)."""

    def __init__(self, pid: int, interval_s: float = 0.5):
        import psutil

        self.proc = psutil.Process(pid)
        self.interval_s = interval_s
        self.peak_rss: int | None = None
        self._read0 = self._read_bytes()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_bytes(self) -> int | None:
        try:
            return int(self.proc.io_counters().read_bytes)
        except Exception:  # noqa: BLE001 (AccessDenied, NoSuchProcess, unsupported)
            return None

    def _sample(self) -> None:
        try:
            rss = int(self.proc.memory_info().rss)
        except Exception:  # noqa: BLE001
            return
        if self.peak_rss is None or rss > self.peak_rss:
            self.peak_rss = rss

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def start(self) -> ProcSampler:
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample()
        out: dict[str, int] = {}
        if self.peak_rss is not None:
            out["peak_rss_bytes"] = self.peak_rss
        r1 = self._read_bytes()
        if self._read0 is not None and r1 is not None and r1 >= self._read0:
            out["disk_read_bytes"] = r1 - self._read0
        return out


def start_proc_sampler(env: Mapping[str, str] | None = None) -> ProcSampler | None:
    e = os.environ if env is None else env
    if e.get(ENV_SSD_STREAM) != "1":
        return None
    try:
        pid = int(e.get(ENV_SERVER_PID, ""))
    except ValueError:
        return None
    try:
        return ProcSampler(pid).start()
    except Exception:  # noqa: BLE001 (no such process, psutil missing)
        return None


def available_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:  # noqa: BLE001
        return None


def floor_breach(
    disk_path: Path,
    *,
    disk_floor_gb: float = DISK_FLOOR_GB_DEFAULT,
    ram_floor_gb: float = RAM_FLOOR_GB_DEFAULT,
    disk_free=None,
    ram_avail=None,
) -> str | None:
    """DISK_BELOW_FLOOR / RAM_BELOW_FLOOR, or None. `disk_free` / `ram_avail` (GB
    callables) are for tests."""
    try:
        free = disk_free() if disk_free else shutil.disk_usage(disk_path).free / 1024**3
    except OSError:
        free = None
    if free is not None and free < disk_floor_gb:
        return DISK_BELOW_FLOOR
    avail = ram_avail() if ram_avail else available_ram_gb()
    if avail is not None and avail < ram_floor_gb:
        return RAM_BELOW_FLOOR
    return None


def floors_from_env(env: Mapping[str, str] | None = None) -> tuple[float, float]:
    """Local overrides: $LLM_SPEED_SSD_DISK_FLOOR_GB, $LLM_SPEED_SSD_RAM_FLOOR_GB."""
    e = os.environ if env is None else env

    def num(k: str, d: float) -> float:
        try:
            v = float(e.get(k, ""))
        except ValueError:
            return d
        return v if 0 <= v < 100_000 else d

    return (
        num("LLM_SPEED_SSD_DISK_FLOOR_GB", DISK_FLOOR_GB_DEFAULT),
        num("LLM_SPEED_SSD_RAM_FLOOR_GB", RAM_FLOOR_GB_DEFAULT),
    )
