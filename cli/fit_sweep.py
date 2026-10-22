"""Memory-fit sweep for day-one launches, wrapping `llm-speed bench` (shared engine).

Moved into the CLI in 0.0.8 from Launch Lab's `launches/tools/fit_sweep.py`, which is
now a thin deprecated wrapper around this module; `llm-speed bench --preset ...`
(cli/commands/preset.py) drives the same `Sweep`. Stdlib only (plus the CLI itself). Runs on Boss's own machines (RTX 5090 Linux/WSL, Apple
Silicon Macs). It never measures speed itself: every number comes from the team
CLI (`llm-speed bench --json ...`). This script only:

  * downloads one model at a time (optional), checking free disk first,
  * starts llama-server per (file, context) with -ngl 999 [--n-cpu-moe N],
  * waits for /health or classifies the load failure (OOM / unsupported arch /
    bad flag / crash / timeout) from the server log,
  * runs `llm-speed bench --backend llama.cpp` against it via LLAMA_CPP_SERVER_URL
    (or `--backend mlx` in-process on Macs),
  * samples peak GPU memory (nvidia-smi) or unified memory (vm_stat + RSS) while
    the attempt runs,
  * stops the server and appends one record per attempt (failures and skips
    included) to <launch_dir>/results.json,
  * uploads the signed artifact only with --upload (via `llm-speed bench --resume`).

Stepping: entries run in plan order (write them high-bit -> low-bit); contexts
run low -> high; after the first failed context of an entry, the larger
contexts of that entry are recorded as skipped_after_failure.

Usage:
  fit_sweep.py plan  PLAN.json                 # dry run: print attempts + commands
  fit_sweep.py run   PLAN.json [--download --one-at-a-time] [--upload] ...
  fit_sweep.py merge A.json B.json -o results.json
Stop gracefully: touch <launch_dir>/STOP
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt

try:  # POSIX only; on Windows the lock is skipped
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "fit-sweep/1"
SGT = dt.timezone(dt.timedelta(hours=8))
DEFAULT_WORKLOADS_BY_CTX = {
    "0": "chat-short",
    "8192": "chat-short,chat-long",
    "32768": "chat-short,chat-long,long-context-decay",
}
DEFAULT_SERVER_ARGS = [
    "--parallel",
    "1",
    "--no-cache-prompt",
    "--cache-ram",
    "0",
    "--temp",
    "0",
    "--seed",
    "42",
]

# Server-log classification. Order matters: first match wins.
FAIL_PATTERNS = [
    (
        "bad_server_args",
        re.compile(
            r"invalid argument|unknown argument|error: unrecognized|usage: .*llama-server",
            re.I,
        ),
    ),
    (
        "unsupported_model",
        re.compile(
            r"unknown model architecture|unsupported model|unknown (?:tensor )?type|invalid ggml type|wrong number of tensors|key not found in model",
            re.I,
        ),
    ),
    (
        "oom",
        re.compile(
            r"out of memory|cudaMalloc failed|CUDA error: out of memory|failed to allocate|unable to allocate|"
            r"alloc(?:ate)?_buffer.*failed|buffer allocation failed|kIOGPUCommandBufferCallbackErrorOutOfMemory|"
            r"Insufficient Memory|not enough memory|exceeds the recommended|failed to create context|"
            r"failed to initialize the context|ggml_metal.*error.*memory",
            re.I,
        ),
    ),
    (
        "load_failed",
        re.compile(
            r"error loading model|failed to load model|llama_model_load: error|unable to load model|failed to open",
            re.I,
        ),
    ),
]
LOG_SIGNAL_RE = re.compile(
    r"offloaded \d+/\d+ layers|model buffer size|KV (?:self |buffer )?size|kv_cache.*size|compute buffer size|"
    r"n_ctx(?:_seq)?\s*=|n_ctx_train|build:|system_info|CPU_Mapped|n_cpu_moe|override|flash_attn|"
    r"recommendedMaxWorkingSetSize|GPU name|ggml_cuda_init|Device \d+:",
    re.I,
)


# ----------------------------------------------------------------- helpers


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def now_sgt() -> str:
    return dt.datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S SGT")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(SGT).strftime('%H:%M:%S')} SGT] {msg}", flush=True)


def run_cmd(cmd: list[str], timeout: float = 15.0) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_status(url: str, timeout: float = 2.0) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def http_json(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def expand(p: str | None) -> Path | None:
    if not p:
        return None
    return Path(os.path.expandvars(os.path.expanduser(p)))


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def tail(path: Path, n: int = 40) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def decode_jws_payload(artifact: Path) -> dict | None:
    """Read the CLI's signed artifact ({"jws": "<h>.<p>.<s>"}) without verifying."""
    try:
        tok = json.loads(artifact.read_text(encoding="utf-8"))["jws"]
        seg = tok.split(".")[1]
        seg += "=" * (-len(seg) % 4)
        return json.loads(base64.urlsafe_b64decode(seg))
    except Exception:
        return None


# ----------------------------------------------------------------- machine facts


def machine_facts(gpu_index: int) -> dict:
    f: dict = {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
    }
    if f["os"] == "Darwin":
        f["macos"] = (run_cmd(["sw_vers", "-productVersion"]) or "").strip() or None
        f["chip"] = (
            run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
        ).strip() or None
        try:
            f["ram_gb"] = round(
                int((run_cmd(["sysctl", "-n", "hw.memsize"]) or "0").strip()) / 1024**3,
                1,
            )
        except ValueError:
            f["ram_gb"] = None
        wl = (run_cmd(["sysctl", "-n", "iogpu.wired_limit_mb"]) or "").strip()
        f["iogpu_wired_limit_mb"] = (
            int(wl) if wl.isdigit() else None
        )  # 0 = macOS default (~75% of RAM)
        f["kind"] = "apple"
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    f["ram_gb"] = round(int(line.split()[1]) / 1024**2, 1)
        except OSError:
            f["ram_gb"] = None
        f["wsl"] = "microsoft" in platform.release().lower()
        q = run_cmd(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=name,driver_version,memory.total,power.limit,vbios_version",
                "--format=csv,noheader,nounits",
            ]
        )
        if q and "," in q:
            parts = [x.strip() for x in q.strip().splitlines()[0].split(",")]
            f.update(
                kind="nvidia",
                gpu=parts[0],
                driver_version=parts[1],
                gpu_memory_total_mib=_num(parts[2]),
                power_limit_w=_num(parts[3]),
                vbios=parts[4] if len(parts) > 4 else None,
            )
            m = re.search(r"CUDA Version:\s*([\d.]+)", run_cmd(["nvidia-smi"]) or "")
            f["cuda_driver_api"] = m.group(1) if m else None
        else:
            f["kind"] = "cpu"
    f["disk_note"] = None
    return f


def _num(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def machine_slug(facts: dict) -> str:
    if facts.get("kind") == "nvidia":
        base = re.sub(r"(?i)nvidia|geforce", "", facts.get("gpu") or "gpu")
    elif facts.get("kind") == "apple":
        base = re.sub(r"(?i)apple", "", facts.get("chip") or "mac")
    else:
        base = "cpu"  # never the hostname
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    if facts.get("kind") == "apple" and facts.get("ram_gb"):
        slug += f"-{int(round(facts['ram_gb']))}gb"
    return slug


def llama_version(server_bin: str) -> dict:
    from cli.launch_telemetry import parse_llama_build
    from cli.pathscrub import scrub_paths

    out = run_cmd([server_bin, "--version"], timeout=20) or ""
    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", out)
    tag = parse_llama_build(out)
    number = (
        int(m.group(1)) if m else (int(tag["build"][1:]) if tag.get("build") else None)
    )
    flags = [
        k
        for k in ("CUDA", "Metal", "Vulkan", "BLAS")
        if re.search(rf"\b{k}\b", out, re.I)
    ]
    return {
        "raw": scrub_paths(out.strip()[-400:]),
        "build_number": number,
        "build": tag.get("build"),
        "commit": (m.group(2) if m else tag.get("commit")),
        "flags": flags,
        "binary": Path(server_bin).name,
    }


# ----------------------------------------------------------------- memory sampling


class MemSampler(threading.Thread):
    """Samples GPU memory (nvidia-smi) or unified memory (vm_stat + RSS) every interval."""

    def __init__(self, kind: str, gpu_index: int = 0, interval: float = 0.5):
        super().__init__(daemon=True)
        self.kind, self.gpu_index, self.interval = kind, gpu_index, interval
        self.pid: int | None = None
        self._halt = threading.Event()
        self.baseline: float | None = None
        self.peak: float | None = None
        self.peak_proc: float | None = None
        self.total: float | None = None
        self.samples = 0
        self.swapouts_start: int | None = None
        self.swapouts_end: int | None = None
        self.series: list[tuple[float, float]] = []
        self.t0 = time.monotonic()

    # --- one reading: (system_used_mib, process_mib or None)
    def read(self) -> tuple[float | None, float | None]:
        if self.kind == "nvidia":
            out = run_cmd(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
            )
            used = None
            if out:
                try:
                    a, b = [float(x) for x in out.strip().splitlines()[0].split(",")]
                    used, self.total = a, b
                except ValueError:
                    pass
            proc = None
            if self.pid:
                apps = (
                    run_cmd(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=5,
                    )
                    or ""
                )
                for line in apps.splitlines():
                    bits = [x.strip() for x in line.split(",")]
                    if len(bits) == 2 and bits[0] == str(self.pid):
                        proc = _num(bits[1])  # [N/A] on WSL -> None
            return used, proc
        if self.kind == "apple":
            out = run_cmd(["vm_stat"], timeout=5) or ""
            ps = re.search(r"page size of (\d+) bytes", out)
            page = int(ps.group(1)) if ps else 16384

            def pages(label):
                m = re.search(rf"{label}:\s+(\d+)", out)
                return int(m.group(1)) if m else 0

            used = (
                (
                    pages("Pages active")
                    + pages("Pages wired down")
                    + pages("Pages occupied by compressor")
                )
                * page
                / 2**20
            )
            so = pages("Swapouts")
            if self.swapouts_start is None:
                self.swapouts_start = so
            self.swapouts_end = so
            proc = None
            if self.pid:
                rss = (
                    run_cmd(["ps", "-o", "rss=", "-p", str(self.pid)], timeout=5) or ""
                ).strip()
                proc = int(rss) / 1024 if rss.isdigit() else None
            return used, proc
        # plain Linux fallback: MemTotal - MemAvailable
        try:
            mi = {
                ln.split(":")[0]: int(ln.split()[1])
                for ln in Path("/proc/meminfo").read_text().splitlines()
            }
            return (mi["MemTotal"] - mi["MemAvailable"]) / 1024, None
        except Exception:
            return None, None

    def snapshot_baseline(self) -> None:
        used, _ = self.read()
        self.baseline = used

    def run(self) -> None:
        while not self._halt.is_set():
            used, proc = self.read()
            if used is not None:
                self.samples += 1
                self.peak = used if self.peak is None else max(self.peak, used)
                if len(self.series) < 4000:
                    self.series.append(
                        (round(time.monotonic() - self.t0, 1), round(used))
                    )
            if proc is not None:
                self.peak_proc = (
                    proc if self.peak_proc is None else max(self.peak_proc, proc)
                )
            self._halt.wait(self.interval)

    def stop(self) -> dict:
        self._halt.set()
        self.join(timeout=10)
        method = {
            "nvidia": "nvidia-smi memory.used (whole GPU), 0.5 s samples",
            "apple": "vm_stat active+wired+compressed (whole machine) + ps RSS of the serving process, 0.5 s samples",
        }.get(self.kind, "/proc/meminfo MemTotal-MemAvailable")
        d = {
            "method": method,
            "baseline_mib": _r(self.baseline),
            "peak_mib": _r(self.peak),
            "peak_delta_mib": _r(self.peak - self.baseline)
            if self.peak is not None and self.baseline is not None
            else None,
            "process_peak_mib": _r(self.peak_proc),
            "total_mib": _r(self.total),
            "samples": self.samples,
        }
        if self.kind == "apple":
            d["swapouts_delta"] = (
                (self.swapouts_end - self.swapouts_start)
                if self.swapouts_start is not None and self.swapouts_end is not None
                else None
            )
        return d


def _r(x):
    return None if x is None else round(x)


# ----------------------------------------------------------------- downloads


def hf_tree(repo: str, revision: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1"
    data = http_json(url, timeout=30)
    return [x for x in (data or []) if x.get("type") == "file"]


def expected_bytes(entry: dict) -> int | None:
    if entry.get("bytes"):
        return int(entry["bytes"])
    if not entry.get("repo"):
        return None
    tree = hf_tree(entry["repo"], entry.get("revision", "main"))
    if entry.get("backend") == "mlx":
        return sum(x.get("size", 0) for x in tree) or None
    for x in tree:
        if x["path"] == entry.get("file"):
            return x.get("size")
    return None


def _hf_cli() -> list[str] | None:
    if shutil.which("hf"):
        return ["hf", "download"]
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli", "download"]
    return None


def expected_bytes_offline(entry: dict) -> int | None:
    """Size from the plan only (no network)."""
    return int(entry["bytes"]) if entry.get("bytes") else None


def download_cmd(entry: dict, dest_dir: Path) -> list[str]:
    cli = _hf_cli() or ["hf", "download"]
    cmd = cli + [entry["repo"]]
    if entry.get("backend") != "mlx":
        cmd.append(entry["file"])
    cmd += ["--revision", entry.get("revision", "main"), "--local-dir", str(dest_dir)]
    return cmd


def _progress(done: int, total: int | None, t0: float, name: str) -> None:
    if not sys.stderr.isatty() and done != total:
        return
    el = max(1e-6, time.monotonic() - t0)
    rate = done / el
    if total:
        pct = 100 * done / total
        left = (total - done) / rate if rate > 0 else 0
        bar = "#" * int(pct / 4)
        msg = (
            f"\r  {name[:40]:<40} [{bar:<25}] {pct:5.1f}%  {done / 1e9:6.2f}/{total / 1e9:.2f} GB  "
            f"{rate / 1e6:6.1f} MB/s  {left / 60:4.0f} min left "
        )
    else:
        msg = f"\r  {name[:40]:<40} {done / 1e9:6.2f} GB  {rate / 1e6:6.1f} MB/s "
    sys.stderr.write(msg)
    if total and done >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def stdlib_download(entry: dict, dest_dir: Path, *, progress=_progress) -> None:
    """Fallback download (no hf tooling). Resumes a partial file with an HTTP Range request
    and shows a progress bar. Writes to <file>.part and renames when complete."""
    rev = entry.get("revision", "main")
    if entry.get("backend") == "mlx":
        files = [(x["path"], x.get("size")) for x in hf_tree(entry["repo"], rev)]
    else:
        files = [(entry["file"], entry.get("bytes"))]
    for rel, size in files:
        target = dest_dir / rel
        if target.exists() and (not size or target.stat().st_size >= size):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        url = f"https://huggingface.co/{entry['repo']}/resolve/{rev}/{rel}"
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={have}-"} if have else {}
        )
        tok = os.environ.get("HF_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if (
                    have and getattr(r, "status", 206) == 200
                ):  # server ignored Range: restart
                    have = 0
                total = (
                    size or (have + int(r.headers.get("Content-Length") or 0)) or None
                )
                with part.open("ab" if have else "wb") as f:
                    done = have
                    while True:
                        chunk = r.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        progress(
                            done - have, (total - have) if total else None, t0, rel
                        )
        except urllib.error.HTTPError as e:
            if e.code != 416:  # 416 = the .part file is already complete
                raise
        part.replace(target)


def download(entry: dict, dest_dir: Path) -> None:
    """Resumable download with progress: huggingface_hub (if importable), else the `hf` /
    `huggingface-cli` tool, else plain HTTPS. All three reuse a partial download."""
    try:
        import huggingface_hub  # type: ignore[import-not-found]
    except ImportError:
        huggingface_hub = None
    if huggingface_hub is not None:
        kw = {
            "repo_id": entry["repo"],
            "revision": entry.get("revision") or None,
            "local_dir": str(dest_dir),
        }
        if entry.get("backend") == "mlx":
            huggingface_hub.snapshot_download(**kw)
        else:
            huggingface_hub.hf_hub_download(filename=entry["file"], **kw)
        return
    if _hf_cli():
        rc = subprocess.run(download_cmd(entry, dest_dir)).returncode
        if rc != 0:
            raise RuntimeError(f"hf download exited {rc}")
        return
    stdlib_download(entry, dest_dir)


# ----------------------------------------------------------------- the sweep


class Sweep:
    def __init__(
        self,
        plan_path: Path | None,
        args,
        *,
        plan: dict | None = None,
        launch_dir: Path | None = None,
    ):
        self.args = args
        if plan is not None:
            self.plan_path = (launch_dir or Path.cwd()) / "plan.json"
            self.plan = plan
        else:
            self.plan_path = Path(plan_path).resolve()
            self.plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        p = self.plan
        self.launch_dir = (
            launch_dir or (self.plan_path.parent / p.get("launch_dir", "."))
        ).resolve()
        self.results_path = (
            Path(args.results).resolve()
            if getattr(args, "results", None)
            else self.launch_dir / "results.json"
        )
        self.models_dir = expand(
            os.environ.get("FIT_MODELS_DIR") or p.get("models_dir", "~/models")
        )
        self.mlx_dir = expand(
            os.environ.get("FIT_MLX_DIR") or p.get("mlx_models_dir", "~/Models/mlx")
        )
        self.server_bin = os.environ.get("LLAMA_SERVER") or p.get(
            "llama_server", "llama-server"
        )
        self.server_bin = shutil.which(str(expand(self.server_bin))) or str(
            expand(self.server_bin)
        )
        self.cli_bin = os.environ.get("LLM_SPEED") or p.get("llm_speed", "llm-speed")
        self.cli_bin = shutil.which(str(expand(self.cli_bin))) or str(
            expand(self.cli_bin)
        )
        # Command prefix for the CLI. The preset runs the CLI it lives in.
        self.cli_prefix: list[str] = list(
            getattr(args, "cli_prefix", None) or [self.cli_bin]
        )
        self.strict_anon = bool(getattr(args, "strict_anon", False))
        # "auto": delete a model only if this sweep downloaded it AND disk is tight.
        self.cleanup_mode = getattr(args, "cleanup_mode", None) or (
            "after_entry" if getattr(args, "one_at_a_time", False) else "never"
        )
        self.on_attempt = getattr(args, "on_attempt", None)  # hook(rec, adir) -> None
        self._resolved: dict[str, Path] = {}  # entry id -> path found in HF cache
        self._mlx_ids: dict[
            str, str
        ] = {}  # entry id -> HF id when served from HF cache
        self.llama: dict | None = None
        self.cli_version: str | None = None
        self.gpu_index = int(p.get("gpu_index", 0))
        self.wbc = p.get("workloads_by_ctx", DEFAULT_WORKLOADS_BY_CTX)
        self.facts: dict = {}
        self.machine = getattr(args, "machine", None) or p.get("machine", "auto")
        self.stop_requested = False

    # ---- plan expansion
    def entries(self) -> list[dict]:
        only = set(filter(None, (getattr(self.args, "only", "") or "").split(",")))
        out = []
        for e in self.plan["entries"]:
            if only and e["id"] not in only:
                continue
            if (
                e.get("optional")
                and not getattr(self.args, "include_optional", False)
                and e["id"] not in only
            ):
                continue
            out.append(e)
        return out

    @staticmethod
    def contexts(entry: dict) -> list[dict]:
        cs = [
            c if isinstance(c, dict) else {"ctx": int(c)}
            for c in entry.get("contexts", [4096])
        ]
        return sorted(cs, key=lambda c: c["ctx"])

    def workloads_for(self, entry: dict, c: dict, ctx: int | None = None) -> str:
        """Workloads for a step, minus any whose prompt + output would not fit in the
        step's context (`ctx` = the server's real n_ctx when known)."""
        if c.get("workloads"):
            wl = c["workloads"]
        elif entry.get("workloads"):
            wl = entry["workloads"]
        else:
            best = None
            for k, v in self.wbc.items():
                if int(k) <= c["ctx"] and (best is None or int(k) > best[0]):
                    best = (int(k), v)
            wl = best[1] if best else "chat-short"
        limit = int(ctx or c["ctx"])
        names = [w.strip() for w in wl.split(",") if w.strip()]
        keep = [w for w in names if workload_ctx_need(w, c) <= limit]
        if len(keep) < len(names):
            dropped = [w for w in names if w not in keep]
            log(
                f"  {entry['id']} ctx={limit}: not running {','.join(dropped)} "
                f"(needs more than {limit} tokens of context)"
            )
        return ",".join(keep) or "chat-short"

    def model_path(self, entry: dict) -> Path:
        if entry["id"] in self._resolved:
            return self._resolved[entry["id"]]
        if entry.get("backend") == "mlx":
            return self.mlx_dir / entry.get("dir", entry["repo"].split("/")[-1])
        return self.models_dir / entry["file"]

    def server_cmd(self, entry: dict, c: dict, port: int) -> list[str]:
        mp = self.model_path(entry)
        cmd = [
            self.server_bin,
            "-m",
            str(mp),
            "--alias",
            mp.name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(c["ctx"]),
            "-ngl",
            str(entry.get("ngl", self.plan.get("ngl", 999))),
        ]
        if entry.get("n_cpu_moe") is not None:
            cmd += ["--n-cpu-moe", str(entry["n_cpu_moe"])]
        if entry.get("reasoning") in ("on", "off", "auto"):
            cmd += ["--reasoning", entry["reasoning"]]
        cmd += list(self.plan.get("server_args", DEFAULT_SERVER_ARGS)) + list(
            entry.get("server_args", [])
        )
        return cmd

    def bench_cmd(self, entry: dict, wl: str, artifact: Path) -> list[str]:
        backend = entry.get("backend", "llama.cpp")
        model = self._mlx_ids.get(entry["id"]) or str(self.model_path(entry).resolve())
        cmd = (
            self.cli_prefix
            + [
                "bench",
                "--backend",
                backend,
                "--model",
                model,
                "--workload",
                wl,
                "--json",
                str(artifact),
            ]
            + list(self.plan.get("cli_args", []))
        )
        if self.strict_anon and "--strict-anon" not in cmd:
            cmd.append("--strict-anon")
        return cmd

    @staticmethod
    def declared_server_args(cmd: list[str]) -> str:
        """The server flags the CLI can't see in external mode (ctx, -ngl, --n-cpu-moe,
        ...), declared to it via $LLM_SPEED_LLAMA_ARGS so the run records them."""
        skip_with_value = {"-m", "--model", "--alias", "--host", "--port"}
        out, i = [], 1
        while i < len(cmd):
            if cmd[i] in skip_with_value:
                i += 2
                continue
            out.append(cmd[i])
            i += 1
        return " ".join(out)

    # ---- results file
    def load_results(self) -> dict:
        if self.results_path.exists():
            try:
                return json.loads(self.results_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                bak = self.results_path.with_suffix(f".corrupt-{int(time.time())}.json")
                self.results_path.rename(bak)
                log(f"results.json was unreadable; moved to {bak.name}")
        return {
            "schema": SCHEMA,
            "launch": self.plan.get("launch"),
            "machines": {},
            "attempts": [],
        }

    def append(self, rec: dict) -> None:
        data = self.load_results()
        data.setdefault("machines", {})[self.machine] = self.facts
        data["attempts"].append(rec)
        data["updated_at"] = now_utc()
        atomic_write_json(self.results_path, data)

    def synthetic(self) -> str | None:
        from cli.upload_guard import synthetic_reason

        return synthetic_reason((self.server_bin,))

    def base_record(self, entry: dict, c: dict | None, status: str, **kw) -> dict:
        rec = {
            "attempt_id": f"{self.machine}/{entry['id']}/{(c or {}).get('ctx', '-')}/{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "launch": self.plan.get("launch"),
            "machine": self.machine,
            "entry_id": entry["id"],
            "model": entry.get("model"),
            "backend": entry.get("backend", "llama.cpp"),
            "quant": entry.get("quant"),
            "control": bool(entry.get("control")),
            "repo": entry.get("repo"),
            "revision": entry.get("revision"),
            "file": entry.get("file") or entry.get("dir"),
            "ctx": (c or {}).get("ctx"),
            "ngl": entry.get("ngl", self.plan.get("ngl", 999))
            if entry.get("backend", "llama.cpp") == "llama.cpp"
            else None,
            "n_cpu_moe": entry.get("n_cpu_moe"),
            "reasoning": entry.get("reasoning"),
            "status": status,
            "fits": None,
            "started_at": now_utc(),
            "started_sgt": now_sgt(),
            "llama_cpp": self.llama
            if entry.get("backend", "llama.cpp") == "llama.cpp"
            else None,
            "cli_version": self.cli_version,
            "synthetic": self.synthetic(),
        }
        rec.update(kw)
        return rec

    # ---- preflight
    def preflight(self, need_llama: bool, need_mlx: bool) -> list[str]:
        problems = []
        self.facts = machine_facts(self.gpu_index)
        if self.machine == "auto":
            self.machine = machine_slug(self.facts)
        self.cli_version = (
            run_cmd(self.cli_prefix + ["--version"]) or ""
        ).strip() or None
        if not self.cli_version:
            problems.append(
                f"llm-speed CLI not runnable via {' '.join(self.cli_prefix)} (pip install -U llm-speed)"
            )
        self.llama = None
        if need_llama:
            if not (Path(self.server_bin).exists() or shutil.which(self.server_bin)):
                problems.append(f"llama-server not found at {self.server_bin}")
            else:
                self.llama = llama_version(self.server_bin)
                mn = self.plan.get("min_llama_build")
                bn = self.llama.get("build_number")
                force = getattr(self.args, "force", False)
                if mn and bn and bn > 1 and bn < int(mn) and not force:
                    problems.append(
                        f"llama.cpp build b{bn} is older than the required b{mn} "
                        f"(Flint support landed in b{mn}); upgrade llama.cpp"
                    )
                if mn and (not bn or bn <= 1):
                    msg = (
                        f"llama-server reports build '{bn}' (a shallow clone or packaged build hides the number), "
                        f"commit {self.llama.get('commit')}; cannot confirm it is b{mn} or later"
                    )
                    if force:
                        log("WARNING: " + msg)
                    else:
                        problems.append(
                            msg
                            + "; use a full clone at the release tag or pass --force"
                        )
                # The CLI's backend_version comes from the FIRST llama-server on PATH, so put ours first.
                os.environ["PATH"] = (
                    str(Path(self.server_bin).resolve().parent)
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )
        if (
            need_mlx
            and self.facts.get("kind") != "apple"
            and not getattr(self.args, "force", False)
        ):
            problems.append("plan has MLX entries but this is not an Apple Silicon Mac")
        return problems

    def gpu_busy(self) -> float | None:
        if self.facts.get("kind") != "nvidia":
            return None
        s = MemSampler("nvidia", self.gpu_index)
        used, _ = s.read()
        return used

    # ---- one attempt (llama.cpp)
    def attempt_llama(self, entry: dict, c: dict, adir: Path) -> dict:
        port = free_port()
        cmd = self.server_cmd(entry, c, port)
        wl = self.workloads_for(entry, c)
        artifact = adir / "llm-speed-run.json"
        rec = self.base_record(entry, c, "running", workloads=wl, server_cmd=cmd)
        kind = self.facts.get("kind", "cpu")
        sampler = MemSampler(kind, self.gpu_index)
        sampler.snapshot_baseline()
        srv_log = adir / "server.log"
        t0 = time.monotonic()
        with srv_log.open("w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        sampler.pid = proc.pid
        sampler.start()
        base = f"http://127.0.0.1:{port}"
        status = None
        deadline = t0 + float(self.plan.get("health_timeout_s", 900))
        try:
            while True:
                if proc.poll() is not None:
                    status = "server_exited"
                    break
                if http_status(base + "/health") == 200:
                    break
                if time.monotonic() > deadline:
                    status = "load_timeout"
                    break
                time.sleep(0.5)
            rec["load_seconds"] = round(time.monotonic() - t0, 1)
            if status is None:
                props = http_json(base + "/props") or {}
                gs = props.get("default_generation_settings") or {}
                rec["server_props"] = {
                    k: props.get(k)
                    for k in (
                        "build_info",
                        "total_slots",
                        "model_path",
                        "chat_template_caps",
                    )
                    if k in props
                }
                rec["server_props"]["n_ctx"] = gs.get("n_ctx") or props.get("n_ctx")
                real = rec["server_props"]["n_ctx"]
                if isinstance(real, int) and real != c["ctx"]:
                    log(
                        f"  llama-server is running with n_ctx={real}, not the requested {c['ctx']}"
                    )
                    rec.setdefault("flags", []).append(f"server-n-ctx:{real}")
                    wl = self.workloads_for(entry, c, ctx=real)
                    rec["workloads"] = wl
                env = dict(
                    os.environ,
                    LLAMA_CPP_SERVER_URL=base,
                    LLAMA_CPP_MODELS_DIR=str(self.model_path(entry).parent.resolve()),
                    LLM_SPEED_LLAMA_ARGS=self.declared_server_args(cmd),
                    **{k: str(v) for k, v in (c.get("env") or {}).items()},
                )
                bcmd = self.bench_cmd(entry, wl, artifact)
                rec["bench_cmd"] = bcmd
                rec.update(self._run_bench(bcmd, env, adir, artifact))
                if proc.poll() is not None:
                    rec["server_died_during_bench"] = True
        finally:
            self._stop_server(proc)
            rec["memory"] = sampler.stop()
        rec["log_signals"] = [
            ln.strip()[:240] for ln in tail(srv_log, 4000) if LOG_SIGNAL_RE.search(ln)
        ][-30:]
        if status is not None:
            rec["server_rc"] = proc.returncode
            rec["status"] = self._classify_log(srv_log) or (
                "load_timeout" if status == "load_timeout" else "server_crashed"
            )
            rec["fits"] = False if rec["status"] == "oom" else None
            rec["server_log_tail"] = tail(srv_log, 25)
        else:
            self._status_from_bench(rec, srv_log)
        self._wait_memory_release(sampler.baseline)
        return rec

    # ---- one attempt (mlx, in-process inside the CLI)
    def attempt_mlx(self, entry: dict, c: dict, adir: Path) -> dict:
        wl = self.workloads_for(entry, c)
        artifact = adir / "llm-speed-run.json"
        rec = self.base_record(
            entry,
            c,
            "running",
            workloads=wl,
            mlx_note="MLX runs in-process in the CLI (no server, no ctx allocation); "
            "'ctx' selects workloads only",
        )
        sampler = MemSampler("apple" if self.facts.get("kind") == "apple" else "cpu")
        sampler.snapshot_baseline()
        sampler.start()
        bcmd = self.bench_cmd(entry, wl, artifact)
        rec["bench_cmd"] = bcmd
        try:
            rec.update(
                self._run_bench(
                    bcmd,
                    dict(
                        os.environ,
                        **{k: str(v) for k, v in (c.get("env") or {}).items()},
                    ),
                    adir,
                    artifact,
                    sampler=sampler,
                )
            )
        finally:
            rec["memory"] = sampler.stop()
        self._status_from_bench(rec, None)
        return rec

    def _run_bench(
        self, bcmd, env, adir: Path, artifact: Path, sampler: MemSampler | None = None
    ) -> dict:
        out: dict = {}
        t0 = time.monotonic()
        with (adir / "bench.log").open("w") as lf:
            p = subprocess.Popen(
                bcmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            if sampler is not None:
                sampler.pid = p.pid
            try:
                rc = p.wait(timeout=float(self.plan.get("bench_timeout_s", 3600)))
            except subprocess.TimeoutExpired:
                p.kill()
                rc = "timeout"
            except KeyboardInterrupt:
                try:
                    p.terminate()
                    p.wait(timeout=10)
                except Exception:  # noqa: BLE001
                    p.kill()
                raise
        out["bench_seconds"] = round(time.monotonic() - t0, 1)
        out["bench_rc"] = rc
        payload = decode_jws_payload(artifact) if artifact.exists() else None
        out["artifact"] = str(artifact) if artifact.exists() else None
        if payload:
            out["cli_results"] = [
                summarize_result(r) for r in payload.get("results", [])
            ]
            fp = payload.get("fingerprint") or {}
            out["cli_fingerprint"] = {
                "accelerator_summary": fp.get("accelerator_summary"),
                "backends": (fp.get("extras") or {}).get("backends"),
            }
        else:
            out["cli_results"] = []
            out["bench_log_tail"] = tail(adir / "bench.log", 25)
        return out

    def _status_from_bench(self, rec: dict, srv_log: Path | None) -> None:
        res = rec.get("cli_results") or []
        errs = [r for r in res if r.get("error")]
        if rec.get("bench_rc") == "timeout":
            rec["status"] = "bench_timeout"
        elif rec.get("server_died_during_bench"):
            rec["status"] = (
                self._classify_log(srv_log) if srv_log else None
            ) or "server_crashed"
            rec["server_log_tail"] = tail(srv_log, 25) if srv_log else None
        elif not res:
            blob = "\n".join(rec.get("bench_log_tail") or [])
            rec["status"] = (
                "oom"
                if re.search(r"memory|Killed|SIGKILL|-9", blob, re.I)
                or rec.get("bench_rc") in (-9, 137)
                else "bench_failed"
            )
        elif not errs:
            rec["status"] = "ok"
        elif len(errs) == len(res):
            blob = " ".join(str(r.get("error")) for r in errs)
            if re.search(
                r"out of memory|metal.*memory|insufficient memory", blob, re.I
            ):
                rec["status"] = "oom"
            elif re.search(r"ReadTimeout|timed out", blob, re.I):
                rec["status"] = (
                    "cli_read_timeout"  # CLI's 30 s per-chunk read timeout; not a fit failure
                )
            else:
                rec["status"] = "bench_failed"
        else:
            rec["status"] = "partial"
        rec["fits"] = (
            True
            if rec["status"] in ("ok", "partial")
            else (False if rec["status"] == "oom" else None)
        )
        if rec["fits"] and rec.get("memory", {}).get("swapouts_delta"):
            rec["flags"] = rec.get("flags", []) + ["swapped-during-run"]
        rec["ctx_actual"] = actual_ctx(rec)
        rec["finished_at"] = now_utc()

    @staticmethod
    def _classify_log(path: Path | None) -> str | None:
        if not path:
            return None
        text = "\n".join(tail(path, 400))
        for label, pat in FAIL_PATTERNS:
            if pat.search(text):
                return label
        return None

    @staticmethod
    def _stop_server(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except Exception:
                pass

    def _wait_memory_release(self, baseline: float | None) -> None:
        if self.facts.get("kind") != "nvidia" or baseline is None:
            time.sleep(2)
            return
        s = MemSampler("nvidia", self.gpu_index)
        for _ in range(60):
            used, _ = s.read()
            if used is None or used <= baseline + 512:
                return
            time.sleep(0.5)
        log("WARNING: GPU memory did not return to baseline within 30 s")

    # ---- download / cleanup
    def hf_cache_hit(self, entry: dict) -> Path | None:
        """A complete copy already in the Hugging Face cache (pre-existing, never deleted)."""
        if not entry.get("repo") or not entry.get("revision"):
            return None
        hub = Path(
            os.environ.get("HF_HUB_CACHE")
            or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
            / "hub"
        )
        snap = (
            hub
            / ("models--" + entry["repo"].replace("/", "--"))
            / "snapshots"
            / entry["revision"]
        )
        exp = expected_bytes_offline(entry)
        if entry.get("backend") == "mlx":
            if (
                snap.is_dir()
                and (snap / "config.json").exists()
                and (exp is None or path_bytes(snap) >= exp)
            ):
                return snap
            return None
        f = snap / entry.get("file", "")
        if f.is_file() and (exp is None or f.stat().st_size >= exp):
            return f
        return None

    def disk_need(self, entry: dict) -> int:
        exp = expected_bytes_offline(entry) or 0
        mp = self.model_path(entry)
        have = path_bytes(mp) if mp.exists() else 0
        need = max(0, exp - have) + int(
            float(self.plan.get("disk_margin_gb", 5)) * 1024**3
        )
        if entry.get("min_free_disk_gb") and not (exp and have >= exp):
            need = max(need, int(float(entry["min_free_disk_gb"]) * 1e9))
        return need

    def ensure_model(self, entry: dict) -> tuple[str, dict]:
        """Returns (state, info). state: present | downloaded | missing | skipped_disk | download_failed.

        Reuses a complete file in models_dir or the HF cache; resumes a partial file;
        checks free disk before every download."""
        cached = self.hf_cache_hit(entry)
        if cached is not None and not self.model_path(entry).exists():
            self._resolved[entry["id"]] = cached
            if entry.get("backend") == "mlx":
                self._mlx_ids[entry["id"]] = entry["repo"]
        mp = self.model_path(entry)
        exp = expected_bytes(entry)
        info = {"path": mp.name, "expected_bytes": exp}
        have = path_bytes(mp) if mp.exists() else 0
        if mp.exists() and (exp is None or have >= exp):
            info["bytes"] = have
            info["source"] = (
                "hf_cache" if entry["id"] in self._resolved else "models_dir"
            )
            return "present", info
        if not getattr(self.args, "download", False):
            return "missing", info
        mp.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(mp.parent).free
        need = (
            (exp or 0) - have + int(float(self.plan.get("disk_margin_gb", 5)) * 1024**3)
        )
        if entry.get(
            "min_free_disk_gb"
        ):  # e.g. Q8_0 on a disk-tight Mac: ~45 GB free or skip
            need = max(need, int(float(entry["min_free_disk_gb"]) * 1e9))
        info.update(free_bytes=free, needed_bytes=need, resumed_from_bytes=have or None)
        if (exp or entry.get("min_free_disk_gb")) and free < need:
            log(
                f"SKIP {entry['id']}: needs {need / 1e9:.1f} GB free incl. margin, have {free / 1e9:.1f} GB "
                f"(free disk, then rerun the same command to resume)"
            )
            return "skipped_disk", info
        dest = mp.parent if entry.get("backend") != "mlx" else mp
        log(
            f"DOWNLOAD {entry['id']} {entry.get('repo')}@{(entry.get('revision') or 'main')[:10]} "
            f"({(exp or 0) / 1e9:.1f} GB{', resuming' if have else ''})"
        )
        t0 = time.monotonic()
        try:
            download(entry, dest)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            info["error"] = str(exc)[:300]
            return "download_failed", info
        info["download_seconds"] = round(time.monotonic() - t0)
        info["bytes"] = path_bytes(mp)
        if (
            entry.get("sha256")
            and mp.is_file()
            and not getattr(self.args, "no_verify", False)
        ):
            log(f"sha256 {mp.name} ...")
            got = sha256_file(mp)
            info["sha256_ok"] = got == entry["sha256"]
            if not info["sha256_ok"]:
                info["error"] = f"sha256 mismatch: {got}"
                return "download_failed", info
        return "downloaded", info

    def cleanup_model(
        self, entry: dict, state: str, next_entries: list[dict] | None = None
    ) -> None:
        mp = self.model_path(entry)
        mode = self.cleanup_mode
        if mode == "never":
            return
        if state != "downloaded" and not getattr(
            self.args, "delete_preexisting", False
        ):
            log(f"keep {mp.name} (it was already on disk before this run)")
            return
        if mode == "auto":
            try:
                free = shutil.disk_usage(mp.parent).free
            except OSError:
                free = 0
            size = path_bytes(mp) if mp.exists() else 0
            needs = [
                self.disk_need(e)
                for e in (next_entries or [])
                if not e.get("skip_status")
            ]
            # Delete only if a later model doesn't fit now AND deleting this one makes it fit.
            helped = [n for n in needs if free < n <= free + size]
            if not helped:
                why = (
                    "disk is not tight"
                    if all(n <= free for n in needs)
                    else "deleting it would not make room for the next model"
                )
                log(f"keep {mp.name} ({why}: {free / 1e9:.1f} GB free)")
                return
            need = helped[0]
            log(
                f"DELETE {mp} ({path_bytes(mp) / 1e9:.1f} GB, downloaded by this run) to make room for the "
                f"next model ({need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free)"
            )
        else:
            log(f"DELETE {mp}")
        if mp.is_dir():
            shutil.rmtree(mp, ignore_errors=True)
        else:
            try:
                mp.unlink()
            except FileNotFoundError:
                pass
        hfc = mp.parent / ".cache" / "huggingface"
        if entry.get("backend") != "mlx" and hfc.exists():
            for f in hfc.rglob(mp.name + "*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    # ---- upload
    def upload(self, rec: dict) -> None:
        if not rec.get("artifact"):
            return
        cmd = (
            self.cli_prefix
            + ["bench", "--resume", rec["artifact"]]
            + list(self.plan.get("upload_args", []))
        )
        if self.strict_anon:
            cmd.append("--strict-anon")
        out = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(r"Submitted:\s*(\S+)", (out.stdout or "") + (out.stderr or ""))
        rec["upload"] = {
            "rc": out.returncode,
            "url": m.group(1) if m else None,
            "tail": ((out.stdout or "") + (out.stderr or "")).strip()[-300:],
        }

    # ---- resume support
    def previous_attempts(self) -> dict[tuple[str, int], dict]:
        """Last recorded attempt per (entry, ctx) on this machine."""
        out: dict[tuple[str, int], dict] = {}
        for a in self.load_results().get("attempts", []):
            if a.get("machine") == self.machine and a.get("ctx") is not None:
                out[(a["entry_id"], int(a["ctx"]))] = a
        return out

    # ---- main loop
    def run(self) -> int:
        entries = self.entries()
        live = [e for e in entries if not e.get("skip_status")]
        need_llama = any(e.get("backend", "llama.cpp") == "llama.cpp" for e in live)
        need_mlx = any(e.get("backend") == "mlx" for e in live)
        problems = self.preflight(need_llama, need_mlx)
        if problems:
            for p in problems:
                log(f"PREFLIGHT FAIL: {p}")
            return 2
        self.launch_dir.mkdir(parents=True, exist_ok=True)
        lock = (self.launch_dir / ".fit_sweep.lock").open("a")
        if fcntl is not None:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                log("another sweep is running in this folder")
                return 2
        stop_file = self.launch_dir / "STOP"
        log(
            f"machine={self.machine} facts={json.dumps({k: v for k, v in self.facts.items() if v is not None})}"
        )
        if self.llama:
            log(
                f"llama.cpp build={self.llama.get('build_number')} commit={self.llama.get('commit')} flags={self.llama.get('flags')}"
            )
        ram = self.facts.get("ram_gb") or 0
        vram_mib = self.facts.get("gpu_memory_total_mib") or 0
        runs_root = self.launch_dir / "runs" / self.machine
        counts: dict[str, int] = {}
        prev = self.previous_attempts()
        total_cells = sum(len(self.contexts(e)) for e in entries)
        done_cells = 0
        durations: list[float] = []
        self.interrupted = False

        def record(rec):
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            self.append(rec)
            line = f"  -> {rec['status']}"
            for r in rec.get("cli_results") or []:
                line += f" | {r['workload']}: {r.get('decode_tps') or '-'} tok/s"
                if r.get("time_to_first_answer_ms") is not None:
                    line += (
                        f" (answer after {r['time_to_first_answer_ms'] / 1000:.1f}s)"
                    )
            mem = rec.get("memory") or {}
            if mem.get("peak_mib") is not None:
                line += f" | peak {mem['peak_mib']} MiB (baseline {mem.get('baseline_mib')})"
            log(line)

        def eta() -> str:
            left = total_cells - done_cells
            if not durations or left <= 0:
                return ""
            sec = left * (sum(durations) / len(durations))
            return f", ETA <= {sec / 60:.0f} min for {left} remaining step(s)"

        try:
            for idx, entry in enumerate(entries):
                ctxs = self.contexts(entry)
                if stop_file.exists():
                    log("STOP file present; ending sweep")
                    break
                # Resume: cells already measured are kept; a recorded failure stops the climb.
                todo, failed = [], None
                for c in ctxs:
                    p = prev.get((entry["id"], c["ctx"]))
                    if p and p.get("status") in ("ok", "partial"):
                        done_cells += 1
                        log(
                            f"done earlier: {entry['id']} ctx={c['ctx']} ({p['status']}); skipping"
                        )
                    elif (
                        p
                        and not str(p.get("status", "")).startswith("skipped")
                        and p.get("status")
                        not in ("missing", "download_failed", "interrupted")
                        and not entry.get("control")
                    ):
                        failed = failed or f"ctx {c['ctx']} {p['status']} (earlier run)"
                        done_cells += 1
                    elif failed and not entry.get("control"):
                        done_cells += 1
                    else:
                        todo.append(c)
                if not todo:
                    continue
                if entry.get(
                    "skip_status"
                ):  # decided up front (e.g. the preset's memory-fit check)
                    record(
                        self.base_record(
                            entry,
                            todo[0],
                            entry["skip_status"],
                            note=entry.get("skip_note"),
                        )
                    )
                    done_cells += len(todo)
                    continue
                if entry.get("min_ram_gb") and ram and ram < entry["min_ram_gb"]:
                    record(
                        self.base_record(
                            entry,
                            None,
                            "skipped_ram",
                            note=f"needs >= {entry['min_ram_gb']} GB RAM",
                        )
                    )
                    done_cells += len(todo)
                    continue
                exp = expected_bytes_offline(entry)
                if (
                    self.facts.get("kind") == "nvidia"
                    and entry.get("backend", "llama.cpp") == "llama.cpp"
                    and vram_mib
                    and exp
                    and exp > vram_mib * 2**20
                    and entry.get("n_cpu_moe") is None
                ):
                    record(
                        self.base_record(
                            entry,
                            None,
                            "skipped_vram",
                            note=f"file {exp} bytes > {vram_mib:.0f} MiB VRAM (no CPU offload configured)",
                        )
                    )
                    done_cells += len(todo)
                    continue
                state, minfo = self.ensure_model(entry)
                if state in ("missing", "skipped_disk", "download_failed"):
                    record(self.base_record(entry, None, state, model_file=minfo))
                    done_cells += len(todo)
                    continue
                try:
                    for c in todo:
                        if stop_file.exists():
                            break
                        if c.get("min_ram_gb") and ram and ram < c["min_ram_gb"]:
                            record(
                                self.base_record(
                                    entry,
                                    c,
                                    "skipped_ram",
                                    note=f"needs >= {c['min_ram_gb']} GB RAM",
                                )
                            )
                            done_cells += 1
                            continue
                        if failed and not entry.get("control"):
                            record(
                                self.base_record(
                                    entry,
                                    c,
                                    "skipped_after_failure",
                                    note=f"smaller context failed: {failed}",
                                )
                            )
                            done_cells += 1
                            continue
                        # Never abort or kill anything for a busy GPU: record what was in use.
                        busy = self.gpu_busy()
                        adir = (
                            runs_root
                            / entry["id"]
                            / f"ctx{c['ctx']}-{dt.datetime.now(dt.timezone.utc).strftime('%H%M%S')}"
                        )
                        adir.mkdir(parents=True, exist_ok=True)
                        log(
                            f"[{done_cells + 1}/{total_cells}] {entry['id']} ctx={c['ctx']} "
                            f"workloads={self.workloads_for(entry, c)}{eta()}"
                        )
                        t_att = time.monotonic()
                        if entry.get("backend") == "mlx":
                            rec = self.attempt_mlx(entry, c, adir)
                        else:
                            rec = self.attempt_llama(entry, c, adir)
                        durations.append(time.monotonic() - t_att)
                        done_cells += 1
                        rec["model_file"] = minfo
                        if busy is not None:
                            rec["gpu_memory_in_use_at_start_mib"] = round(busy)
                            thr = float(self.plan.get("busy_gpu_threshold_mib", 2048))
                            if busy > thr:
                                rec.setdefault("flags", []).append(
                                    f"gpu-not-idle-at-start:{busy:.0f}MiB"
                                )
                        if self.on_attempt is not None:
                            self.on_attempt(rec, adir)
                        if (
                            getattr(self.args, "upload", False)
                            and rec["status"] == "ok"
                        ):
                            self.upload(rec)
                        record(rec)
                        if rec["status"] not in ("ok", "partial"):
                            failed = f"ctx {c['ctx']} {rec['status']}"
                finally:
                    if (
                        self.cleanup_mode == "after_entry"
                        or self.cleanup_mode == "auto"
                    ):
                        self.cleanup_model(entry, state, entries[idx + 1 :])
        except KeyboardInterrupt:
            self.interrupted = True
            log(
                "interrupted: servers we started were stopped; completed steps are saved in "
                f"{self.results_path}. Rerun the same command to resume."
            )
            return 130
        log(f"done: {counts}; results -> {self.results_path}")
        return 0

    # ---- dry run
    def print_plan(self) -> int:
        entries = self.entries()
        self.facts = machine_facts(self.gpu_index)
        if self.machine == "auto":
            self.machine = machine_slug(self.facts)
        ram = self.facts.get("ram_gb") or 0
        print(
            f"plan: {self.plan_path}\nlaunch_dir: {self.launch_dir}\nresults: {self.results_path}\n"
            f"machine: {self.machine} (ram {ram} GB, kind {self.facts.get('kind')})\nmodels_dir: {self.models_dir}\n"
        )
        total = 0
        for e in entries:
            mp = self.model_path(e)
            exp = e.get("bytes")
            total = max(total, exp or 0)
            tag = (
                " [control]"
                if e.get("control")
                else (" [optional]" if e.get("optional") else "")
            )
            print(
                f"== {e['id']}{tag}  {e.get('backend', 'llama.cpp')}  {e.get('quant')}  "
                f"{(exp or 0) / 1e9:.1f} GB  present={mp.exists()}"
            )
            if e.get("min_ram_gb") and ram and ram < e["min_ram_gb"]:
                print(f"   SKIP: needs {e['min_ram_gb']} GB RAM")
                continue
            print(
                f"   download: {' '.join(download_cmd(e, mp.parent if e.get('backend') != 'mlx' else mp))}"
            )
            for c in self.contexts(e):
                skip = c.get("min_ram_gb") and ram and ram < c["min_ram_gb"]
                wl = self.workloads_for(e, c)
                art = Path("<attempt>/llm-speed-run.json")
                if e.get("backend") == "mlx":
                    print(
                        f"   ctx {c['ctx']:>6}{'  SKIP(ram)' if skip else ''}: {' '.join(self.bench_cmd(e, wl, art))}"
                    )
                else:
                    print(
                        f"   ctx {c['ctx']:>6}{'  SKIP(ram)' if skip else ''}: {' '.join(self.server_cmd(e, c, 0))}\n"
                        f"              then LLAMA_CPP_SERVER_URL=http://127.0.0.1:<port> {' '.join(self.bench_cmd(e, wl, art))}"
                    )
        if total:
            print(
                f"\nlargest single model: {total / 1e9:.1f} GB (+{self.plan.get('disk_margin_gb', 5)} GB margin) "
                f"needed free with --one-at-a-time"
            )
        return 0


# Tokens of context each workload needs (prompt + output budget + chat-template margin).
# chat-long is ~4,096 prompt + 1,024 output, so it never runs in a 4K step.
WORKLOAD_CTX_NEED = {
    "chat-short": 512,
    "chat-long": 5376,
    "agent-trace": 8192,
    "concurrent-decode": 1024,
}


def workload_ctx_need(name: str, c: dict | None = None) -> int:
    if name == "long-context-decay":
        raw = ((c or {}).get("env") or {}).get("LLM_SPEED_DECAY_CTX") or "128000"
        try:
            return max(int(x) for x in str(raw).split(",") if x.strip()) + 512
        except ValueError:
            return 128000 + 512
    return WORKLOAD_CTX_NEED.get(name, 0)


def actual_ctx(rec: dict) -> int | None:
    """Context the attempt really ran with: llama-server's n_ctx from /props; for MLX
    (no preallocated context) the largest prompt + output actually processed."""
    n = (rec.get("server_props") or {}).get("n_ctx")
    if isinstance(n, int) and n > 0:
        return n
    if rec.get("backend") != "mlx":
        return None
    best = 0
    for r in rec.get("cli_results") or []:
        if r.get("error"):
            continue
        best = max(
            best, int(r.get("prompt_tokens") or 0) + int(r.get("output_tokens") or 0)
        )
        for pt in r.get("context_curve") or []:
            if (
                not pt.get("error")
                and not pt.get("skipped")
                and isinstance(pt.get("ctx"), int)
            ):
                best = max(best, pt["ctx"])
    return best or None


def summarize_result(r: dict) -> dict:
    m = r.get("model") or {}
    ex = r.get("extras") or {}
    out = {
        k: r.get(k)
        for k in (
            "workload",
            "backend_version",
            "decode_tps",
            "prefill_tps",
            "ttft_ms",
            "decode_p50_latency_ms",
            "prompt_tokens",
            "output_tokens",
            "context_tokens",
            "wall_ms",
            "error",
            "flags",
        )
    }
    for k in (
        "decode_tps",
        "prefill_tps",
        "ttft_ms",
        "decode_p50_latency_ms",
        "wall_ms",
    ):
        if isinstance(out.get(k), float):
            out[k] = round(out[k], 2)
    out["model"] = {k: m.get(k) for k in ("identifier", "name", "quant", "digest")}
    if ex.get("context_curve"):
        out["context_curve"] = [
            {
                k: (round(p[k], 2) if isinstance(p.get(k), float) else p.get(k))
                for k in (
                    "ctx",
                    "decode_tps",
                    "prefill_tps",
                    "ttft_ms",
                    "skipped",
                    "error",
                )
            }
            for p in ex["context_curve"]
        ]
    if ex.get("tps_source"):
        out["tps_source"] = ex["tps_source"]
    for k in (
        "time_to_first_answer_ms",
        "reasoning_tokens_est",
        "answer_started",
        "peak_vram_mib",
        "peak_vram_source",
        "failure_kind",
        "llama_cpp_build",
        "llama_cpp_commit",
        "llama_server_args",
        "server_n_ctx",
        "model_file",
        "model_file_bytes",
    ):
        if ex.get(k) is not None:
            out[k] = ex[k]
    return out


def cmd_merge(args) -> int:
    merged = {"schema": SCHEMA, "launch": None, "machines": {}, "attempts": []}
    seen = set()
    for p in args.inputs:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        merged["launch"] = merged["launch"] or d.get("launch")
        merged["machines"].update(d.get("machines", {}))
        for a in d.get("attempts", []):
            if a["attempt_id"] not in seen:
                seen.add(a["attempt_id"])
                merged["attempts"].append(a)
    merged["attempts"].sort(key=lambda a: a.get("started_at") or "")
    merged["updated_at"] = now_utc()
    atomic_write_json(Path(args.output), merged)
    print(f"merged {len(merged['attempts'])} attempts -> {args.output}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        p = sub.add_parser(name)
        p.add_argument("plan")
        p.add_argument("--only", default="", help="comma-separated entry ids")
        p.add_argument("--include-optional", action="store_true")
        p.add_argument(
            "--machine", default=None, help="override machine slug (default: auto)"
        )
        p.add_argument(
            "--results",
            default=None,
            help="results file (default <launch_dir>/results.json)",
        )
        if name == "run":
            p.add_argument(
                "--download",
                action="store_true",
                help="download missing model files (hf CLI or stdlib)",
            )
            p.add_argument(
                "--one-at-a-time",
                action="store_true",
                help="delete each model the sweep downloaded after its attempts (disk-tight machines)",
            )
            p.add_argument(
                "--delete-preexisting",
                action="store_true",
                help="with --one-at-a-time, also delete model files that were already on disk",
            )
            p.add_argument(
                "--no-verify",
                action="store_true",
                help="skip sha256 check of downloaded GGUFs",
            )
            p.add_argument(
                "--upload",
                action="store_true",
                help="upload each ok attempt's signed artifact (llm-speed bench --resume). Off by default.",
            )
            p.add_argument(
                "--force",
                action="store_true",
                help="skip the min llama.cpp build and Apple-only-MLX preflight checks (testing)",
            )
            p.add_argument(
                "--allow-busy-gpu",
                action="store_true",
                help="run even if the GPU already holds memory (attempts get flagged)",
            )
    m = sub.add_parser("merge")
    m.add_argument("inputs", nargs="+")
    m.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "merge":
        return cmd_merge(args)
    sw = Sweep(Path(args.plan), args)
    if args.cmd == "plan":
        return sw.print_plan()

    def _sig(signum, frame):
        raise SystemExit(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, _sig)
    return sw.run()


if __name__ == "__main__":
    sys.exit(main())
