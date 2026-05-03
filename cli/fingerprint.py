"""Hardware + environment fingerprint for the current machine.

Captured at the start of every benchmark run; the deterministic key fields
are hashed to produce a stable per-machine identity used for run signatures
and outlier clustering. Best-effort everywhere: missing CLIs, unsupported
OSes, and odd parse failures degrade to None / empty rather than raising.

Probe order (cheap → expensive, common → rare):
  1. CPU / RAM / OS via stdlib + psutil.
  2. GPUs via vendor CLIs in priority order: nvidia-smi → rocm-smi →
     system_profiler (Apple) → lspci (Intel discrete on Linux).
  3. Power / thermal state.
  4. Backend versions (llama.cpp, ollama, CUDA, ROCm, Metal).
  5. Hash + accelerator_summary derived from the above.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import shutil
import subprocess
from collections import Counter
from typing import Any

import psutil

from .types import Fingerprint, GpuInfo

log = logging.getLogger("cli.fingerprint")


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout: float = 5.0) -> str | None:
    """Run `cmd`, return stdout on success, None on any failure (no raises)."""
    if not cmd or shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.debug("command failed: %s (%s)", " ".join(cmd), exc)
        return None
    if proc.returncode != 0:
        log.debug("command exit %d: %s", proc.returncode, " ".join(cmd))
        return None
    return proc.stdout or ""


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

# Canonicalize raw `nvidia-smi name` strings to short labels matching seed/extractors.
_NVIDIA_SHORT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"RTX\s*(\d{4})\s*Ti\s*Super", re.I), "RTX {0} Ti Super"),
    (re.compile(r"RTX\s*(\d{4})\s*Super", re.I), "RTX {0} Super"),
    (re.compile(r"RTX\s*(\d{4})\s*Ti", re.I), "RTX {0} Ti"),
    (re.compile(r"RTX\s*(\d{4})\b", re.I), "RTX {0}"),
    (re.compile(r"RTX\s*A(\d{4})\b", re.I), "RTX A{0}"),
    (re.compile(r"RTX\s*(\d{4})\s*Ada", re.I), "RTX {0} Ada"),
    (re.compile(r"\b(A100|H100|H200|B200|L40S?|L4|T4|V100)\b"), "{0}"),
]


def _canonical_nvidia_name(raw: str) -> str:
    s = raw.strip()
    # Strip the "NVIDIA " or "NVIDIA GeForce " prefixes most consumer cards carry.
    s = re.sub(r"^NVIDIA\s+(?:GeForce\s+)?", "", s, flags=re.I)
    for pat, tpl in _NVIDIA_SHORT_PATTERNS:
        m = pat.search(s)
        if m:
            return tpl.format(*m.groups())
    return s


def _detect_nvidia() -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=...`. Returns [] if nvidia-smi missing or fails."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,pci.bus_id,vbios_version,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: list[GpuInfo] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, driver, mem_mib, bus_id = parts[0], parts[1], parts[2], parts[3]
        vbios = parts[4] if len(parts) > 4 else ""
        power_limit = parts[5] if len(parts) > 5 else ""

        memory_gb: float | None = None
        try:
            memory_gb = round(float(mem_mib) / 1024.0, 1)
        except ValueError:
            pass

        extras: dict[str, Any] = {"raw_name": name}
        if vbios and vbios.lower() not in {"n/a", ""}:
            extras["vbios_version"] = vbios
        if power_limit and power_limit.lower() not in {"n/a", ""}:
            try:
                extras["power_limit_w"] = float(power_limit)
            except ValueError:
                pass

        gpus.append(
            GpuInfo(
                name=_canonical_nvidia_name(name),
                kind="nvidia",
                memory_gb=memory_gb,
                driver_version=driver or None,
                pci_bus_id=bus_id or None,
                extras=extras,
            )
        )

    # Throttle reasons (best-effort): annotate each GPU.
    throttle_out = _run(
        [
            "nvidia-smi",
            "--query-gpu=clocks_throttle_reasons.active",
            "--format=csv,noheader",
        ]
    )
    if throttle_out:
        reasons = [r.strip() for r in throttle_out.splitlines() if r.strip()]
        for gpu, reason in zip(gpus, reasons, strict=False):
            gpu.extras["throttle_reasons_active"] = reason

    return gpus


def _detect_amd() -> list[GpuInfo]:
    """Parse `rocm-smi` CSV. Schema is annoyingly stable-ish; tolerate column shuffles."""
    out = _run(
        [
            "rocm-smi",
            "--showproductname",
            "--showdriverversion",
            "--showmeminfo",
            "vram",
            "--csv",
        ]
    )
    if not out:
        return []
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    gpus: list[GpuInfo] = []
    for row in lines[1:]:
        cells = [c.strip() for c in row.split(",")]
        if len(cells) != len(header):
            continue
        rec = dict(zip(header, cells, strict=False))
        # rocm-smi emits one "card0", "card1" row per GPU and one summary "system" row;
        # skip the latter.
        device = rec.get("device") or rec.get("gpu") or ""
        if device.lower().startswith("system"):
            continue
        name = (
            rec.get("card series")
            or rec.get("card model")
            or rec.get("product name")
            or rec.get("gpu series")
            or "AMD GPU"
        )
        driver = rec.get("driver version") or None
        memory_gb: float | None = None
        for key in ("vram total memory (b)", "vram total (b)", "vram total"):
            if key in rec:
                try:
                    memory_gb = round(float(rec[key]) / (1024**3), 1)
                    break
                except ValueError:
                    continue
        gpus.append(
            GpuInfo(
                name=_canonical_amd_name(name),
                kind="amd",
                memory_gb=memory_gb,
                driver_version=driver,
                pci_bus_id=rec.get("pci bus") or None,
                extras={"raw_name": name, "device": device},
            )
        )
    return gpus


def _canonical_amd_name(raw: str) -> str:
    s = raw.strip()
    for token in ("MI300X", "MI325X", "MI355X", "MI250X", "MI210"):
        if token.lower() in s.lower():
            return token
    m = re.search(r"(7900\s*XTX|7900\s*XT|9070\s*XT|6900\s*XT|6800\s*XT)", s, re.I)
    if m:
        return f"Radeon RX {m.group(1).upper().replace('  ', ' ')}"
    return s


# Apple Silicon SoC recognition. Order matters — match longer suffixes first.
_APPLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bM(\d)\s*Ultra\b", re.I), "M{0} Ultra"),
    (re.compile(r"\bM(\d)\s*Max\b", re.I), "M{0} Max"),
    (re.compile(r"\bM(\d)\s*Pro\b", re.I), "M{0} Pro"),
    (re.compile(r"\bM(\d)\b(?!\d)", re.I), "M{0}"),
]


def _canonical_apple_name(raw: str) -> str | None:
    for pat, tpl in _APPLE_PATTERNS:
        m = pat.search(raw)
        if m:
            return tpl.format(*m.groups())
    return None


def _detect_apple() -> tuple[list[GpuInfo], str | None]:
    """Return (gpus, metal_family). Apple Silicon only — no-op on Intel Macs without GPUs."""
    if platform.system() != "Darwin":
        return [], None

    metal_family: str | None = None
    gpus: list[GpuInfo] = []

    out = _run(["system_profiler", "SPDisplaysDataType", "-json"])
    if out:
        try:
            data = json.loads(out)
            for entry in data.get("SPDisplaysDataType", []):
                chipset = entry.get("sppci_model") or entry.get("_name") or ""
                short = _canonical_apple_name(chipset)
                if not short:
                    continue
                cores = entry.get("sppci_cores") or entry.get(
                    "spdisplays_ndrvs"
                )  # may be None
                vendor = entry.get("spdisplays_vendor", "")
                metal = entry.get("spdisplays_metalfamily") or entry.get("metal_family")
                if metal and not metal_family:
                    metal_family = str(metal)

                extras: dict[str, Any] = {"raw_chipset": chipset}
                if cores:
                    try:
                        extras["gpu_cores"] = (
                            int(re.sub(r"[^\d]", "", str(cores)) or 0) or None
                        )
                    except ValueError:
                        pass
                if vendor:
                    extras["vendor"] = vendor

                gpus.append(
                    GpuInfo(
                        name=short,
                        kind="apple",
                        memory_gb=None,  # unified memory; reported via RAM
                        driver_version=None,
                        pci_bus_id=None,
                        extras=extras,
                    )
                )
        except (json.JSONDecodeError, AttributeError, TypeError) as exc:
            log.debug("system_profiler json parse failed: %s", exc)

    if not gpus:
        # Fallback: infer from CPU brand string.
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or ""
        short = _canonical_apple_name(brand)
        if short:
            gpus.append(
                GpuInfo(
                    name=short,
                    kind="apple",
                    extras={"raw_chipset": brand.strip(), "source": "sysctl"},
                )
            )

    return gpus, metal_family


def _detect_intel_linux() -> list[GpuInfo]:
    """Look for Intel Arc / discrete in lspci output. Linux only."""
    if platform.system() != "Linux":
        return []
    out = _run(["lspci", "-mm"])
    if not out:
        return []
    gpus: list[GpuInfo] = []
    for line in out.splitlines():
        if "VGA compatible controller" not in line and "3D controller" not in line:
            continue
        if "Intel" not in line:
            continue
        # crude name extraction
        m = re.search(r'"([^"]*Arc[^"]*)"', line)
        name = m.group(1) if m else "Intel GPU"
        short = name
        m2 = re.search(r"\bArc\s*([AB]\d{3})\b", name, re.I)
        if m2:
            short = f"Arc {m2.group(1).upper()}"
        gpus.append(
            GpuInfo(
                name=short,
                kind="intel",
                extras={"raw_name": name, "source": "lspci"},
            )
        )
    return gpus


def _detect_gpus() -> list[GpuInfo]:
    """Run all GPU probes and concatenate. Order: NVIDIA, AMD, Apple, Intel."""
    gpus: list[GpuInfo] = []
    gpus.extend(_detect_nvidia())
    gpus.extend(_detect_amd())
    apple_gpus, _ = _detect_apple()
    gpus.extend(apple_gpus)
    gpus.extend(_detect_intel_linux())
    return gpus


# ---------------------------------------------------------------------------
# CPU / RAM / OS
# ---------------------------------------------------------------------------


def _detect_cpu_model() -> str:
    sysname = platform.system()
    if sysname == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("model name"):
                        _, _, val = line.partition(":")
                        return val.strip()
        except OSError as exc:
            log.debug("/proc/cpuinfo read failed: %s", exc)
    elif sysname == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    elif sysname == "Windows":
        out = _run(["wmic", "cpu", "get", "name"])
        if out:
            for line in out.splitlines():
                line = line.strip()
                if line and line.lower() != "name":
                    return line
    return platform.processor() or "unknown"


def _detect_cpu_cores() -> int:
    try:
        physical = psutil.cpu_count(logical=False)
        if physical:
            return int(physical)
    except Exception as exc:  # psutil rarely raises but be safe
        log.debug("psutil.cpu_count failed: %s", exc)
    return psutil.cpu_count(logical=True) or 1


def _detect_ram_gb() -> float:
    return round(psutil.virtual_memory().total / (1024**3), 1)


def _detect_os_extras() -> dict[str, Any]:
    extras: dict[str, Any] = {}
    if platform.system() == "Linux":
        try:
            with open("/etc/os-release", encoding="utf-8") as f:
                rel: dict[str, str] = {}
                for line in f:
                    if "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    rel[k.strip()] = v.strip().strip('"')
            distro = rel.get("PRETTY_NAME") or rel.get("NAME")
            if distro:
                extras["distro"] = distro
        except OSError:
            pass
    elif platform.system() == "Darwin":
        # macOS marketing version is more useful than Darwin kernel version.
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            extras["macos_version"] = mac_ver
    return extras


# ---------------------------------------------------------------------------
# Power / thermal state
# ---------------------------------------------------------------------------


def _detect_power_state() -> dict[str, Any]:
    info: dict[str, Any] = {}
    sysname = platform.system()

    if sysname == "Darwin":
        batt = _run(["pmset", "-g", "batt"])
        if batt:
            lower = batt.lower()
            if "ac power" in lower:
                info["power_source"] = "ac"
            elif "battery power" in lower:
                info["power_source"] = "battery"
            else:
                info["power_source"] = "unknown"
        therm = _run(["pmset", "-g", "therm"])
        if therm:
            for line in therm.splitlines():
                line = line.strip()
                if line.startswith("CPU_Scheduler_Limit"):
                    info["cpu_scheduler_limit"] = line.split("=", 1)[-1].strip()
                elif line.startswith("CPU_Speed_Limit"):
                    info["cpu_speed_limit"] = line.split("=", 1)[-1].strip()

    elif sysname == "Linux":
        # /sys/class/power_supply/AC*/online: "1" → on AC.
        try:
            import glob

            ac_files = glob.glob("/sys/class/power_supply/AC*/online")
            for path in ac_files:
                try:
                    with open(path, encoding="utf-8") as f:
                        val = f.read().strip()
                    info["power_source"] = "ac" if val == "1" else "battery"
                    break
                except OSError:
                    continue
        except Exception as exc:
            log.debug("ac power probe failed: %s", exc)

    return info


# ---------------------------------------------------------------------------
# Backend versions
# ---------------------------------------------------------------------------


def _first_line(s: str | None) -> str | None:
    if not s:
        return None
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _detect_backends(
    metal_family: str | None, have_nvidia: bool, have_amd: bool
) -> dict[str, str]:
    versions: dict[str, str] = {}

    for binary in ("llama-cli", "llama-server"):
        if shutil.which(binary):
            v = _first_line(_run([binary, "--version"]))
            if v:
                versions["llama.cpp"] = v
                break

    ollama_v = _first_line(_run(["ollama", "--version"]))
    if ollama_v:
        versions["ollama"] = ollama_v

    if have_nvidia:
        nvcc = _run(["nvcc", "--version"])
        if nvcc:
            # Last line typically: "Build cuda_13.0.r13.0/...". Prefer the "release" line.
            for line in nvcc.splitlines():
                if "release" in line.lower():
                    versions["cuda"] = line.strip()
                    break
            else:
                versions["cuda"] = _first_line(nvcc) or "unknown"
        else:
            # Fallback: parse from `nvidia-smi`.
            smi = _run(["nvidia-smi"])
            if smi:
                m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
                if m:
                    versions["cuda"] = m.group(1)

    if have_amd:
        rocm_v = _first_line(_run(["rocm-smi", "--version"]))
        if rocm_v:
            versions["rocm"] = rocm_v

    if metal_family:
        versions["metal"] = metal_family

    return versions


# ---------------------------------------------------------------------------
# Summary + hash
# ---------------------------------------------------------------------------


def _accelerator_summary(
    gpus: list[GpuInfo], cpu_model: str, cpu_cores: int, ram_gb: float
) -> str:
    """Human-readable headline. Apple unified, NVIDIA/AMD/Intel discrete."""
    parts: list[str] = []

    if gpus and all(g.kind == "apple" for g in gpus):
        # Apple: collapse to "M3 Max (40-core GPU) + 64GB unified".
        primary = gpus[0]
        cores = primary.extras.get("gpu_cores")
        if cores:
            parts.append(f"{primary.name} ({cores}-core GPU)")
        else:
            parts.append(primary.name)
        parts.append(f"{int(round(ram_gb))}GB unified")
        return " + ".join(parts)

    if gpus:
        # Group identical GPUs: "2x RTX 3090".
        counts = Counter(g.name for g in gpus)
        gpu_chunks: list[str] = []
        for name, n in counts.items():
            mem = next(
                (g.memory_gb for g in gpus if g.name == name and g.memory_gb), None
            )
            mem_s = f" ({int(round(mem))}GB)" if mem else ""
            gpu_chunks.append(f"{n}x {name}{mem_s}" if n > 1 else f"{name}{mem_s}")
        parts.append(" + ".join(gpu_chunks))

    if cpu_model:
        parts.append(f"{cpu_model} ({cpu_cores}c)")
    parts.append(f"{int(round(ram_gb))}GB")
    return " + ".join(parts)


def _major_os_version(version: str) -> str:
    """Strip patch components: '26.3.1' -> '26', '13.5.7' -> '13', '6.8.0-31-generic' -> '6'."""
    if not version:
        return ""
    # Take the leading run of digits as the major version. Tolerate kernel-style
    # strings ('6.8.0-31-generic') and marketing strings ('26.3').
    m = re.match(r"^\s*(\d+)", version)
    return m.group(1) if m else ""


def _round_ram_gb(ram_gb: float) -> int:
    """Round to nearest 8 GB bucket. Floor at 8 GB so a probe failure (0.0) stays 0."""
    if ram_gb is None or ram_gb <= 0:
        return 0
    return int((ram_gb + 4) // 8) * 8


def _round_gpu_memory_gb(memory_gb: float | None) -> int | None:
    """Round GPU memory to nearest 8 GB bucket. None stays None (Apple unified)."""
    if memory_gb is None:
        return None
    if memory_gb <= 0:
        return 0
    return int((memory_gb + 4) // 8) * 8


def _trimmed_gpu_dicts(gpus: list[GpuInfo]) -> list[dict[str, Any]]:
    """Privacy-safe per-GPU payload: name, kind, bucketed memory_gb only."""
    return [
        {
            "name": g.name,
            "kind": g.kind,
            "memory_gb": _round_gpu_memory_gb(g.memory_gb),
        }
        for g in gpus
    ]


def _fingerprint_hash(
    gpus: list[GpuInfo],
    cpu_model: str,
    cpu_cores: int,
    ram_gb: float,
    os_name: str,
    os_version: str,
) -> str:
    """SHA-256 (truncated) over the trimmed, class-level fields.

    Hashes the same fields that get uploaded so the hash is deterministic over
    the privacy-safe payload — and so two physically identical machines (same
    SoC, same RAM bucket, same major OS) produce the SAME fingerprint_hash.
    Class-level identity, not user-level.
    """
    trimmed_gpus = sorted(
        _trimmed_gpu_dicts(gpus),
        key=lambda d: (d["name"], d["kind"], d["memory_gb"] or 0),
    )
    payload = {
        "gpus": trimmed_gpus,
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "ram_gb": _round_ram_gb(ram_gb),
        "os_name": os_name,
        "os_version": _major_os_version(os_version),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_fingerprint() -> Fingerprint:
    """Capture full HW + env fingerprint for the current machine.
    Best-effort: missing tools degrade gracefully; never raise."""
    try:
        os_name = platform.system() or "unknown"
        os_version = platform.release() or "unknown"

        cpu_model = _detect_cpu_model()
        cpu_cores = _detect_cpu_cores()
        ram_gb = _detect_ram_gb()

        gpus = _detect_gpus()
        # Re-run apple detector once to harvest metal_family without re-emitting GPUs.
        _, metal_family = _detect_apple()

        have_nvidia = any(g.kind == "nvidia" for g in gpus)
        have_amd = any(g.kind == "amd" for g in gpus)

        extras: dict[str, Any] = {}
        os_extras = _detect_os_extras()
        if os_extras:
            extras.update(os_extras)
        power = _detect_power_state()
        if power:
            extras["power"] = power
        backends = _detect_backends(metal_family, have_nvidia, have_amd)
        if backends:
            extras["backends"] = backends

        summary = _accelerator_summary(gpus, cpu_model, cpu_cores, ram_gb)
        fp_hash = _fingerprint_hash(
            gpus, cpu_model, cpu_cores, ram_gb, os_name, os_version
        )

        return Fingerprint(
            os_name=os_name,
            os_version=os_version,
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpus=gpus,
            accelerator_summary=summary,
            fingerprint_hash=fp_hash,
            extras=extras,
        )
    except Exception as exc:
        # Last-ditch: never raise out of fingerprint; emit a stub so callers proceed.
        log.exception("fingerprint detection failed: %s", exc)
        return Fingerprint(
            os_name=platform.system() or "unknown",
            os_version=platform.release() or "unknown",
            cpu_model="unknown",
            cpu_cores=1,
            ram_gb=0.0,
            extras={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Upload serialization (privacy-trimmed)
# ---------------------------------------------------------------------------


def to_uploadable_dict(fp: Fingerprint, *, strict_anon: bool = False) -> dict[str, Any]:
    """Serialize a Fingerprint with only the privacy-safe subset of fields.

    Default mode includes:
      - os_name, os_version (major only), cpu_model, cpu_cores
      - ram_gb bucketed to nearest 8 GB
      - per-GPU: name, kind, memory_gb (bucketed). NO driver_version, NO pci_bus_id.
      - accelerator_summary
      - fingerprint_hash (class-level over the same trimmed fields)
      - extras: only `backends`. NO power, macos_version, distro, thermal.

    In strict_anon mode, fingerprint_hash is omitted entirely — no class-level ID either.
    """
    backends = (fp.extras or {}).get("backends") or {}
    extras_out: dict[str, Any] = {}
    if backends:
        extras_out["backends"] = dict(backends)

    payload: dict[str, Any] = {
        "os_name": fp.os_name,
        "os_version": _major_os_version(fp.os_version),
        "cpu_model": fp.cpu_model,
        "cpu_cores": fp.cpu_cores,
        "ram_gb": _round_ram_gb(fp.ram_gb),
        "gpus": _trimmed_gpu_dicts(fp.gpus),
        "accelerator_summary": fp.accelerator_summary,
        "extras": extras_out,
    }
    if not strict_anon:
        payload["fingerprint_hash"] = fp.fingerprint_hash
    return payload


# ---------------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    fp = detect_fingerprint()
    print(json.dumps(to_uploadable_dict(fp), indent=2, sort_keys=True, default=str))
