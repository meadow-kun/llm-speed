"""Shared types contract.

This is THE source of truth for everything that crosses module boundaries in
the CLI. Drivers, workloads, fingerprint, upload — all agree on these shapes.
Do not redefine these; extend via the typed `extras` dicts where needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Hardware / environment fingerprint
# ---------------------------------------------------------------------------


@dataclass
class GpuInfo:
    name: str  # 'RTX 4090', 'M3 Max', 'MI300X', ...
    kind: str  # 'nvidia' | 'amd' | 'apple' | 'intel' | 'cpu'
    memory_gb: float | None = None
    driver_version: str | None = None
    pci_bus_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fingerprint:
    """Captured at the start of every benchmark run. Hashed for the run signature."""

    os_name: str  # 'Darwin' | 'Linux' | 'Windows'
    os_version: str
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    gpus: list[GpuInfo] = field(default_factory=list)
    accelerator_summary: str = (
        ""  # human-readable, e.g. '1x RTX 4090 + Ryzen 9 7950X3D + 64GB'
    )
    fingerprint_hash: str = ""  # SHA-256 over deterministic key fields
    extras: dict[str, Any] = field(
        default_factory=dict
    )  # backend versions, governor, thermal, etc.


# ---------------------------------------------------------------------------
# Models and backends
# ---------------------------------------------------------------------------


@dataclass
class ModelRef:
    """A backend-specific reference to a model. Each driver knows how to load it."""

    backend: str  # 'llama.cpp' | 'ollama' | 'vllm' | 'mlx' | 'exllamav2' | 'hosted-api'
    identifier: str  # path / hf id / ollama tag / openrouter slug
    name: str  # normalized human name, e.g. 'Qwen3-Coder-Next'
    size: str | None = None  # '8B' | '70B' | '80B-A3B'
    quant: str | None = None  # 'Q4_K_M' | 'IQ3_XXS' | 'FP16' | 'FP8'
    digest: str | None = (
        None  # sha256-prefix of weights file (mandatory for canonical results)
    )
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendDetection:
    """Result of probing for a backend on this machine."""

    available: bool
    name: str
    version: str | None = None
    build_flags: list[str] = field(
        default_factory=list
    )  # e.g. ['CUDA', 'Metal', 'AVX2']
    runtime_versions: dict[str, str] = field(
        default_factory=dict
    )  # {'cuda': '13.0', 'driver': '580.x'}
    notes: str = ""


# ---------------------------------------------------------------------------
# Workload params and results
# ---------------------------------------------------------------------------


@dataclass
class BenchParams:
    """Per-workload configuration, plus universal knobs."""

    workload: str
    suite_version: str
    prompt_tokens: int = 0  # workload-defined
    output_tokens: int = 0  # workload-defined
    batch_size: int = 1
    context_length: int = 0  # for long-context workload
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadResult:
    """Output of one workload run on one backend × model. Uploaded as-is."""

    workload: str
    suite_version: str
    backend: str
    backend_version: str | None
    model: ModelRef

    ttft_ms: float | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    decode_p50_latency_ms: float | None = None
    decode_p95_latency_ms: float | None = None

    prompt_tokens: int = 0
    output_tokens: int = 0
    batch_size: int = 1
    context_tokens: int = 0
    wall_ms: float = 0.0

    prefix_cache_hit_rate: float | None = None
    raw_timings_ms: list[float] = field(
        default_factory=list
    )  # per-token decode times; uploaded only on request

    error: str | None = None
    flags: list[str] = field(
        default_factory=list
    )  # 'thermal-throttle', 'on-battery', etc.
    extras: dict[str, Any] = field(default_factory=dict)  # backend-specific telemetry


@dataclass
class RunReport:
    """Full per-run artifact. One Fingerprint + N WorkloadResults."""

    suite_version: str
    cli_version: str
    fingerprint: Fingerprint
    results: list[WorkloadResult] = field(default_factory=list)
    started_at: str = ""  # ISO 8601 UTC
    finished_at: str = ""


# ---------------------------------------------------------------------------
# Protocols (drivers + workloads)
# ---------------------------------------------------------------------------


@runtime_checkable
class BackendDriver(Protocol):
    """Each backend (llama.cpp, ollama, vllm, mlx, exllamav2, hosted-api) implements this."""

    name: str

    def detect(self) -> BackendDetection: ...
    def list_models(self) -> list[ModelRef]: ...
    def run_chat(
        self,
        model: ModelRef,
        prompt: str,
        *,
        max_output_tokens: int,
        stream: bool = True,
        extras: dict[str, Any] | None = None,
    ) -> ChatRunOutcome: ...


@dataclass
class ChatRunOutcome:
    """Low-level result of a single prompt → completion call. Workloads compose these."""

    success: bool
    output_text: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float | None
    decode_token_times_ms: list[float]  # per-token wall-clock deltas during decode
    wall_ms: float
    backend_extras: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class Workload(Protocol):
    """A standardized benchmark scenario. The same workload runs across all backends."""

    name: str
    suite_version: str

    def run(self, driver: BackendDriver, model: ModelRef) -> WorkloadResult: ...
