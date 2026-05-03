"""Shared pytest fixtures for the llm-speed test suite.

Provides synthetic `Fingerprint`, `WorkloadResult`, `ModelRef`, and `RunReport`
objects so individual tests can build off a known-good baseline without
duplicating long constructor calls.

The signing fixtures isolate the on-disk keypair under a tmp_path-rooted
`LLM_SPEED_CONFIG`, so tests never touch the user's real `~/.config/llm-speed`.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from cli.types import (
    Fingerprint,
    GpuInfo,
    ModelRef,
    RunReport,
    WorkloadResult,
)


@pytest.fixture
def sample_gpu() -> GpuInfo:
    return GpuInfo(
        name="RTX 4090",
        kind="nvidia",
        memory_gb=24.0,
        driver_version="580.65",
        pci_bus_id="0000:01:00.0",
        extras={"raw_name": "NVIDIA GeForce RTX 4090"},
    )


@pytest.fixture
def sample_fingerprint(sample_gpu: GpuInfo) -> Fingerprint:
    return Fingerprint(
        os_name="Linux",
        os_version="6.8.0",
        cpu_model="AMD Ryzen 9 7950X3D",
        cpu_cores=16,
        ram_gb=64.0,
        gpus=[sample_gpu],
        accelerator_summary="RTX 4090 (24GB) + AMD Ryzen 9 7950X3D (16c) + 64GB",
        fingerprint_hash="abcdef0123456789",
        extras={"backends": {"llama.cpp": "b3500", "cuda": "13.0"}},
    )


@pytest.fixture
def sample_model() -> ModelRef:
    return ModelRef(
        backend="llama.cpp",
        identifier="/models/qwen3-coder-next.gguf",
        name="Qwen3-Coder-Next",
        size="80B-A3B",
        quant="Q4_K_M",
        digest="sha256:deadbeef",
    )


@pytest.fixture
def sample_workload_result(sample_model: ModelRef) -> WorkloadResult:
    return WorkloadResult(
        workload="chat-short",
        suite_version="suite-v1",
        backend="llama.cpp",
        backend_version="b3500",
        model=sample_model,
        ttft_ms=142.3,
        prefill_tps=8421.1,
        decode_tps=187.4,
        decode_p50_latency_ms=5.3,
        decode_p95_latency_ms=7.1,
        prompt_tokens=128,
        output_tokens=256,
        batch_size=1,
        wall_ms=1530.0,
        prefix_cache_hit_rate=0.0,
        raw_timings_ms=[5.1, 5.2, 5.3, 5.4, 5.5],
    )


@pytest.fixture
def sample_run_report(
    sample_fingerprint: Fingerprint,
    sample_workload_result: WorkloadResult,
) -> RunReport:
    return RunReport(
        suite_version="suite-v1",
        cli_version="0.0.1-dev",
        fingerprint=sample_fingerprint,
        results=[sample_workload_result],
        started_at="2026-04-25T12:00:00Z",
        finished_at="2026-04-25T12:01:30Z",
    )


@pytest.fixture
def isolated_signing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Redirect `cli.config.CONFIG_DIR` and `cli.signing.KEY_PATH` to a tmp dir.

    Because `cli.signing` reads CONFIG_DIR at import time, we set the env var
    AND reload the modules so the new path takes effect. This keeps the user's
    real `~/.config/llm-speed/keys/ed25519.key` out of the test loop.
    """
    cfg_dir = tmp_path / "llm-speed-config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LLM_SPEED_CONFIG", str(cfg_dir))
    # Reload config + signing so the new env var is picked up.
    import cli.config as _config
    import cli.signing as _signing

    importlib.reload(_config)
    importlib.reload(_signing)
    yield cfg_dir
    # Reload back to the user's normal config so other tests see the default.
    monkeypatch.delenv("LLM_SPEED_CONFIG", raising=False)
    importlib.reload(_config)
    importlib.reload(_signing)
