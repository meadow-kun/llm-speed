"""W4 — concurrent-decode.

Batch sizes [1, 4, 8, 16]. Each request: 1k-token input -> 256-token output.

Critical for hosted-API comparison and self-hosted multi-user setups: shows
how aggregate throughput scales when many decode streams share GPU resources.

Concurrency is implemented via `concurrent.futures.ThreadPoolExecutor`. For
HTTP-backed drivers (ollama / hosted-api / vllm-http) this parallelizes the
I/O wait and exercises real server-side batching. For pure in-process
backends (mlx, exllamav2 embedded, vllm-embedded) the GIL or single-process
model means threads serialize through the engine — in that case the work is
effectively sequential and we tag the result with the `no-real-concurrency`
flag so consumers can interpret aggregate_tps correctly.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .. import SUITE_VERSION
from ..registry import register_workload
from ..timing import now_ms, percentile, summarize_decode
from ..types import ChatRunOutcome, ModelRef, WorkloadResult
from .prompts.fixtures import build_prompt_of_chars

LOG = logging.getLogger("cli.workloads.concurrent-decode")

BATCH_SIZES: tuple[int, ...] = (1, 4, 8, 16)
PROMPT_TOKEN_TARGET = 1024
OUTPUT_TOKENS = 256

# Backends where threading does not yield real overlap (single-process,
# blocking compute). Concurrent runs against these are reported but flagged.
_NON_CONCURRENT_BACKENDS = {
    "mlx",
    "exllamav2",
    "vllm-embedded",
    "llama-cpp-embedded",
}

PROMPT = build_prompt_of_chars(PROMPT_TOKEN_TARGET * 4) + (
    "\n\n---\n\nSummarize the passages above in eight to twelve sentences. Be specific."
)


def _run_one(driver, model: ModelRef) -> tuple[bool, ChatRunOutcome, float, float]:
    """Run a single request; return (success, outcome, started_ms, finished_ms)."""
    started = now_ms()
    try:
        outcome = driver.run_chat(
            model,
            PROMPT,
            max_output_tokens=OUTPUT_TOKENS,
            stream=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.exception("concurrent-decode: stream raised")
        finished = now_ms()
        return (
            False,
            ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=finished - started,
                error=f"driver-exception: {exc!r}",
            ),
            started,
            finished,
        )
    finished = now_ms()
    return (outcome.success, outcome, started, finished)


def _measure_batch(driver, model: ModelRef, batch_size: int) -> dict[str, Any]:
    """Run `batch_size` parallel streams and aggregate metrics."""
    LOG.debug("concurrent-decode batch=%d starting", batch_size)

    completions: list[tuple[bool, ChatRunOutcome, float, float]] = []
    if batch_size == 1:
        completions.append(_run_one(driver, model))
    else:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futs = [pool.submit(_run_one, driver, model) for _ in range(batch_size)]
            for f in as_completed(futs):
                completions.append(f.result())

    successes = [(o, s, fi) for ok, o, s, fi in completions if ok]
    failures = [c for c in completions if not c[0]]

    if not successes:
        return {
            "batch": batch_size,
            "aggregate_tps": None,
            "per_stream_tps": None,
            "p50_ms": None,
            "p95_ms": None,
            "skipped": True,
            "error": "all-streams-failed",
            "n_failures": len(failures),
        }

    earliest_start = min(s for _, s, _ in successes)
    latest_finish = max(fi for _, _, fi in successes)
    wall_s = max((latest_finish - earliest_start) / 1000.0, 1e-9)

    total_output_tokens = sum(o.output_tokens for o, _, _ in successes)
    aggregate_tps = total_output_tokens / wall_s if wall_s > 0 else None

    per_stream: list[float] = []
    all_p50: list[float] = []
    all_p95: list[float] = []
    for o, _, _ in successes:
        s = summarize_decode(o.decode_token_times_ms)
        if s["decode_tps"] is not None:
            per_stream.append(float(s["decode_tps"]))
        if s["p50"] is not None:
            all_p50.append(float(s["p50"]))
        if s["p95"] is not None:
            all_p95.append(float(s["p95"]))

    per_stream_tps_mean = (sum(per_stream) / len(per_stream)) if per_stream else None
    p50_overall = percentile(all_p50, 0.5) if all_p50 else None
    p95_overall = percentile(all_p95, 0.95) if all_p95 else None

    return {
        "batch": batch_size,
        "aggregate_tps": aggregate_tps,
        "per_stream_tps": per_stream_tps_mean,
        "p50_ms": p50_overall,
        "p95_ms": p95_overall,
        "wall_ms": (latest_finish - earliest_start),
        "n_success": len(successes),
        "n_failures": len(failures),
        "total_output_tokens": total_output_tokens,
        "skipped": False,
        "error": None,
    }


class ConcurrentDecodeWorkload:
    """W4: aggregate vs per-stream tok/s across batch sizes 1, 4, 8, 16."""

    name = "concurrent-decode"
    suite_version = SUITE_VERSION

    def run(self, driver, model: ModelRef) -> WorkloadResult:
        flags: list[str] = []
        backend = model.backend
        if backend in _NON_CONCURRENT_BACKENDS:
            flags.append("no-real-concurrency")
            LOG.info("concurrent-decode: backend %r serializes; flagging", backend)

        curve: list[dict[str, Any]] = []
        any_success = False
        first_point: dict[str, Any] | None = None
        last_wall_ms = 0.0

        for bs in BATCH_SIZES:
            point = _measure_batch(driver, model, bs)
            curve.append(point)
            if not point["skipped"]:
                any_success = True
                last_wall_ms = point.get("wall_ms") or last_wall_ms
                if first_point is None:
                    first_point = point
            else:
                LOG.warning(
                    "concurrent-decode batch=%d failed: %s", bs, point.get("error")
                )

        if not any_success:
            return WorkloadResult(
                workload="concurrent-decode",
                suite_version=SUITE_VERSION,
                backend=backend,
                backend_version=None,
                model=model,
                error="all-batches-failed",
                flags=flags,
                extras={
                    "concurrency_curve": curve,
                    "batch_sizes": list(BATCH_SIZES),
                },
            )

        # Top-level decode_tps reflects batch=1 (or earliest successful batch).
        return WorkloadResult(
            workload="concurrent-decode",
            suite_version=SUITE_VERSION,
            backend=backend,
            backend_version=None,
            model=model,
            ttft_ms=None,
            prefill_tps=None,
            decode_tps=first_point.get("per_stream_tps") if first_point else None,
            decode_p50_latency_ms=first_point.get("p50_ms") if first_point else None,
            decode_p95_latency_ms=first_point.get("p95_ms") if first_point else None,
            prompt_tokens=PROMPT_TOKEN_TARGET,
            output_tokens=OUTPUT_TOKENS,
            batch_size=first_point.get("batch", 1) if first_point else 1,
            context_tokens=PROMPT_TOKEN_TARGET,
            wall_ms=last_wall_ms,
            flags=flags,
            extras={
                "concurrency_curve": curve,
                "batch_sizes": list(BATCH_SIZES),
                "prompt_chars": len(PROMPT),
                "target_prompt_tokens": PROMPT_TOKEN_TARGET,
                "target_output_tokens": OUTPUT_TOKENS,
            },
        )


register_workload("concurrent-decode", lambda: ConcurrentDecodeWorkload())


if __name__ == "__main__":
    print(f"concurrent-decode prompt: {len(PROMPT)} chars (~{len(PROMPT) // 4} tokens)")
    print(f"concurrent-decode batch sizes: {list(BATCH_SIZES)}")
    print(f"concurrent-decode output budget per stream: {OUTPUT_TOKENS}")
