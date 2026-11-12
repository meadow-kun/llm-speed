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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .. import SUITE_VERSION
from ..registry import register_workload
from ..timing import (
    PREFILL_TPS_OVER_TTFT,
    PREFILL_TPS_SERVER,
    RECORD_SERVER_PREFILL,
    combine_decode_tps_sources,
    now_ms,
    percentile,
    select_decode_tps,
    summarize_decode,
)
from ..types import ChatRunOutcome, ModelRef, WorkloadResult
from .prompts.fill import fill_prompt

LOG = logging.getLogger("cli.workloads.concurrent-decode")

# Default when the preset does not set LLM_SPEED_CONCURRENT_STREAMS.
# 32 lanes is not in this list: the preset sets it, for example "1,2,4,8,16,32".
# 0.0.23 records at most 6 counts. A 7th is refused before a server starts.
BATCH_SIZES: tuple[int, ...] = (1, 4, 8, 16)
PROMPT_TOKEN_TARGET = 1024
STREAM_MIN, STREAM_MAX = 1, 64
MAX_STREAM_COUNTS = 6
PROMPT_MIN, PROMPT_MAX = 1024, 1_048_576
OUTPUT_TOKENS = 256
INSTRUCTION = (
    "\n\n---\n\nSummarize the passages above in eight to twelve sentences. Be specific."
)

# Backends where threading does not yield real overlap (single-process,
# blocking compute). Concurrent runs against these are reported but flagged.
_NON_CONCURRENT_BACKENDS = {
    "mlx",
    "exllamav2",
    "vllm-embedded",
    "llama-cpp-embedded",
}


def too_many_streams_message(n: int) -> str:
    """The refusal shown before a server starts. `n` is the count that was passed."""
    return (
        f"at most {MAX_STREAM_COUNTS} stream counts per run (got {n}); "
        "split into separate runs"
    )


def parse_stream_list(raw: str) -> tuple[int, ...] | None:
    """Comma-separated stream counts, each 1-64. None when any value is not.

    Shared by the workload, the fit sweep, and preset validation. An empty
    string is None (the caller keeps the default only when the variable is unset).
    Length is not checked here; `stream_list_error` refuses more than 6.
    """
    try:
        pts = tuple(int(x) for x in str(raw).split(",") if str(x).strip())
    except ValueError:
        return None
    if not pts or any(p < STREAM_MIN or p > STREAM_MAX for p in pts):
        return None
    return pts


def stream_list_error(raw: str) -> str | None:
    """None when `raw` may run. A message when it must be refused first."""
    parsed = parse_stream_list(raw)
    if parsed is None:
        return "LLM_SPEED_CONCURRENT_STREAMS must be comma-separated integers 1-64"
    if len(parsed) > MAX_STREAM_COUNTS:
        return too_many_streams_message(len(parsed))
    return None


def parse_prompt_target(raw: str) -> int | None:
    """Prompt-token target in 1024-1048576, or None."""
    try:
        n = int(str(raw).strip())
    except ValueError:
        return None
    if PROMPT_MIN <= n <= PROMPT_MAX:
        return n
    return None


def streams_from_env(raw: str | None) -> tuple[int, ...]:
    """Unset or blank keeps 1, 4, 8, 16. A bad value, or more than 6, is an error."""
    if raw is None or not str(raw).strip():
        return BATCH_SIZES
    err = stream_list_error(str(raw))
    if err:
        raise ValueError(err)
    parsed = parse_stream_list(str(raw))
    if parsed is None:
        raise ValueError(
            "LLM_SPEED_CONCURRENT_STREAMS must be comma-separated integers 1-64"
        )
    return parsed


def prompt_from_env(raw: str | None) -> int:
    """Unset or blank keeps 1024. A set value outside 1024-1048576 is an error."""
    if raw is None or not str(raw).strip():
        return PROMPT_TOKEN_TARGET
    parsed = parse_prompt_target(str(raw))
    if parsed is None:
        raise ValueError("LLM_SPEED_CONCURRENT_PROMPT must be an integer 1024-1048576")
    return parsed


def _streams() -> tuple[int, ...]:
    return streams_from_env(os.environ.get("LLM_SPEED_CONCURRENT_STREAMS"))


def _prompt_target() -> int:
    return prompt_from_env(os.environ.get("LLM_SPEED_CONCURRENT_PROMPT"))


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def _round2(value: float | None) -> float | None:
    """Two decimal places. The Worker extras cap is 4,096 bytes."""
    if value is None:
        return None
    return round(float(value), 2)


def _stream_prefill(outcome: ChatRunOutcome) -> tuple[float | None, str | None]:
    """Prefill tok/s for one lane, and which clock it came from.

    llama-server `prompt_per_second` is the server's prompt rate. Without it,
    the rate is prompt tokens over this lane's time to first token. With
    `--parallel` at least the lane count there is no slot wait, so that time
    is request overhead and contention, not prefill throughput.
    """
    extras = outcome.backend_extras or {}
    server = _positive(extras.get("prefill_tps"))
    if server is not None and extras.get("prefill_tps_source") == PREFILL_TPS_SERVER:
        return server, PREFILL_TPS_SERVER
    ttft = _positive(outcome.ttft_ms)
    if ttft is not None and outcome.prompt_tokens > 0:
        return outcome.prompt_tokens / (ttft / 1000.0), PREFILL_TPS_OVER_TTFT
    return None, None


def _prefill_basis(sources: list[str]) -> str | None:
    order = (PREFILL_TPS_SERVER, PREFILL_TPS_OVER_TTFT)
    present = [name for name in order if name in sources]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return "+".join(present)


def _one_lane(curve: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The successful batch-1 point. Row-level prefill and TTFT use it."""
    for point in curve:
        if point.get("batch") == 1 and not point.get("skipped"):
            return point
    return None


def _row_metrics_basis(curve: list[dict[str, Any]]) -> str:
    """Why the row did or did not copy prefill and TTFT from batch 1.

    A list with no batch of 1 says so. A batch of 1 that ran and failed says
    that, instead of claiming the point was absent.
    """
    for point in curve:
        if point.get("batch") != 1:
            continue
        if point.get("skipped"):
            return "1-lane point ran but failed"
        return "1-lane"
    return "no 1-lane point"


def _slot_extras(slot: dict[str, Any] | None) -> dict[str, Any]:
    if not slot:
        return {}
    return {"n_ctx_slot": slot.get("n_ctx_slot"), "kv_unified": slot.get("kv_unified")}


def _context_slot(driver) -> dict[str, Any] | None:
    """{n_ctx_slot, kv_unified} from a llama-server driver, or None."""
    fn = getattr(driver, "context_slot", None)
    if not callable(fn):
        return None
    try:
        info = fn()
    except Exception as exc:  # noqa: BLE001
        LOG.debug("context slot probe failed: %s", exc)
        return None
    return info if isinstance(info, dict) else None


def _count_with(driver, model):
    fn = getattr(driver, "count_prompt_tokens", None) if driver is not None else None
    if not callable(fn):

        def _unknown(_text: str) -> int | None:
            return None

        return _unknown

    def _count(text: str) -> int | None:
        try:
            n = fn(model, text)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("prompt fill count failed: %s", exc)
            return None
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            return None
        return n

    return _count


def _run_one(
    driver, model: ModelRef, prompt: str
) -> tuple[bool, ChatRunOutcome, float, float]:
    """Run a single request; return (success, outcome, started_ms, finished_ms)."""
    started = now_ms()
    try:
        outcome = driver.run_chat(
            model,
            prompt,
            max_output_tokens=OUTPUT_TOKENS,
            stream=True,
            # Server prompt tok/s stays on this workload. Chat and long-context
            # rows do not set this, so they keep prompt tokens over client TTFT.
            extras={RECORD_SERVER_PREFILL: True},
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


def _measure_batch(
    driver, model: ModelRef, batch_size: int, prompt: str
) -> dict[str, Any]:
    """Run `batch_size` parallel streams and aggregate metrics."""
    LOG.debug("concurrent-decode batch=%d starting", batch_size)

    completions: list[tuple[bool, ChatRunOutcome, float, float]] = []
    if batch_size == 1:
        completions.append(_run_one(driver, model, prompt))
    else:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futs = [
                pool.submit(_run_one, driver, model, prompt) for _ in range(batch_size)
            ]
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
            "n_success": 0,
            "n_failures": len(failures),
            "decode_tps_source": None,
            "prefill_tps": None,
            "prefill_tps_source": None,
            "ttft_ms": None,
            "ttft_max_ms": None,
        }

    earliest_start = min(s for _, s, _ in successes)
    latest_finish = max(fi for _, _, fi in successes)
    wall_s = max((latest_finish - earliest_start) / 1000.0, 1e-9)

    total_output_tokens = sum(o.output_tokens for o, _, _ in successes)
    aggregate_tps = total_output_tokens / wall_s if wall_s > 0 else None

    per_stream: list[float] = []
    stream_rates: list[float] = []
    sources: list[str] = []
    ttfts: list[float] = []
    prefills: list[float] = []
    prefill_sources: list[str] = []
    all_p50: list[float] = []
    all_p95: list[float] = []
    for o, _, _ in successes:
        s = summarize_decode(o.decode_token_times_ms)
        stream_tps = s["decode_tps"]
        if stream_tps is not None:
            stream_rates.append(float(stream_tps))
        rate, source = select_decode_tps(stream_tps, o.backend_extras)
        if rate is not None:
            per_stream.append(rate)
        if source is not None:
            sources.append(source)
        ttft = _positive(o.ttft_ms)
        if ttft is not None:
            ttfts.append(ttft)
        prefill, prefill_source = _stream_prefill(o)
        if prefill is not None:
            prefills.append(prefill)
        if prefill_source is not None:
            prefill_sources.append(prefill_source)
        if s["p50"] is not None:
            all_p50.append(float(s["p50"]))
        if s["p95"] is not None:
            all_p95.append(float(s["p95"]))

    per_stream_tps_mean = (sum(per_stream) / len(per_stream)) if per_stream else None
    p50_overall = percentile(all_p50, 0.5) if all_p50 else None
    p95_overall = percentile(all_p95, 0.95) if all_p95 else None
    reported = [o.prompt_tokens for o, _, _ in successes if o.prompt_tokens > 0]
    prompt_tokens = reported[0] if reported else None

    return {
        "batch": batch_size,
        "aggregate_tps": _round2(aggregate_tps),
        "per_stream_tps": _round2(per_stream_tps_mean),
        "prompt_tokens": prompt_tokens,
        "p50_ms": _round2(p50_overall),
        "p95_ms": _round2(p95_overall),
        "wall_ms": _round2(latest_finish - earliest_start),
        "n_success": len(successes),
        "n_failures": len(failures),
        "total_output_tokens": total_output_tokens,
        "skipped": False,
        "error": None,
        "decode_tps_stream": _round2(
            sum(stream_rates) / len(stream_rates) if stream_rates else None
        ),
        "decode_tps_source": combine_decode_tps_sources(sources),
        "prefill_tps": _round2((sum(prefills) / len(prefills)) if prefills else None),
        "prefill_tps_source": _prefill_basis(prefill_sources),
        # ttft_ms is the p50 across lanes. ttft_max_ms is the slowest lane.
        # n_ctx_slot and kv_unified stay on the row. Repeating them here
        # pushes an 8-point curve over the Worker extras cap.
        "ttft_ms": _round2(percentile(ttfts, 0.50) if ttfts else None),
        "ttft_max_ms": _round2(max(ttfts) if ttfts else None),
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

        try:
            streams = _streams()
            target = _prompt_target()
        except ValueError as exc:
            return WorkloadResult(
                workload="concurrent-decode",
                suite_version=SUITE_VERSION,
                backend=backend,
                backend_version=None,
                model=model,
                error=str(exc),
                flags=flags,
            )
        prompt, measured = fill_prompt(
            target, _count_with(driver, model), suffix=INSTRUCTION
        )
        slot = _context_slot(driver)
        prompt_count = measured if isinstance(measured, int) else target
        if slot is not None and isinstance(slot.get("n_ctx_slot"), int):
            need = prompt_count + OUTPUT_TOKENS
            slot_n = slot["n_ctx_slot"]
            slot_extra = {
                "n_ctx_slot": slot_n,
                "kv_unified": slot.get("kv_unified"),
                "batch_sizes": list(streams),
                "target_prompt_tokens": target,
                "prompt_tokens_actual": measured,
                "target_output_tokens": OUTPUT_TOKENS,
            }
            if need > slot_n:
                LOG.warning(
                    "concurrent-decode: prompt plus max tokens (%d) exceeds n_ctx_slot (%d)",
                    need,
                    slot_n,
                )
                return WorkloadResult(
                    workload="concurrent-decode",
                    suite_version=SUITE_VERSION,
                    backend=backend,
                    backend_version=None,
                    model=model,
                    error=(
                        f"invalid config: prompt plus max tokens ({need}) "
                        f"exceeds n_ctx_slot ({slot_n})"
                    ),
                    flags=flags,
                    extras=slot_extra,
                )
        for bs in streams:
            point = _measure_batch(driver, model, bs, prompt)
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
                    "batch_sizes": list(streams),
                    "target_prompt_tokens": target,
                    "prompt_tokens_actual": measured,
                    "row_metrics_basis": _row_metrics_basis(curve),
                    **_slot_extras(slot),
                },
            )

        # decode_tps is per-stream on the first successful batch. The total for
        # that batch is extras.concurrency_curve[].aggregate_tps (already a
        # result field). prompt_tokens is the filled count, not the target.
        # Row-level prefill_tps and ttft_ms are the successful 1-lane point.
        # A list with no batch of 1 says `no 1-lane point`. A batch of 1 that
        # ran and failed says so, and those two row fields stay unset.
        lane1 = _one_lane(curve)
        row_basis = _row_metrics_basis(curve)
        actual = None
        if first_point is not None and first_point.get("prompt_tokens"):
            actual = int(first_point["prompt_tokens"])
        elif measured is not None:
            actual = measured
        return WorkloadResult(
            workload="concurrent-decode",
            suite_version=SUITE_VERSION,
            backend=backend,
            backend_version=None,
            model=model,
            ttft_ms=lane1.get("ttft_ms") if lane1 else None,
            prefill_tps=lane1.get("prefill_tps") if lane1 else None,
            decode_tps=first_point.get("per_stream_tps") if first_point else None,
            decode_p50_latency_ms=first_point.get("p50_ms") if first_point else None,
            decode_p95_latency_ms=first_point.get("p95_ms") if first_point else None,
            prompt_tokens=actual or 0,
            output_tokens=OUTPUT_TOKENS,
            batch_size=first_point.get("batch", 1) if first_point else 1,
            context_tokens=actual or target,
            wall_ms=last_wall_ms,
            flags=flags,
            extras={
                "concurrency_curve": curve,
                "batch_sizes": list(streams),
                "prompt_chars": len(prompt),
                "target_prompt_tokens": target,
                "prompt_tokens_actual": actual,
                "target_output_tokens": OUTPUT_TOKENS,
                "decode_tps_stream": (
                    first_point.get("decode_tps_stream") if first_point else None
                ),
                "decode_tps_source": (
                    first_point.get("decode_tps_source") if first_point else None
                ),
                "prefill_tps_source": (
                    lane1.get("prefill_tps_source") if lane1 else None
                ),
                "row_metrics_basis": row_basis,
                **_slot_extras(slot),
            },
        )


register_workload("concurrent-decode", lambda: ConcurrentDecodeWorkload())


if __name__ == "__main__":
    _prompt, _n = fill_prompt(_prompt_target(), lambda _t: None, suffix=INSTRUCTION)
    print(
        f"concurrent-decode prompt: {len(_prompt)} chars"
        f" (~{_n if _n is not None else len(_prompt) // 4} tokens)"
    )
    print(f"concurrent-decode batch sizes: {list(_streams())}")
    print(f"concurrent-decode output budget per stream: {OUTPUT_TOKENS}")
