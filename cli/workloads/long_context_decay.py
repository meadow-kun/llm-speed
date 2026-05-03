"""W3 — long-context-decay.

For each of [32k, 64k, 128k] input contexts: 256-token output, batch=1.
Skipped if model context max < target. Reports a curve of
(context_length, prefill_tps, decode_tps, ttft_ms) measurements.

The point of this workload is to expose how prefill cost (quadratic in
context for vanilla attention, linear with FlashAttention) and decode cost
(linear KV-cache streaming) scale across backends and quantizations.
Different backends handle long contexts very differently — paged attention,
prefix-cache reuse, KV offload to RAM, attention sinks — and a single number
hides those differences.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import SUITE_VERSION
from ..registry import register_workload
from ..timing import summarize_decode
from ..types import ChatRunOutcome, ModelRef, WorkloadResult
from .prompts.fixtures import build_prompt_of_chars

LOG = logging.getLogger("cli.workloads.long-context-decay")

CONTEXT_TOKEN_TARGETS: tuple[int, ...] = (32_000, 64_000, 128_000)
OUTPUT_TOKENS = 256
INSTRUCTION = "\n\n---\n\nBriefly summarize the document above in three sentences."


def _build_prompt_for_ctx(ctx_tokens: int) -> str:
    """Build a prompt that approximates `ctx_tokens` tokens via 4 chars/token."""
    target_chars = ctx_tokens * 4 - len(INSTRUCTION)
    body = build_prompt_of_chars(max(target_chars, 0))
    return body + INSTRUCTION


def _model_max_ctx(model: ModelRef) -> int | None:
    """Best-effort retrieve model's max context window."""
    n_ctx = model.extras.get("n_ctx") if model.extras else None
    if isinstance(n_ctx, int) and n_ctx > 0:
        return n_ctx
    # Some drivers store under different keys; check common ones.
    for key in ("context_length", "max_context", "max_ctx"):
        v = model.extras.get(key) if model.extras else None
        if isinstance(v, int) and v > 0:
            return v
    return None


def _measure_one(
    driver,
    model: ModelRef,
    ctx_tokens: int,
) -> dict[str, Any]:
    """Run one context-size point. Return dict with metrics + status."""
    prompt = _build_prompt_for_ctx(ctx_tokens)
    LOG.debug("long-context-decay ctx=%d prompt_chars=%d", ctx_tokens, len(prompt))

    try:
        outcome: ChatRunOutcome = driver.run_chat(
            model,
            prompt,
            max_output_tokens=OUTPUT_TOKENS,
            stream=True,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("long-context-decay ctx=%d driver raised: %r", ctx_tokens, exc)
        return {
            "ctx": ctx_tokens,
            "prefill_tps": None,
            "decode_tps": None,
            "ttft_ms": None,
            "wall_ms": None,
            "skipped": True,
            "error": f"driver-exception: {exc!r}",
        }

    if not outcome.success:
        return {
            "ctx": ctx_tokens,
            "prefill_tps": None,
            "decode_tps": None,
            "ttft_ms": None,
            "wall_ms": outcome.wall_ms,
            "skipped": True,
            "error": outcome.error or "driver-failure",
        }

    summary = summarize_decode(outcome.decode_token_times_ms)
    decode_tps = summary["decode_tps"]
    backend_decode_tps = (
        outcome.backend_extras.get("decode_tps") if outcome.backend_extras else None
    )
    if isinstance(backend_decode_tps, (int, float)) and backend_decode_tps > 0:
        decode_tps = float(backend_decode_tps)

    prefill_tps: float | None = None
    backend_prefill_tps = (
        outcome.backend_extras.get("prefill_tps") if outcome.backend_extras else None
    )
    if isinstance(backend_prefill_tps, (int, float)) and backend_prefill_tps > 0:
        prefill_tps = float(backend_prefill_tps)
    elif (
        outcome.ttft_ms is not None
        and outcome.ttft_ms > 0
        and outcome.prompt_tokens > 0
    ):
        prefill_tps = outcome.prompt_tokens / (outcome.ttft_ms / 1000.0)

    return {
        "ctx": ctx_tokens,
        "prompt_tokens_actual": outcome.prompt_tokens,
        "output_tokens": outcome.output_tokens,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "ttft_ms": outcome.ttft_ms,
        "wall_ms": outcome.wall_ms,
        "p50_ms": summary["p50"],
        "p95_ms": summary["p95"],
        "skipped": False,
        "error": None,
    }


class LongContextDecayWorkload:
    """W3: probe (prefill_tps, decode_tps) at several increasing context lengths."""

    name = "long-context-decay"
    suite_version = SUITE_VERSION

    def run(self, driver, model: ModelRef) -> WorkloadResult:
        max_ctx = _model_max_ctx(model)
        curve: list[dict[str, Any]] = []
        skipped_too_large: list[int] = []
        any_success = False
        backend_version: str | None = None
        last_wall_ms = 0.0
        first_point: dict[str, Any] | None = None

        for ctx in CONTEXT_TOKEN_TARGETS:
            if max_ctx is not None and ctx > max_ctx:
                LOG.info("skip ctx=%d (model max_ctx=%d)", ctx, max_ctx)
                curve.append(
                    {
                        "ctx": ctx,
                        "prefill_tps": None,
                        "decode_tps": None,
                        "ttft_ms": None,
                        "wall_ms": None,
                        "skipped": True,
                        "error": f"exceeds-model-max-ctx({max_ctx})",
                    }
                )
                skipped_too_large.append(ctx)
                continue

            point = _measure_one(driver, model, ctx)
            curve.append(point)
            if not point["skipped"]:
                any_success = True
                last_wall_ms = point.get("wall_ms") or 0.0
                if first_point is None:
                    first_point = point
            else:
                # If we got an error at a particular ctx, do not waste time
                # on larger contexts — they will likely fail too.
                LOG.info("ctx=%d failed (%s); halting curve", ctx, point.get("error"))
                for remaining in CONTEXT_TOKEN_TARGETS[
                    CONTEXT_TOKEN_TARGETS.index(ctx) + 1 :
                ]:
                    curve.append(
                        {
                            "ctx": remaining,
                            "prefill_tps": None,
                            "decode_tps": None,
                            "ttft_ms": None,
                            "wall_ms": None,
                            "skipped": True,
                            "error": "halted-after-prior-failure",
                        }
                    )
                break

        flags: list[str] = []
        if skipped_too_large:
            flags.append("skipped-over-model-ctx")

        if not any_success:
            return WorkloadResult(
                workload="long-context-decay",
                suite_version=SUITE_VERSION,
                backend=model.backend,
                backend_version=backend_version,
                model=model,
                wall_ms=last_wall_ms,
                error="all-context-points-failed",
                flags=flags,
                extras={
                    "context_curve": curve,
                    "context_targets": list(CONTEXT_TOKEN_TARGETS),
                },
            )

        # Top-level summary numbers reflect the smallest successful context.
        return WorkloadResult(
            workload="long-context-decay",
            suite_version=SUITE_VERSION,
            backend=model.backend,
            backend_version=backend_version,
            model=model,
            ttft_ms=first_point.get("ttft_ms") if first_point else None,
            prefill_tps=first_point.get("prefill_tps") if first_point else None,
            decode_tps=first_point.get("decode_tps") if first_point else None,
            decode_p50_latency_ms=first_point.get("p50_ms") if first_point else None,
            decode_p95_latency_ms=first_point.get("p95_ms") if first_point else None,
            prompt_tokens=(first_point.get("prompt_tokens_actual") or 0)
            if first_point
            else 0,
            output_tokens=(first_point.get("output_tokens") or 0) if first_point else 0,
            batch_size=1,
            context_tokens=first_point.get("ctx") if first_point else 0,
            wall_ms=last_wall_ms,
            flags=flags,
            extras={
                "context_curve": curve,
                "context_targets": list(CONTEXT_TOKEN_TARGETS),
                "model_max_ctx": max_ctx,
            },
        )


register_workload("long-context-decay", lambda: LongContextDecayWorkload())


if __name__ == "__main__":
    for ctx in CONTEXT_TOKEN_TARGETS:
        p = _build_prompt_for_ctx(ctx)
        print(f"long-context-decay ctx={ctx}: {len(p)} chars (~{len(p) // 4} tokens)")
