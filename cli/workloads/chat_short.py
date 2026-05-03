"""W1 — chat-short.

128-token prompt -> 256-token output, batch=1.

Reports: TTFT, prefill tok/s, decode tok/s, p50/p95 decode latency.
This is the baseline workload that every backend must pass cleanly; numbers
from this row are what casual readers will compare across hardware.
"""

from __future__ import annotations

import logging

from .. import SUITE_VERSION
from ..registry import register_workload
from ..timing import summarize_decode
from ..types import ChatRunOutcome, ModelRef, WorkloadResult
from .prompts.fixtures import SHORT_USER_MSG

LOG = logging.getLogger("cli.workloads.chat-short")

OUTPUT_TOKENS = 256
TARGET_PROMPT_TOKENS = 128


def _result_from_outcome(
    outcome: ChatRunOutcome,
    model: ModelRef,
    backend_version: str | None,
) -> WorkloadResult:
    """Translate a `ChatRunOutcome` into a populated `WorkloadResult`.

    Prefer backend-reported tps over wall-clock-derived numbers when present
    (Ollama's `eval_count`/`eval_duration`, vLLM's metrics, etc.). Tag the
    source via `extras['tps_source']` so the upstream knows which to trust.
    """
    summary = summarize_decode(outcome.decode_token_times_ms)

    decode_tps = summary["decode_tps"]
    tps_source = "wall_clock" if decode_tps is not None else "none"

    backend_decode_tps = (
        outcome.backend_extras.get("decode_tps") if outcome.backend_extras else None
    )
    if isinstance(backend_decode_tps, (int, float)) and backend_decode_tps > 0:
        decode_tps = float(backend_decode_tps)
        tps_source = "backend"

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

    return WorkloadResult(
        workload="chat-short",
        suite_version=SUITE_VERSION,
        backend=model.backend,
        backend_version=backend_version,
        model=model,
        ttft_ms=outcome.ttft_ms,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        decode_p50_latency_ms=summary["p50"],
        decode_p95_latency_ms=summary["p95"],
        prompt_tokens=outcome.prompt_tokens,
        output_tokens=outcome.output_tokens,
        batch_size=1,
        context_tokens=outcome.prompt_tokens,
        wall_ms=outcome.wall_ms,
        prefix_cache_hit_rate=(
            outcome.backend_extras.get("prefix_cache_hit_rate")
            if outcome.backend_extras
            else None
        ),
        raw_timings_ms=list(outcome.decode_token_times_ms),
        error=None,
        flags=[],
        extras={
            "tps_source": tps_source,
            "target_prompt_tokens": TARGET_PROMPT_TOKENS,
            "target_output_tokens": OUTPUT_TOKENS,
            "backend_extras": dict(outcome.backend_extras)
            if outcome.backend_extras
            else {},
        },
    )


class ChatShortWorkload:
    """W1 baseline: short prompt, short response, single stream."""

    name = "chat-short"
    suite_version = SUITE_VERSION

    def run(self, driver, model: ModelRef) -> WorkloadResult:
        prompt = SHORT_USER_MSG
        LOG.debug("chat-short prompt: %d chars", len(prompt))

        try:
            outcome = driver.run_chat(
                model,
                prompt,
                max_output_tokens=OUTPUT_TOKENS,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("chat-short driver raised")
            return WorkloadResult(
                workload="chat-short",
                suite_version=SUITE_VERSION,
                backend=model.backend,
                backend_version=None,
                model=model,
                error=f"driver-exception: {exc!r}",
            )

        backend_version = (
            outcome.backend_extras.get("backend_version")
            if outcome.backend_extras
            else None
        )

        if not outcome.success:
            return WorkloadResult(
                workload="chat-short",
                suite_version=SUITE_VERSION,
                backend=model.backend,
                backend_version=backend_version,
                model=model,
                wall_ms=outcome.wall_ms,
                error=outcome.error or "unknown driver failure",
            )

        return _result_from_outcome(outcome, model, backend_version)


register_workload("chat-short", lambda: ChatShortWorkload())


if __name__ == "__main__":
    print(
        f"chat-short prompt: {len(SHORT_USER_MSG)} chars (target ~512, ~{TARGET_PROMPT_TOKENS} tokens)"
    )
    print(f"chat-short output budget: {OUTPUT_TOKENS} tokens")
