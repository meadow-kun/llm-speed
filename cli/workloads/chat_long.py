"""W2 — chat-long.

4k-token prompt -> 1k-token output, batch=1.

Reports the same set as W1 (TTFT, prefill tok/s, decode tok/s, p50/p95 decode
latency). Tests prefill scaling: a long prompt exposes attention cost growth
and KV-cache allocation behavior that the short-prompt workload does not.

The 4k-token prompt is constructed by concatenating fixture paragraphs to
~16,000 characters using the ~4-chars-per-token heuristic.
"""

from __future__ import annotations

import logging

from .. import SUITE_VERSION
from ..registry import register_workload
from ..timing import summarize_decode
from ..types import ChatRunOutcome, ModelRef, WorkloadResult
from .prompts.fixtures import build_prompt_of_chars

LOG = logging.getLogger("cli.workloads.chat-long")

TARGET_PROMPT_TOKENS = 4096
TARGET_PROMPT_CHARS = TARGET_PROMPT_TOKENS * 4  # ~16,000 chars
OUTPUT_TOKENS = 1024

# Build once at import; this is a fixed fixture for suite-v1.
PROMPT_BODY = build_prompt_of_chars(TARGET_PROMPT_CHARS)
INSTRUCTION = (
    "Please summarize the key points of the passages above in five bullet "
    "points, then answer this question: which performance regime is most "
    "likely to bottleneck a multi-user inference deployment of a 70B-class "
    "model on a single 80GB GPU, and why?"
)
PROMPT = PROMPT_BODY + "\n\n---\n\n" + INSTRUCTION


def _result_from_outcome(
    outcome: ChatRunOutcome,
    model: ModelRef,
    backend_version: str | None,
) -> WorkloadResult:
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
        workload="chat-long",
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
            "prompt_chars": len(PROMPT),
            "backend_extras": dict(outcome.backend_extras)
            if outcome.backend_extras
            else {},
        },
    )


class ChatLongWorkload:
    """W2: long prompt, moderate response, single stream. Stresses prefill."""

    name = "chat-long"
    suite_version = SUITE_VERSION

    def run(self, driver, model: ModelRef) -> WorkloadResult:
        LOG.debug(
            "chat-long prompt: %d chars (~%d tokens target)",
            len(PROMPT),
            TARGET_PROMPT_TOKENS,
        )

        try:
            outcome = driver.run_chat(
                model,
                PROMPT,
                max_output_tokens=OUTPUT_TOKENS,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.exception("chat-long driver raised")
            return WorkloadResult(
                workload="chat-long",
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
                workload="chat-long",
                suite_version=SUITE_VERSION,
                backend=model.backend,
                backend_version=backend_version,
                model=model,
                wall_ms=outcome.wall_ms,
                error=outcome.error or "unknown driver failure",
            )

        return _result_from_outcome(outcome, model, backend_version)


register_workload("chat-long", lambda: ChatLongWorkload())


if __name__ == "__main__":
    print(
        f"chat-long prompt: {len(PROMPT)} chars (target ~{TARGET_PROMPT_CHARS}, ~{TARGET_PROMPT_TOKENS} tokens)"
    )
    print(f"chat-long output budget: {OUTPUT_TOKENS} tokens")
