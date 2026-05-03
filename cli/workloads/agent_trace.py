"""W5 — `agent-trace` workload (the differentiator).

Plays a canned multi-turn coding-agent transcript against a backend driver.
Each turn appends to a growing chat history; on every assistant turn we
actually call `driver.run_chat` (so we measure decode/prefill/TTFT for that
specific turn's prompt) — but we then DISCARD the model's real output and
substitute the canned assistant content into the history before continuing.

Why discard real output?
    The benchmark must be deterministic across models. If model A produces
    a 200-token answer and model B produces a 50-token answer, every
    subsequent turn's prefill differs — the comparison is no longer apples
    to apples. By replaying a fixed transcript we guarantee that model A
    and model B see identical prior context at every turn; we only time
    how fast each one would respond to that context.

The point of this workload, beyond raw decode tps, is to surface
**prefix-cache hit rate** — backends that reuse the KV cache across turns
(vLLM, llama.cpp with caching, hosted APIs) win this workload by 3-5x.
We pull each turn's `backend_extras` for any cache-hit signal a driver
chooses to expose; see `_extract_cache_hit_rate` for the keys we look for.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

from .. import SUITE_VERSION
from ..registry import register_workload
from ..timing import now_ms, percentile
from ..types import (
    BackendDriver,
    ChatRunOutcome,
    ModelRef,
    WorkloadResult,
)
from .prompts.agent_trace_trace import SYSTEM_PROMPT, TRACE, TraceTurn

LOG = logging.getLogger("cli.workloads.agent_trace")

# Rough heuristic: 4 chars/token is the cheap, model-agnostic estimate we use
# when a driver doesn't return a real token count for the prompt.
_CHARS_PER_TOKEN = 4

# Backend extras keys we know about that might carry a prefix-cache signal.
# Drivers are free to use any of these; we accept the first that's present.
_CACHE_HIT_RATE_KEYS = (
    "prefix_cache_hit_rate",
    "cache_hit_rate",
    "kv_cache_hit_rate",
    "prompt_cache_hit_rate",
)
_CACHE_HIT_TOKEN_KEYS = (
    # (hit_count_key, total_count_key) — vLLM-style and llama.cpp-style
    ("prefix_cache_hit_tokens", "prompt_tokens"),
    ("cached_tokens", "prompt_tokens"),
    ("prompt_cache_hit_tokens", "prompt_tokens"),
    ("n_cached", "n_prompt"),
)


def _format_history(history: list[TraceTurn]) -> str:
    """Serialize the conversation into a single prompt string with role labels.

    Drivers that apply a chat template internally will see role-tagged blocks
    they can split on; drivers that pass the prompt through verbatim get a
    sane fallback. Either way, output is fully deterministic for a given
    history prefix — which is exactly what backends with prefix caching need
    to register a hit.
    """
    parts: list[str] = [f"<|system|>\n{SYSTEM_PROMPT}\n"]
    for turn in history:
        parts.append(f"<|{turn.role}|>\n{turn.content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _extract_cache_hit_rate(backend_extras: dict, prompt_tokens: int) -> float | None:
    """Pull a prefix-cache hit rate (0.0..1.0) from a driver's `backend_extras`.

    Returns None if the driver didn't expose anything cache-related — that's
    expected on backends without prefix caching, and we report it as a missing
    value rather than zero so the consumer can tell "no cache" from "cache,
    cold". Order matters: explicit ratios beat token-count division.
    """
    if not backend_extras:
        return None
    for k in _CACHE_HIT_RATE_KEYS:
        v = backend_extras.get(k)
        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
            return float(v)
    for hit_key, total_key in _CACHE_HIT_TOKEN_KEYS:
        hit = backend_extras.get(hit_key)
        total = backend_extras.get(total_key, prompt_tokens)
        if (
            isinstance(hit, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            return max(0.0, min(1.0, float(hit) / float(total)))
    return None


class AgentTraceWorkload:
    """Implements `cli.types.Workload`."""

    name = "agent-trace"
    suite_version = SUITE_VERSION

    def run(self, driver: BackendDriver, model: ModelRef) -> WorkloadResult:
        # Pre-compute the list of assistant turn indices we need to actually call
        # the model on. Non-assistant turns and assistant turns with
        # expect_output_tokens == 0 are pure context-builders.
        benchmarked = [
            i
            for i, t in enumerate(TRACE)
            if t.role == "assistant" and t.expect_output_tokens > 0
        ]
        if not benchmarked:
            return WorkloadResult(
                workload=self.name,
                suite_version=self.suite_version,
                backend=driver.name,
                backend_version=None,
                model=model,
                error="trace has no benchmarkable assistant turns",
            )

        history: list[TraceTurn] = []
        per_turn_records: list[dict] = []
        all_decode_times: list[float] = []
        cache_hit_rates: list[float] = []
        last_outcome: ChatRunOutcome | None = None
        last_prompt_tokens = 0

        wall_start = now_ms()

        for idx, turn in enumerate(TRACE):
            if turn.role != "assistant" or turn.expect_output_tokens <= 0:
                history.append(turn)
                continue

            prompt = _format_history(history)
            prompt_tokens_est = _approx_tokens(prompt)

            try:
                outcome = driver.run_chat(
                    model,
                    prompt=prompt,
                    max_output_tokens=turn.expect_output_tokens,
                    stream=True,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("driver.run_chat raised on turn %d", idx)
                return WorkloadResult(
                    workload=self.name,
                    suite_version=self.suite_version,
                    backend=driver.name,
                    backend_version=None,
                    model=model,
                    error=f"turn {idx}: driver raised: {exc!r}",
                )

            if not outcome.success:
                return WorkloadResult(
                    workload=self.name,
                    suite_version=self.suite_version,
                    backend=driver.name,
                    backend_version=None,
                    model=model,
                    error=f"turn {idx}: driver reported failure: {outcome.error!r}",
                )

            decode_times = list(outcome.decode_token_times_ms or [])
            all_decode_times.extend(decode_times)

            # Per-turn decode tps from this turn's per-token deltas (excludes prefill).
            turn_decode_tps: float | None = None
            if decode_times:
                total_s = sum(decode_times) / 1000.0
                if total_s > 0:
                    turn_decode_tps = len(decode_times) / total_s

            prompt_tokens = outcome.prompt_tokens or prompt_tokens_est

            # Per-turn prefill tps: prompt tokens divided by TTFT (the time
            # until the first decoded token = the time spent prefilling).
            turn_prefill_tps: float | None = None
            if outcome.ttft_ms and outcome.ttft_ms > 0 and prompt_tokens > 0:
                turn_prefill_tps = prompt_tokens / (outcome.ttft_ms / 1000.0)

            cache_rate = _extract_cache_hit_rate(
                outcome.backend_extras or {}, prompt_tokens
            )
            if cache_rate is not None:
                cache_hit_rates.append(cache_rate)

            per_turn_records.append(
                {
                    "turn_idx": idx,
                    "ttft_ms": outcome.ttft_ms,
                    "decode_tps": turn_decode_tps,
                    "prefill_tps": turn_prefill_tps,
                    "output_tokens": outcome.output_tokens or len(decode_times),
                    "prompt_tokens": prompt_tokens,
                    "wall_ms": outcome.wall_ms,
                    "prefix_cache_hit_rate": cache_rate,
                }
            )

            last_outcome = outcome
            last_prompt_tokens = prompt_tokens

            # Discard real model output; substitute canned content so the next
            # turn's prefix is identical across runs / models.
            history.append(turn)

        wall_ms = now_ms() - wall_start

        # ---- aggregate ----
        ttfts = [r["ttft_ms"] for r in per_turn_records if r["ttft_ms"] is not None]
        prefills = [
            r["prefill_tps"] for r in per_turn_records if r["prefill_tps"] is not None
        ]

        # Decode tps as the weighted average of per-turn decode tps,
        # weighted by output tokens. (Equivalent to total_decoded_tokens /
        # total_decode_seconds, but computed defensively if any turn lacked
        # per-token times.)
        weighted_num = 0.0
        weighted_den = 0.0
        for r in per_turn_records:
            if r["decode_tps"] is not None and r["output_tokens"]:
                weighted_num += r["decode_tps"] * r["output_tokens"]
                weighted_den += r["output_tokens"]
        decode_tps_agg: float | None = (
            (weighted_num / weighted_den) if weighted_den > 0 else None
        )

        median_ttft = statistics.median(ttfts) if ttfts else None
        median_prefill = statistics.median(prefills) if prefills else None
        p50 = percentile(all_decode_times, 0.50)
        p95 = percentile(all_decode_times, 0.95)
        median_cache = statistics.median(cache_hit_rates) if cache_hit_rates else None

        total_output = sum(
            t.expect_output_tokens for t in TRACE if t.role == "assistant"
        )

        extras: dict[str, Any] = {
            "per_turn": per_turn_records,
            "n_assistant_turns_benchmarked": len(benchmarked),
            "trace_chars_total": sum(len(t.content) for t in TRACE)
            + len(SYSTEM_PROMPT),
        }
        if median_cache is not None:
            extras["prefix_cache_hit_rate"] = median_cache
        if last_outcome is not None and last_outcome.backend_extras:
            extras["last_turn_backend_extras"] = dict(last_outcome.backend_extras)

        return WorkloadResult(
            workload=self.name,
            suite_version=self.suite_version,
            backend=driver.name,
            backend_version=None,
            model=model,
            ttft_ms=median_ttft,
            prefill_tps=median_prefill,
            decode_tps=decode_tps_agg,
            decode_p50_latency_ms=p50,
            decode_p95_latency_ms=p95,
            prompt_tokens=last_prompt_tokens,
            output_tokens=total_output,
            batch_size=1,
            context_tokens=last_prompt_tokens,
            wall_ms=wall_ms,
            prefix_cache_hit_rate=median_cache,
            raw_timings_ms=all_decode_times,
            extras=extras,
        )


register_workload("agent-trace", lambda: AgentTraceWorkload())


if __name__ == "__main__":
    total_chars = sum(len(t.content) for t in TRACE) + len(SYSTEM_PROMPT)
    n_bench = sum(
        1 for t in TRACE if t.role == "assistant" and t.expect_output_tokens > 0
    )
    # No driver needed for self-test; print headline shape numbers.
    print(f"total trace chars: {total_chars}; assistant turns to benchmark: {n_bench}")
