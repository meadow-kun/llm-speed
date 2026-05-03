"""Frozen prompt fixtures for suite-v1 workloads.

These strings are part of the suite definition. Editing any of them is a
breaking change and requires bumping `cli.SUITE_VERSION`. They are intentionally
verbose so character-length heuristics (~4 chars/token) yield the desired
token budget without invoking a real tokenizer.

Why no tokenizer? The CLI must be backend-agnostic and stdlib-only. A real
tokenizer count would have to load model-specific BPE merges, which we cannot
do at fixture-construction time. The 4 chars/token heuristic is good enough
for the suite's purpose: each user runs the *same* prompt across backends,
so absolute token counts only need to be in the right order of magnitude.
For reporting we ask drivers to return their tokenizer's authoritative
`prompt_tokens` count from the backend response.
"""

from __future__ import annotations

# Conventional rule of thumb across modern BPE tokenizers (GPT-2/4, Llama,
# Qwen, etc.) for English prose: ~4 characters per token. Workloads use this
# to size their target prompt-character budgets.
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Short prompt — W1 (chat-short). Target ~128 tokens (~512 chars).
# A realistic developer question that elicits a substantive technical answer,
# so decode produces a real response (not just refusal/short-form).
# ---------------------------------------------------------------------------

SHORT_USER_MSG = (
    "I'm building a CLI tool in Python that benchmarks local LLM backends "
    "across hardware. I want to capture time-to-first-token, prefill tokens "
    "per second, decode tokens per second, and p50/p95 inter-token latency. "
    "What is a robust way to measure these from a streaming HTTP response, "
    "and how should I separate prefill time from decode time when the "
    "server only emits tokens after the prompt has been fully ingested? "
    "Walk through the wall-clock instrumentation step by step. Be concrete."
)


# ---------------------------------------------------------------------------
# Medium fixture paragraph — building block for moderate prompts (~1500 chars,
# roughly 375 tokens). Concatenated by workloads to compose longer prompts.
# ---------------------------------------------------------------------------

MEDIUM_USER_MSG_BASE = (
    "Modern transformer inference splits naturally into two regimes with very "
    "different performance characteristics. The prefill phase ingests the "
    "entire prompt in a single forward pass; arithmetic intensity is high, "
    "the GPU is compute-bound, and the relevant figure of merit is prompt "
    "tokens processed per second. The decode phase, by contrast, generates "
    "one token per forward pass and is dominated by memory bandwidth: every "
    "step must stream the full KV cache and the model weights through the "
    "tensor cores, so decode tokens per second is roughly inversely "
    "proportional to model size in bytes. This asymmetry is why a 70B model "
    "at 4-bit on a single 4090 may prefill at thousands of tokens per second "
    "yet decode at only 20 to 40 tokens per second, and why batching multiple "
    "decode streams together amortizes the bandwidth cost and dramatically "
    "improves aggregate throughput. Long contexts add a third wrinkle: "
    "attention cost scales quadratically in context length during prefill, "
    "and the KV cache itself grows linearly, eventually exceeding VRAM and "
    "forcing offload to system RAM with a ten-to-twentyfold slowdown. "
    "Different backends handle these regimes with very different trade-offs. "
    "Some prioritize single-stream latency, others maximize multi-stream "
    "throughput, and a few do clever things with paged attention or prefix "
    "cache reuse to make agentic workloads, which reissue overlapping "
    "contexts repeatedly, far cheaper than a naive accounting would suggest. "
    "A serious benchmark suite has to measure each of these regimes "
    "independently, because the right answer to which backend is fastest "
    "depends entirely on which regime your application lives in."
)


# ---------------------------------------------------------------------------
# Long-paragraph pool — building blocks for W2/W3 long contexts.
# Each ~300 chars (~75 tokens). Eight distinct paragraphs of original prose
# covering different topics so concatenations look like realistic mixed text
# rather than obviously-repeated boilerplate.
# ---------------------------------------------------------------------------

LONG_PARAGRAPH_POOL: list[str] = [
    (
        "The history of computing hardware can be read as a long series of "
        "compromises between latency and throughput. Pipelining traded "
        "single-instruction completion time for higher aggregate instruction "
        "rates; superscalar execution traded silicon area for parallelism; "
        "out-of-order execution traded verification complexity for the "
        "ability to fill stalls with useful work. Each compromise stuck."
    ),
    (
        "Memory hierarchies exist because there is no single technology that "
        "is simultaneously fast, dense, cheap, and persistent. SRAM is fast "
        "but expensive; DRAM is dense but slow; flash is dense and cheap "
        "but slow and wear-prone; magnetic media is the densest of all yet "
        "wholly unsuited to random access. Caching policies exist to bridge "
        "these tiers without exposing their seams to the programmer."
    ),
    (
        "Numerical analysts have known for a century that floating-point "
        "addition is not associative. The order in which a sum is evaluated "
        "can move the result by many ulps, and parallel reductions on GPUs "
        "expose this to anyone who has compared two runs of the same kernel "
        "with subtly different launch configurations. Reproducibility under "
        "data-parallelism remains an active and contested research topic."
    ),
    (
        "Distributed systems folklore has long held that the network is the "
        "computer, but the more useful mantra for building correct services "
        "is that every remote call can hang, retry, double-deliver, or "
        "arrive out of order. Protocol designers who internalize this build "
        "idempotent operations, monotonic clocks, and explicit timeouts; "
        "those who do not eventually rediscover all three under duress."
    ),
    (
        "The transformer architecture, introduced in a 2017 paper that "
        "famously declared attention all that one needs, has since absorbed "
        "almost every adjacent idea worth keeping: rotary position "
        "embeddings, mixture-of-experts routing, grouped-query attention, "
        "FlashAttention kernels, and a dozen variants of normalization. "
        "What has not changed is the basic shape of the computational graph."
    ),
    (
        "Quantization is the practice of replacing high-precision weights "
        "and activations with lower-bit approximations chosen to preserve "
        "model behavior on representative inputs. The art is in the "
        "calibration set, the per-channel versus per-tensor scale choice, "
        "and the placement of mixed-precision operators around layers that "
        "are unusually sensitive, such as embedding lookups and final logits."
    ),
    (
        "Operating system schedulers have grown progressively more "
        "interested in the workload above them. Linux's completely-fair "
        "scheduler defers to cgroups, energy-aware policies on phones favor "
        "big.LITTLE migration, and macOS's quality-of-service classes let "
        "applications hint which threads are user-facing. Each addition is "
        "an admission that fairness alone does not capture user intent."
    ),
    (
        "Benchmarking is harder than it looks because the act of measuring "
        "perturbs the system being measured. Cold caches inflate first-run "
        "numbers, thermal limits compress sustained ones, frequency boost "
        "rewards short bursts at the expense of long ones, and background "
        "noise on shared machines adds variance that swamps the effect of "
        "interest. Honest benchmarks publish their methodology in detail."
    ),
]


# ---------------------------------------------------------------------------
# Reserved for the agent-trace workload (separate agent fills this in).
# ---------------------------------------------------------------------------

AGENT_TRACE_PRELUDE: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_prompt_of_chars(target_chars: int) -> str:
    """Concat paragraphs from `LONG_PARAGRAPH_POOL` (cycling) until length >= target.

    Joined with double newlines so the result reads like a stack of paragraphs.
    Returns a string of length >= `target_chars` (typically a few hundred over).
    """
    if target_chars <= 0:
        return ""
    if not LONG_PARAGRAPH_POOL:
        raise RuntimeError("LONG_PARAGRAPH_POOL is empty")

    parts: list[str] = []
    total = 0
    sep = "\n\n"
    i = 0
    while total < target_chars:
        para = LONG_PARAGRAPH_POOL[i % len(LONG_PARAGRAPH_POOL)]
        parts.append(para)
        total += len(para) + (len(sep) if len(parts) > 1 else 0)
        i += 1
    return sep.join(parts)


def approx_tokens_for_chars(n_chars: int) -> int:
    """Heuristic: characters / 4. Used for sizing prompts, not for reporting."""
    return max(1, n_chars // CHARS_PER_TOKEN)


__all__ = [
    "CHARS_PER_TOKEN",
    "SHORT_USER_MSG",
    "MEDIUM_USER_MSG_BASE",
    "LONG_PARAGRAPH_POOL",
    "AGENT_TRACE_PRELUDE",
    "build_prompt_of_chars",
    "approx_tokens_for_chars",
]


if __name__ == "__main__":
    print(
        f"SHORT_USER_MSG: {len(SHORT_USER_MSG)} chars (~{approx_tokens_for_chars(len(SHORT_USER_MSG))} tokens)"
    )
    print(
        f"MEDIUM_USER_MSG_BASE: {len(MEDIUM_USER_MSG_BASE)} chars (~{approx_tokens_for_chars(len(MEDIUM_USER_MSG_BASE))} tokens)"
    )
    print(f"LONG_PARAGRAPH_POOL: {len(LONG_PARAGRAPH_POOL)} paragraphs")
    for i, p in enumerate(LONG_PARAGRAPH_POOL):
        print(f"  [{i}] {len(p)} chars")
    sample = build_prompt_of_chars(4000)
    print(f"build_prompt_of_chars(4000) -> {len(sample)} chars")
