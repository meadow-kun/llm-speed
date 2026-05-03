## TL;DR

- **What we measure.** Decode tok/s — output tokens per wall-clock second after
  the first token. Prefill tok/s — prompt tokens per wall-clock second before
  generation starts. TTFT — milliseconds between request leaving the client and
  the first token arriving back.
- **How we sign.** Every upload is an Ed25519 signature over a canonicalised
  JSON payload, served as a JWS (RFC 7515) compact token. The 32-byte raw
  public key rides in the protected header so anyone can verify without
  contacting us.
- **What we don't do.** No cherry-picking — every workload result is uploaded.
  No retry-and-keep-best — each `(workload, model, backend)` runs once per
  invocation. No hidden outliers — runs more than 3σ from their cluster are
  flagged on the page, not deleted.
- **What you can audit.** `https://llm-speed.com/r/<id>` for any signed run,
  the exact bytes the CLI signs via `llm-speed bench --dry-run --print-payload`,
  and this document at
  [`docs/METHODOLOGY.md`](https://github.com/meadow-kun/llm-speed/blob/main/docs/METHODOLOGY.md).

---

## 1. Workload suite

Every benchmark run executes a fixed sequence of workloads. The suite is
versioned (`suite-v1`); each result is pinned to the exact version that
produced it. New suites get a new version; old runs stay valid and filterable.

### W1 — `chat-short`

A short prompt followed by a short completion. Baseline number for any model.

| Field | Value |
|---|---|
| Prompt | 128 tokens (a fixed natural-language question, byte-identical on every machine) |
| Output | 256 tokens |
| Batch | 1 |
| Reports | TTFT (ms), prefill tok/s, decode tok/s, p50 / p95 per-token decode latency |

### W2 — `chat-long`

A 4 096-token prompt and a 1 024-token completion. Surfaces prefill scaling.

| Field | Value |
|---|---|
| Prompt | 4 096 tokens (fixture paragraphs concatenated to length, byte-identical across machines) |
| Output | 1 024 tokens |
| Batch | 1 |
| Reports | TTFT, prefill tok/s, decode tok/s, p50 / p95 latency |

### W3 — `long-context-decay`

The same model evaluated at 32 k, 64 k, and 128 k input tokens (where the
backend supports each tier), each producing a 256-token completion. Captures
how prefill and decode degrade with context length.

- Reports: an array of `{ context_tokens, prefill_tps, decode_tps, ttft_ms }`.
- Opt-in (skipped by default — slow on small rigs and exceeds memory on
  short-VRAM hardware). Run with `--workload long-context-decay`.

### W4 — `concurrent-decode`

Multiple simultaneous decode streams at batch sizes 1, 4, 8, 16. Measures the
trade-off between aggregate throughput and per-stream latency.

- Each stream: 1 024-token prompt → 256-token completion.
- Reports: per-batch `{ batch, aggregate_tps, per_stream_tps, p50_ms, p95_ms }`.
- Backends without true concurrent decoding (MLX runs serially in-process)
  are flagged on the result so the number is not misread as parallel
  throughput.

### W5 — `agent-trace`

A canned multi-turn coding-agent trace with realistic tool calls (read-file,
write-file, run-tests, fix, repeat). The trace is fixed bytes — every machine
sees identical context at every turn — so timings are comparable. Context
grows to roughly 16 k tokens by the final turn.

- Reports: per-turn TTFT and decode tok/s, end-to-end wall time, and
  prefix-cache hit rate where the backend exposes it.

---

## 2. Hardware fingerprint

Captured automatically on every run. Always coarse — class, not device.

| Field | Example |
|---|---|
| OS | `Darwin 26` (major version only — no patch level) |
| CPU | `Apple M3 Pro`, 12 cores |
| RAM | 40 GB (rounded to nearest 8 GB bucket) |
| GPU | `M3 Pro`, 18-core GPU (no driver build, no PCI bus ID) |
| Backend versions | `mlx 0.31.3`, `llama.cpp 7410`, `cuda 13.0` |
| Accelerator summary | `M3 Pro (18-core GPU) + 36 GB unified` |

The `fingerprint_hash` is a SHA-256 over only the bucketed fields above. Two
machines with identical hardware produce identical hashes — a class
identifier, not a device identifier. In `--strict-anon` mode the hash is
omitted entirely.

The exact list of collected and excluded fields is documented in
[`docs/PRIVACY.md`](/privacy).

---

## 3. Captured signals per workload result

```json
{
  "workload": "chat-short",
  "ttft_ms": 142.3,
  "prefill_tps": 8421.1,
  "decode_tps": 187.4,
  "decode_p50_latency_ms": 5.3,
  "decode_p95_latency_ms": 7.1,
  "prompt_tokens": 128,
  "output_tokens": 256,
  "wall_ms": 1530,
  "prefix_cache_hit_rate": null,
  "backend": "llama.cpp",
  "backend_version": "7410",
  "model": {
    "identifier": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    "name": "Qwen2.5-7B-Instruct",
    "size": "7B",
    "quant": "Q4_K_M",
    "digest": "<sha256 prefix of the weights file>"
  }
}
```

`model.digest` pins the exact weights — two `Llama 3.3 70B Q4_K_M` results
from different sources are not blended unless the digests match.

Per-token decode timings (`raw_timings_ms`) are kept locally for replay and
dispute. They are not uploaded by default.

---

## 4. Supported backends

`llm-speed` autodetects each backend via a thin driver module. A missing
dependency means the driver is skipped, not crashed.

| Backend | Detection |
|---|---|
| `llama.cpp` | `llama-server` or `llama-cli` on PATH |
| `ollama` | daemon reachable on `localhost:11434` |
| `mlx` | Apple Silicon + `mlx-lm` installed |
| `vllm` | running HTTP server, or `vllm` Python package |
| `exllamav2` | CUDA + `exllamav2` Python package |
| `hosted-api` | provider API key in environment: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY` (OpenAI-compatible HTTP, plus a direct Anthropic adapter) |

`TensorRT-LLM` and `SGLang` are not yet supported.

---

## 5. Signing and verification

Every submission is signed. The CLI generates an Ed25519 keypair on first run
(`~/.config/llm-speed/keys/ed25519.key`, mode 0600). Each upload is a JWS
compact token whose payload is the canonicalised result JSON and whose
protected header contains the 32-byte raw public key. The server rejects any
submission whose signature does not verify against the embedded key.

The same canonicalisation runs on the client and on the server, so the bytes
the CLI hashes are exactly the bytes the server hashes — modulo the
`signature` and `public_key` fields, which are excluded from the canonical
form.

Inspect the bytes that would be signed before any upload:

```
llm-speed bench --quick --dry-run --print-payload
```

`--dry-run` runs the workloads but skips upload and local save.
`--print-payload` writes the exact JSON payload to stdout.

---

## 6. What we don't do

- **No cherry-picking.** Every workload result produced by a single
  `llm-speed bench` invocation is uploaded together. The CLI cannot drop
  individual workloads from a submission; the only choice is "submit all of
  this run, or none of it."
- **No retry-and-keep-best.** Each `(workload, model, backend)` triple runs
  exactly once per invocation. To produce multiple measurements, run the CLI
  multiple times — every invocation is a new signed submission, visible on
  the leaderboard, with its own permalink.
- **No hidden outliers.** Server-side outlier detection flags runs that fall
  more than 3σ from their `(model × hardware × backend)` cluster. Flagged
  runs stay on their permalinks and on the per-model and per-hardware pages,
  marked for community review. The dispute thread is public.
- **No silent backend drift.** `backend_version` is captured per result, so a
  number does not change meaning when llama.cpp or vLLM ships a new release —
  it produces a new row pinned to the new version.
- **No quality scoring.** Whether a model produces coherent output is
  orthogonal to how fast it produces it. Quality benchmarks exist
  (LMSYS Chatbot Arena, GDPval, MMLU-Pro, HumanEval); we do not replicate
  them.
- **No power efficiency yet.** `tok/s/W` is not measured. Coming when the
  per-platform power-meter integration lands.
- **No multi-node throughput.** Single-node only.

---

## 7. Reproducibility

Anyone reading this page should be able to do three things.

### 7.1 Verify the published wheel matches the open-source CLI

```
llm-speed verify
```

`verify` prints the SHA-256 of the installed CLI's pinned dependency tree, the
git commit it was built from, and the URL of the wheel. Compare against the
GitHub release asset to confirm the wheel and the source agree.

### 7.2 Re-run the same workload against any signed run

Every `/r/<id>` page lists `suite_version`, `backend`, `backend_version`,
`model.identifier`, `model.quant`, and `model.digest`. Reproduce the run with:

```
llm-speed bench \
  --suite suite-v1 \
  --workload chat-short \
  --backend mlx \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

On hardware in the same `accelerator_summary` class, your numbers should land
within roughly 5 % of the original on `decode_tps` and within roughly 10 % on
`ttft_ms` — the larger TTFT band reflects thermal state, background load, and
KV-cache warmth, none of which the suite controls.

### 7.3 Audit any specific run

Click the `/r/<id>` permalink. The page shows the full fingerprint, the
public key, the signature, the suite version, the backend version, and every
captured signal per workload. The exact JSON the server stores is at
`https://api.llm-speed.com/v1/results/<id>`.

To check a signature locally:

```
llm-speed verify-run https://llm-speed.com/r/<id>
```

This fetches the JSON, recomputes the canonical form, and verifies the
Ed25519 signature against the embedded public key. Exit code is 0 on a valid
signature, non-zero otherwise.

---

## 8. Signed runs you can audit

The following five permalinks are the highest decode tok/s currently on the
leaderboard, as of 2026-05-01. Each is a real, signed, unedited run; the
links resolve to the canonical run page with the full fingerprint, signature,
and per-workload signals.

| # | Decode tok/s | Model | Hardware | Backend | Run |
|---:|---:|---|---|---|---|
| 1 | 432.2 | meta-llama/llama-4-scout | M3 Pro (18-core GPU) + 36 GB unified | hosted-api (OpenRouter) | [r_g4ph40ae0e1](/r/r_g4ph40ae0e1) |
| 2 | 286.5 | mlx-community/Qwen2.5-0.5B-Instruct-4bit | M3 Pro (18-core GPU) + 36 GB unified | mlx 0.31.3 | [r_akcbpx5vcqa](/r/r_akcbpx5vcqa) |
| 3 | 192.5 | mlx-community/stable-code-instruct-3b-4bit | M3 Ultra (60-core GPU) + 96 GB unified | mlx 0.31.3 | [r_y2_5y8oo97d](/r/r_y2_5y8oo97d) |
| 4 | 175.4 | qwen/qwen3-coder-next | M3 Pro (18-core GPU) + 36 GB unified | hosted-api (OpenRouter) | [r_zrnrkjvkvvi](/r/r_zrnrkjvkvvi) |
| 5 | 168.3 | mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit | M3 Ultra (60-core GPU) + 96 GB unified | mlx 0.31.3 | [r_l_v1-zq_qaz](/r/r_l_v1-zq_qaz) |

The fastest signed run as of 2026-05-01 is 432.2 decode tok/s for
`meta-llama/llama-4-scout` via OpenRouter on the `chat-short` workload —
[r_g4ph40ae0e1](/r/r_g4ph40ae0e1). The hosted-API number reflects upstream
provider speed (OpenRouter routes Llama-4-Scout to wafer-scale serving), not
local hardware. The fastest local-hardware run is 286.5 decode tok/s for
`Qwen2.5-0.5B-Instruct-4bit` on M3 Pro via MLX —
[r_akcbpx5vcqa](/r/r_akcbpx5vcqa).

The full live ranking is at [`/cheatsheet`](/cheatsheet), recomputed from the
public API on every page render.

---

## 9. How to dispute a number

Every `/r/<id>` page has a stable permalink and a dispute link. Open a GitHub
issue referencing the run ID and the specific number you are challenging. The
submitter is invited to reproduce; if they cannot, the result is withdrawn.
The exchange is public.

If a result is found to violate the methodology (different workload bytes,
modified backend, edited payload), it is removed from the leaderboard and the
run page returns a 404 with the message: "The result was withdrawn during a
dispute."

---

## 10. Versioning

The current methodology is `suite-v1`. Every result records its
`suite_version`. When the suite changes — new prompts, new workloads,
different output lengths — the version increments. Old results stay valid and
filterable; they are not invalidated by a new version, only distinguishable
from it.

The CLI ships pinned to one suite version per release; a release does not
silently change which workloads ran on which result.

---

## 11. Open source

The full client, the API, and this methodology are in the public repository:

- Workload definitions: [`cli/workloads/`](https://github.com/meadow-kun/llm-speed/blob/main/cli/workloads)
- Signing and canonicalisation: [`cli/signing.py`](https://github.com/meadow-kun/llm-speed/blob/main/cli/signing.py)
- Server-side ingest is closed-source. Signatures are verifiable
  client-side against the embedded public key (RFC 7515 / EdDSA).
- This document: [`docs/METHODOLOGY.md`](https://github.com/meadow-kun/llm-speed/blob/main/docs/METHODOLOGY.md)

File an issue if anything in this document does not match the code.
