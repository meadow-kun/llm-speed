# `llm-speed` CLI — Requirements & Design

## Goal

A single-command, reproducible benchmark harness that:
1. Detects the user's hardware and software environment.
2. Runs a standardized workload suite.
3. Captures every signal needed for a comparable, dispute-resistant result.
4. Uploads to llm-speed.com with provenance, or runs offline.

Adoption is the moat. Every pipx install is a vote that our methodology is the canonical one. So the CLI must be **trivial to install, trivial to run, and produce results people share**.

## Install / UX targets

- `pipx install llm-speed` (also `uv tool install llm-speed`) — one command.
- `llm-speed bench` with no args runs the default suite against an auto-detected backend + first-available model.
- First-time run completes in **< 2 minutes** on a 4090-class GPU. (Slower on CPU is fine.)
- `llm-speed bench --quick` runs in < 30s for a smoke test.
- Standard suite produces a shareable URL: `https://llm-speed.com/r/<id>`.
- Offline mode: `llm-speed bench --no-upload --json out.json`.

## Supported backends (Phase 1)

| Backend | Detection | Workload runner |
|---|---|---|
| `llama.cpp` (incl. `llama-server`) | binary in PATH or `--llama-cpp-path` | direct subprocess + HTTP |
| `ollama` | `ollama list` succeeds | HTTP `localhost:11434` |
| `vLLM` | python import or running server | OpenAI-compatible HTTP |
| `MLX` (Apple) | `mlx-lm` importable + Apple Silicon | python invocation |
| `exllamav2` | python import + CUDA | python invocation |
| Hosted APIs | env-var keys present | OpenAI-compatible HTTP (OpenRouter / Together / Fireworks / Groq / direct OpenAI/Anthropic adapters) |

`TensorRT-LLM` and `SGLang` deferred to Phase 2 — they're enterprise-flavored and AA-AgentPerf already partially covers them.

## Hardware & environment fingerprint

Captured automatically every run; uploaded as opaque metadata so disputes can verify.

**GPU:** `nvidia-smi --query-gpu=name,driver_version,vbios_version,pci.bus_id,memory.total,power.limit --format=csv` (NVIDIA), `rocm-smi --showproductname --showdriverversion` (AMD), `system_profiler SPDisplaysDataType` (Apple).

**CPU:** model, cores, clock, NUMA from `/proc/cpuinfo` (Linux), `sysctl -a hw` (Mac), `wmic cpu` (Windows).

**RAM:** total + available, channels where detectable.

**OS:** kernel/build, distro, governor (Linux), thermal state (Mac via `pmset`).

**Backend:** version (e.g. `llama.cpp` git sha or release tag), build flags where retrievable (CUDA / Metal / ROCm / CPU), driver/runtime versions (CUDA 13.x, ROCm 6.x, Metal version).

**Quant:** quant scheme + GGUF/safetensors digest (SHA-256 of the model file truncated). Pinning a specific quantization is **mandatory** — same model name at Q4_K_M vs IQ3_XXS gives wildly different numbers.

**Power state:** AC vs battery (laptops), thermal throttle status before/after run.

## Workload suite

Five workloads, each with fixed prompts/contexts so numbers compare across users.

### W1 — `chat-short` (baseline)
- 128-token prompt → 256-token output
- batch=1
- Reports: TTFT, prefill tok/s, decode tok/s.

### W2 — `chat-long`
- 4k-token prompt → 1k-token output
- batch=1
- Reports: TTFT, prefill tok/s, decode tok/s. Tests prefill scaling.

### W3 — `long-context-decay`
- 32k / 64k / 128k input contexts → 256-token output (skipped if model max < context)
- Reports: prefill tok/s and decode tok/s curve as function of context length. Catches KV-cache and attention scaling differences across backends.

### W4 — `concurrent-decode`
- batch sizes 1, 4, 8, 16
- 1k input → 256 output each
- Reports: aggregate tok/s and per-stream tok/s vs batch size. Critical for hosted-API comparison and for self-hosted multi-user setups.

### W5 — `agent-trace` (the differentiator)
- A canned multi-turn tool-call trace mimicking a real coding agent: read-file → analyze → write-file → run-tests → fix → repeat. ~10 turns, context grows to 16k.
- Each turn is a real prompt/response cycle with prefix-cache reuse.
- Reports: end-to-end wall-clock, time-per-turn p50/p95, prefix cache hit rate (where backend exposes it), total tokens generated.
- This is the workload **nobody else benchmarks** and the one that matches how Qwen3-Coder-Next-class models are actually used.

## Captured signals (per workload run)

```
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
  "prefix_cache_hit_rate": 0.0,
  "raw_timings": [...]   // per-token decode times for replay
}
```

`raw_timings` is bulky but disposable — kept locally, only the percentile summary uploads by default. Available for disputes if requested.

## Anti-gaming / data-trust mechanisms

1. **Run signature.** CLI signs uploads with an ephemeral keypair tied to the HW fingerprint hash. Resubmitting the same fingerprint with wildly different numbers triggers review.
2. **Power/thermal state checks.** Throttling detected mid-run → result flagged, not rejected (we want honest data, not perfect data).
3. **Model digest verification.** SHA-256 of the GGUF/safetensors prefix is mandatory — prevents "I ran Llama 70B" results that were actually a 7B.
4. **Outlier flagging.** Server-side: results >3σ from the (model × backend × HW) cluster get auto-flagged for community review.
5. **Public dispute thread per result.** Anyone can challenge; submitter can reproduce or withdraw.
6. **No editing.** Once submitted, a result is immutable. Re-runs create new entries. Trends are visible.

## Privacy

- HW fingerprint is hashed; raw serial numbers never leave the machine.
- No prompts/outputs uploaded by default — the workload prompts are part of the suite, not user data.
- Optional anon mode: hashed fingerprint, no IP logging server-side.
- Opt-in to claim a profile (link results to a GitHub username for the contributor leaderboard).

## Portability strategy (without sacrificing signals)

The tension: cross-platform support pulls toward a thin abstraction layer (Llamafile's bet); deep signal capture pulls toward backend-specific code. Resolution: **drivers, not abstraction.**

- Each backend is a thin "driver" module implementing a small interface: `detect()`, `run_workload(workload, model_ref) -> WorkloadResult`. ~150–250 lines each.
- Drivers can call backend-specific APIs and parse backend-specific telemetry — they're allowed to be different. The harness only requires they fill the common `WorkloadResult` shape.
- Driver metadata (`raw` JSON blob) carries backend-specific extras (e.g. vLLM's prefix-cache stats, llama.cpp's KV cache layout) without forcing the schema to know about them.
- The CLI itself is pure Python with stdlib + a small dependency set (`httpx`, `pydantic`, `psutil`, `rich`). No PyTorch or CUDA bindings in the core — those only get imported when the relevant driver is selected.
- Fallback chain: if an optional driver dep is missing, that driver is skipped with a clear message ("ollama not detected; skip"); the CLI keeps running on whatever's available.
- Single binary distribution path (Phase 1.5): `pyoxidizer` or `shiv` to produce a no-Python-needed executable for users who don't have Python — but `pipx`/`uv` is the primary install path because most local-LLM users already have Python.

## CLI surface

```
llm-speed bench [--backend BACKEND] [--model MODEL_REF] [--workload W1,W2,...]
                [--quick] [--no-upload] [--json PATH] [--api-key KEY] [--anon]

llm-speed detect              # print fingerprint + available backends
llm-speed list-models         # list locally available models per backend
llm-speed compare RUN_ID...   # local diff of saved runs
llm-speed login               # GitHub OAuth, claim contributor profile
llm-speed self-update         # update suite definitions (workloads can change)
```

## Versioning & reproducibility

- The workload suite itself is versioned (`suite-v1`, `suite-v2`...). Submitted results reference the exact suite version. Methodology changes don't invalidate old data.
- CLI bundles an exact pin of suite definitions; `self-update` pulls the latest from a signed manifest.
- Server tracks suite-version in every result so leaderboards can filter to a single version.

## Out of scope (Phase 1)

- GPU power/efficiency measurement (`tok/s/W`) — interesting but adds wall-clock instrumentation complexity. Phase 2.
- Quality regression checks (does this quant produce coherent output?) — orthogonal to speed; defer.
- Distributed multi-node — not the consumer audience.
