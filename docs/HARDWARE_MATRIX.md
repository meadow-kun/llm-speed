# Hardware support matrix

This document describes which (OS × architecture × backend) combinations
`llm-speed` officially supports, what we test on every release, and what we
accept community contributions for.

## Tier definitions

- **Tier 1 — Fully supported.** Both *correctness* (CLI runs, schema is valid,
  signing+upload round-trip works) and *perf numbers* (decode tok/s, TTFT,
  prefill tok/s) are verified before each release. A regression in a Tier-1
  configuration is a release-blocker.
- **Tier 2 — Correctness only.** The CLI is expected to start, fingerprint,
  and produce a valid `WorkloadResult`. Performance numbers are not validated
  by the maintainers — they may be off-trend until a community contributor or
  hardware partner verifies them. A perf regression on Tier 2 is *not* a
  release-blocker.
- **Tier 3 — Best-effort.** No formal support. We accept PRs that fix bugs in
  this configuration and we promise not to actively break it; that's all.

## Supported configurations

| Tier | Configuration | Tested every release |
|---|---|---|
| 1 | macOS arm64 (M1–M5) + MLX + llama.cpp | Manual on maintainer's Mac mini |
| 1 | Linux x86_64 + NVIDIA + llama.cpp / vLLM / ollama / exllamav2 | Maintainer's self-hosted 3090 runner |
| 1 | Linux x86_64 + ollama (CPU) | GH-hosted `ubuntu-latest` |
| 1 | Linux arm64 + llama.cpp CPU | GH-hosted ARM runner |
| 1 | Windows 11 + ollama | GH-hosted `windows-latest` |
| 2 | Linux x86_64 + AMD ROCm | Community contribution |
| 2 | macOS x86_64 (Intel) | Community contribution |
| 2 | Intel Arc + Linux | Community contribution |
| 3 | Anything else (BSD, Android, Nvidia Jetson, Raspberry Pi, etc.) | Community-best-effort |

## How CI maps to the matrix

The `test` and `smoke-cli` jobs in `.github/workflows/ci.yml` exercise:

- `ubuntu-latest` (x86_64) on Python 3.10 / 3.11 / 3.12 — Tier 1 Linux x86_64.
- `macos-latest` (arm64) on Python 3.10 / 3.11 / 3.12 — Tier 1 macOS arm64.
- `windows-latest` on Python 3.10 / 3.11 / 3.12 — Tier 1 Windows 11.

CI runners do not have GPUs or local LLM backends installed, so CI verifies
*correctness* on Tier 1 and Tier 2 platforms. The *perf-numbers* arm of Tier 1
is checked by the maintainer on the self-hosted hardware mentioned above.

## How to add a configuration

If you want a configuration moved to a higher tier:

1. Open an issue describing the platform and which backends you can run.
2. For Tier 2 → Tier 1, you must commit to running the suite on every release
   and reporting back. The most reliable way is to host a GitHub Actions
   self-hosted runner that the `ci.yml` workflow can target.
3. For Tier 3 → Tier 2, a one-off correctness report (output of
   `llm-speed detect` plus a successful `llm-speed bench --quick`) is enough.

## Non-goals

- Datacenter racks (multi-host H100/B200 clusters). Out of scope.
- Mobile devices. Phase 3+, not yet.
- Cross-compilation pipelines. We ship a single Python package; users install
  via `pipx`, `uv`, or `pip`.
