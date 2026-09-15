# llm-speed

> Measure LLM inference speed and explore the results at [llm-speed.com](https://llm-speed.com/).

A reproducible benchmark client for local + hosted LLM inference. One install, one command, one signed result.

## Install

Install the published [PyPI package](https://pypi.org/project/llm-speed/) with
Python 3.10 or later and pipx:

```sh
pipx install llm-speed
llm-speed bench
```

## Explore the results

- [Benchmark leaderboard](https://llm-speed.com/): find model and hardware results,
  then open a source run to inspect its workloads and configuration.
- [Datasets and downloads](https://llm-speed.com/data): focused studies in CSV and
  JSON, plus a separately dated July 8, 2026 bulk snapshot of run summaries.
- [VRAM and Mac memory calculator](https://llm-speed.com/tools/vram-fit): estimate
  model memory requirements, including your installed Mac memory. Estimates are
  separate from measured benchmark results.
- [Methodology](https://llm-speed.com/methodology): understand workloads, signing
  and the limits of comparisons between different setups.

## Original RTX 5090 studies

**[Qwen3.8-27B versus Gemma 4 12B](https://llm-speed.com/blog/qwen3-8-27b-vs-gemma-4-12b-rtx-5090).**
Measured September 5, 2026: 12 workload rows from six source runs, with three
repetitions per model. Compare latency and generation speed with the exact model
files and runtime settings. The models use different quantizations; these speed
results do not establish a coding-quality winner.
[CSV](https://llm-speed.com/data/qwen-gemma-5090-chat-2026-09-05.csv) ·
[JSON and configuration](https://llm-speed.com/data/qwen-gemma-5090-chat-2026-09-05.json).

**[Gemma 4: how input length changes the wait for the first token](https://llm-speed.com/m/gemma-4-12b-it-qat#long-context).**
Measured September 6, 2026: three repetitions at each of 23,872, 47,652 and 95,212
actual input tokens, using Q4_0 on one RTX 5090. The study separates first-token
latency from generation speed. It does not establish maximum supported context
or answer quality.
[CSV](https://llm-speed.com/data/gemma-4-5090-context-2026-09-06.csv) ·
[JSON and configuration](https://llm-speed.com/data/gemma-4-5090-context-2026-09-06.json) ·
[Reusable chart](https://llm-speed.com/data/gemma-4-5090-context-2026-09-06-figure.png) ·
[Citation and limitations](https://llm-speed.com/data/gemma-4-5090-context-2026-09-06-citation.txt).

Data and the chart are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
When reusing them, credit llm-speed, link to the specific study and license, and
identify any changes. Retain the measurement date and configuration; the July
bulk archive does not contain these September studies.

## What it does

- **Auto-detects** your installed backends: llama.cpp, Ollama, vLLM, MLX, exllamav2, plus any OpenAI-compatible hosted API.
- **Standardised workloads** — chat-short, chat-long, agent-trace, prefill-stress, long-context decay. Same protocol on every machine.
- **Captures** decode tok/s, prefill tok/s, TTFT, p50/p95 latency, plus a hardware fingerprint (bucketed; see [PRIVACY.md](docs/PRIVACY.md)).
- **Signs every run** with an Ed25519 keypair on your machine (JWS / RFC 7515). The public key rides in the JWS header so anyone can verify the result without contacting us.
- **Uploads** the signed result to [llm-speed.com](https://llm-speed.com) where it joins the public leaderboard. You can pass `--anon`, `--strict-anon`, or `--no-upload` to control identity / network behaviour.

## Why use this

- The output is **reproducible**. Run the same `llm-speed bench` on someone else's rig and you can compare numbers directly — same workload definitions, same prefill/decode breakdown, same timing window.
- The output is **auditable**. Every signed run has a permalink at `https://llm-speed.com/r/<id>` showing the exact bytes the CLI signed.
- The output is **honest**. No retry-and-keep-best, no cherry-picking, no hidden outliers. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Privacy

Every field that leaves your machine is enumerated in [docs/PRIVACY.md](docs/PRIVACY.md). Hardware fingerprint is bucketed (RAM rounded to 8 GB, OS to major version). No PCI bus IDs, driver build numbers, hostname, username, or prompt/output text. EU readers: see PRIVACY §6a (GDPR).

To preview exactly what the CLI would upload, without uploading:

```sh
llm-speed bench --quick --dry-run --print-payload
```

## Verifying the binary

```sh
llm-speed verify
```

The CLI computes the SHA-256 of its own wheel and compares against a sidecar published from BOTH `llm-speed.com/dist/` and the matching [GitHub Releases page](https://github.com/meadow-kun/llm-speed/releases). Disagreement triggers a hard-fail "do not trust this binary" verdict — closes the single-CDN compromise vector.

## What's audited

- [docs/security/cli_pii_audit_2026-05-01.md](docs/security/cli_pii_audit_2026-05-01.md) — PII leak audit
- [docs/security/cli_oss_ready_2026-05-01.md](docs/security/cli_oss_ready_2026-05-01.md) — open-source readiness
- [docs/security/pentest_2026-05-01.md](docs/security/pentest_2026-05-01.md) — penetration test
- [docs/security/trust_chain_v2.md](docs/security/trust_chain_v2.md) — supply-chain trust posture

## Reporting a security issue

See [SECURITY.md](SECURITY.md) — `privacy@llm-speed.com`, 90-day disclosure timeline, no legal action against good-faith research.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Maintainer

`meadow-kun` on GitHub. Single-maintainer open-source project. No company. Issues + DMs welcome.
