# llm-speed

> Benchmark any LLM on any hardware. CLI for the [llm-speed.com](https://llm-speed.com) flywheel.

A reproducible benchmark client for local + hosted LLM inference. One install, one command, one signed result.

```sh
pipx install https://llm-speed.com/dist/llm_speed-0.0.1-py3-none-any.whl && llm-speed bench
```

Or once published to PyPI:

```sh
pipx install llm-speed && llm-speed bench
```

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
