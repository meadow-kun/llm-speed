# CLI changelog

## 0.0.7 — 2026-09-23

[Published release](https://github.com/meadow-kun/llm-speed/releases/tag/v0.0.7) · [PyPI](https://pypi.org/project/llm-speed/0.0.7/)

- Benchmark output highlights measured decode throughput and the suite version.
- Per-workload results show token, batch, and context settings.
- Successful uploads show a clear link for viewing and sharing the result.
- Offline runs retain local-save and resume instructions.
- The public source mirror now matches the package version. A tested Homebrew formula is available through `brew install meadow-kun/tap/llm-speed`.

Benchmark execution, upload consent, strict anonymity, offline behavior, and the measurement payload are unchanged. No automatic update checks or telemetry were added.

Release assets include the same wheel/source bytes as PyPI, a wheel checksum, and a verified Sigstore bundle. Historical release labels are preserved; the old `v1.0.2` label did not match the package version.
