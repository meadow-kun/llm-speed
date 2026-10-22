# CLI changelog

## 0.0.9 — 2026-10-22

[Published release](https://github.com/meadow-kun/llm-speed/releases/tag/v0.0.9) · [PyPI](https://pypi.org/project/llm-speed/0.0.9/)

- **`llm-speed agent`**: runner mode: a machine takes preset jobs from a queue. `pipx install -U llm-speed && llm-speed agent install` pairs the machine (prints a code for the queue admin to approve), asks for consent once, stores the per-machine token (macOS Keychain / 0600 file) and installs a launchd / systemd user service. The service claims jobs from the queue (`ops/AGENT_QUEUE_API.md`; default `https://api.llm-speed.com`, or a self-hosted one via the documented `--api-base URL`, remembered at enroll; the token only goes to that server), runs only reviewed packaged presets whose `(name, version, sha256)` matches the install, through the normal `llm-speed bench --preset` path, always `--strict-anon`, and reports status + fit_attempts rows. Gates: pause file, quiet hours, busy GPU (other processes' VRAM / util), low unified memory, low disk. `pause [--now]`, `resume`, `status`, `presets`, `approve`/`decline`, `config`, `uninstall [--forget]`. Built-in `selftest` preset. Design: `docs/cli_v2_design.md` part 2.
- **Shared preset schema** `cli/schemas/llm-speed-preset-1.schema.json` (CLI + Worker); `cli/preset_schema.py` validates packaged presets with it.
- Phase 2 signed presets (pinned Worker key `llm-speed-presets-2026-10`, selected by the envelope `key_id`; canonical bundle bytes required; signed revocations, monotonic versions) behind the `remote_presets` agent setting, off by default.
- Agent matches GM's live queue contract: pairing 202 pending / 200 token once / 410 used / 404 unknown (stops at once); `partial` results when a crash or the runtime cap comes after some measured cells; claims with `attempt` outside 1-3 are rejected.
- `llm-speed bench --preset` no longer tries `POST /v1/fit-attempts` (the route does not exist); fit_attempts rows stay local unless an agent job reports them.
- No team-only switches: `--api-base` is visible and documented (docs/CLI.md), `cost_class` is accepted and ignored (classification is server-side), comments use neutral wording, agent env vars documented.
- `worker/migrations/0010_fit_attempts_flint.sql`: the Flint fit_attempts columns (ttfa_ms, n_cpu_moe, context_requested, ...) and `n_cpu_moe` in the dedupe key (table rebuild, ids kept, re-run refused). The queue's result route accepts them (applied 2026-10-22), so agent job results now include them (`SEND_FLINT_FIT_COLUMNS`).

## 0.0.8 — 2026-10-22

[Published release](https://github.com/meadow-kun/llm-speed/releases/tag/v0.0.8) · [PyPI](https://pypi.org/project/llm-speed/0.0.8/)

Launch-readiness for day-one runs of large MoE models (first target: Recursive Flint 40B-A2.5B on RTX 3090/4090/5090 and Apple Silicon).

- **Load and out-of-memory failures are results, not crashes.** Each failed workload carries `extras.failure_kind` (`oom`, `unsupported_arch`, `load_failed`, `load_timeout`, `server_unreachable`, `runtime_error`). The llama-server log is kept locally (`~/.cache/llm-speed/logs/llama-server-last.log`, never uploaded) and the matching line (e.g. `cudaMalloc failed: out of memory`) goes into the error. After a load failure the remaining workloads are recorded as `skipped_after_load_failure` instead of re-loading the model each time. Failure-only runs still stay local unless you pass the new `--upload-failures`.
- **Peak memory per workload:** `extras.peak_vram_mib` (rounded to 16 MiB) from `nvidia-smi` device-wide sampling on NVIDIA, or MLX's own peak allocation on Apple Silicon; `extras.peak_vram_source` says which.
- **Time to first answer:** `extras.time_to_first_answer_ms` (first non-reasoning token), `extras.reasoning_tokens_est` and `extras.answer_started` for single-stream workloads. `ttft_ms` is unchanged (first token of any kind).
- **llama-server passthrough:** `--llama-args='-c 4096 -ngl 999 --n-cpu-moe 20'` (or `$LLM_SPEED_LLAMA_ARGS`) is applied when the CLI starts llama-server and recorded as a declared label when reusing `LLAMA_CPP_SERVER_URL`. Only allowlisted flags and values are uploaded (`extras.llama_server_args`); other flags and any path-like value are redacted.
- **Exact build and file labels:** `extras.llama_cpp_build` (e.g. `b11803`, null for shallow-clone builds that report build number 1), `extras.llama_cpp_commit`, `extras.llama_cpp_build_source` (server `/props` preferred over local `--version`), `extras.model_file` (GGUF basename) and `extras.model_file_bytes`. Build numbers of any length parse (b11803+). NVFP4 and MXFP4 GGUFs now get a quant label.
- **MLX:** fails fast with `unsupported_arch` when the installed mlx-lm has no module for the model's `model_type` (Flint needs mlx-lm with `models/flint.py`, PR ml-explore/mlx-lm#1998, unmerged as of 2026-10-22).
- **`llm-speed bench --preset flint-40b`:** one command per machine. Detects NVIDIA vs Apple Silicon, picks the quants that fit its memory (others are recorded as skipped, a 16 GB card gets a `--n-cpu-moe` offload probe), finds llama-server (PATH, Homebrew, `~/llama.cpp*/build/bin`) and requires build b11803 or newer, printing the exact install command and asking y/N before running it (no sudo, nothing silent). Checks free disk before each download, downloads from Hugging Face with progress and resume (reuses partial files and the HF cache), runs 4K, 16K and 32K context (stops climbing at the first failure), records every OOM / failed load / skip, prints a summary table, then uploads through the usual consent flow (`--strict-anon` honoured). Rerunning resumes; Ctrl-C stops the servers and keeps finished steps. A model is deleted after its steps only if the preset downloaded it and the next model needs the space. `--with-control` adds the Qwen3.8-27B control. Model repos, pinned revisions, sizes and per-machine context ladders (up to 262K on the RTX 5090 and 64 GB+ Macs, MLX long-context decay at 32K and 131K, disk-gated Q8_0 on Macs) live in `cli/presets/flint-40b.json` (versioned). `llm-speed preset hash <name>` prints the sha256 of the preset's canonical serialization for review-then-publish; `llm-speed preset show|list` too.
- `long-context-decay` honours `$LLM_SPEED_DECAY_CTX` (e.g. `131072`) to measure only those context points.
- **fit_attempts records:** the preset (and the shared fit sweep, `cli/fit_sweep.py`, moved here from Launch Lab's tool) writes one row per attempt in the shape of the site's `fit_attempts` table plus Flint columns (time to first answer, reasoning tokens, `n_cpu_moe`, driver major, llama.cpp build/commit, GGUF basename, baseline memory). Saved locally as `fit_attempts.json`; uploaded to `/v1/fit-attempts` once the site accepts them (until then the CLI says so and keeps them local). Every string passes the path sanitizer.
- **Recorded context = the context that ran.** Each fit step records the llama-server's real `n_ctx` (from `/props`) or, on MLX, the tokens actually processed; the planned step is kept as `context_requested`. A workload that would not fit a step is not run in it (chat-long needs ~5.4K tokens, so it never runs in a 4K step), and if the server reports a smaller `n_ctx` than requested the workloads are re-picked for it. The preset's MLX steps are now 4K chat-short, 8K chat-long, and decay at 32,000 and 128,000 tokens.
- **Upload hard guard.** Every upload path refuses, with the reason, when `LLM_SPEED_OFFLINE=1` is set, or when the environment is synthetic (a test runner, `MOCK_*` variables, a mock/fixture `nvidia-smi` or `llama-server`, or `LLM_SPEED_SYNTHETIC=1`) and the target is not a reserved test host (localhost, `*.test`, `*.example`, `*.invalid`). `--resume` honours `--no-upload`. Fit-sweep records from such environments carry `"synthetic": "<reason>"` and the preset will not upload them.
- `--resume` with `--strict-anon` now sends no User-Agent, like a fresh strict-anon upload.
- llama-server boot timeout raised from 60 s to 600 s (`LLM_SPEED_LLAMA_BOOT_TIMEOUT_S`); `--log-disable` is no longer passed so failures can be explained.

Consent, `--strict-anon`, `--anon`, local-save and `--no-upload` behavior are unchanged. New uploaded fields are listed in `cli/launch_telemetry.py` `UPLOADED_EXTRAS_KEYS` and docs/PRIVACY.md.

## 0.0.7 — 2026-09-23

[Published release](https://github.com/meadow-kun/llm-speed/releases/tag/v0.0.7) · [PyPI](https://pypi.org/project/llm-speed/0.0.7/)

- Benchmark output highlights measured decode throughput and the suite version.
- Per-workload results show token, batch, and context settings.
- Successful uploads show a clear link for viewing and sharing the result.
- Offline runs retain local-save and resume instructions.
- The public source mirror now matches the package version. A tested Homebrew formula is available through `brew install meadow-kun/tap/llm-speed`.

Benchmark execution, upload consent, strict anonymity, offline behavior, and the measurement payload are unchanged. No automatic update checks or telemetry were added.

Release assets include the same wheel/source bytes as PyPI, a wheel checksum, and a verified Sigstore bundle. Historical release labels are preserved; the old `v1.0.2` label did not match the package version.
