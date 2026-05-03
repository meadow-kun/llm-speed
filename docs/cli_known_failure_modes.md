# CLI known failure modes

Every way the CLI is currently known to break, plus trigger / symptom / fix /
status. Sourced from `docs/sweep_2026-04-29.md` operational issues + the
audit done while writing `docs/cli_test_matrix.md`.

Each entry has a unique short id (KFM-NNN) so issues / commits / docs can
reference it stably even when this file is reordered.

| Status legend |
|---|
| **shipped**   — fix is in main and a test pins the regression |
| **partial**   — fix is in main but no test pins the regression yet |
| **pending**   — known broken, no fix |
| **wontfix**   — out of scope by design |

---

## KFM-001 — Kingston SSD disconnects mid-bench, partial run is captured

- **Status:** partial
- **Trigger:** HF_HOME or model cache lives on a removable drive (USB SSD,
  network mount). The drive disconnects mid-run.
- **Symptom:** Every workload after the disconnect fails with
  `[Errno 13] Permission denied: '/Volumes/KINGSTON'`. The error IS captured
  as a proper run record on the API (no silent failure), but no perf number
  lands. Disk fills 100% if the retry path falls back to the internal cache.
- **Fix:** None shipped. Workaround: re-mount the drive and run
  `bash /tmp/m3pro-retry.sh` with HF_HOME repointed.
- **Where:** any backend that delegates to HuggingFace cache (mlx, llama.cpp
  via HF snapshot, ollama via blob store).
- **Repro:** `unplug Kingston during a 9-cell sweep` (literally what
  happened on 2026-04-29).
- **Action item:** Add a pre-flight check in `cmd_bench` that confirms the
  resolved model path is on a mounted volume before the sweep starts.

## KFM-002 — Reasoning-content tokens were not counted (Qwen3.6 / DeepSeek-R1)

- **Status:** shipped (test pins it)
- **Trigger:** Any reasoning model emits `delta.reasoning_content` instead
  of (or before) `delta.content` in the OpenAI-compatible streaming response.
- **Symptom:** Driver reports 0 tok/s decode for reasoning models. RTX 5090
  Qwen3.6-27B in-place would have shown 0 tps without the patch.
- **Fix:** `cli/drivers/llama_cpp.py` `_run_chat_stream` now sums both
  `content` and `reasoning_content`. Pinned by
  `tests/cli/test_cross_env.py::test_llama_cpp_stream_counts_reasoning_content`.
- **Note:** The hosted-api driver currently does NOT count
  `reasoning_content`. Open question whether OpenRouter / Anthropic emit it
  the same way; not yet confirmed.

## KFM-003 — `mlx-community/CodeLlama-13B-*-MLX` ships old `.npz` format

- **Status:** wontfix (upstream)
- **Trigger:** Any mlx-lm 0.31+ trying to load
  `mlx-community/CodeLlama-13B-Python-4bit-MLX` or `-Instruct-hf-4bit-MLX`.
- **Symptom:** mlx-lm refuses to load the weights — old `.npz` format is no
  longer supported.
- **Fix:** Dropped from the test matrix on 2026-04-29. We don't auto-detect
  this case; the user sees the upstream error.
- **Action item:** Skip these slugs in the curated catalog; tag them as
  "deprecated upstream".

## KFM-004 — OpenRouter `nousresearch/hermes-3-llama-3.1-405b` 404s despite being in /v1/models

- **Status:** wontfix (upstream)
- **Trigger:** Listing /v1/models then querying for hermes-3 405B.
- **Symptom:** 404 from OpenRouter even though the slug is in their /models
  response. Looks like an upstream stale-listing bug.
- **Fix:** Replaced with Llama-4-Scout + Llama-4-Maverick in the matrix.
- **Action item:** Surface the 404 with a clear "this provider lists but
  doesn't serve this model" message, instead of just the raw HTTP error.

## KFM-005 — M3 Pro miniconda env has numpy/sklearn binary conflict

- **Status:** shipped (workaround)
- **Trigger:** miniconda3 Python 3.11 with stale numpy on Apple Silicon
  trying to import transformers for any Llama-architecture model.
- **Symptom:** ImportError chain ending in a numpy ABI mismatch.
- **Fix:** Switched to `uv tool install` with Python 3.12. Recipe at
  `tools/m3ultra-setup.sh`. Documented in CLAUDE.md.
- **Action item:** Add a fingerprint warning when we detect conda + Apple
  Silicon + transformers, prompting the user toward the uv recipe.

## KFM-006 — `--strict-anon` + `--api-key` did NOT drop the bearer (FIXED in test)

- **Status:** shipped (test pins it)
- **Trigger:** User passes `--strict-anon --api-key sk-...` together (e.g.
  in a script that always passes both).
- **Symptom (would have been):** Bearer token leaks to api.llm-speed.com
  despite the user explicitly choosing strict-anon. Privacy regression.
- **Fix:** `cli/upload.py:upload_report` short-circuits the Authorization
  header when `strict_anon or anon` is true. Pinned by
  `tests/cli/test_cross_env.py::test_strict_anon_drops_auth_header`.

## KFM-007 — Non-tty stdin could silently auto-consent

- **Status:** shipped (test pins it)
- **Trigger:** CI runner / `cron` / `ssh some-box llm-speed bench` where
  stdin is not a TTY. Without the TTY check, an empty `readline()` would
  match the `[Y/n]` "empty defaults to yes" rule.
- **Symptom (would have been):** Silent uploads on every CI invocation, no
  consent record.
- **Fix:** `cli/consent.py:prompt_for_consent` checks `sys.stdin.isatty()`
  and refuses to prompt when it's false. Pinned by
  `tests/cli/test_cross_env.py::test_consent_non_tty_refuses_to_auto_consent`.

## KFM-008 — Disk full during offline save could leave a half-written runs file

- **Status:** shipped (test pins it)
- **Trigger:** `~/.cache/llm-speed/runs/` fills up; `Path.write_text` raises
  ENOSPC mid-write.
- **Symptom (would have been):** Half-written JSON file under runs/ that
  later confuses `--resume`.
- **Fix:** `Path.write_text` is atomic-ish in CPython (writes to fd then
  closes), and the path doesn't exist if the write raised. Pinned by
  `tests/cli/test_cross_env.py::test_save_offline_disk_full_no_partial_file`.
- **Caveat:** We do NOT use `os.replace` for the runs file (we DO use it for
  the consent file and key file). A truncate-then-fail mid-write on a
  pre-existing file at the same path could leave a truncated old file.

## KFM-009 — 429 / Retry-After is NOT honored

- **Status:** pending
- **Trigger:** Hosted-api provider returns 429 (rate-limited) during upload
  or during list_models.
- **Symptom:** `_post_with_retries` sees 4xx and raises immediately
  ("server rejected upload"). The retry loop only retries on 5xx and
  TransportError. The user sees a hard fail and might double-submit.
- **Fix:** None. Need to (a) parse Retry-After, (b) sleep for that long,
  (c) retry up to N times, (d) bound the total time.
- **Action item:** Update `cli/upload.py:_post_with_retries` to special-case
  429 the way it handles 5xx, with the Retry-After parse.

## KFM-010 — `LLAMA_CPP_SERVER_URL` external mode skips model existence check

- **Status:** by design
- **Trigger:** Set `LLAMA_CPP_SERVER_URL` to point at an existing server,
  pass any model identifier (e.g. a path that doesn't exist on this host).
- **Symptom:** The driver happily forwards to the external server, which
  may serve a totally different model than what the ModelRef says. The
  uploaded `model.identifier` is wrong.
- **Fix:** None. By design — we trust the operator who set
  LLAMA_CPP_SERVER_URL to know what's loaded. But this is a footgun.
- **Action item:** When LLAMA_CPP_SERVER_URL is set, fetch `/v1/models` from
  the external server and pin the actual served model name into the report's
  backend_extras so the audit trail is preserved.

## KFM-011 — Plain `http://` API base sends bearer over plaintext (warns but proceeds)

- **Status:** partial
- **Trigger:** `--api-base http://...` + an api_key in default mode.
- **Symptom:** `warn_if_insecure_api_base` prints a warning to stderr, but
  the upload still proceeds and the bearer travels in cleartext.
- **Fix:** Warning is shipped; the *block* is not.
- **Action item:** Promote to a hard error unless the user passes
  `--allow-insecure-bearer` (or similar).

## KFM-012 — Autopicker priority order is not pinned by a test

- **Status:** pending
- **Trigger:** Refactor `cmd_bench._BACKEND_PRIORITY` and accidentally swap
  vllm and llama.cpp.
- **Symptom:** Users on multi-backend hosts get a different default backend
  than they did yesterday.
- **Fix:** None — there's no test that asserts "given (X, Y, Z) all
  available, the picker chooses X first".
- **Action item:** Add `tests/cli/test_pick_backend_priority.py` with a
  parameterized test that stubs `detect()` for each backend.

## KFM-013 — Windows native path is completely untested

- **Status:** pending
- **Trigger:** Anyone running `pip install llm-speed; llm-speed bench` on
  Windows native (not WSL2).
- **Symptom:** Unknown. We expect: (a) consent file path with backslashes,
  (b) cmd.exe quoting around the auto-install command strings, (c) keypair
  file lacking 0o600 chmod (Windows).
- **Fix:** None. The auto_install plans don't have a Windows branch — they
  return `available=False` on Windows.
- **Action item:** Run the proposed `ci.yml` on `windows-latest` and
  triage what breaks.

## KFM-014 — Air-gapped / corporate-MITM TLS

- **Status:** wontfix (or special-case)
- **Trigger:** Corporate firewall doing TLS interception with a custom CA.
- **Symptom:** httpx fails cert verification, all uploads fail, no clear
  guidance.
- **Fix:** Document `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` workarounds, but
  don't auto-disable cert verification.

## KFM-015 — Hosted-api ModelRef.extras would have leaked upstream JSON

- **Status:** shipped (test pins it)
- **Trigger:** OpenRouter (and similar) /v1/models returns 10+ KB of pricing,
  description, license metadata per model.
- **Symptom (would have been):** Upload payload's `extras` field carries
  arbitrary upstream JSON, leaking metadata + tripping the 256-char privacy
  invariant on long descriptions.
- **Fix:** `cli/drivers/hosted_api.py:_list_provider_models` builds ModelRef
  with only `provider` + `base_url` in extras. Pinned by
  `tests/cli/test_cross_env.py::test_hosted_api_no_raw_in_modelref_extras`.

## KFM-016 — `assert_no_long_strings` allowlist match is exact (not suffix)

- **Status:** shipped (test pins it)
- **Trigger:** Someone tries to add a long string to the upload payload at
  a path NOT on the allowlist (e.g. `results.extras.error`).
- **Symptom (would have been):** A long error blob slips through if path
  matching is by suffix.
- **Fix:** `_path_is_allowed` uses set membership (exact match). Pinned by
  `tests/cli/test_cross_env.py::test_assert_no_long_strings_path_match_is_exact`.

## KFM-017 — psutil + cryptography wheels lag on Python 3.13 / Windows arm64

- **Status:** pending
- **Trigger:** User runs Python 3.13 or Windows-arm64.
- **Symptom:** `pip install llm-speed` fails to install the deps.
- **Fix:** Pin the supported Python versions in CI and document
  3.13 as "expected to work but not in CI yet".

## KFM-018 — Half-implemented hosted-api: no Retry-After, no per-provider rate-limit awareness

- **Status:** pending
- **Trigger:** Hammering Groq or Anthropic with rapid sequential bench cells.
- **Symptom:** First few work, then 429s, then we hard-fail. The fingerprint
  upload AFTER the bench succeeds (it's a different endpoint) so the run
  record contains an empty results array.
- **Fix:** Same as KFM-009 — implement Retry-After + global rate budgeting.

## KFM-019 — Build-time prerender of `/hw/<slug>` and `/m/<slug>` was empty

- **Status:** shipped (web side, not CLI)
- **Trigger:** Cloudflare Pages prerender exceeds the per-request fetch budget.
- **Symptom:** "/hw/rtx-5090" rendered "No benchmarks" for 12+ hours after
  the launch sweep landed.
- **Fix:** `dynamic = "force-dynamic"` on those routes + filter at summary
  level before hydrating. Documented in `docs/SESSION_HANDOFF_2026-04-29.md`.
- **Why this is in the CLI doc:** because if the CLI uploads succeed but the
  web side doesn't render them, users (especially HN-day commenters)
  conclude the CLI is broken. Worth tracking together.

## KFM-020 — Workloads can return non-WorkloadResult objects (defensive guard exists)

- **Status:** shipped (defensive)
- **Trigger:** A workload's `run()` returns the wrong type (e.g. a tuple).
- **Symptom (caught by guard):** `cmd_bench` substitutes a `_failed_result`
  with the type name in the error message instead of crashing.
- **Fix:** Already in `cli/commands/bench.py`. No test pins it.
- **Action item:** Unit-test the substitution path.

---

## Open ledger — fixes pending vs shipped

| Status   | Count |
|----------|-------|
| shipped  | 8 (KFM-002, 005, 006, 007, 008, 015, 016, 020) |
| partial  | 3 (KFM-001, 009-as-pending, 011) |
| pending  | 5 (KFM-009, 012, 013, 017, 018) |
| wontfix  | 4 (KFM-003, 004, 010, 014) |
| total    | 20 |

The 5 pending entries are the launch-blocker shortlist. KFM-013 (Windows)
has the highest user-impact-per-fix — adding the proposed `ci.yml` flushes
out whatever we don't know.
