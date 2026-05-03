# llm-speed CLI — PII / Doxxing Audit

- **Date:** 2026-05-01
- **Scope:** the `pipx`-installed CLI (`cli/` package), runtime behavior of every subcommand, every backend driver, and every byte that can leave the user's machine.
- **Method:** read-only static review of `cli/`, `tests/test_signing*.py`, traced data-flow from `dataclasses.asdict(result)` in `cli/signing.py:228-232` through every driver's `ModelRef`/`ChatRunOutcome`/`WorkloadResult` constructions.
- **Auditor's stance:** skeptical. The PRIVACY.md contract is the bar; the CLI must not exceed it.

---

## 0. Verdict (one paragraph)

**FAILS the PRIVACY.md contract on three concrete leaks** — all in the per-result payload, all caused by trusting the privacy invariant's 256-char string check to do per-field policy when it only does per-string-length policy. The most serious is **F-1**: `results.model.identifier` is uploaded verbatim and, for the `llama.cpp`, `mlx` (local Models dir), and `exllamav2` drivers, is set to `str(absolute_path)` — i.e. on macOS that's `/Users/<username>/...` and on Linux often `/home/<username>/...`. Username leak. Combined with **F-2** (driver-side `model.extras` carrying additional absolute paths under non-allowlisted but short keys) and **F-3** (`results.error` allowed to be unbounded length and routinely contains raw paths from `RuntimeError(f"model not found: {model_path}")`), this is a launch-blocking discrepancy between the documented contract ("we don't collect hostname / username") and the actual upload payload. The remaining ~50 audit items pass cleanly: fingerprint bucketing, `--strict-anon` semantics, `--anon` bearer-suppression, JWS signing, consent flow, on-disk file modes, no `shell=True` on user-controlled input, no log lines that print the bearer token. The fix surface is small and local to the drivers + one extra trim step in `build_uploadable_report`.

---

## 1. Audit table

Verdict legend: ✅ matches contract · ⚠️ contract-adjacent risk · ❌ violation.

### A. Fingerprint surface

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| A1 | `cli/fingerprint.py:699-729` | `to_uploadable_dict` produces only `os_name`, `os_version` (major), `cpu_model`, `cpu_cores`, `ram_gb` (bucketed), `gpus[].{name,kind,memory_gb}`, `accelerator_summary`, `extras.backends`, optionally `fingerprint_hash`. No driver/build/PCI/serial fields. Matches PRIVACY.md §1. | ✅ |
| A2 | `cli/fingerprint.py:588-597` (per-GPU trim) and `cli/fingerprint.py:712-728` (per-fp trim) | PCI bus IDs (`GpuInfo.pci_bus_id`), `driver_version`, `vbios_version`, `power.power_limit_w`, `throttle_reasons_active`, `power_state`, `macos_version` patch component, `distro` patchlevel — all dropped from upload. Verified against PRIVACY.md §2 forbidden list. | ✅ |
| A3 | `cli/fingerprint.py:527-559` | `_accelerator_summary` only consumes `gpu.name`, `gpu.kind`, `gpu.memory_gb`, `cpu_model`, `cpu_cores`, `ram_gb`. Never embeds driver build, PCI, vBIOS, hostname. | ✅ |
| A4 | `cli/fingerprint.py:572-585` (`_round_ram_gb`, `_round_gpu_memory_gb`) and `cli/fingerprint.py:562-569` (`_major_os_version`) | RAM bucketed to 8 GB, GPU memory bucketed to 8 GB, OS version stripped to leading digit run. Tested (call site `to_uploadable_dict`). | ✅ |
| A5 | `cli/fingerprint.py:600-628` | `_fingerprint_hash` SHA-256s only the trimmed dict (`{gpus(trimmed), cpu_model, cpu_cores, ram_gb(bucketed), os_name, os_version(major)}`), not the raw `Fingerprint`. Note: `cpu_model` is the FULL brand string ("AMD Ryzen 9 7950X3D") — that's the one declared field in §1, not a leak. | ✅ |
| A6 | `cli/signing.py:258` (`ephemeral_keypair() if strict_anon else get_or_create_keypair()`) and `cli/fingerprint.py:727-728` (omits `fingerprint_hash` when `strict_anon`) | Verified by `tests/test_signing_round_trip.py::test_strict_anon_uses_fresh_keypair_per_run`. Two strict-anon JWS tokens differ. | ✅ |

### B. Upload path

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| B1 | `cli/upload.py:174` | `if api_key and not strict_anon and not anon: headers["Authorization"] = ...` — both `--anon` and `--strict-anon` suppress the bearer. | ✅ |
| B2 | `cli/upload.py:161-168` | `if strict_anon: headers = {"Content-Type": ...}` — no `User-Agent: llm-speed-cli/...`, no `X-LLM-Speed-Anon: 1`. The httpx default UA ships, indistinguishable from a generic httpx client. Matches PRIVACY.md §3 modes table. | ✅ |
| B3 | `cli/signing.py:283`, `cli/signing.py:186-202` | `assert_no_long_strings` runs BEFORE the JWS is created. ValueError aborts before any network call. Allowlist enforced by exact path match. | ✅ |
| B4 | `cli/signing.py:159-183` (`_LONG_STRING_ALLOWED` set + `_path_is_allowed = path in allow`) | Exact-match, not suffix-match. Tested by `tests/test_signing.py::test_nested_extras_error_rejected`, `test_fingerprint_extras_backends_error_rejected`, etc. The post-2026-04 fix is in place and locked down. | ✅ |
| B5 | `cli/signing.py:159-179` (privacy invariant) | The 256-char invariant bounds *long* strings on non-allowlisted paths. **It does NOT bound short strings carrying PII (e.g. a 50-char absolute path) on non-allowlisted paths.** Section 3 details this. So driver `extras` are bounded for length but not for content. | ⚠️ |

### C. Per-driver leak audit

#### llama_cpp (`cli/drivers/llama_cpp.py`)

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| C1 | `:243-255` | `ModelRef.identifier = str(path)` (absolute fs path), `extras = {"path": str(path), "size_bytes": ...}`. Both flow into upload. | ❌ (F-1, F-2) |
| C2 | `:191-208` (`_parse_version_output` from `--version` proc.stdout/stderr) | The full `--version` blob is parsed into a short `version` string + a small `flags` list (`["CUDA","Metal",...]`). Raw blob is not retained. ✅. Server SSE bodies are kept in `error` (`:322`, `:405`) truncated to 400 chars — non-200 server bodies CAN end up in `results.error` which is allowlisted. Borderline. | ⚠️ |
| C3 | `:177`, `:227`, `:435` | Reads `LLAMA_CPP_SERVER_URL` and `LLAMA_CPP_MODELS_DIR`. Used for control flow (subprocess argv, model scan) — not forwarded into the upload payload. | ✅ |
| C4 | `:243` (`identifier=str(path)`), `:251` (`"path": str(path)` in extras), `:441-443` (external URL into `_ServerHandle.base_url`) | Yes — full absolute paths to GGUF files leak via both `model.identifier` AND `model.extras.path`. | ❌ (F-1, F-2) |
| C5 | n/a | not applicable to this driver | ✅ |

#### mlx (`cli/drivers/mlx.py`)

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| C1 | `:147-160` (local `~/Models/mlx/`) | `identifier = str(child.resolve())` — absolute path. | ❌ (F-1) |
| C1 | `:189-199` (HF-cache scan) | `identifier = hf_id` (clean), `extras = {"snapshot_path": str(chosen)}` — absolute path under `~/.cache/huggingface/hub/...`. | ❌ (F-2) |
| C2 | n/a | No subprocess; uses `mlx_lm` import. | ✅ |
| C3 | `:438` only | `os.environ.get("LOGLEVEL", ...)` only in `__main__` smoke block; not in production path. | ✅ |
| C4 | `:147`, `:197` | Yes — see C1. | ❌ |
| C5 | n/a | | ✅ |

#### ollama (`cli/drivers/ollama.py`)

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| C1 | `:151-165` | `identifier = tag` (e.g. `qwen3:8b`) — clean. `extras = {"modified_at": ..., "size_bytes": ..., "details": {...}}`. `modified_at` is a timestamp from the daemon (correlatable across runs but not PII per se), `details` is the daemon's metadata blob (`format`, `family`, `parameter_size`, `quantization_level`) — small, no paths. | ⚠️ (F-4 — `modified_at` is a per-machine fingerprint signal that PRIVACY.md §1 doesn't mention) |
| C2 | n/a | All HTTP, no subprocess. Server response bodies on error truncated to 200 chars and only logged at WARNING, not put in payload (line 127). ✅ | ✅ |
| C3 | `:35` | Reads `OLLAMA_HOST` — used to set `base_url`, not forwarded into upload. ✅ | ✅ |
| C4 | n/a | No filesystem scanning. | ✅ |
| C5 | n/a | | ✅ |

#### vllm (`cli/drivers/vllm.py`)

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| C1 | `:121-128` | `identifier = mid` (HF-style id, clean). `extras = {"served_by": "http", "url": url}`. The `url` is the local vLLM HTTP base. If the user pointed `VLLM_URL` at a LAN host (`http://192.168.x.x:8000`), that LAN URL ships. | ⚠️ (F-2 variant) |
| C2 | `:212`, `:301`, `:320` | Server response body on error → `error` field, truncated to 300 chars. `results.error` is allowlisted (no length cap), so a chatty server can fill it. Used purely as a string, not parsed. | ⚠️ |
| C3 | `:43`, `:172` | `VLLM_URL` and `VLLM_API_KEY`. The API key is passed into the request `Authorization` header but **never** placed into `payload`, `backend_extras`, `error`, or any logged field. Verified by reading every log statement in the file. | ✅ |
| C4 | `:127` | `extras["url"]` carries the local vLLM URL. Mostly harmless (`http://localhost:8000`) but not zero. | ⚠️ |
| C5 | n/a | | ✅ |

#### exllamav2 (`cli/drivers/exllamav2.py`)

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| C1 | `:131-139` (local `~/Models/exl2/`) | `identifier = str(child.resolve())` — absolute path. No `extras`. | ❌ (F-1) |
| C1 | `:164-173` (HF-cache scan) | `identifier = str(chosen.resolve())` (absolute path inside `~/.cache/huggingface/hub/...`). `extras = {"hf_id": hf_id}`. The `hf_id` is clean, but `identifier` is an absolute path. | ❌ (F-1) |
| C2 | n/a | Uses `exllamav2` Python imports; no subprocess. | ✅ |
| C3 | `:390` | `os.environ.get("LOGLEVEL", ...)` only in `__main__`. | ✅ |
| C4 | `:127`, `:167` | Yes — `model.identifier` is an absolute filesystem path. | ❌ |
| C5 | n/a | | ✅ |

#### hosted_api (`cli/drivers/hosted_api.py`) — highest-risk driver

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| C1 | `:163-173` | `identifier = f"{provider}/{model_id}"` (e.g. `openai/gpt-4o-mini`), `extras = {"provider": provider, "base_url": base_url}`. `base_url` is the public provider URL (e.g. `https://api.openai.com/v1`) — disclosed by PRIVACY.md (§3 mentions hosted-api auto-pick warns the user). The driver explicitly drops the upstream provider's `/models` row to avoid leaking pricing/license/description text (`:158-162` comment). Good defense-in-depth. | ✅ |
| C2 | `:300`, `:322`, `:368`, `:421`, `:462` | Provider response body on error truncated to 300-400 chars, stuffed into `_failed_outcome(...)` → `results.error`. **The body content is the upstream provider's raw error JSON.** Some providers include rate-limit hashes, request IDs, account hints in error bodies. These would ship verbatim under `results.error`. | ⚠️ |
| C3 | `:120`, `:194` (only) | API keys read with `os.environ.get(env_var)`; placed into request header by `_auth_headers`. **Never** placed in `payload`, `backend_extras`, error message, or any log statement. Verified line-by-line. The auth header is the only place the key lives. | ✅ |
| C4 | n/a | No filesystem reads. | ✅ |
| C5 | `:64-75` (`_auth_headers`), `:138`, `:300` | `Authorization: Bearer <api_key>` and `x-api-key: <api_key>` are constructed locally and passed only to `httpx.{post,stream}`. Logging at line 138 is `r.text[:200]` — the response BODY only, not the request HEADERS. **`r.headers` is never logged anywhere.** No driver code path puts the key into `backend_extras`, `error`, payload, or stdout. Confirmed safe. | ✅ |

### D. Error / logging surface

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| D1 | grep summary across `cli/` | All `LOG.{debug,info,warning,error}` calls reviewed. None log full Authorization headers. None log full prompt text (workloads only log `len(PROMPT)` and `len(prompt) // 4` token estimates — see e.g. `cli/workloads/chat_long.py:113-117`, `cli/workloads/long_context_decay.py:60`). None log model output text. **However, several DO log absolute paths**: `cli/drivers/llama_cpp.py:486` (`LOG.info("starting llama-server: %s", " ".join(cmd))` — the cmd line includes `-m <abs_path>`), `cli/drivers/llama_cpp.py:541` (proc pid), `cli/drivers/exllamav2.py:371` (`LOG.info("exllamav2 loading %s", identifier)`), `cli/drivers/mlx.py:400` (`LOG.info("mlx loading %s", identifier)`). These are LOCAL stderr-only at INFO level and require `--verbose` or env `LOGLEVEL=INFO` to surface. They are NOT uploaded; they're stderr. Acceptable for local-only debug, but a nervous user pasting their stderr into a GitHub issue would leak. | ⚠️ (logging only, not upload) |
| D2 | `cli/signing.py:199-202` | `raise ValueError(f"upload payload contains unexpected long string at {_path or '<root>'}; refusing to upload (privacy invariant)")` — emits the *path* (good), not the *value* (good). | ✅ |
| D3 | `cli/__main__.py:249-253` | `except Exception as exc: ... print(f"error: {exc}", file=sys.stderr)`. Tracebacks only printed when `--verbose`. So an unhandled exception with a path in its repr (e.g. `RuntimeError(f"model not found: {model_path}")`) would print the path to stderr. Acceptable for local CLI behavior. The PAYLOAD path through `cli/commands/bench.py:310` — `_failed_result(..., f"{type(exc).__name__}: {exc}")` — does upload the exception's `str()` into `results.error`, which is unbounded by the privacy invariant. **This is F-3.** | ❌ (F-3) |

### E. Persistence on disk

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| E1 | `cli/signing.py:74` (`os.open(tmp, ..., 0o600)`) and `:86` (`os.chmod(path, 0o600)`) | Mode 0600 confirmed — both at create time and as a re-chmod after `os.replace`. | ✅ |
| E2 | `cli/consent.py:128-138` | Persisted record: `{"consented": True, "consented_at": <iso>, "cli_version": ..., "api_base": ..., "fingerprint_hash": ...}`. PRIVACY.md §7 promises "timestamp, CLI version, trimmed fingerprint_hash" — file actually also persists `api_base`. That's not PII (it's the user's choice of API URL) but **it's not in PRIVACY.md's list**. Doc-bug, not leak. | ⚠️ (F-5 — doc/code mismatch) |
| E3 | `cli/upload.py:60-77` | `save_offline(... include_raw_timings=True)` — local saves DO include raw timings. PRIVACY.md §2 says "kept locally for replay/dispute, only included in upload if you opt in". Local save with raw timings is consistent with §2's "kept locally" framing. ✅. The signed JWS payload uploaded via `--resume` was signed at save time with `include_raw_timings=True`, so when resumed, raw timings DO ship. PRIVACY.md says "only included in upload if you opt in" — and `--resume` re-uploading a saved file does upload them without explicit opt-in. | ⚠️ (F-6 — resume re-uploads raw timings) |

### F. Subprocess / argv

| ID | File:line | Prose | Verdict |
|---|---|---|---|
| F1 | `cli/drivers/llama_cpp.py:465-484` | `cmd = [binary, "-m", model_path, "--host", DEFAULT_HOST, "--port", ...]`. `model_path` comes from `model.identifier` which the user supplied. Passed as a list (not via shell), so no shell-injection risk; worst case is a bogus path that fails the `Path(model_path).exists()` check at `:460`. | ✅ |
| F2 | `cli/auto_install.py:109,138` | `shell=True` is used, but only for the canned strings `"curl -fsSL https://ollama.com/install.sh \| sh"` and `"sudo apt-get update && sudo apt-get install -y llama.cpp"`. Both are constants (no user input), only run after explicit y/n via `confirm_and_install`. No driver code uses `shell=True`. | ✅ |
| F3 | All driver subprocess calls (`subprocess.run` in `cli/fingerprint.py`, `cli/drivers/llama_cpp.py`) | None pass `env=`. The child process inherits the parent's full env including `OPENAI_API_KEY` etc. **`llama-server`, `nvidia-smi`, etc. don't read those keys, so this isn't a credential exfiltration; but a hostile binary on `$PATH` named `nvidia-smi` would inherit them.** Threat model trade-off, not a contract violation. | ⚠️ |

---

## 2. Findings (in priority order)

### F-1 — `model.identifier` leaks absolute filesystem path including username (❌ launch-blocking)

- **Where:** `cli/drivers/llama_cpp.py:247`, `cli/drivers/mlx.py:152`, `cli/drivers/exllamav2.py:131`, `cli/drivers/exllamav2.py:167`.
- **What:** The `ModelRef.identifier` field is set to `str(path.resolve())`. `dataclasses.asdict(result)` in `cli/signing.py:229` serializes the entire `ModelRef`, and `results.model.identifier` is on the privacy-invariant allowlist (`cli/signing.py:172`), so even an unbounded-length absolute path passes through to the JWS payload.
- **Concrete example payload:** `results[0].model.identifier == "/Users/<user>/.cache/huggingface/hub/models--mlx-community--Llama-3-8B/snapshots/abc123/model.safetensors"`. PRIVACY.md §2 explicitly says **"Hostname / username … never leave your machine"**.
- **Fix proposal:** in `build_uploadable_report` (or in each driver's `list_models`), normalize `model.identifier` to a non-fs identifier before signing. Two clean options:
  1. Drivers store the path on `model.extras["_local_path"]` (private), and set `identifier = "<backend>:<digest>"` or `<basename>` for upload-time use.
  2. `build_uploadable_report` walks `results[*].model` and replaces any `identifier` that looks like an absolute path (`startswith("/")` or `Path(s).is_absolute()`) with `Path(s).name`.
  Option (2) is the surgical fix and adds ~5 lines to `cli/signing.py`.

### F-2 — `model.extras` carries absolute paths under non-allowlisted but short keys (❌)

- **Where:**
  - `cli/drivers/llama_cpp.py:251-253` — `extras = {"path": str(path), "size_bytes": ...}`
  - `cli/drivers/mlx.py:197` — `extras = {"snapshot_path": str(chosen)}`
  - `cli/drivers/vllm.py:127` — `extras = {"served_by": ..., "url": url}` (LAN URL leak risk)
- **What:** these key names are NOT in the privacy-invariant allowlist. The 256-char check passes them through because the values are short (paths < 256 chars). So the privacy invariant doesn't catch them.
- **Fix proposal:** drop `model.extras` entirely from upload payload (it's driver-internal scratch, not part of PRIVACY.md §1). Add to `_trim_workload_result` in `cli/signing.py:210-216`:
  ```python
  if "model" in out and isinstance(out["model"], dict):
      out["model"] = {k: v for k, v in out["model"].items() if k != "extras"}
  ```
  Equivalent for the `extras["backend_extras"]` field on `WorkloadResult` — strip down to a known-allowlisted subset (`provider`, `model_id`, `usage`, `done_reason`, `stop_reason`, `prefix_cache_hit_rate`).

### F-3 — `results.error` is allowlisted (no length cap) and routinely contains absolute paths (❌)

- **Where:** `cli/signing.py:170` puts `"results.error"` on the long-string allowlist with no length cap. Drivers feed exceptions into `error` strings, e.g. `cli/drivers/llama_cpp.py:461` `RuntimeError(f"model not found: {model_path}")` → `cli/commands/bench.py:310` `f"{type(exc).__name__}: {exc}"` → upload.
- **What:** A failed run uploads `results[0].error == "RuntimeError: model not found: /Users/<user>/Models/qwen3.gguf"`. Username again.
- **Fix proposal:** add a sanitizer step in `cli/signing.py` that scrubs absolute paths from `error` strings before signing:
  ```python
  _PATH_RE = re.compile(r"(/Users/[^/\s]+|/home/[^/\s]+|C:\\Users\\[^\\s]+)")
  def _scrub_paths(s: str) -> str:
      return _PATH_RE.sub("<path-redacted>", s)
  ```
  Apply to all `results[*].error` values in `_trim_workload_result`.

### F-4 — `model.extras["modified_at"]` (ollama) is a per-machine timestamp (⚠️)

- **Where:** `cli/drivers/ollama.py:160`. The daemon's `modified_at` is when the local user last pulled the model. Not PII per se, but it's a stable per-machine signal not enumerated in PRIVACY.md §1.
- **Fix:** subsumed by F-2's "drop `model.extras` entirely from upload".

### F-5 — `consent.json` persists `api_base` not listed in PRIVACY.md §7 (⚠️ doc bug)

- **Where:** `cli/consent.py:128-138`.
- **Fix:** either remove `api_base` from the record (`api_base` already lives in argv anyway), or update PRIVACY.md §7 to list it. Either is fine; the data is local-only and the user supplied it.

### F-6 — `--resume` uploads `raw_timings_ms` without re-asking the user (⚠️)

- **Where:** `cli/upload.py:67` (`save_offline(... include_raw_timings=True)`) + `cli/upload.py:216-251` (`upload_saved_payload` re-posts the JWS as-is).
- **What:** PRIVACY.md §2 says raw timings only ship "if you opt in via `include_raw_timings=True`". The save flow always sets it to True, and resume re-uploads the saved JWS verbatim. So the user's choice of "save now, upload later" silently includes raw timings.
- **Fix:** in `save_offline`, default `include_raw_timings` to `False` and require an explicit flag (`--include-raw-timings`) to enable. OR sign two payloads at save time (one with, one without) and have `--resume` pick the lean one by default.

### F-7 — INFO-level logs include absolute paths to stderr (⚠️ local only)

- **Where:** `cli/drivers/llama_cpp.py:486`, `cli/drivers/exllamav2.py:371`, `cli/drivers/mlx.py:400`.
- **What:** Stderr only, default level is WARNING (so these don't surface unless the user passes `--verbose`). Not uploaded. But a user copy-pasting stderr into a bug report leaks their username. Low severity.
- **Fix:** when emitting these logs, redact via the same `_scrub_paths` from F-3, or at minimum use `Path(model_path).name` instead of the full path.

---

## 3. Quotable summary (one paragraph the maintainer can paste)

> An independent audit of the llm-speed CLI on 2026-05-01 verified that the documented privacy contract (`docs/PRIVACY.md`) is faithful for **fingerprint bucketing** (RAM/GPU memory rounded to 8 GB, OS to major version), the **`--anon` and `--strict-anon` modes** (Authorization bearer dropped under both, ephemeral keypair and default `User-Agent` under strict-anon, verified by `tests/test_signing_round_trip.py`), the **JWS signing path** (Ed25519, deterministic for the same payload, tamper-detected), the **first-run consent gate** (no auto-consent on non-TTY stdin), the **on-disk file modes** (keypair and consent file at `0600`), and the **server-side IP non-persistence** (out of scope for this CLI audit but consistent with the contract). The audit also identified **three concrete leaks in the per-result upload payload** that exceed the contract: (F-1) `results.model.identifier` ships as a raw absolute filesystem path for the `llama.cpp`, `mlx` (local Models dir), and `exllamav2` drivers, leaking the user's macOS/Linux username; (F-2) driver-side `model.extras` carries additional absolute paths under non-allowlisted but short-string keys that the 256-char privacy invariant does not catch; (F-3) `results.error` is allowlisted without a length cap and routinely contains exception strings of the form `"model not found: /Users/<user>/..."`. All three are local to the upload-payload-build step in `cli/signing.py:build_uploadable_report` and a small per-driver `list_models` cleanup; the fix surface is well under fifty lines and does not require any cryptographic or protocol change. With those three fixes, the CLI matches PRIVACY.md as written.

---

## 4. Negative findings (things deliberately checked and clean)

- The hosted-API driver does NOT log, upload, or otherwise expose the `Authorization`/`x-api-key` value on any code path. (cli/drivers/hosted_api.py — confirmed via line-by-line read of every `LOG.*`, `print`, `_failed_outcome`, and payload-construction path.)
- `assert_no_long_strings` uses exact-path match, not suffix-match. Tests in `tests/test_signing.py` lock down the regression cases.
- `subprocess.run` is never invoked with `shell=True` on user-controlled data.
- Workload prompts and model output text are never put into `WorkloadResult.extras`. Workloads log only `len(prompt)` / `prompt_chars`, not the prompt body. Output text (`outcome.output_text`) is computed for return but never propagated into a payload field.
- `--strict-anon` correctly skips the persistent keypair (`get_or_create_keypair` is never called on that branch — verified at `cli/signing.py:258`).
- The `--print-payload` flag emits exactly the body that would be POSTed (`cli/commands/bench.py:339-342`), letting a paranoid user diff against `docs/PRIVACY.md` §1 directly.
