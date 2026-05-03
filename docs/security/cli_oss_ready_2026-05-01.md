# llm-speed CLI — Open-source readiness audit (2026-05-01)

Scope: read-only audit of `cli/`, `tests/`, `pyproject.toml`, `README.md`, the
release workflow, and packaging surfaces (`Formula/`, `npm/`, `Dockerfile*`)
ahead of widening public scrutiny. Static review at `main` (commit `2453f99`,
"init"). No code changes were made.

## 1. Verdict

**Open-source ready *with notes*.** The CLI itself (the `cli/` Python package
that strangers would actually run) is in good shape: crypto is correct,
consent UX fails closed on non-TTY, the auto-pick-to-hosted-API surprise has a
loud stderr notice, and there are zero `the maintainer` / `<maintainer>` / work-org
identifier leaks in any source under `cli/`. A careful r/LocalLLaMA reader
who reads the `cli/` package will not find anything alarming.

The "with notes" qualifier comes from the *packaging* and *publishing* layer:
five tracked files outside `cli/` still point at a non-existent
`<old-handle>` GitHub username (README install instructions, all three
Dockerfiles, npm), the release workflow does not actually publish the
`.sha256` sidecar that the brand-new `llm-speed verify` subcommand depends on,
and the test suite covers 18% of the `cli/` package by line — the upload,
consent, resume, and driver paths are entirely untested. None of these block
publication, but they are what a sophisticated reader will catch.

## 2. Identity scrub verification

`grep -rni` over `cli/`, `tests/`, `pyproject.toml`, `README.md`, `LICENSE`
for `<dev>` / `<maintainer>` / `<work-domain>` / `<maintainer-domain>` / `<work-org>`:

| Hit | File:line | Verdict |
|---|---|---|
| "the live cases the maintainer ran by hand" | `tests/test_extractors.py:4,21` | comment in fixture file; benign but cosmetic — strangers see "the maintainer" with no context |
| `assert "<dev>" not in theme.REPO_URL.lower()` | `tests/cli/ui/test_theme.py:27-29` | guard test — desirable. Keep. |
| `Copyright 2026 the maintainer` | `LICENSE:189` | required by Apache-2.0 attribution; legal name is fine here |
| `https://github.com/<old-handle>/...` | `README.md:15` | **identity bleed** — wrong username (404), neither personal nor org |
| same URL | `Dockerfile:33`, `Dockerfile.cuda:8`, `Dockerfile.rocm:7` | **identity bleed** in OCI image labels |
| same URL | `npm/install.js:39`, `npm/package.json:10,13` | **identity bleed** in npm package metadata |
| `<dev-mac-1>`, `<dev-mac-2>` | `docs/SESSION_HANDOFF_2026-04-29.md:141`, `docs/TEST_MATRIX.md:9-10` | tailscale hostnames in tracked docs — minor; reveals the maintainer's machine names |
| `<old-handle>` | many `tests/playwright/test-results/**/error-context.md` | playwright run-output captures — these are crawled artifacts and should not be tracked, but they are. |

`pyproject.toml` `authors`, `urls.Source`, `urls.Issues` all correctly use
`meadow-kun` (lines 6, 57, 59). `cli/ui/theme.py:26` uses `meadow-kun`. So
the *runtime CLI* points at the right place; the *distribution metadata*
points at a fictional username.

`grep -rn '/Users/<user>/'` over the same set: zero hits. Good.

## 3. Code-smell findings (prioritized)

| # | File:line | Smell | Suggested fix |
|---|---|---|---|
| S-1 | `README.md:15`, `Dockerfile:33`, `Dockerfile.cuda:8`, `Dockerfile.rocm:7`, `npm/install.js:39`, `npm/package.json:10,13` | `https://github.com/<old-handle>/<old-repo>` — wrong username; URL 404s | Replace with `https://github.com/meadow-kun/llm-speed` (matches `pyproject.toml`, `cli/ui/theme.py`). |
| S-2 | `.github/workflows/release.yml:1-256` | The `verify` subcommand depends on a `<wheel>.sha256` sidecar served at `https://llm-speed.com/dist/`, but `release.yml` never computes or publishes that sidecar (only PyPI upload). `verify` will always 404 against the public dist as currently published. | After `python -m build`, `sha256sum dist/*.whl > dist/*.whl.sha256` (one-line); push the `.whl.sha256` files to the same CDN path as `.whl`. Document in `docs/RELEASE.md`. |
| S-3 | `cli/commands/login.py:35`, `cli/commands/self_update.py:55` | `TODO: ...` comments in user-visible help/error text. A first-time visitor running `llm-speed login` literally sees "TODO: server-side OAuth callback handler is not yet wired". | Replace with neutral "not yet implemented in 0.0.1; tracking issue: <link>" or hide the subcommand under `--experimental`. |
| S-4 | `tests/playwright/test-results/**` | Tracked playwright artifacts contain `<old-handle>` URLs (the rendered page footer). These are generated, large, and continually re-leak the wrong URL on every test run. | `.gitignore` `tests/playwright/test-results/`. They should never have been committed. |
| S-5 | `docs/TEST_MATRIX.md:9-10`, `docs/SESSION_HANDOFF_2026-04-29.md` | Personal Tailscale hostnames (`<dev-mac-1>`, `<dev-mac-2>`) tracked in a public repo. Low-grade information leak; tells strangers the maintainer's machine fleet. | Either rename to generic labels (`mac-m3-pro`, `mac-m3-ultra`) or move handoff/test-matrix docs out of the tracked tree. |
| S-6 | `tests/test_extractors.py:4,21` | "the maintainer ran by hand" left as a literal comment in a test fixture. Cosmetic. | s/the maintainer/the maintainer/ or drop. |
| S-7 | `scripts/build_shiv.sh:79` | `echo "TODO: pull python-build-standalone, repack as a single-file launcher."` — visible in script output. | Remove or move to docs. |
| S-8 | `cli/__main__.py:189` | Bare `print(...)` to stderr for a registry-load warning. Other modules use `logger`. | Route through `logging.getLogger(...).warning(...)` with `--verbose` for the traceback (the next line already covers that case). |
| S-9 | `cli/output.py:84,88` | `console.print(f"[green]Submitted:[/green] {result_url}")` — server-supplied `url` flows through Rich markup. Documented in `docs/security/cli-review.md` F-10 as low-severity, still open. | `console.print(text, markup=False)` or `rich.markup.escape(url)` before formatting. |
| S-10 | `pyproject.toml:40-46` | `psutil`, `rich`, `cryptography`, `joserfc` have no upper version bound (only `httpx` does). | Add `<7`, `<15`, `<50`, `<2` upper bounds. Documented in `cli-review.md` F-12. |

No `print()` in production code paths beyond S-8 — all other `print(...)`
hits are inside `if __name__ == "__main__":` smoke-test blocks at the foot
of each driver, which is fine. No `# DEBUG` / `# HACK` / `# REMOVE BEFORE
COMMIT` comments. No "this shouldn't happen" error text. No commented-out
code blocks. No hardcoded IPs or paths beyond the documented
provider URLs in `cli/drivers/hosted_api.py:40-51`.

## 4. Cryptographic correctness check

| Item | Result | Citation |
|---|---|---|
| Ed25519 keypair persistence is atomic | **PASS** — `os.open(tmp, O_WRONLY|O_CREAT|O_TRUNC, 0o600)` + write + `os.replace(tmp, path)`. No write-then-rename race; no world-readable window. | `cli/signing.py:66-88` |
| JWS signing pins algorithm | **PASS** — `algorithms=[ALG]` passed to `joserfc.serialize_compact` AND to `deserialize_compact`. `ALG = "EdDSA"` is the only accepted value. | `cli/signing.py:43, 267, 320` |
| Algorithm-confusion attack rejected | **PASS** — `verify_jws_token` parses the header BEFORE invoking joserfc and explicitly checks `header.get("alg") != ALG` first. Verified empirically by running a forged HS256 token through it: rejected with "unexpected alg: HS256". | `cli/signing.py:304-305` |
| Embedded JWK trust is intentional and documented | **PARTIAL** — the docstring on `verify_jws_token` says "Trusts the embedded jwk — the pubkey IS the identity here (anonymous-but-pinned-to-a-keypair model)." That's clear. But the docstring does not warn that `--resume` therefore verifies *integrity*, not *provenance* (an attacker who can write to `~/.cache/llm-speed/runs/` can forge a valid JWS — server is the trust boundary). Documented at `cli-review.md:F-11`, still open. | `cli/signing.py:292-294` |
| Privacy-invariant guard exists and is path-pinned | **PASS** — `_LONG_STRING_ALLOWED` is a closed allowlist of fully-qualified dotted paths, not a suffix match. `cli-review.md:F-5` (over-permissive suffix matching) was closed in the 2026-04-27 fix-pass. | `cli/signing.py:159-202` |
| `--strict-anon` actually drops persistent identity | **PASS** — `ephemeral_keypair()` is used per run; fingerprint_hash is omitted; User-Agent and Authorization are dropped. | `cli/upload.py:161-176`, `cli/signing.py:104-106` |
| `--anon` drops the bearer token | **PASS (recently fixed)** — `if api_key and not strict_anon and not anon:` correctly suppresses the Authorization header under either anon flag. The earlier finding `cli-review.md:F-3` is closed. | `cli/upload.py:174` |
| Atomic consent write | **PASS** — same tmp+rename pattern as the keypair. | `cli/consent.py:42-60` |

The signing module is the strongest part of the codebase.

## 5. Test coverage report

`python -m pytest tests/ --tb=short --ignore=tests/playwright -q`:
**108 passed, 2 skipped, 1 xfailed** (1.74s). All green.

`python -m pytest --cov=cli tests/ --ignore=tests/playwright`: **18% overall**
across `cli/`. Per-module:

| Module | Coverage | Notes |
|---|---|---|
| `cli/types.py` | 100% | dataclasses, fully covered |
| `cli/config.py` | 100% | constants only |
| `cli/registry.py` | 96% | 1 line missed |
| `cli/timing.py` | 94% | |
| `cli/ui/progress.py` | 84% | |
| `cli/verify.py` | 82% | new in v2 |
| `cli/signing.py` | 80% | strong sign-then-tamper round-trip suite |
| `cli/ui/auto.py` | 74% | |
| `cli/ui/welcome.py` | 57% | wizard flow partially mocked |
| `cli/ui/share.py` | 51% | |
| `cli/fingerprint.py` | 15% | NVIDIA/Apple/Linux paths skipped on macOS |
| `cli/__main__.py` | 0% | argparse dispatch — never invoked from a test |
| `cli/upload.py` | 0% | **uncovered**: upload, retry, resume, save-offline |
| `cli/consent.py` | 0% | **uncovered**: TTY/EOF gating logic |
| `cli/output.py` | 0% | |
| `cli/auto_install.py` | 0% (driver tests touch it) | tests live but coverage isn't tagged for the file |
| `cli/commands/bench.py` | 0% | the main user flow has no integration test |
| `cli/commands/{compare,detect,list_models,login,self_update}.py` | 0% | |
| `cli/drivers/{exllamav2,hosted_api,llama_cpp,mlx,ollama,vllm}.py` | 0% | drivers shell out / hit network — fair, but dummy-driver coverage of the protocol surface should exist |
| `cli/workloads/{agent_trace,chat_long,chat_short,concurrent_decode,long_context_decay}.py` | 0% | |

**Modules <70% coverage:** `fingerprint.py` (15%), `ui/welcome.py` (57%),
`ui/share.py` (51%), and every module at 0% above.

**Code paths that are entirely untested:**
- `--strict-anon` end-to-end (the `sign_report_jws` round-trip test covers the
  signing primitive at `tests/test_signing_round_trip.py:146-155`, but the
  bench command path that wires `strict_anon → upload_or_save → no
  Authorization header → no consent prompt` is never exercised).
- `--resume PATH` (`upload_saved_payload` and `_handle_resume`).
- Consent prompt: TTY gate, EOF handling, "y/yes/empty" branches.
- Every driver's failure modes (no daemon running, malformed model file,
  network timeout, hosted-API 401/429).
- `compare`, `detect`, `list-models`, `login`, `self-update` subcommands.
- The `--anon`-vs-`--strict-anon` header-difference matrix.

The 80% on `signing.py` is doing a lot of trust-load-bearing work that the
rest of the codebase isn't currently underwriting.

## 6. README + docs gap list

| Gap | Severity | Notes |
|---|---|---|
| No `CONTRIBUTING.md` | medium | Strangers don't know how to run the test suite, what the dev loop is, or how to send a PR. The README mentions `scripts/check.sh`; that's the right starting point. |
| No `SECURITY.md` | medium | No documented disclosure address. `docs/security/cli-review.md` is excellent internal-style content but doesn't tell a researcher where to send a finding. |
| No `CHANGELOG.md` | low | Version is `0.0.1-dev` everywhere; nothing for an outsider to follow. |
| No `.github/ISSUE_TEMPLATE/` or `PULL_REQUEST_TEMPLATE.md` | low | Optional but customary on public repos. |
| `README.md:15` install link 404s | high | `<old-handle>` username doesn't exist; should be `meadow-kun`. (See S-1.) |
| `README.md` doesn't tell a stranger what `llm-speed` *does* in 5 lines | medium | Lines 1-4 say "Source for **llm-speed.com**" and pivot immediately into install + seeders. The CLI's actual job ("benchmark any LLM on any hardware, get a tok/s number, optionally submit") is in `pyproject.toml:4` but not at the top of the README. |
| CLI subcommands not enumerated | medium | The README talks about the seeder ("Per-seeder usage", "Quick start" runs `seed.run`) but never lists `llm-speed bench`, `detect`, `list-models`, `compare`, `verify`, `about`. Help is only discoverable via `llm-speed --help`. |
| Privacy/consent story not in README | medium | `docs/PRIVACY.md` is great; the README doesn't link to it. A stranger evaluating "should I run this on my machine?" wants the privacy summary at the install instructions, not 3 doc-tree clicks away. |
| LICENSE present and Apache-2.0 | **PASS** | `LICENSE:1-202`, matches `pyproject.toml:7`. |

## 7. Top 3 priorities before launch

1. **Fix the 5 broken `<old-handle>/<old-repo>` URLs** in `README.md`,
   `Dockerfile{,cuda,rocm}`, `npm/install.js`, `npm/package.json`. They 404,
   they leak a partial real name, and they undermine every "how do I install
   this?" path in the repo. Mechanical sed-replace to `meadow-kun`. (S-1)

2. **Publish the `.sha256` sidecar in the release workflow.** The brand-new
   `llm-speed verify` subcommand is the literal answer to "should I trust
   this binary?" for the r/LocalLLaMA audience. Right now it always fails
   against the public dist because the sidecar isn't there. One `sha256sum`
   line in `release.yml` after `python -m build` plus uploading the
   `.whl.sha256` next to the `.whl`. (S-2)

3. **Rewrite the README opener and add a CONTRIBUTING.md / SECURITY.md.**
   Lead with "what `llm-speed` does in 5 lines, with one example invocation,
   and the privacy summary linked." List the CLI subcommands. Add a
   one-paragraph CONTRIBUTING (`./scripts/check.sh`, `pytest tests/`) and a
   one-paragraph SECURITY (where to disclose, that the on-disk threat model
   is single-user-machine, link to `docs/security/cli-review.md`). This is
   the cheapest single change that converts a stranger from "is this real?"
   to "let me try it." (Section 6)

The remaining items in §3 (TODO comments in user-visible help, the tracked
playwright artifacts, the missing upper-bounds, the markup-escape on server-
supplied URLs, and the Tailscale hostnames in docs) can ride along on the
same launch-prep PR but won't change a careful reader's verdict on their own.
