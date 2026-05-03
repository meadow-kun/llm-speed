# CLI cross-environment test matrix

The matrix the CLI must survive before HN launch. Each cell is one of:

- ✅ verified — actually run on this combo, with a date and a run id / commit
- 🧪 unit-tested only — no real backend was invoked; logic exercised via mocks
- ⚠️ deduced-likely-works — code path has no platform-specific branch; we expect
  it to work but haven't run it
- ❌ untested — gap, blocks "tier 1" claim, probably the source of HN-day pain

Last updated: 2026-04-29 (this commit).

## Platform × backend × setup-state

### Setup-state: at least one backend installed + at least one model cached

| Platform                       | llama.cpp | mlx | ollama | vllm | exllamav2 | hosted-api |
|--------------------------------|:---------:|:---:|:------:|:----:|:---------:|:----------:|
| macOS arm64 (M3 Pro)           |    ✅     | ✅  |   🧪   |  —   |     —     |     ✅     |
| macOS arm64 (M3 Ultra)         |    ✅     | ✅  |   🧪   |  —   |     —     |     ✅     |
| macOS x86_64 (Intel)           |    ❌     | —   |   ❌   |  —   |     —     |     ❌     |
| Linux x86_64 (Ubuntu, NVIDIA)  |    ✅     | —   |   🧪   |  ⚠️  |    ⚠️     |     ✅     |
| Linux x86_64 (Debian)          |    ⚠️     | —   |   ⚠️   |  ⚠️  |    ⚠️     |     ⚠️     |
| Linux x86_64 (Arch)            |    ❌     | —   |   ❌   |  ❌  |     ❌    |     ❌     |
| Linux arm64 (Raspberry Pi)     |    ❌     | —   |   ❌   |  —   |     —     |     ❌     |
| Linux arm64 (Jetson)           |    ❌     | —   |   ❌   |  —   |     —     |     ❌     |
| Linux arm64 (AWS Graviton)     |    ❌     | —   |   ❌   |  —   |     —     |     ❌     |
| WSL2 Ubuntu (5090 box)         |    ✅     | —   |   🧪   |  ⚠️  |    ⚠️     |     ✅     |
| Windows native (PowerShell)    |    ❌     | —   |   ❌   |  —   |     —     |     ❌     |
| Windows native (cmd.exe)       |    ❌     | —   |   ❌   |  —   |     —     |     ❌     |

Evidence for ✅:
- `docs/sweep_2026-04-29.md` (M3 Pro 9 cells, M3 Ultra 20 cells, RTX 5090 1 cell, hosted-api 5 cells)
- `bots/github-action/` self-test on `ubuntu-latest`

### Setup-state matrix per platform

These cut across all platforms. Tested via mocks except where noted.

| Setup state                                          | Status | Notes |
|------------------------------------------------------|:------:|-------|
| Nothing installed locally → fall back to hosted-api  |   🧪   | `cmd_bench._pick_backend` returns `hosted-api` only as last-resort fallback. Covered by tests/cli/test_cross_env.py:test_hosted_api_no_raw_in_modelref_extras as a proxy. |
| llama.cpp installed but no GGUFs cached              |   🧪   | `_resolve_model` errors with "no models available …" — pin in a future test |
| Multiple backends installed → autopicker deterministic |  🧪  | Priority order is `(vllm, llama.cpp, ollama, mlx, exllamav2, hosted-api)`. No test pins this — gap. |
| HF_HOME on removable drive that disappears mid-run   |   ❌   | Documented in known-failure-modes; needs a real removable mount to repro |
| Disk full mid-download                               |   ❌   | We don't intercept the HF download path |
| Disk full during local save                          |   🧪   | tests/cli/test_cross_env.py:test_save_offline_disk_full_no_partial_file |
| API rate-limit (429) mid-upload                      |   ⚠️   | `_post_with_retries` retries on 5xx but treats 4xx (incl. 429) as a hard fail; gap. |

### Network states

| Network state                                | Status | Notes |
|----------------------------------------------|:------:|-------|
| No internet → graceful error                 |   🧪   | `_post_with_retries` catches httpx.TransportError and surfaces UploadError after N retries; offline save fallback runs. |
| HTTPS only (no plain HTTP)                   |   🧪   | `warn_if_insecure_api_base` warns on http://; no automatic insecure fallback. |
| API endpoint 5xx → CLI retries N then fails  |   🧪   | Implicit; covered by the upload retry loop. No dedicated test pins the count. |
| Rate limit (429) → honor Retry-After         |   ❌   | We do NOT honor Retry-After. 429 is treated as a 4xx and surfaces immediately. |
| TLS cert pinning / corporate MITM            |   ❌   | We don't pin certs and don't test corporate root CA stores. |

### Privacy modes

| Mode                          | Status | Test |
|-------------------------------|:------:|------|
| Default                       |   🧪   | tests/test_signing_round_trip.py |
| `--anon`                      |   🧪   | tests/cli/test_cross_env.py:test_anon_drops_auth_keeps_persistent_keypair |
| `--strict-anon`               |   🧪   | tests/cli/test_cross_env.py:test_strict_anon_drops_auth_header + test_strict_anon_drops_user_agent_header |
| `--no-upload`                 |   🧪   | covered indirectly via cmd_bench logic; no dedicated test |
| `--dry-run --print-payload`   |   🧪   | smoke covered in scripts/check.sh; no unit test |
| `--include-raw-timings`       |   🧪   | sign_report_jws supports the flag; signing round-trip exercises both branches |

## Python versions

Declared `requires-python = ">=3.10"`.

| Version | Tested | Notes |
|---------|:------:|-------|
| 3.10    |   ❌   | NOT verified locally; should be checked in CI |
| 3.11    |   ✅   | local dev env (miniconda3) |
| 3.12    |   ✅   | M3 Ultra runner (uv tool install) |
| 3.13    |   ❌   | UNTESTED. Anything with `from __future__ import annotations` should be fine, but psutil + cryptography + joserfc need 3.13 wheels. |

## Highest-leverage gaps still uncovered (post-this-commit)

1. **No integration test for the consent → upload → save fallback chain.** We
   test pieces but not the full cmd_bench path under different `--anon` /
   `--no-upload` / `--dry-run` combinations.
2. **No test for autopicker priority ordering.** `_pick_backend` could silently
   reorder priority and we'd never catch it. Add a parameterized test that
   stubs `detect()` for each registered driver.
3. **Windows path quoting in install plans is untested.** `auto_install.plan_for`
   returns a `command: str` that the wizard prints verbatim. On Windows cmd.exe,
   commands like `sudo apt-get …` are nonsense — but we don't ship a Windows
   plan, so the wizard hits the `available=False` branch. No test pins that.
4. **429 / Retry-After handling.** Untested AND unimplemented. Filed in
   `cli_known_failure_modes.md`.
5. **HF_HOME on removable drive disappearing mid-run.** Repros need a USB SSD;
   not a unit-test target. Documented as a known failure mode; mitigation is to
   surface the OSError clearly instead of swallowing it.

## CI matrix proposal — `ci.yml`

The repo currently has no `ci.yml` (the workflow folder has only
`daily-metrics.yml`, `reddit-poster.yml`, `release.yml`, `scheduled-seed.yml`).
The proposal below ADDS a new `ci.yml` that runs on every push to main and
every pull request.

Proposed YAML (do not apply yet):

```yaml
# .github/workflows/ci.yml
name: ci

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: test (${{ matrix.os }} / py${{ matrix.python }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.11", "3.12", "3.13"]
        # 3.13 wheels for cryptography on Windows can lag; allow that one
        # combo to fail without blocking the matrix on launch day.
        exclude:
          - os: windows-latest
            python: "3.13"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip
      - name: install
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test]"
      - name: pytest
        run: python -m pytest tests/ --ignore=tests/playwright --tb=short -q
      - name: smoke (no network)
        env:
          # Hard-fail any test that tries to hit api.llm-speed.com
          LLM_SPEED_API: http://127.0.0.1:1
        run: |
          python -m cli --version
          python -m cli --help
          python -m cli detect
          python -m cli list-models

  lint:
    name: lint (ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check --select=E,F,I,B,UP --ignore=E501 seed cli api tests
      - run: ruff format --check seed cli api tests

  arm64-smoke:
    # GH-hosted ARM runners. Use to cover Linux arm64 detection paths.
    name: smoke (linux-arm64)
    runs-on: ubuntu-24.04-arm
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[test]"
      - run: python -m pytest tests/ --ignore=tests/playwright --tb=short -q
      - run: |
          python -m cli --version
          python -m cli detect
```

Why this shape:

- **3 OSes × 4 Pythons** = 12 cells. The exclude trims 1, leaving 11. All
  ~30s each. Total CI time ≈ 6 min wall-clock with parallelism.
- **No GPU runners.** GH-hosted CI has no GPU; correctness on Tier-1 platforms
  is the goal. Perf numbers stay on the maintainer's self-hosted runner per
  `docs/HARDWARE_MATRIX.md`.
- **arm64 smoke job** uses the new `ubuntu-24.04-arm` runner family
  (free for public repos). Catches arm64-specific issues in fingerprint /
  cryptography / psutil before WSL2 / Pi users hit them.
- **No backend-real tests in CI.** We deliberately don't `apt-get install
  llama.cpp` in CI — it'd drift from what users experience. The mocked tests
  are the contract.
- **Smoke step pins LLM_SPEED_API to 127.0.0.1:1** so any test that
  accidentally hits the real API in CI fails loudly instead of leaking
  fingerprint data to api.llm-speed.com from GH-hosted runners.

## What this matrix doesn't cover

- Real corporate networks with TLS interception (BlueCoat, Zscaler). Out of
  scope; documented as a known failure in `cli_known_failure_modes.md`.
- Air-gapped deployments. The `--no-upload` + `--dry-run` flags exist for
  this audience but aren't exercised end-to-end.
- Battery-powered laptops on power-save mode. Listed as a `flags`
  side-effect in WorkloadResult but no test pins the flag.
- Mobile / Android / iOS. Out of scope per HARDWARE_MATRIX.md.

## How to use this matrix

When a new bug surfaces in the wild:

1. Find the (platform × backend × setup-state) cell.
2. If the cell is ❌ or 🧪 and the bug is real, downgrade to ⚠️ if a fix is
   shipped or ❌ if not, and add an entry to `cli_known_failure_modes.md`.
3. If the cell was ✅ and the bug is real, the cell is now WRONG — re-run the
   sweep that originally certified it and mark the regression date.
4. New tests (mocks) go under `tests/cli/`; new real-backend bench cells go
   under `tools/runners/`.
