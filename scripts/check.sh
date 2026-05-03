#!/usr/bin/env bash
#
# Local CI replacement. Run before pushing main.
#
# Phases (skip with the env vars on the right):
#   1. lint     — ruff check + format check          (SKIP_LINT=1)
#   2. test     — pytest --cov                       (SKIP_TEST=1)
#   3. smoke    — CLI version/help/detect/list       (SKIP_SMOKE=1)
#   4. build    — python -m build (sdist + wheel)    (SKIP_BUILD=1)
#
# Quick mode for the inner-loop:
#   ./scripts/check.sh --quick      # lint + test only, ~10s
#
# Pre-push wiring (opt-in once):
#   git config core.hooksPath .githooks
# Then `git push` runs this script via .githooks/pre-push.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

QUICK=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --help|-h)
            sed -n '/^#$/,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
    esac
done

# Colors when stdout is a tty.
if [ -t 1 ]; then
    BOLD=$'\033[1m'
    GREEN=$'\033[32m'
    RED=$'\033[31m'
    YELLOW=$'\033[33m'
    DIM=$'\033[2m'
    RESET=$'\033[0m'
else
    BOLD="" GREEN="" RED="" YELLOW="" DIM="" RESET=""
fi

PY="${PYTHON:-python3}"
FAILED=()
START=$(date +%s)

# Set up a project-local venv at .venv/ to avoid PEP 668 conflicts on
# Homebrew Python. Idempotent — re-uses .venv if already present. Honours
# uv when available (faster), falls back to stdlib venv otherwise. The
# venv is .gitignore'd.
_ensure_venv() {
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        return 0
    fi
    if [ ! -d .venv ]; then
        echo "${DIM}creating project .venv…${RESET}"
        if command -v uv >/dev/null 2>&1; then
            uv venv --quiet .venv
        else
            $PY -m venv .venv
        fi
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="python"
}

_install_pkg() {
    local pkg="$1"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --quiet "$pkg"
    else
        $PY -m pip install --quiet "$pkg"
    fi
}

_ensure_venv

step() { echo; echo "${BOLD}── $1 ──${RESET}"; }
ok() { echo "${GREEN}✓${RESET} $1"; }
fail() { echo "${RED}✗${RESET} $1"; FAILED+=("$1"); }
skip() { echo "${YELLOW}↷${RESET} $1 ${DIM}(skipped)${RESET}"; }

# ----------------------------------------------------------------------------
# 1. lint
# ----------------------------------------------------------------------------
if [ "${SKIP_LINT:-0}" = "1" ]; then
    skip "lint"
else
    step "lint (ruff)"
    if ! command -v ruff >/dev/null 2>&1; then
        echo "${DIM}installing ruff…${RESET}"
        _install_pkg ruff
    fi
    if ruff check --select=E,F,I,B,UP --ignore=E501 cli tests; then
        ok "ruff check"
    else
        fail "ruff check"
    fi
    if ruff format --check cli tests; then
        ok "ruff format"
    else
        fail "ruff format (run: ruff format cli tests)"
    fi
fi

# ----------------------------------------------------------------------------
# 2. test
# ----------------------------------------------------------------------------
if [ "${SKIP_TEST:-0}" = "1" ]; then
    skip "test"
else
    step "test (pytest)"
    if ! $PY -c "import pytest" >/dev/null 2>&1; then
        echo "${DIM}installing test deps via pip install -e '.[test]'…${RESET}"
        _install_pkg "-e .[test]"
    fi
    if $PY -m pytest -v --cov=cli --cov-report=term; then
        ok "pytest"
    else
        fail "pytest"
    fi
fi

if [ "$QUICK" = "1" ]; then
    step "quick mode — skipping smoke + build"
    skip "smoke"; skip "build"
else
    # ------------------------------------------------------------------------
    # 3. smoke CLI
    # ------------------------------------------------------------------------
    if [ "${SKIP_SMOKE:-0}" = "1" ]; then
        skip "smoke-cli"
    else
        step "smoke (CLI)"
        if $PY -m cli --version >/dev/null && \
           $PY -m cli --help    >/dev/null && \
           $PY -m cli detect    >/dev/null && \
           $PY -m cli list-models >/dev/null; then
            ok "cli version/help/detect/list-models"
        else
            fail "cli smoke"
        fi
    fi

    # ------------------------------------------------------------------------
    # 4. build
    # ------------------------------------------------------------------------
    if [ "${SKIP_BUILD:-0}" = "1" ]; then
        skip "build"
    else
        step "build (sdist + wheel)"
        if ! $PY -c "import build" >/dev/null 2>&1; then
            echo "${DIM}installing build…${RESET}"
            _install_pkg build
        fi
        if $PY -m build --outdir dist/local-check >/tmp/check-build.log 2>&1; then
            ok "python -m build ($(ls dist/local-check/ | wc -l | tr -d ' ') artifacts)"
            rm -rf dist/local-check
        else
            fail "python -m build (see /tmp/check-build.log)"
            tail -20 /tmp/check-build.log
        fi
    fi
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START ))
echo
echo "${BOLD}── summary ──${RESET}"
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "${GREEN}${BOLD}all checks passed${RESET} (${ELAPSED}s)"
    exit 0
else
    echo "${RED}${BOLD}${#FAILED[@]} check(s) failed${RESET} (${ELAPSED}s):"
    for f in "${FAILED[@]}"; do
        echo "  ${RED}✗${RESET} $f"
    done
    exit 1
fi
