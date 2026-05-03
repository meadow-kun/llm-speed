#!/usr/bin/env bash
# Build a shiv-packaged single-file zipapp of llm-speed.
#
# Output: dist/llm-speed-${VERSION}-${PLATFORM}-${ARCH}.pyz
#
# A .pyz is a Python zipapp: it bundles every dep into a single file but still
# requires a Python interpreter on the target machine (it embeds the shebang
# `/usr/bin/env python3`). For users who already have Python (most local-LLM
# users do), this is a "no-pip-needed" install.
#
# For a true single-file native binary that does NOT need Python on the target,
# we'd use PyOxidizer in a future epic. Shiv is dramatically simpler and
# covers the 80% case for free.
#
# Optional: BUNDLE_PYTHON=1 will bundle a portable Python runtime via
# python-build-standalone. NOT IMPLEMENTED yet — leave for a future epic.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Read version directly from pyproject.toml (no extra deps).
VERSION="$(
  python3 - <<'PY'
import re, sys
text = open("pyproject.toml").read()
m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
if not m:
    sys.exit("could not find version in pyproject.toml")
print(m.group(1))
PY
)"

# Detect platform / arch tags compatible with our release naming.
case "$(uname -s)" in
  Linux*)  PLATFORM="linux" ;;
  Darwin*) PLATFORM="darwin" ;;
  MINGW*|MSYS*|CYGWIN*) PLATFORM="win32" ;;
  *) PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')" ;;
esac

case "$(uname -m)" in
  x86_64|amd64) ARCH="x64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *) ARCH="$(uname -m)" ;;
esac

OUT_DIR="${ROOT}/dist"
mkdir -p "${OUT_DIR}"
OUT="${OUT_DIR}/llm-speed-${VERSION}-${PLATFORM}-${ARCH}.pyz"

# Ensure shiv is available.
if ! command -v shiv >/dev/null 2>&1; then
  echo "shiv not on PATH; installing into a throwaway venv..."
  python3 -m venv "${ROOT}/.shiv-venv"
  "${ROOT}/.shiv-venv/bin/pip" install --upgrade pip
  "${ROOT}/.shiv-venv/bin/pip" install shiv
  SHIV="${ROOT}/.shiv-venv/bin/shiv"
else
  SHIV="$(command -v shiv)"
fi

echo "Building ${OUT} ..."
"${SHIV}" \
  --console-script llm-speed \
  --output-file "${OUT}" \
  --python "/usr/bin/env python3" \
  --site-packages-cache \
  --compile-pyc \
  --reproducible \
  .

echo "Built: ${OUT}"
ls -lh "${OUT}"

if [[ "${BUNDLE_PYTHON:-0}" == "1" ]]; then
  echo "BUNDLE_PYTHON=1 is not yet implemented; skipping portable-Python step."
  echo "TODO: pull python-build-standalone, repack as a single-file launcher."
fi
