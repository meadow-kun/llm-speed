"""Interpreter flags for every `python -m cli ...` the agent starts (Security Desk R-71, R-74).

`-m` puts the working directory first on sys.path, so a stray top-level `cli/` package
there (a llm-speed-web checkout has one) would run instead of the installed CLI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BREW_FORMULA_DIR = "llm-speed"


def isolation_flags() -> list[str]:
    """-I in a virtual environment (uv tool / pipx / Homebrew): no cwd on sys.path, PYTHON*
    env and user site ignored. Outside a venv the package may live in user site
    (`pip install --user`), which -I would hide: there -P (3.11+) still keeps the cwd off
    sys.path, and on 3.10 there is no flag for that, so callers also pin the cwd."""
    if sys.prefix != sys.base_prefix:
        return ["-I"]
    return ["-P"] if sys.version_info >= (3, 11) else []


def _opt_path(path: str) -> str | None:
    """<brew prefix>/Cellar/llm-speed/<version>/<rest> -> <brew prefix>/opt/llm-speed/<rest>."""
    parts = Path(path).parts
    for i in range(len(parts) - 2):
        if parts[i] == "Cellar" and parts[i + 1] == BREW_FORMULA_DIR:
            return str(Path(*parts[:i], "opt", BREW_FORMULA_DIR, *parts[i + 3 :]))
    return None


def stable_python(executable: str | None = None, prefix: str | None = None) -> str:
    """The interpreter path to register in service files and to re-exec (0.0.14).
    Homebrew runs llm-speed from a versioned keg, <brew prefix>/Cellar/llm-speed/<ver>/,
    which `brew upgrade` + cleanup deletes; its <brew prefix>/opt/llm-speed link always
    points at the installed version, so use that. pipx / uv / venv paths are already
    stable and come back unchanged. Kept unresolved so the venv stays active."""
    exe = os.path.abspath(executable or sys.executable)
    for cand in (exe, os.path.realpath(exe)):
        opt = _opt_path(cand)
        if opt and os.path.exists(opt):
            return opt
    opt = _opt_path(os.path.realpath(prefix or sys.prefix))
    if opt:
        py = os.path.join(opt, "bin", os.path.basename(exe))
        if os.path.exists(py):
            return py
    return exe
