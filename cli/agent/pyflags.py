"""Interpreter flags for every `python -m cli ...` the agent starts (Security Desk R-71, R-74).

`-m` puts the working directory first on sys.path, so a stray top-level `cli/` package
there (a llm-speed-web checkout has one) would run instead of the installed CLI.
"""

from __future__ import annotations

import sys


def isolation_flags() -> list[str]:
    """-I in a virtual environment (uv tool / pipx / Homebrew): no cwd on sys.path, PYTHON*
    env and user site ignored. Outside a venv the package may live in user site
    (`pip install --user`), which -I would hide: there -P (3.11+) still keeps the cwd off
    sys.path, and on 3.10 there is no flag for that, so callers also pin the cwd."""
    if sys.prefix != sys.base_prefix:
        return ["-I"]
    return ["-P"] if sys.version_info >= (3, 11) else []
