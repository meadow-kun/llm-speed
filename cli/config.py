"""Constants, paths, defaults."""

from __future__ import annotations

import os
from pathlib import Path

from . import SUITE_VERSION

CACHE_DIR = Path(
    os.environ.get("LLM_SPEED_CACHE", str(Path.home() / ".cache" / "llm-speed"))
)
CONFIG_DIR = Path(
    os.environ.get("LLM_SPEED_CONFIG", str(Path.home() / ".config" / "llm-speed"))
)
DEFAULT_API_BASE = os.environ.get("LLM_SPEED_API", "https://api.llm-speed.com")

# Default workloads run by `llm-speed bench` with no flags.
# FACTORY SURFACE (per-vertical): the property factory renders this tuple per
# vertical. Names must be a subset of the registered workloads in
# cli.workloads._WORKLOAD_MODULES. New metrics ride workload_results.extras,
# not this list.
DEFAULT_WORKLOADS = (
    "chat-short",
    "chat-long",
    "concurrent-decode",
    "agent-trace",
)
QUICK_WORKLOADS = ("chat-short",)

__all__ = [
    "CACHE_DIR",
    "CONFIG_DIR",
    "DEFAULT_API_BASE",
    "DEFAULT_WORKLOADS",
    "QUICK_WORKLOADS",
    "SUITE_VERSION",
]
