"""Workloads. Importing this package side-effect-registers every workload.

Each workload module must:
1. Define a class implementing `cli.types.Workload`.
2. Call `cli.registry.register_workload(NAME, factory)` at import time.

Workload names (canonical, must match `cli.config.DEFAULT_WORKLOADS`):
    'chat-short', 'chat-long', 'long-context-decay', 'concurrent-decode', 'agent-trace'

FACTORY SURFACE: `_WORKLOAD_MODULES` below is one of the per-vertical extraction
points the property factory renders. To add/replace workloads for a vertical,
edit this tuple (each module self-registers on import). New *metrics* do NOT go
here — they ride the existing `workload_results.extras` JSON emitted per workload.
"""

from __future__ import annotations

import importlib
import logging

LOG = logging.getLogger("cli.workloads")

# FACTORY SURFACE (per-vertical): the set of workload modules to import+register.
_WORKLOAD_MODULES = (
    "cli.workloads.chat_short",
    "cli.workloads.chat_long",
    "cli.workloads.long_context_decay",
    "cli.workloads.concurrent_decode",
    "cli.workloads.agent_trace",
    "cli.workloads.txt2img",
)

for _modname in _WORKLOAD_MODULES:
    try:
        importlib.import_module(_modname)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("workload %s did not load: %s", _modname, exc)
