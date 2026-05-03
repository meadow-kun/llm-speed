"""Workloads. Importing this package side-effect-registers every workload.

Each workload module must:
1. Define a class implementing `cli.types.Workload`.
2. Call `cli.registry.register_workload(NAME, factory)` at import time.

Workload names (canonical, must match `cli.config.DEFAULT_WORKLOADS`):
    'chat-short', 'chat-long', 'long-context-decay', 'concurrent-decode', 'agent-trace'
"""

from __future__ import annotations

import importlib
import logging

LOG = logging.getLogger("cli.workloads")

_WORKLOAD_MODULES = (
    "cli.workloads.chat_short",
    "cli.workloads.chat_long",
    "cli.workloads.long_context_decay",
    "cli.workloads.concurrent_decode",
    "cli.workloads.agent_trace",
)

for _modname in _WORKLOAD_MODULES:
    try:
        importlib.import_module(_modname)
    except Exception as exc:  # noqa: BLE001
        LOG.debug("workload %s did not load: %s", _modname, exc)
