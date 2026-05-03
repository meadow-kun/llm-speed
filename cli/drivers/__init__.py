"""Backend drivers. Importing this package side-effect-registers every driver.

Each driver module must:
1. Define a class implementing `cli.types.BackendDriver`.
2. Call `cli.registry.register_driver(NAME, factory)` at module import.

Driver names (canonical, must match):
    'llama.cpp', 'ollama', 'vllm', 'mlx', 'exllamav2', 'hosted-api'
"""

from __future__ import annotations

import importlib
import logging

LOG = logging.getLogger("cli.drivers")

# Import every driver module; each is responsible for registering itself.
# A driver that fails to import (missing optional dep) MUST log and continue.
_DRIVER_MODULES = (
    "cli.drivers.llama_cpp",
    "cli.drivers.ollama",
    "cli.drivers.hosted_api",
    "cli.drivers.mlx",
    "cli.drivers.vllm",
    "cli.drivers.exllamav2",
)

for _modname in _DRIVER_MODULES:
    try:
        importlib.import_module(_modname)
    except Exception as exc:  # noqa: BLE001 — drivers self-isolate
        LOG.debug("driver %s did not load: %s", _modname, exc)
