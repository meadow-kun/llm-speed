"""Driver and workload registration. Each backend / workload module calls
`register_driver` / `register_workload` at import time. The CLI entry point
imports the relevant subpackages to populate these maps before dispatch.

Re-registration is idempotent: the second call silently replaces the factory.
This matters because `python -m cli.workloads.foo` reimports the module under
`__main__`, which would otherwise raise on the duplicate registration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .types import BackendDriver, Workload

LOG = logging.getLogger("cli.registry")

_DRIVERS: dict[str, Callable[[], BackendDriver]] = {}
_WORKLOADS: dict[str, Callable[[], Workload]] = {}


def register_driver(name: str, factory: Callable[[], BackendDriver]) -> None:
    if name in _DRIVERS:
        LOG.debug("driver %r re-registered (replacing previous factory)", name)
    _DRIVERS[name] = factory


def register_workload(name: str, factory: Callable[[], Workload]) -> None:
    if name in _WORKLOADS:
        LOG.debug("workload %r re-registered (replacing previous factory)", name)
    _WORKLOADS[name] = factory


def get_driver(name: str) -> BackendDriver:
    if name not in _DRIVERS:
        raise KeyError(f"unknown driver {name!r}; available: {sorted(_DRIVERS)}")
    return _DRIVERS[name]()


def get_workload(name: str) -> Workload:
    if name not in _WORKLOADS:
        raise KeyError(f"unknown workload {name!r}; available: {sorted(_WORKLOADS)}")
    return _WORKLOADS[name]()


def all_driver_names() -> list[str]:
    return sorted(_DRIVERS)


def all_workload_names() -> list[str]:
    return sorted(_WORKLOADS)
