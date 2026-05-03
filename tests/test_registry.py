"""Tests for `cli.registry`. Verifies driver/workload registration, idempotent
re-registration, and `KeyError` on lookup miss with a helpful message.

We use throwaway fake driver/workload objects rather than importing real ones,
because the real drivers do environment probing that is expensive and slow.
"""

from __future__ import annotations

import pytest

from cli import registry
from cli.types import (
    BackendDetection,
    ChatRunOutcome,
    ModelRef,
    WorkloadResult,
)


class _FakeDriver:
    """Minimum surface to satisfy `BackendDriver` for registry tests."""

    name = "fake-driver"

    def detect(self) -> BackendDetection:
        return BackendDetection(available=True, name=self.name, version="0.0.0")

    def list_models(self) -> list[ModelRef]:
        return []

    def run_chat(self, model, prompt, **kwargs) -> ChatRunOutcome:
        return ChatRunOutcome(
            success=True,
            output_text="",
            prompt_tokens=0,
            output_tokens=0,
            ttft_ms=None,
            decode_token_times_ms=[],
            wall_ms=0.0,
        )


class _FakeWorkload:
    name = "fake-workload"
    suite_version = "suite-v1"

    def run(self, driver, model) -> WorkloadResult:
        return WorkloadResult(
            workload=self.name,
            suite_version=self.suite_version,
            backend=driver.name,
            backend_version=None,
            model=model,
        )


@pytest.fixture(autouse=True)
def _restore_registry_state():
    """Snapshot + restore the global registry maps so tests don't pollute each other.
    This also lets us run alongside the real driver imports without breaking them.
    """
    drivers_before = dict(registry._DRIVERS)
    workloads_before = dict(registry._WORKLOADS)
    yield
    registry._DRIVERS.clear()
    registry._DRIVERS.update(drivers_before)
    registry._WORKLOADS.clear()
    registry._WORKLOADS.update(workloads_before)


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


def test_register_and_get_driver_round_trip():
    registry.register_driver("fake-x", lambda: _FakeDriver())
    drv = registry.get_driver("fake-x")
    assert isinstance(drv, _FakeDriver)
    assert drv.name == "fake-driver"


def test_register_and_get_workload_round_trip():
    registry.register_workload("fake-w", lambda: _FakeWorkload())
    wl = registry.get_workload("fake-w")
    assert isinstance(wl, _FakeWorkload)
    assert wl.name == "fake-workload"


# ---------------------------------------------------------------------------
# Idempotent re-registration (the bug-fix this test pins down)
# ---------------------------------------------------------------------------


def test_re_registering_a_driver_replaces_the_factory():
    sentinel_a = _FakeDriver()
    sentinel_b = _FakeDriver()
    registry.register_driver("dup-d", lambda: sentinel_a)
    registry.register_driver("dup-d", lambda: sentinel_b)
    # Second call must win silently (no exception).
    assert registry.get_driver("dup-d") is sentinel_b


def test_re_registering_a_workload_replaces_the_factory():
    sentinel_a = _FakeWorkload()
    sentinel_b = _FakeWorkload()
    registry.register_workload("dup-w", lambda: sentinel_a)
    registry.register_workload("dup-w", lambda: sentinel_b)
    assert registry.get_workload("dup-w") is sentinel_b


# ---------------------------------------------------------------------------
# Unknown lookups
# ---------------------------------------------------------------------------


def test_get_driver_unknown_name_raises_keyerror_with_available_list():
    with pytest.raises(KeyError) as excinfo:
        registry.get_driver("nonexistent-backend")
    msg = str(excinfo.value)
    assert "nonexistent-backend" in msg
    assert "available" in msg.lower()


def test_get_workload_unknown_name_raises_keyerror_with_available_list():
    with pytest.raises(KeyError) as excinfo:
        registry.get_workload("nonexistent-workload")
    msg = str(excinfo.value)
    assert "nonexistent-workload" in msg
    assert "available" in msg.lower()


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def test_listing_helpers_return_sorted_names():
    registry.register_driver("aaa-driver", lambda: _FakeDriver())
    registry.register_driver("zzz-driver", lambda: _FakeDriver())
    names = registry.all_driver_names()
    assert "aaa-driver" in names and "zzz-driver" in names
    assert names == sorted(names)
