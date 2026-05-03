"""Tests for `cli.auto_install`.

Two invariants we must pin down:

* No install is ever run without an explicit y/n confirm. We assert the
  runner is NOT invoked when confirm() returns False.
* Every install plan exposes its full command string verbatim - the user is
  shown the exact thing we would run before any confirm prompt fires.
"""

from __future__ import annotations

import platform

import pytest

from cli import auto_install

# ---------------------------------------------------------------------------
# plan_for: shape per backend per OS
# ---------------------------------------------------------------------------


def test_plan_for_unknown_backend_is_unavailable():
    plan = auto_install.plan_for("nonsense")
    assert plan.available is False
    assert plan.skip_reason


def test_plan_for_ollama_includes_command_string():
    plan = auto_install.plan_for("ollama")
    # The command field is what the user is shown verbatim, regardless of OS.
    assert isinstance(plan.command, str)
    assert plan.command  # not empty


def test_plan_for_mlx_unavailable_off_apple_silicon():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        pytest.skip("can't assert mlx-unavailability on Apple Silicon hosts")
    plan = auto_install.plan_for("mlx")
    assert plan.available is False
    assert "MLX" in (plan.skip_reason or "") or "Apple" in (plan.skip_reason or "")


# ---------------------------------------------------------------------------
# confirm_and_install: refusal short-circuits
# ---------------------------------------------------------------------------


def test_confirm_and_install_refusal_does_not_run(monkeypatch: pytest.MonkeyPatch):
    plan = auto_install.InstallPlan(
        backend="testbackend",
        description="test",
        command="echo hi",
        argv=["echo", "hi"],
    )
    invocations: list[bool] = []

    def _runner(_plan):
        invocations.append(True)
        return 0

    out = auto_install.confirm_and_install(
        plan,
        confirm=lambda prompt: False,
        runner=_runner,
    )
    assert out is False
    assert invocations == []


def test_confirm_and_install_consent_runs_runner():
    plan = auto_install.InstallPlan(
        backend="testbackend",
        description="test",
        command="true",
        argv=["true"],
    )
    seen: list[auto_install.InstallPlan] = []

    def _runner(plan):
        seen.append(plan)
        return 0

    out = auto_install.confirm_and_install(
        plan,
        confirm=lambda prompt: True,
        runner=_runner,
    )
    assert out is True
    assert seen and seen[0].argv == ["true"]


def test_confirm_and_install_unavailable_plan_returns_false():
    plan = auto_install.InstallPlan(
        backend="testbackend",
        description="test",
        command="(no path)",
        argv=[],
        available=False,
        skip_reason="not supported",
    )
    runner_calls: list[bool] = []
    out = auto_install.confirm_and_install(
        plan,
        confirm=lambda prompt: True,  # would consent, but plan is unavailable
        runner=lambda p: runner_calls.append(True) or 0,
    )
    assert out is False
    assert runner_calls == []


# ---------------------------------------------------------------------------
# detect_missing: plain-list invariants
# ---------------------------------------------------------------------------


def test_detect_missing_returns_a_list_of_strings():
    out = auto_install.detect_missing()
    assert isinstance(out, list)
    for s in out:
        assert isinstance(s, str)


def test_detect_missing_only_offers_supported_backends():
    out = auto_install.detect_missing()
    for s in out:
        assert s in {"llama.cpp", "ollama", "mlx"}
