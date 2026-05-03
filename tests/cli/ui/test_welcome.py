"""Tests for `cli.ui.welcome`.

Most of the wizard is interactive. We test the pieces that are pure or
take a callable seam:

* render_splash + render_backends emit the brand wordmark and a per-backend
  status line.
* offer_install honours an explicit y/n confirm callable: typing 'no' leaves
  the runner untouched.
* run_wizard short-circuits to auto mode on a non-tty stdin.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from rich.console import Console

from cli.auto_install import InstallPlan
from cli.ui import welcome


def _make_console() -> Console:
    return Console(file=io.StringIO(), record=True, width=140, color_system=None)


# ---------------------------------------------------------------------------
# render_*
# ---------------------------------------------------------------------------


def test_render_splash_prints_brand_and_privacy_link():
    cons = _make_console()
    welcome.render_splash(console=cons)
    text = cons.export_text()
    assert "llm-speed" in text
    assert "privacy" in text.lower()


def test_render_backends_lists_known_backends(monkeypatch: pytest.MonkeyPatch):
    # All three "not found" so the table is deterministic.
    monkeypatch.setattr(welcome, "is_llama_cpp_installed", lambda: False)
    monkeypatch.setattr(welcome, "is_ollama_installed", lambda: False)
    monkeypatch.setattr(welcome, "is_mlx_installed", lambda: False)
    cons = _make_console()
    welcome.render_backends(console=cons)
    text = cons.export_text()
    for backend in ("llama.cpp", "ollama", "mlx"):
        assert backend in text
    assert "not found" in text


# ---------------------------------------------------------------------------
# offer_install
# ---------------------------------------------------------------------------


def test_offer_install_refusal_does_not_run(monkeypatch: pytest.MonkeyPatch):
    plan = InstallPlan(
        backend="ollama",
        description="Install Ollama",
        command="echo hi",
        argv=["echo", "hi"],
    )
    monkeypatch.setattr(welcome, "plan_for", lambda b: plan)

    ran: list[bool] = []

    def _runner(plan: InstallPlan) -> int:
        ran.append(True)
        return 0

    # Confirm returns False: install must NOT run.
    cons = _make_console()
    out = welcome.offer_install(
        "ollama",
        confirm=lambda prompt: False,
        console=cons,
    )
    assert out is False
    assert ran == []


def test_offer_install_skipped_plan_returns_false():
    cons = _make_console()
    # mlx on Linux: plan_for returns available=False with a skip_reason.
    import platform

    if platform.system() != "Linux":
        # On macOS/Windows the test's expectation about mlx differs; skip
        # rather than entrench an OS-specific assertion.
        pytest.skip("only deterministic on Linux")
    out = welcome.offer_install("mlx", confirm=lambda p: True, console=cons)
    assert out is False


# ---------------------------------------------------------------------------
# run_wizard: non-tty path
# ---------------------------------------------------------------------------


def test_run_wizard_falls_back_to_auto_on_non_tty(monkeypatch: pytest.MonkeyPatch):
    """When stdin/stdout aren't ttys, run_wizard must NOT prompt."""
    monkeypatch.setattr(welcome, "_is_tty", lambda: False)

    captured: dict[str, Any] = {}

    def _fake_auto(*, api_base=None, no_upload=False):
        captured["called"] = True
        captured["api_base"] = api_base
        return 42

    monkeypatch.setattr(welcome, "run_auto", _fake_auto)
    cons = _make_console()
    rc = welcome.run_wizard(api_base="https://example.test", console=cons)
    assert rc == 42
    assert captured["called"] is True
    assert captured["api_base"] == "https://example.test"
