"""Tests for `cli.ui.progress` - StepReporter + spinner.

We feed each helper a recording rich Console and assert on the lines emitted
in order. The spinner test only checks the non-tty fallback path; the live
path involves Rich's Live renderer which is harder to record deterministically.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from cli.ui import progress


def _make_console() -> Console:
    return Console(file=io.StringIO(), record=True, width=120, color_system=None)


# ---------------------------------------------------------------------------
# StepReporter
# ---------------------------------------------------------------------------


def test_step_reporter_emits_status_lines_in_order():
    cons = _make_console()
    sr = progress.StepReporter(
        ["fingerprint", "pick backend", "smoke run"], console=cons
    )

    sr.start("fingerprint", detail="probing hardware")
    sr.ok("fingerprint", detail="M3 Pro / 36GB")

    sr.start("pick backend")
    sr.warn("pick backend", detail="ollama unavailable; falling back to llama.cpp")

    sr.start("smoke run")
    sr.fail("smoke run", detail="model load timeout")

    text = cons.export_text()
    # Lines must show up in the order we emitted them.
    idx = lambda needle: text.index(needle)  # noqa: E731
    assert idx("fingerprint") < idx("pick backend") < idx("smoke run")
    # Status markers from the theme are present (ASCII bare words, no
    # square brackets so rich doesn't eat them as markup).
    assert "ok" in text
    assert "warn" in text
    assert "err" in text


def test_step_reporter_summary_lists_all_steps():
    cons = _make_console()
    sr = progress.StepReporter(["a", "b"], console=cons)
    sr.ok("a", detail="done")
    sr.fail("b", detail="boom")
    sr.summary()
    text = cons.export_text()
    assert "Run summary" in text
    assert "a" in text and "b" in text


# ---------------------------------------------------------------------------
# spinner: non-tty fallback
# ---------------------------------------------------------------------------


def test_spinner_emits_plain_line_on_non_tty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(progress, "_tty", lambda: False)
    cons = _make_console()
    with progress.spinner("loading model", console=cons):
        pass
    text = cons.export_text()
    assert "loading model" in text
