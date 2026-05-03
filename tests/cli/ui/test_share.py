"""Tests for `cli.ui.share` - the post-run share affordance helpers.

These cover the pure URL-builder layer (run id extraction, badge URL,
markdown snippet, x/reddit intents) plus a recording test for the share
panel that asserts the lines emitted to a recording rich Console.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from cli.ui import share

# ---------------------------------------------------------------------------
# extract_run_id
# ---------------------------------------------------------------------------


def test_extract_run_id_from_full_url():
    assert share.extract_run_id("https://llm-speed.com/r/r_abc123") == "r_abc123"


def test_extract_run_id_from_bare_id():
    assert share.extract_run_id("r_abc123") == "r_abc123"


def test_extract_run_id_returns_none_for_garbage():
    assert share.extract_run_id("") is None
    assert share.extract_run_id("not a url with weird chars !!!") is None


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def test_badge_url_uses_site_default():
    assert share.badge_url("r_abc") == "https://llm-speed.com/badge/r_abc.svg"


def test_run_url_uses_site_default():
    assert share.run_url("r_abc") == "https://llm-speed.com/r/r_abc"


def test_markdown_snippet_links_badge_to_run_page():
    out = share.markdown_snippet("r_abc")
    assert "https://llm-speed.com/badge/r_abc.svg" in out
    assert "https://llm-speed.com/r/r_abc" in out
    # Markdown image-in-link shape: [![alt](image)](link)
    assert out.startswith("[![") and out.endswith(")")


def test_x_intent_url_includes_text_and_url():
    out = share.x_intent_url("r_abc")
    assert "twitter.com/intent/tweet" in out
    assert "url=https" in out
    assert "text=" in out


def test_reddit_intent_url_includes_title_and_url():
    out = share.reddit_intent_url("r_abc")
    assert "reddit.com/submit" in out
    assert "url=https" in out
    assert "title=" in out


# ---------------------------------------------------------------------------
# print_share_block - recorded console output
# ---------------------------------------------------------------------------


def _make_recording_console() -> Console:
    return Console(file=io.StringIO(), record=True, width=120, color_system=None)


def test_print_share_block_emits_required_urls():
    cons = _make_recording_console()
    rid = share.print_share_block("https://llm-speed.com/r/r_abc", console=cons)
    assert rid == "r_abc"
    text = cons.export_text()
    assert "https://llm-speed.com/r/r_abc" in text
    assert "https://llm-speed.com/badge/r_abc.svg" in text
    assert "twitter.com/intent/tweet" in text
    assert "reddit.com/submit" in text


def test_print_share_block_returns_none_for_unparseable_url():
    cons = _make_recording_console()
    rid = share.print_share_block("not a url", console=cons)
    assert rid is None


# ---------------------------------------------------------------------------
# interactive_share_prompt - mocks stdin + clipboard
# ---------------------------------------------------------------------------


class _FakeStdin:
    def __init__(self, line: str) -> None:
        self._line = line

    def readline(self) -> str:
        return self._line

    def isatty(self) -> bool:
        return True


def test_interactive_share_prompt_skips_on_non_tty():
    """When stdin is not a tty we must not block waiting for input."""

    class _NonTty:
        def isatty(self) -> bool:
            return False

        def readline(self) -> str:  # pragma: no cover - never reached
            raise AssertionError("must not read")

    cons = _make_recording_console()
    share.interactive_share_prompt(
        "https://llm-speed.com/r/r_xyz",
        console=cons,
        stdin=_NonTty(),
    )
    text = cons.export_text()
    # Block was rendered, but no key was consumed.
    assert "r_xyz" in text


def test_interactive_share_prompt_copy_path(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, str] = {}

    def _fake_copy(text: str) -> bool:
        captured["text"] = text
        return True

    monkeypatch.setattr(share, "copy_to_clipboard", _fake_copy)
    cons = _make_recording_console()
    share.interactive_share_prompt(
        "https://llm-speed.com/r/r_zzz",
        console=cons,
        stdin=_FakeStdin("c\n"),
    )
    assert "r_zzz" in captured["text"]
    assert "https://llm-speed.com/badge/r_zzz.svg" in captured["text"]
    text = cons.export_text()
    assert "copied" in text.lower()
