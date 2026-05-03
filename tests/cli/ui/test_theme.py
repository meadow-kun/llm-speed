"""Tests for `cli.ui.theme` - small but worth pinning so accidental palette
churn (e.g. introducing emoji) breaks CI."""

from __future__ import annotations

from cli.ui import theme


def test_no_emoji_in_brand_strings():
    """The project style guide forbids emoji in CLI output."""
    for s in (
        theme.BRAND,
        theme.TAGLINE,
        theme.SYM_OK,
        theme.SYM_WARN,
        theme.SYM_ERR,
        theme.SYM_RUN,
    ):
        # Emoji codepoints live above U+1F000 (Misc Symbols and Pictographs +).
        for ch in s:
            assert ord(ch) < 0xF000, f"unexpected high codepoint in {s!r}: {ch!r}"


def test_brand_url_constants_match_meadow_kun_identity():
    assert theme.SITE_URL == "https://llm-speed.com"
    assert theme.REPO_URL == "https://github.com/meadow-kun/llm-speed"
    # Personal identity, not "Philip" or work orgs.
    assert "philip" not in theme.REPO_URL.lower()
    assert "nordenfelt" not in theme.REPO_URL.lower()


def test_kbd_renders_a_key_label_pair():
    out = theme.kbd("c", "copy")
    assert "c" in out
    assert "copy" in out


def test_hint_bar_joins_pairs():
    out = theme.hint_bar(("c", "copy"), ("q", "quit"))
    assert "copy" in out and "quit" in out
