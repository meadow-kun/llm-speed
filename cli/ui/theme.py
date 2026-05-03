"""Colors, labels, and chrome strings used across the CLI TUI.

Centralised so a tweak to the brand palette or a wording change is a one-line
edit. NO emoji per the project style guide; status uses bracketed words like
[ok] / [warn] / [err]. Rich markup tags in this file map to a small palette:

* ``brand``  - primary teal accent (matches the badge accent + site)
* ``muted``  - dim chrome, used for separators and labels
* ``ok``     - green, success/affirmative
* ``warn``   - yellow, soft warnings the user should notice
* ``err``    - red, hard failures
* ``hint``   - cyan, secondary info / keyboard hints
"""

from __future__ import annotations

from rich.console import Console

# ---------------------------------------------------------------------------
# Brand wordmark + tagline
# ---------------------------------------------------------------------------

BRAND = "llm-speed"
SITE = "llm-speed.com"
SITE_URL = "https://llm-speed.com"
REPO_URL = "https://github.com/meadow-kun/llm-speed"
PRIVACY_URL = "https://llm-speed.com/privacy"

TAGLINE = "Benchmark any LLM on any hardware."

# ---------------------------------------------------------------------------
# Rich colors. Keep these as plain strings so the tests can grep markup.
# ---------------------------------------------------------------------------

C_BRAND = "bold #34d399"  # the badge accent green-teal
C_BRAND_DIM = "#34d399"
C_MUTED = "dim white"
C_OK = "bold green"
C_WARN = "bold yellow"
C_ERR = "bold red"
C_HINT = "cyan"
C_KBD = "reverse cyan"  # keyboard-key visual

# Symbols. ASCII only - safe in any tty, no emoji. Rich treats square
# brackets as markup, so we use plain runs of letters / dots and rely on
# the surrounding color tag for emphasis instead of brackets.
SYM_OK = "ok"
SYM_WARN = "warn"
SYM_ERR = "err"
SYM_RUN = ".."
SYM_BULLET = "-"
SYM_RULE = "-" * 60

# ---------------------------------------------------------------------------
# Console singleton
# ---------------------------------------------------------------------------

_console: Console | None = None


def get_console() -> Console:
    """Return the shared rich Console.

    Re-using one Console across modules keeps interleaved progress output and
    error messages on the same width / theme.
    """
    global _console
    if _console is None:
        _console = Console(highlight=False, soft_wrap=False)
    return _console


def banner() -> str:
    """Single-line wordmark used at the top of interactive screens."""
    return f"[{C_BRAND}]{BRAND}[/]  [dim]{TAGLINE}[/]"


def kbd(key: str, label: str) -> str:
    """Render a `[K]eybinding label` style hint."""
    return f"[{C_KBD}] {key} [/] [dim]{label}[/]"


def hint_bar(*pairs: tuple[str, str]) -> str:
    """Format a footer hint bar from (key, label) tuples."""
    return "   ".join(kbd(k, lbl) for k, lbl in pairs)
