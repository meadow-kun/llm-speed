"""Post-run share affordances.

After a successful upload we print:

* the run page URL
* the badge SVG URL
* a Markdown snippet ready to paste into a README
* an X (Twitter) intent URL
* a Reddit submit URL

The user can press a single key (``c`` to copy the snippet, ``x`` to open the X
intent, ``r`` to open Reddit, ``s`` to skip). Cross-platform clipboard /
browser-open use stdlib ``subprocess`` only - ``pbcopy`` on macOS, ``xclip`` /
``wl-copy`` on Linux, ``clip.exe`` on Windows; ``open`` / ``xdg-open`` /
``start`` for browsers.

This module extracts the run id from the result URL the server hands back, so
it works whether the server returns ``{"id": "r_abc"}`` or
``{"url": "https://llm-speed.com/r/r_abc"}``.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import urllib.parse

from rich.console import Console
from rich.panel import Panel

from .theme import (
    C_BRAND,
    C_HINT,
    C_MUTED,
    C_OK,
    C_WARN,
    SITE_URL,
    get_console,
    hint_bar,
)

log = logging.getLogger("cli.ui.share")

# Server may return the bare id or a full URL. Accept both.
_ID_RE = re.compile(r"/r/([A-Za-z0-9_-]+)")


def extract_run_id(result_url: str) -> str | None:
    """Return the canonical run id from a result URL or pass-through string."""
    if not result_url:
        return None
    m = _ID_RE.search(result_url)
    if m:
        return m.group(1)
    # Accept a bare id like 'r_abc123'.
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", result_url):
        return result_url
    return None


def badge_url(run_id: str, *, site: str = SITE_URL) -> str:
    return f"{site.rstrip('/')}/badge/{run_id}.svg"


def run_url(run_id: str, *, site: str = SITE_URL) -> str:
    return f"{site.rstrip('/')}/r/{run_id}"


def markdown_snippet(
    run_id: str, *, label: str = "llm-speed", site: str = SITE_URL
) -> str:
    """Generic Markdown badge embed. Mirrors web/lib/distribution.ts shape.

    The web side knows the model + tps to put in the alt text; we don't have
    the run object here so we use a generic alt. The badge SVG itself carries
    the canonical labels.
    """
    return f"[![{label}]({badge_url(run_id, site=site)})]({run_url(run_id, site=site)})"


def x_intent_url(
    run_id: str, *, headline: str | None = None, site: str = SITE_URL
) -> str:
    text = headline or f"My LLM speed run on {site.replace('https://', '')}"
    params = urllib.parse.urlencode({"text": text, "url": run_url(run_id, site=site)})
    return f"https://twitter.com/intent/tweet?{params}"


def reddit_intent_url(
    run_id: str, *, title: str | None = None, site: str = SITE_URL
) -> str:
    title_str = title or "Crowdsourced LLM inference speed result"
    params = urllib.parse.urlencode(
        {"title": title_str, "url": run_url(run_id, site=site)}
    )
    return f"https://www.reddit.com/submit?{params}"


# ---------------------------------------------------------------------------
# Cross-platform helpers
# ---------------------------------------------------------------------------


def copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard write using native CLI tools. Returns True on success.

    Tries in order:
      * macOS: ``pbcopy``
      * Linux/Wayland: ``wl-copy``
      * Linux/X11: ``xclip -selection clipboard``, then ``xsel --clipboard``
      * Windows: ``clip.exe``
    """
    candidates: list[list[str]] = []
    if sys.platform == "darwin":
        candidates.append(["pbcopy"])
    elif sys.platform.startswith("linux"):
        candidates.append(["wl-copy"])
        candidates.append(["xclip", "-selection", "clipboard"])
        candidates.append(["xsel", "--clipboard", "--input"])
    elif sys.platform.startswith("win"):
        candidates.append(["clip"])

    for cmd in candidates:
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(
                cmd, input=text.encode("utf-8"), check=False, timeout=5
            )
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("clipboard tool %s failed: %s", cmd[0], exc)
    return False


def _is_safe_https_url(url: str) -> bool:
    """Reject anything that isn't a plain http(s) URL before passing it
    to a child process.

    The fallback ``cmd /c start "" <url>`` form on Windows interprets ``&``,
    ``^``, ``"`` in the URL argument — a malicious URL like
    ``"https://x.com/x\\"&calc&\\""`` could fork a `calc.exe` child.
    The realistic prerequisite is "the API server returned a hostile
    string" (api.llm-speed.com is hardened, but defence in depth costs
    nothing). Restrict to https-only with no embedded shell metachars.
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except (ValueError, AttributeError):
        return False
    if parsed.scheme not in {"https", "http"}:
        return False
    if not parsed.netloc:
        return False
    # Refuse any character that's a shell metachar on cmd.exe or POSIX shells.
    # Real URLs don't legitimately contain these — they'd be percent-encoded.
    for ch in url:
        if ch in '"\'`$\\&|;<>(){}[]*?!\n\r\t':
            return False
    return True


def open_in_browser(url: str) -> bool:
    """Open ``url`` in the user's default browser via stdlib ``webbrowser``.

    Falls back to platform `open` / `xdg-open` / `start` if webbrowser is
    unable to dispatch (rare but possible on minimal Linux containers).

    Refuses any URL that isn't a clean http(s) scheme with no shell
    metacharacters — see ``_is_safe_https_url``. Returns False on reject.
    """
    if not _is_safe_https_url(url):
        log.debug("refusing to open unsafe URL: %r", url[:200])
        return False
    try:
        import webbrowser

        if webbrowser.open(url):
            return True
    except Exception as exc:  # noqa: BLE001
        log.debug("webbrowser.open failed: %s", exc)

    cmds: list[list[str]] = []
    if sys.platform == "darwin":
        cmds.append(["open", url])
    elif sys.platform.startswith("linux"):
        cmds.append(["xdg-open", url])
    elif sys.platform.startswith("win"):
        cmds.append(["cmd", "/c", "start", "", url])
    for cmd in cmds:
        if not shutil.which(cmd[0]):
            continue
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.debug("browser-open via %s failed: %s", cmd[0], exc)
    return False


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def print_share_block(
    result_url: str,
    *,
    site: str = SITE_URL,
    console: Console | None = None,
) -> str | None:
    """Print the share affordance panel. Returns the run id if extractable."""
    cons = console or get_console()
    rid = extract_run_id(result_url)
    if not rid:
        cons.print(f"[{C_WARN}]could not extract run id from[/] {result_url}")
        return None

    page = run_url(rid, site=site)
    badge = badge_url(rid, site=site)
    snippet = markdown_snippet(rid, site=site)
    x = x_intent_url(rid, site=site)
    rd = reddit_intent_url(rid, site=site)

    body = (
        f"[{C_BRAND}]Run page[/]   {page}\n"
        f"[{C_BRAND}]Badge SVG[/]  {badge}\n"
        f"[{C_BRAND}]Markdown[/]\n"
        f"  [dim]{snippet}[/]\n"
        f"\n"
        f"[{C_HINT}]Share on X[/]      {x}\n"
        f"[{C_HINT}]Share on Reddit[/] {rd}\n"
    )
    cons.print(
        Panel(body, title="[bold]share this run[/]", border_style=C_BRAND, expand=False)
    )
    cons.print(
        hint_bar(
            ("c", "copy markdown"),
            ("x", "share on X"),
            ("r", "share on Reddit"),
            ("o", "open run page"),
            ("s", "skip"),
        )
    )
    return rid


def interactive_share_prompt(
    result_url: str,
    *,
    site: str = SITE_URL,
    console: Console | None = None,
    stdin=None,
) -> None:
    """Print the share block, then read a single keystroke and act on it.

    No-op when stdin is not a TTY (so CI / piped runs don't block).
    """
    cons = console or get_console()
    rid = print_share_block(result_url, site=site, console=cons)
    if rid is None:
        return

    inp = stdin if stdin is not None else sys.stdin
    if not (hasattr(inp, "isatty") and inp.isatty()):
        return

    cons.print()
    cons.print("[dim]press a key (c / x / r / o / s):[/] ", end="")
    try:
        line = inp.readline()
    except KeyboardInterrupt:
        cons.print()
        return
    if not line:
        return
    answer = line.strip().lower()[:1]

    if answer == "c":
        snippet = markdown_snippet(rid, site=site)
        if copy_to_clipboard(snippet):
            cons.print(f"[{C_OK}]copied markdown snippet to clipboard[/]")
        else:
            cons.print(f"[{C_WARN}]no clipboard tool found[/] - here it is:")
            cons.print(snippet)
    elif answer == "x":
        url = x_intent_url(rid, site=site)
        if open_in_browser(url):
            cons.print(f"[{C_OK}]opened X intent in browser[/]")
        else:
            cons.print(f"[{C_WARN}]could not open browser[/] - paste this:")
            cons.print(url)
    elif answer == "r":
        url = reddit_intent_url(rid, site=site)
        if open_in_browser(url):
            cons.print(f"[{C_OK}]opened Reddit submit in browser[/]")
        else:
            cons.print(f"[{C_WARN}]could not open browser[/] - paste this:")
            cons.print(url)
    elif answer == "o":
        url = run_url(rid, site=site)
        if open_in_browser(url):
            cons.print(f"[{C_OK}]opened run page in browser[/]")
        else:
            cons.print(f"[{C_WARN}]could not open browser[/] - paste this:")
            cons.print(url)
    else:
        cons.print(
            f"[{C_MUTED}]skipped sharing.[/] you can come back to {run_url(rid, site=site)} any time."
        )
