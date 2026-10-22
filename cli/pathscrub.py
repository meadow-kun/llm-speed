"""Strip filesystem paths out of anything that leaves the machine (0.0.8).

Rule: no upload, and no signed local artifact, may contain a filesystem path.
A path becomes its basename at most; home directories and usernames never
survive. Applied to every string in the payload by ``signing.build_uploadable_report``
(which also produces the local-save copy), so new fields are covered by default.

Not touched: URLs (``https://host/path``), HF repo ids (``org/name``), relative
fragments without a leading root (``a/b``) — none of those name a local file.
"""

from __future__ import annotations

import re
from typing import Any

# POSIX absolute path or ~-path. The leading "/" or "~/" must not follow a word
# char, ":", "/" or "." so URLs ("https://x/y"), "host:8080/health" and relative
# "org/model" ids are left alone.
_POSIX_PATH_RE = re.compile(
    r"""(?<![\w:/.~-])(?:~[\w.-]*)?/(?:[^\s/'"`,;()<>\[\]{}|]+(?: [^\s/'"`,;()<>\[\]{}|]+){0,3}/)*([^\s/'"`,;()<>\[\]{}|]*)"""
)
# Windows drive or UNC path, either slash direction.
_WIN_PATH_RE = re.compile(
    r"""(?<![\w])(?:[A-Za-z]:|\\\\[^\s\\/'"]+)[\\/](?:[^\s\\/'"`,;()<>|]+(?: [^\s\\/'"`,;()<>|]+){0,3}[\\/])*([^\s\\/'"`,;()<>|]*)"""
)
# Directory names may contain spaces ("C:\\Users\\Jane Doe\\", "/Volumes/Ext SSD/"): a
# non-final segment may hold up to three single-space-separated words, since it is
# delimited by the next slash. The final segment still stops at whitespace.
# file:// URLs name a local file: reduce them to the basename too.
_FILE_URL_RE = re.compile(r"""file://(?:localhost)?(?=/|[A-Za-z]:)""")
# Bare home references that may survive as a token on their own.
_HOME_RE = re.compile(
    r"(?:/(?:Users|home)/[^\s/'\"]+|[A-Za-z]:\\Users\\[^\s\\'\"]+|/root)(?=$|[\s/\\'\"])"
)


def scrub_paths(text: str) -> str:
    """Replace every filesystem path in ``text`` with its basename."""
    if not text or ("/" not in text and "\\" not in text):
        return text

    def _base(m: re.Match[str]) -> str:
        return m.group(1) or ""

    out = _FILE_URL_RE.sub("", text)
    out = _WIN_PATH_RE.sub(_base, out)
    out = _POSIX_PATH_RE.sub(_base, out)
    out = _HOME_RE.sub("~", out)
    return out


def looks_like_path(text: str) -> bool:
    return bool(text) and scrub_paths(text) != text


def scrub_payload(obj: Any) -> Any:
    """Return a copy of a JSON-like tree with paths scrubbed from every string."""
    if isinstance(obj, dict):
        return {k: scrub_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_payload(v) for v in obj]
    if isinstance(obj, str):
        return scrub_paths(obj)
    return obj
