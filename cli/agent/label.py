"""The machine label sent when pairing (`agent enroll` / `agent install`).

Default (no --label): ``<chip-slug>-<random>``, e.g. ``apple-m3-ultra-7f3k`` or
``rtx-5090-a91c``. The chip slug comes only from the coarse capabilities this CLI
already sends (Apple chip name, else NVIDIA GPU name, else ``<os>-<arch>``); never
the hostname, username, home path, serials or any other identifier. It is generated
once, saved in agent/config.json and reused on every re-pairing.

A user --label is checked locally against the queue's rules (worker/src/agent-queue.ts,
handlePair / approve: a string, non-empty after trimming, at most 64 characters, and
not path-like per PATH_RE) so the user gets a clear error before any request.
"""

from __future__ import annotations

import re
import secrets

MAX_LEN = 64  # server limit (agent-queue.ts: label.length > 64 -> 400)
DEFAULT_MAX_LEN = 40  # generated labels stay well under the limit
SUFFIX_LEN = 4
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
# Python port of PATH_RE in worker/src/agent-queue.ts (kept in sync by a parity test).
PATH_RE = re.compile(
    r"""(^|[\s"'=(:,\[])(~[\\/]|/[^\s/"']+/[^\s"']*|[A-Za-z]:[\\/]|\\\\[^\s\\]+\\)"""
)
# Vendor words that only make the slug longer ("NVIDIA GeForce RTX 5090" -> rtx-5090).
_DROP_PREFIXES = ("nvidia-", "geforce-")


class LabelError(ValueError):
    pass


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def chip_slug(caps: dict) -> str:
    name = caps.get("chip") or caps.get("gpu_name")
    slug = slugify(name) if name else ""
    changed = True
    while changed:
        changed = False
        for p in _DROP_PREFIXES:
            if slug.startswith(p) and len(slug) > len(p):
                slug, changed = slug[len(p) :], True
    if not slug:
        slug = slugify(f"{caps.get('os') or ''}-{caps.get('arch') or ''}") or "cpu"
    room = DEFAULT_MAX_LEN - SUFFIX_LEN - 1
    return slug[:room].rstrip("-") or "cpu"


def generate_default_label(caps: dict) -> str:
    suffix = "".join(secrets.choice(_ALPHABET) for _ in range(SUFFIX_LEN))
    return f"{chip_slug(caps)}-{suffix}"


def validate_label(label: object) -> str:
    """Return the trimmed label or raise LabelError with a message for the user."""
    if not isinstance(label, str):
        raise LabelError("label must be text")
    s = label.strip()
    if not s:
        raise LabelError("label is empty")
    if len(s) > MAX_LEN:
        raise LabelError(
            f"label is {len(s)} characters; the queue allows at most {MAX_LEN}"
        )
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        raise LabelError("label contains control characters")
    if PATH_RE.search(s):
        raise LabelError(
            "label looks like a file path (the queue rejects those); use a short name "
            "such as mac-studio-m3-ultra"
        )
    return s


def is_valid(label: object) -> bool:
    try:
        validate_label(label)
        return True
    except LabelError:
        return False
