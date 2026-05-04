"""
Python port of `web/lib/seo.ts` slug helpers.

Keeps deduplication identical between the website (which renders /m/<slug>
and /hw/<slug>) and the MCP server (which has to recognise free-text user
input like "llama 3.3 70b" or "RTX 5090 (32GB)" and route it to the same
canonical slug the website uses).

Source of truth: web/lib/seo.ts. If the TypeScript regex changes, mirror it
here and add a parity case to tests/test_slugs.py.
"""

from __future__ import annotations

import re

# Mirror of ORG_PREFIXES in web/lib/seo.ts. Sorted by length descending so
# longer prefixes (e.g. "mlx-community", "deepseek-ai") match before their
# shorter parents ("mlx", "deepseek"). The TS Set is iterated in insertion
# order — which happens to put the long ones first — but we don't depend on
# Python set iteration order to get the same outcome.
_ORG_PREFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "mlx-community",
            "mlx",
            "lmstudio-community",
            "bartowski",
            "thebloke",
            "unsloth",
            "qwen",
            "meta-llama",
            "google",
            "microsoft",
            "deepseek-ai",
            "deepseek",
            "anthropic",
            "openai",
        ),
        key=len,
        reverse=True,
    )
)

# Mirror of QUANT_TAIL in web/lib/seo.ts. The TS regex is unicode-flagged but
# operates on ASCII, so re.IGNORECASE is enough.
_QUANT_TAIL = re.compile(
    r"(-(?:[0-9]+bit|mxfp[0-9]+|gguf|awq|gptq|exl[23]|q[0-9](?:[-_][a-z0-9]+)?|mlx))+$",
    re.IGNORECASE,
)

# Mirror of toSlugLocal in web/lib/seo.ts. Python's re does not support
# Unicode property escapes (\p{L}), but the `regex` module does — we avoid
# adding a third-party dep and use the ASCII-equivalent approximation that
# matches the website's behavior for every label observed in the wild.
# Anything that isn't ASCII letter/digit gets folded to a separator.
_NON_ALPHANUM = re.compile(r"[^A-Za-z0-9]+")


def _to_slug_local(input_str: str) -> str:
    """Mirror of toSlugLocal in seo.ts."""
    s = input_str.lower()
    s = _NON_ALPHANUM.sub("-", s)
    s = s.strip("-")
    return s


def clean_model_display_name(name: str | None) -> str:
    """Mirror of cleanModelDisplayName in seo.ts."""
    if not name:
        return ""
    lower = name.lower()
    for prefix in _ORG_PREFIXES:
        if lower.startswith(prefix + "-") or lower.startswith(prefix + "/"):
            return name[len(prefix) + 1 :]
    return name


def _strip_quant_suffix(slug: str) -> str:
    """Mirror of stripQuantSuffix in seo.ts."""
    return _QUANT_TAIL.sub("", slug)


def canonical_model_slug(name: str | None) -> str:
    """Mirror of canonicalModelSlug in seo.ts."""
    if not name:
        return ""
    return _strip_quant_suffix(_to_slug_local(clean_model_display_name(name)))


# Mirror of the parenthesised-suffix strip in canonicalHardwareSlug.
_PAREN_CHUNK = re.compile(r"\s*\([^)]*\)")


def canonical_hardware_slug(label: str | None) -> str:
    """Mirror of canonicalHardwareSlug in seo.ts.

    Strips parenthesised annotations (typically VRAM / unified-memory tags
    like "(32GB)" or "(192GB Unified)") before slugifying. Also strips a
    trailing "+ ... + NN GB" tail of accelerator-summary strings so the
    label "RTX 5090 (32GB) + AMD Ryzen 7 9850X3D 8-Core Processor (8c) + 30GB"
    collapses to "rtx-5090".

    The "+ ... " collapse is an MCP-server-only addition: the website
    derives /hw slugs from a curated catalog and does not need this, but
    the MCP server has to canonicalise raw accelerator_summary strings.
    """
    if not label:
        return ""
    stripped = _PAREN_CHUNK.sub("", label).strip()
    # MCP-only: take the first "+"-delimited segment (the GPU/accelerator
    # itself) so we dedupe across CPU/RAM variants of the same accelerator.
    head = stripped.split("+", 1)[0].strip()
    return _to_slug_local(head)
