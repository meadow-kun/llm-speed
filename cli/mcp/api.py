"""
Thin httpx client over api.llm-speed.com.

Exposes:
  - list_runs(limit) -> list[RunSummary]
  - get_run(run_id) -> Run | None
  - get_manifest() -> SuiteManifest

Caching: the full /v1/results listing is wrapped in a 5-minute TTLCache
so an interactive MCP session doesn't hammer the API. Single-run details
get a smaller TTL because we expect users to spot-check different runs.
"""

from __future__ import annotations

import os
from typing import Any, cast

import httpx
from cachetools import TTLCache

DEFAULT_API_BASE = "https://api.llm-speed.com"
DEFAULT_SITE_BASE = "https://llm-speed.com"
API_BASE = os.environ.get("LLM_SPEED_API_BASE", DEFAULT_API_BASE)
SITE_BASE = os.environ.get("LLM_SPEED_SITE_BASE", DEFAULT_SITE_BASE)
USER_AGENT = "llm-speed-mcp/0.1 (+https://llm-speed.com)"

# 5-minute TTL — matches the spec. Single slot for the listing; a small
# LRU for individual runs since detail pages are queried by id.
_listings_cache: TTLCache[int, list[dict[str, Any]]] = TTLCache(maxsize=4, ttl=300)
_run_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=128, ttl=300)
_manifest_cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=1, ttl=900)


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=API_BASE,
        timeout=httpx.Timeout(15.0, connect=5.0),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


def list_runs(limit: int = 500) -> list[dict[str, Any]]:
    """Return run summaries from /v1/results, cached for 5 minutes."""
    cached = _listings_cache.get(limit)
    if cached is not None:
        return cached
    with _client() as c:
        resp = c.get("/v1/results", params={"limit": limit})
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
    runs = cast(list[dict[str, Any]], data.get("runs", []))
    _listings_cache[limit] = runs
    return runs


def get_run(run_id: str) -> dict[str, Any] | None:
    """Return the full run record from /v1/results/{id}, or None on 404."""
    cached = _run_cache.get(run_id)
    if cached is not None:
        return cached
    with _client() as c:
        resp = c.get(f"/v1/results/{run_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
    _run_cache[run_id] = data
    return data


def get_manifest() -> dict[str, Any]:
    """Return the suite manifest from /v1/suite/manifest."""
    cached = _manifest_cache.get("default")
    if cached is not None:
        return cached
    with _client() as c:
        resp = c.get("/v1/suite/manifest")
        resp.raise_for_status()
        data = cast(dict[str, Any], resp.json())
    _manifest_cache["default"] = data
    return data


def run_url(run_id: str) -> str:
    """Canonical /r/<id> URL for citation."""
    return f"{SITE_BASE}/r/{run_id}"


def model_url(slug: str) -> str:
    return f"{SITE_BASE}/m/{slug}"


def hardware_url(slug: str) -> str:
    return f"{SITE_BASE}/hw/{slug}"


def vs_url(a_slug: str, b_slug: str) -> str:
    """Canonical /vs/<a>-vs-<b> deeplink."""
    return f"{SITE_BASE}/vs/{a_slug}-vs-{b_slug}"
