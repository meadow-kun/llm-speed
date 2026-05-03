"""`llm-speed self-update` — STUB.

Fetches the suite manifest from the API. If the bundled `SUITE_VERSION` matches
the server's, prints "already on suite-vX". Otherwise prints what changed and
notes that automatic application is not yet implemented.
"""

from __future__ import annotations

import logging

import httpx
from rich.console import Console

from .. import SUITE_VERSION
from ..config import DEFAULT_API_BASE

log = logging.getLogger("cli.commands.self_update")


def cmd_self_update(args) -> int:
    console = Console()
    api_base = getattr(args, "api_base", None) or DEFAULT_API_BASE
    url = f"{api_base.rstrip('/')}/v1/suite/manifest"
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        console.print(f"[red]could not reach manifest:[/red] {exc}")
        return 1

    if resp.status_code != 200:
        console.print(f"[red]manifest fetch failed:[/red] HTTP {resp.status_code}")
        return 1

    try:
        manifest = resp.json()
    except ValueError as exc:
        console.print(f"[red]manifest is not JSON:[/red] {exc}")
        return 1

    remote_ver = manifest.get("suite_version")
    if remote_ver == SUITE_VERSION:
        console.print(f"already on {SUITE_VERSION}")
        return 0

    console.print(f"[yellow]bundled:[/yellow] {SUITE_VERSION}")
    console.print(f"[yellow]remote: [/yellow] {remote_ver}")
    changes = manifest.get("changes") or []
    if changes:
        console.print("Changes:")
        for c in changes:
            console.print(f"  - {c}")
    console.print(
        "[dim]TODO: automatic update coming in 0.1; for now, "
        "upgrade the package (pipx upgrade llm-speed).[/dim]"
    )
    return 0
