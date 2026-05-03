"""`llm-speed login` — STUB. Generates a state token, prints the URL.

The actual GitHub OAuth callback is server-side and out of scope for this
agent. This command just sets up the client side: a random state token written
to the config dir so a subsequent `claim` step can verify it.
"""

from __future__ import annotations

import logging
import secrets

from rich.console import Console

from ..config import CONFIG_DIR

log = logging.getLogger("cli.commands.login")

STATE_PATH = CONFIG_DIR / "pending-oauth-state.txt"


def cmd_login(args) -> int:
    console = Console()
    state = secrets.token_urlsafe(24)
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(state, encoding="utf-8")
    except OSError as exc:
        console.print(f"[red]could not persist state:[/red] {exc}")
        return 1

    url = f"https://llm-speed.com/cli/auth?state={state}"
    console.print(f"Visit [bold]{url}[/bold] to claim your contributor profile.")
    console.print(
        "[dim]TODO: server-side OAuth callback handler is not yet wired; "
        "this command just stores the state token locally.[/dim]"
    )
    return 0
