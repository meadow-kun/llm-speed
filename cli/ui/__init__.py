"""Interactive TUI surface for the llm-speed CLI.

The modules here drive the post-redesign experience:

* `theme`     - colors / chrome / no-emoji palette.
* `progress`  - rich-based status helpers used by the bench flow.
* `welcome`   - first-run wizard (detect, offer install, smoke run, share).
* `auto`      - non-interactive auto-mode (sensible defaults, no prompts).
* `share`     - post-run share affordances (badge URLs, intent links, copy).

Everything in this package guards against `not sys.stdin.isatty()` so the
existing non-TTY bench flow used by CI keeps working.
"""

from __future__ import annotations
