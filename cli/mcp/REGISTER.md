# Registry submission checklist

Three public registries to list the llm-speed MCP server on. None
auto-submit — these are forms / PRs to fill out by hand.

The MCP server is the `mcp` subcommand of the main `llm-speed` CLI
(boot with `llm-speed mcp`); the install command in every registry
is the standard CLI install plus the `[mcp]` extra.

Don't submit until:

1. The `llm-speed` wheel with `[mcp]` extra is verifiable end-to-end:
   `pipx install 'llm-speed[mcp]' && llm-speed mcp` boots the server
   and `tools/list` returns all five tools.
2. The wheel sha256 chain is aligned across `llm-speed.com/dist/` and
   `github.com/meadow-kun/llm-speed/releases/...` (run `llm-speed verify`).
3. `cli/mcp/README.md` is current.

---

## 1. Smithery — https://smithery.ai

Smithery is the dominant MCP discovery hub.

- **Explicit listing**: visit https://smithery.ai/new and submit the
  GitHub repo URL directly. Form fields:
  - Name: `llm-speed`
  - Description (1-line): `Query the llm-speed.com leaderboard inline — every answer cited.`
  - Category: `Data` (sub-tag: `Benchmarks`)
  - Install command: `pipx install 'llm-speed[mcp]'`
  - Run command: `llm-speed mcp`
  - Tools manifest: paste the `tools/list` response from
    `printf ... | llm-speed mcp` (see README "Verify").
  - Author: `meadow-kun`
  - License: `Apache-2.0`

After listing, Smithery generates a Claude Desktop one-click install
URL. Copy that into the project README's "Install" section.

---

## 2. MCPT — https://mcpt.org (Model Context Protocol Tools)

Curated list. Add via PR to the registry repo:

- Repo: https://github.com/punkpeye/awesome-mcp-servers (or whatever
  the canonical mcpt.org backing repo is at submission time — check the
  site footer; the hub has bounced between custodians).
- File to edit: `README.md` under the appropriate category. We fit
  best under `Data & Analytics` or `Developer Tools`.
- Entry shape (mirror existing entries — usually a one-liner):

  ```md
  - [llm-speed](https://github.com/meadow-kun/llm-speed) — Query the llm-speed.com crowdsourced LLM inference-speed leaderboard inline. Every tool response is cited back to the run that backs it. Activate with `pipx install 'llm-speed[mcp]' && llm-speed mcp`.
  ```

- Open the PR from a fork (per-repo identity = `meadow-kun`). Title:
  `Add llm-speed`. Body: one-line summary + link to the GitHub repo.

---

## 3. Open Tools — https://open-tools.dev

Newer registry, MCP-native (not curated by humans — submission via JSON
manifest). Two paths:

- **Programmatic**: POST to `https://open-tools.dev/api/v1/tools` with:

  ```json
  {
    "name": "llm-speed",
    "package": "llm-speed",
    "package_manager": "pypi",
    "extras": ["mcp"],
    "command": "llm-speed",
    "args": ["mcp"],
    "description": "Query the llm-speed.com inference-speed leaderboard. Every answer cited.",
    "homepage": "https://llm-speed.com",
    "license": "Apache-2.0",
    "tags": ["benchmarks", "llm", "tokens-per-second", "leaderboard"]
  }
  ```

  Confirm the exact endpoint shape against the current docs at
  https://open-tools.dev/docs/submit before sending — the API shape
  has changed once already.

- **Web form**: https://open-tools.dev/submit. Same fields.

After submission they email a verification link to the `mailto:` in
`pyproject.toml` (currently the GitHub noreply). Click through to
publish.

---

## Anti-checklist (do not do)

- Do not submit before the wheel with `[mcp]` extra is published — a
  broken install command tanks registry trust scores.
- Do not paste a tool manifest containing local-only env-var defaults
  (`LLM_SPEED_API_BASE` = localhost). The published manifest must boot
  cleanly with zero env vars set.
- Do not list a paid-tier or auth-required variant. Free, anonymous,
  read-only.
