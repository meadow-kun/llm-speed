# `llm-speed mcp` — Model Context Protocol server

Boots a Model Context Protocol stdio server so any MCP-aware AI assistant
(Claude Desktop, Cline, Cursor, Continue, Claude Code, ChatGPT desktop, etc.)
can ground its answers about local LLM throughput in real benchmark data
from [llm-speed.com](https://llm-speed.com) instead of guessing. Every
tool response includes a `citation: list[str]` of `https://llm-speed.com/r/<id>`
URLs back to the runs that prove the numbers.

## Tools

| Tool | What it does |
|---|---|
| `lookup_speed(model, hardware?)` | Best decode tok/s for a (model, hardware) pair, or every hardware for the model when `hardware` is omitted. |
| `compare(a, b)` | Side-by-side: who's faster, by how many tok/s, with the run backing each side and a `/vs/<slug>` deeplink. Each side accepts `"<model>"`, `"<hardware>"`, or `"<model> on <hardware>"`. |
| `recommend(constraints)` | Up to 5 ranked (model, hardware) cells matching `vram_gb_max`, `ram_gb_max`, `decode_tps_min`, `model_size_min`, `model_size_max`, `backend`, `locality` ("local" / "hosted" / "any"). |
| `top_models(hardware, n=10)` | Fastest models on a given rig, ranked by decode tok/s. |
| `state_of()` | Latest /state-of headline cells (fastest local, fastest 70B+ class, fastest coding agent), computed live from the most recent calendar month in the listing. |

There is also one prompt resource:

- `howto://pick-a-local-llm` — guided template for using the tools to
  pick a local LLM under a given rig + task.

## Install

The MCP server ships as an optional extra of the main `llm-speed` CLI.

Fresh install:
```sh
pipx install 'llm-speed[mcp]'
```

Add MCP support to an existing install:
```sh
pipx inject llm-speed mcp cachetools
```

With uv (preferred on Apple Silicon):
```sh
uv tool install --with mcp --with cachetools llm-speed
```

## Boot

```sh
llm-speed mcp
```

The server speaks the stdio transport. No authentication, no API key;
it hits the public `https://api.llm-speed.com` endpoints.

## Configure your MCP client

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "llm-speed": {
      "command": "llm-speed",
      "args": ["mcp"]
    }
  }
}
```

If `pipx` installed `llm-speed` outside Claude's PATH, use the absolute
path printed by `pipx list` (e.g. `/Users/<you>/.local/bin/llm-speed`).

### Cline / Cursor / Continue

Add to the editor's MCP-server settings:

```json
{
  "mcpServers": {
    "llm-speed": {
      "command": "llm-speed",
      "args": ["mcp"]
    }
  }
}
```

### Claude Code (CLI)

```sh
claude mcp add llm-speed -- llm-speed mcp
```

## Verify

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  | llm-speed mcp
```

You should see `lookup_speed`, `compare`, `recommend`, `top_models`,
`state_of` listed.

## Caching

The full `/v1/results?limit=500` listing is cached in-process via
`cachetools.TTLCache` for 5 minutes, so a single MCP session won't
hammer the API. Single-run details get a 5-minute TTL too. The suite
manifest is cached for 15 minutes.

Override the API base for local development:

```sh
LLM_SPEED_API_BASE=http://localhost:8787 \
LLM_SPEED_SITE_BASE=http://localhost:3000 \
llm-speed mcp
```

## Data freshness

Tools surface the *highest* decode_tps cell per (model, hardware). When
two runs disagree, the faster one wins. Both runs are still resolvable
via their `/r/<id>` URL — point users at the run page to see the full
history.

## Edge cases worth knowing

- Free-text user input ("RTX 5090", "rtx5090", "RTX 5090 (32GB)") all
  collapse to the same canonical hardware slug. CPU/RAM tails of an
  accelerator-summary string ("RTX 5090 (32GB) + AMD ... + 30GB") are
  stripped so the rig dedupes regardless of the host system.
- Hosted models (anthropic/, openai/, google/, deepseek/, qwen/) are
  classified as `locality="hosted"`. `recommend({locality:"local"})`
  filters them out — the default `locality:"any"` keeps both.
- `lookup_speed` does fuzzy slug matching: exact slug match → slug
  prefix → display-name substring. Pass either the canonical slug
  (`qwen2-5-7b-instruct`) or a recognisable name (`Qwen2.5 7B Instruct`).
- `state_of()` headlines are derived from the listing's per-run
  headline rows. Categories with no qualifying headline that month
  (e.g. no 70B-class run as the top decode_tps of any run) come back
  with `available: false` rather than fabricated.

## Contribute a benchmark

The MCP server's value scales linearly with the size of the dataset —
every contribution is a free upgrade.

```sh
llm-speed bench
```

## Registry submission

See `REGISTER.md` for the Smithery / MCPT / Open Tools listing flow.
