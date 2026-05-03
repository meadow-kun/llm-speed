# llm-speed CLI v2 - UX design

The CLI's job is to feel polished and inviting, not just functional. v2 adopts
the patterns Claude Code's interactive surface uses: TUI as the default,
streaming progress, friendly errors, share affordances at the natural moment
of pride (right after a successful run), and a `verify` subcommand that lets
the user prove the binary on their machine matches the open-source code.

## Subcommands

| command | purpose |
|---|---|
| `llm-speed`               | enter the interactive wizard (default on tty) |
| `llm-speed --auto`        | non-interactive: pick a backend + model + run the quick suite |
| `llm-speed bench [...]`   | the existing benchmark flow, now with streaming UI + share screen |
| `llm-speed detect`        | unchanged - print fingerprint + backend matrix |
| `llm-speed list-models`   | unchanged |
| `llm-speed compare`       | unchanged |
| `llm-speed login`         | unchanged stub |
| `llm-speed self-update`   | unchanged stub |
| `llm-speed verify [path]` | NEW - sha256 a local wheel against the publicly-served one |
| `llm-speed about`         | NEW - 5-line "what this is + links" splash |

`llm-speed --help` is unchanged in shape; the wizard is in addition to, not
instead of, the explicit subcommand surface.

## TTY contract

Every TUI surface guards on `sys.stdin.isatty() and sys.stdout.isatty()` and
falls back to a non-blocking auto/print path otherwise. CI runs of `bench`
look identical to v1 - no new prompts, no spinners that mangle log output.

Privacy stays exactly where it was: the consent prompt in `cli/consent.py`
runs on first upload-bound `bench`, the wizard simply hands control to
`cmd_bench` and stops at the prompt rather than skipping past it.

## Screen 1: first-run wizard

```
llm-speed  Benchmark any LLM on any hardware.
  welcome - this is a one-time setup. every byte the cli sends is documented at https://llm-speed.com/privacy

  backend     status        notes
  llama.cpp   not found     GGUF, llama-server
  ollama      installed     daemon at :11434
  mlx         not found     Apple Silicon only

>> ready to benchmark
  we'll run the QUICK workload (about 30s) on the first available backend

Run the quick benchmark now? [Y/n]
```

If no backend is installed the wizard prints the install offer instead, with
the exact shell command surfaced before any execution:

```
no benchmark backend installed yet. llm-speed needs one of: llama.cpp, ollama, mlx.
install offer: Install Ollama via Homebrew

Proposed install for ollama:
  $ brew install ollama

Run this now? [y/N]
```

The default is "no" - the user has to type `y` to proceed.

## Screen 2: auto-bench progress

```
>> auto mode: quick workload, sensible defaults
  running the QUICK workload set against the first available backend
  press Ctrl-C any time to stop

Running 1 workloads on ollama / Qwen2.5-Coder
  - chat-short ... ok  decode=187.4 tps
─────────────── M3 Pro (18-core GPU) + 36GB unified ───────────────
ollama@0.5.7   Qwen2.5-Coder (7B, Q4_K_M)
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ workload ┃ decode tps ┃ prefill tps┃ ttft ms ┃ p50 ms ┃ p95 ms ┃
┃ chat-shor┃     187.4  ┃    8421.1  ┃   142.3 ┃    5.3 ┃    7.1 ┃
┗━━━━━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━━━━┻━━━━━━━━━┻━━━━━━━━┻━━━━━━━━┛
Submitted: https://llm-speed.com/r/r_abc123
```

## Screen 3: post-run share screen

```
╭─ share this run ────────────────────────────────────────────────╮
│ Run page    https://llm-speed.com/r/r_abc123                    │
│ Badge SVG   https://llm-speed.com/badge/r_abc123.svg            │
│ Markdown                                                        │
│   [![llm-speed](https://llm-speed.com/badge/r_abc123.svg)](...) │
│                                                                 │
│ Share on X       https://twitter.com/intent/tweet?text=...      │
│ Share on Reddit  https://www.reddit.com/submit?title=...        │
╰─────────────────────────────────────────────────────────────────╯
 c  copy markdown    x  share on X    r  share on Reddit    o  open run page    s  skip

press a key (c / x / r / o / s):
```

Pressing `c` writes the markdown snippet to the system clipboard via
`pbcopy` / `wl-copy` / `xclip` / `xsel` / `clip.exe` depending on platform.
`x`, `r`, `o` open the relevant intent URL via `webbrowser.open` with a
platform `open` / `xdg-open` / `start` fallback. `s` (or any other key) is a
clean no-op.

## Screen 4: verify output

```
llm-speed verify - prove this binary matches the public artifact

  local wheel    : /Users/.../dist/llm_speed-0.0.1-py3-none-any.whl
  local sha256   : 5f4dcc3b5aa765d61d8327deb882cf99...
  expected sha256: 5f4dcc3b5aa765d61d8327deb882cf99...
  sidecar URL    : https://llm-speed.com/dist/llm_speed-0.0.1-py3-none-any.whl.sha256

MATCH. The bytes you are about to run are byte-identical to the wheel served at
      https://llm-speed.com/dist/llm_speed-0.0.1-py3-none-any.whl
      Source code (audit it yourself): https://github.com/meadow-kun/llm-speed

  installed version: llm-speed 0.0.1-dev
  pip show llm-speed:
    Name: llm-speed
    Version: 0.0.1
    ...
```

A mismatch flips the verdict block to a red `MISMATCH` and exits non-zero so
`verify` is composable in scripts.

## Theme

* `brand`     - teal `#34d399` (matches the badge accent + site)
* `muted`     - dim white, used for chrome / labels
* `ok` / `warn` / `err` - green / yellow / red, consistent across all surfaces
* `hint`      - cyan, used for keyboard hints

ASCII only. No emoji anywhere. Status words are bare (`ok`, `warn`, `err`,
`..`) so Rich's markup parser can colour them without fighting square
brackets.

## Identity

Every URL emitted by the CLI uses the `meadow-kun` GitHub identity per the
2026-04-28 scrub. There's a unit test (`tests/cli/ui/test_theme.py`) that
asserts no `philip` / `nordenfelt` substring appears in the brand URLs to
catch regressions early.
