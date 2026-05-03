# llm-speed (npm wrapper)

This is a thin npm wrapper around the [`llm-speed`](https://llm-speed.com) CLI.
The real implementation is Python; this package downloads a prebuilt binary
that bundles a Python runtime so `npm install -g llm-speed` works without a
local Python install.

## Install

```sh
npm install -g llm-speed
llm-speed --version
```

## Why this exists

Many JS-side devs already have `npm` but not `pipx`. This package mirrors what
`pnpm`, `prettier-plugin-go-template`, and similar polyglot tools do: ship the
implementation in its native ecosystem (PyPI) and expose a download-on-install
shim for the npm ecosystem.

## Recommended install paths (in order of preference)

1. `pipx install llm-speed` (lowest overhead, native install)
2. `uv tool install llm-speed`
3. `brew install llm-speed/tap/llm-speed`
4. `npm install -g llm-speed` (this package)

## License

Apache-2.0
