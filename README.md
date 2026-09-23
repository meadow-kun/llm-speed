# llm-speed

Benchmark local LLM inference and inspect the settings behind your results.
[llm-speed.com](https://llm-speed.com/) publishes submitted measurements across hardware and backends. This repository contains the benchmark CLI.

## Install

With Python 3.10 or newer:

```sh
pipx install llm-speed
```

Or use `uv tool install llm-speed`. On a machine without Python, the [installer](https://llm-speed.com/install.sh) can provision it with your consent:

```sh
curl -fsSL https://llm-speed.com/install.sh | sh
```

Check dependencies and backend setup with `llm-speed doctor`. On an interactive terminal it offers setup steps; otherwise it prints guidance.

## Run

```sh
llm-speed --version
llm-speed bench --quick --no-upload
```

The second command saves results locally without uploading. Choose a backend and model with `--backend` and `--model` when needed. Run `llm-speed bench --help` for options. Uploading is optional; review the [privacy documentation](docs/PRIVACY.md) before sharing results.

Supported backend integrations include Ollama, llama.cpp, MLX, vLLM, and ExLlamaV2. Availability depends on your hardware and installed backend. Measurements with different models, quantization, and workloads are not interchangeable rankings. See the [methodology](docs/METHODOLOGY.md).

## Links and license

- [Results and datasets](https://llm-speed.com/)
- [Report an issue](https://github.com/meadow-kun/llm-speed/issues)
- [Security policy](SECURITY.md)

CLI code is licensed under [Apache-2.0](LICENSE). Shared benchmark data is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
