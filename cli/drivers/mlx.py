"""MLX backend driver (Apple Silicon).

Lazy-imports `mlx_lm` and `mlx.core` only inside method bodies after a feature
check, so this module imports cleanly on machines without MLX installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import platform
import re
from pathlib import Path
from typing import Any

from ..registry import register_driver
from ..timing import now_ms, stopwatch
from ..types import (
    BackendDetection,
    ChatRunOutcome,
    ModelRef,
)

LOG = logging.getLogger("cli.drivers.mlx")

_BACKEND_NAME = "mlx"

# Heuristic regexes for parsing names that look like 'Qwen3-Coder-Next-30B-A3B-4bit'.
_SIZE_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])")
_QUANT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:(\d+)\s*bit|q(\d+)|fp(\d+)|bf(\d+))", re.IGNORECASE
)


def _parse_name_size_quant(label: str) -> tuple[str, str | None, str | None]:
    """Pull a human name, size and quant from a model dir / repo id."""
    base = label.replace("/", "-")
    size = None
    m = _SIZE_RE.search(base)
    if m:
        size = f"{m.group(1)}B"
    quant = None
    qm = _QUANT_RE.search(base)
    if qm:
        if qm.group(1):
            quant = f"{qm.group(1)}bit"
        elif qm.group(2):
            quant = f"Q{qm.group(2)}"
        elif qm.group(3):
            quant = f"FP{qm.group(3)}"
        elif qm.group(4):
            quant = f"BF{qm.group(4)}"
    return base, size, quant


class MlxDriver:
    """Apple Silicon MLX driver. Caches a loaded (model, tokenizer) per identifier."""

    name: str = _BACKEND_NAME

    def __init__(self) -> None:
        self._loaded: dict[str, tuple[Any, Any]] = {}

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def detect(self) -> BackendDetection:
        if platform.system() != "Darwin":
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes="MLX requires macOS",
            )
        if platform.machine() != "arm64":
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes="MLX requires Apple Silicon (arm64)",
            )
        if importlib.util.find_spec("mlx_lm") is None:
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes="mlx_lm not installed (pip install 'llm-speed[mlx]')",
            )

        version: str | None = None
        runtime: dict[str, str] = {}
        try:
            mlx_lm = importlib.import_module("mlx_lm")
            version = getattr(mlx_lm, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("mlx_lm import failed during detect: %s", exc)
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes=f"mlx_lm import failed: {exc}",
            )

        try:
            mlx_core = importlib.import_module("mlx.core")
            runtime["mlx.core"] = (
                getattr(mlx_core, "__version__", "unknown") or "unknown"
            )
            default_device = getattr(mlx_core, "default_device", None)
            if callable(default_device):
                try:
                    runtime["device"] = str(default_device())
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            LOG.debug("mlx.core import failed during detect: %s", exc)

        return BackendDetection(
            available=True,
            name=_BACKEND_NAME,
            version=version,
            build_flags=["Metal"],
            runtime_versions=runtime,
            notes="Apple Silicon MLX",
        )

    # ------------------------------------------------------------------
    # model discovery
    # ------------------------------------------------------------------
    def list_models(self) -> list[ModelRef]:
        results: list[ModelRef] = []
        seen: set[str] = set()

        for entry in self._scan_huggingface_cache():
            key = entry.identifier
            if key in seen:
                continue
            seen.add(key)
            results.append(entry)

        models_dir = Path.home() / "Models" / "mlx"
        if models_dir.exists():
            for child in sorted(models_dir.iterdir()):
                if not child.is_dir():
                    continue
                if not self._dir_looks_like_mlx(child):
                    continue
                key = str(child.resolve())
                if key in seen:
                    continue
                seen.add(key)
                name, size, quant = _parse_name_size_quant(child.name)
                results.append(
                    ModelRef(
                        backend=_BACKEND_NAME,
                        identifier=key,
                        name=name,
                        size=size,
                        quant=quant,
                    )
                )

        return results

    def _scan_huggingface_cache(self) -> list[ModelRef]:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache.exists():
            return []
        out: list[ModelRef] = []
        for repo_dir in cache.iterdir():
            # Repo dirs are named like 'models--mlx-community--Llama-3-8B-Instruct-4bit'
            if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
                continue
            snapshots = repo_dir / "snapshots"
            if not snapshots.exists():
                continue
            chosen: Path | None = None
            for snap in snapshots.iterdir():
                if not snap.is_dir():
                    continue
                if self._dir_looks_like_mlx(snap):
                    chosen = snap
                    break
            if chosen is None:
                continue
            # Reconstruct an HF id like 'mlx-community/Llama-3-8B-Instruct-4bit'.
            tail = repo_dir.name[len("models--") :]
            hf_id = tail.replace("--", "/", 1).replace("--", "/")
            label = hf_id
            name, size, quant = _parse_name_size_quant(label)
            out.append(
                ModelRef(
                    backend=_BACKEND_NAME,
                    identifier=hf_id,
                    name=name,
                    size=size,
                    quant=quant,
                    extras={"snapshot_path": str(chosen)},
                )
            )
        return out

    @staticmethod
    def _dir_looks_like_mlx(path: Path) -> bool:
        try:
            entries = list(path.iterdir())
        except OSError:
            return False
        names = {p.name for p in entries}
        has_npz = any(n.endswith(".npz") for n in names)
        has_safetensors = any(n.endswith(".safetensors") for n in names)
        if has_npz:
            return True
        if not has_safetensors:
            return False
        cfg = path / "config.json"
        if not cfg.exists():
            # mlx-community models almost always have a config.json; bail safe.
            return False
        try:
            payload = json.loads(cfg.read_text())
        except Exception:  # noqa: BLE001
            return False
        # mlx-converted configs commonly carry 'quantization' or an 'mlx_*' key.
        if any(k.startswith("mlx_") for k in payload.keys()):
            return True
        if "quantization" in payload:
            return True
        # A few mlx-community repos don't set the marker; tolerate if the path
        # name itself flags 'mlx'.
        return (
            "mlx" in path.parent.parent.name.lower()
            if path.parent.parent.name
            else False
        )

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------
    def run_chat(
        self,
        model: ModelRef,
        prompt: str,
        *,
        max_output_tokens: int,
        stream: bool = True,
        extras: dict[str, Any] | None = None,
    ) -> ChatRunOutcome:
        if importlib.util.find_spec("mlx_lm") is None:
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error="mlx_lm not installed",
            )

        try:
            mlx_lm = importlib.import_module("mlx_lm")
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"mlx_lm import failed: {exc}",
            )

        try:
            backend_model, tokenizer = self._load(mlx_lm, model.identifier)
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"mlx_lm.load failed: {exc}",
            )

        # Apply chat template if available; otherwise pass the raw prompt.
        formatted_prompt = self._apply_chat_template(tokenizer, prompt)
        try:
            prompt_ids = tokenizer.encode(formatted_prompt)
            prompt_token_count = len(prompt_ids)
        except Exception:  # noqa: BLE001
            prompt_token_count = 0

        decode_times: list[float] = []
        ttft_ms: float | None = None
        output_pieces: list[str] = []
        output_tokens = 0

        stream_generate = getattr(mlx_lm, "stream_generate", None)
        if stream and callable(stream_generate):
            with stopwatch() as wall:
                start = now_ms()
                last = start
                try:
                    for piece in stream_generate(
                        backend_model,
                        tokenizer,
                        formatted_prompt,
                        max_tokens=max_output_tokens,
                    ):
                        token_ts = now_ms()
                        text = self._extract_text(piece)
                        if ttft_ms is None:
                            ttft_ms = token_ts - start
                        else:
                            decode_times.append(token_ts - last)
                        last = token_ts
                        if text:
                            output_pieces.append(text)
                        output_tokens += 1
                except Exception as exc:  # noqa: BLE001
                    return ChatRunOutcome(
                        success=False,
                        output_text="".join(output_pieces),
                        prompt_tokens=prompt_token_count,
                        output_tokens=output_tokens,
                        ttft_ms=ttft_ms,
                        decode_token_times_ms=decode_times,
                        wall_ms=wall[0],
                        error=f"stream_generate failed: {exc}",
                    )
            return ChatRunOutcome(
                success=True,
                output_text="".join(output_pieces),
                prompt_tokens=prompt_token_count,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                decode_token_times_ms=decode_times,
                wall_ms=wall[0],
                backend_extras={"streaming": True},
            )

        # Fallback: non-streaming generate; synthesize timings.
        generate = getattr(mlx_lm, "generate", None)
        if not callable(generate):
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=prompt_token_count,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error="mlx_lm exposes neither stream_generate nor generate",
            )
        with stopwatch() as wall:
            try:
                text = generate(
                    backend_model,
                    tokenizer,
                    formatted_prompt,
                    max_tokens=max_output_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                return ChatRunOutcome(
                    success=False,
                    output_text="",
                    prompt_tokens=prompt_token_count,
                    output_tokens=0,
                    ttft_ms=None,
                    decode_token_times_ms=[],
                    wall_ms=0.0,
                    error=f"generate failed: {exc}",
                )
        try:
            output_ids = tokenizer.encode(text)
            output_tokens = max(0, len(output_ids) - prompt_token_count)
        except Exception:  # noqa: BLE001
            output_tokens = len(text.split())
        return ChatRunOutcome(
            success=True,
            output_text=text or "",
            prompt_tokens=prompt_token_count,
            output_tokens=output_tokens,
            ttft_ms=None,
            decode_token_times_ms=[],
            wall_ms=wall[0],
            backend_extras={"streaming": False},
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _load(self, mlx_lm: Any, identifier: str) -> tuple[Any, Any]:
        if identifier in self._loaded:
            return self._loaded[identifier]
        load = mlx_lm.load
        LOG.info("mlx loading %s", identifier)
        backend_model, tokenizer = load(identifier)
        self._loaded[identifier] = (backend_model, tokenizer)
        return backend_model, tokenizer

    @staticmethod
    def _apply_chat_template(tokenizer: Any, prompt: str) -> str:
        apply = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            out = apply(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(out, str):
                return out
        except Exception as exc:  # noqa: BLE001
            LOG.debug("apply_chat_template failed: %s", exc)
        return prompt

    @staticmethod
    def _extract_text(piece: Any) -> str:
        # mlx_lm.stream_generate has historically yielded either str directly or
        # a small dataclass with a `.text` attribute; tolerate both.
        if isinstance(piece, str):
            return piece
        text = getattr(piece, "text", None)
        if isinstance(text, str):
            return text
        token = getattr(piece, "token", None)
        if isinstance(token, str):
            return token
        return ""


register_driver(_BACKEND_NAME, lambda: MlxDriver())


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    drv = MlxDriver()
    det = drv.detect()
    if not det.available:
        print(f"MLX not available: {det.notes}")
    else:
        print(f"MLX available: version={det.version} runtime={det.runtime_versions}")
        models = drv.list_models()
        print(f"Found {len(models)} model(s):")
        for m in models[:10]:
            print(f"  - {m.identifier}  size={m.size} quant={m.quant}")
