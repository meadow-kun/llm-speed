"""ExLlamaV2 backend driver (NVIDIA / CUDA only).

Heavy imports of `exllamav2` and `torch` are deferred to method bodies so this
module imports cleanly on machines without CUDA.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
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

LOG = logging.getLogger("cli.drivers.exllamav2")

_BACKEND_NAME = "exllamav2"

_BPW_RE = re.compile(r"(\d+(?:\.\d+)?)\s*bpw", re.IGNORECASE)
_SIZE_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])")


def _parse_bpw(label: str) -> str | None:
    m = _BPW_RE.search(label)
    if m:
        return f"{m.group(1)}bpw"
    return None


def _parse_size(label: str) -> str | None:
    m = _SIZE_RE.search(label)
    if m:
        return f"{m.group(1)}B"
    return None


class Exllamav2Driver:
    name: str = _BACKEND_NAME

    def __init__(self) -> None:
        self._loaded: dict[str, tuple[Any, Any, Any, Any]] = {}

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def detect(self) -> BackendDetection:
        if importlib.util.find_spec("exllamav2") is None:
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes="exllamav2 not installed (pip install 'llm-speed[exllamav2]')",
            )

        # CUDA presence — required for any actual generation.
        cuda_ok = False
        runtime: dict[str, str] = {}
        if importlib.util.find_spec("torch") is not None:
            try:
                torch = importlib.import_module("torch")
                cuda_ok = bool(
                    getattr(torch, "cuda", None) and torch.cuda.is_available()
                )
                runtime["torch"] = getattr(torch, "__version__", "unknown")
                cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
                if cuda_version:
                    runtime["cuda"] = cuda_version
            except Exception as exc:  # noqa: BLE001
                LOG.debug("torch import failed: %s", exc)
        if not cuda_ok:
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                runtime_versions=runtime,
                notes="exllamav2 requires CUDA",
            )

        version: str | None = None
        try:
            exl2 = importlib.import_module("exllamav2")
            version = getattr(exl2, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes=f"exllamav2 import failed: {exc}",
            )

        return BackendDetection(
            available=True,
            name=_BACKEND_NAME,
            version=version,
            build_flags=["CUDA"],
            runtime_versions=runtime,
            notes="exllamav2 ready",
        )

    # ------------------------------------------------------------------
    # model discovery
    # ------------------------------------------------------------------
    def list_models(self) -> list[ModelRef]:
        seen: set[str] = set()
        out: list[ModelRef] = []

        for ref in self._scan_huggingface_cache():
            if ref.identifier in seen:
                continue
            seen.add(ref.identifier)
            out.append(ref)

        local = Path.home() / "Models" / "exl2"
        if local.exists():
            for child in sorted(local.iterdir()):
                if not child.is_dir() or not self._dir_looks_like_exl2(child):
                    continue
                key = str(child.resolve())
                if key in seen:
                    continue
                seen.add(key)
                quant = _parse_bpw(child.name) or self._read_bpw_from_config(child)
                out.append(
                    ModelRef(
                        backend=_BACKEND_NAME,
                        identifier=key,
                        name=child.name,
                        size=_parse_size(child.name),
                        quant=quant,
                    )
                )

        return out

    def _scan_huggingface_cache(self) -> list[ModelRef]:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
        if not cache.exists():
            return []
        results: list[ModelRef] = []
        for repo_dir in cache.iterdir():
            if not repo_dir.is_dir() or not repo_dir.name.startswith("models--"):
                continue
            snapshots = repo_dir / "snapshots"
            if not snapshots.exists():
                continue
            chosen: Path | None = None
            for snap in snapshots.iterdir():
                if snap.is_dir() and self._dir_looks_like_exl2(snap):
                    chosen = snap
                    break
            if chosen is None:
                continue
            tail = repo_dir.name[len("models--") :]
            hf_id = tail.replace("--", "/", 1).replace("--", "/")
            quant = _parse_bpw(hf_id) or self._read_bpw_from_config(chosen)
            results.append(
                ModelRef(
                    backend=_BACKEND_NAME,
                    identifier=str(chosen.resolve()),
                    name=hf_id,
                    size=_parse_size(hf_id),
                    quant=quant,
                    extras={"hf_id": hf_id},
                )
            )
        return results

    @staticmethod
    def _dir_looks_like_exl2(path: Path) -> bool:
        cfg = path / "config.json"
        if not cfg.exists():
            return False
        try:
            entries = list(path.iterdir())
        except OSError:
            return False
        if not any(p.suffix == ".safetensors" for p in entries):
            return False
        try:
            payload = json.loads(cfg.read_text())
        except Exception:  # noqa: BLE001
            return False
        # exllamav2 quants stamp config.json with quantization_config or "bits"
        # / "bpw" fields. Tolerate the loose 'exl2' marker on file/dir names.
        if isinstance(payload.get("quantization_config"), dict):
            return True
        if "bpw" in payload or "head_bits" in payload:
            return True
        name = path.name.lower()
        if "exl2" in name or "bpw" in name:
            return True
        # parent (snapshot dir) lookup for HF-cache layout.
        parent = (
            path.parent.parent.name.lower()
            if path.parent and path.parent.parent
            else ""
        )
        return "exl2" in parent or "bpw" in parent

    @staticmethod
    def _read_bpw_from_config(path: Path) -> str | None:
        cfg = path / "config.json"
        if not cfg.exists():
            return None
        try:
            payload = json.loads(cfg.read_text())
        except Exception:  # noqa: BLE001
            return None
        bpw = payload.get("bpw")
        if isinstance(bpw, (int, float)):
            return f"{bpw}bpw"
        qcfg = payload.get("quantization_config")
        if isinstance(qcfg, dict):
            for key in ("bits", "bpw", "wbits"):
                v = qcfg.get(key)
                if isinstance(v, (int, float)):
                    return f"{v}bpw"
        return None

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
        if importlib.util.find_spec("exllamav2") is None:
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error="exllamav2 not installed",
            )

        try:
            exl2 = importlib.import_module("exllamav2")
            generator_mod = importlib.import_module("exllamav2.generator")
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"exllamav2 import failed: {exc}",
            )

        try:
            backend_model, cache, tokenizer, generator = self._load(
                exl2, generator_mod, model.identifier
            )
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"exllamav2 load failed: {exc}",
            )

        # Reset the KV cache between runs so we don't leak state.
        try:
            cache.current_seq_len = 0
        except Exception:  # noqa: BLE001
            pass

        sampler_cls = getattr(generator_mod, "ExLlamaV2Sampler", None)
        settings = sampler_cls.Settings() if sampler_cls is not None else None

        # Encode prompt for token-count and to seed the stream generator.
        try:
            input_ids = tokenizer.encode(prompt)
            prompt_tokens = int(input_ids.shape[-1])
        except Exception:  # noqa: BLE001
            prompt_tokens = 0
            input_ids = None

        decode_times: list[float] = []
        ttft_ms: float | None = None
        output_pieces: list[str] = []
        output_tokens = 0

        # Prefer the streaming generator so we can capture per-token timings.
        try:
            with stopwatch() as wall:
                start = now_ms()
                last = start
                if input_ids is None:
                    raise RuntimeError("tokenizer.encode failed")

                if hasattr(generator, "warmup"):
                    generator.warmup()
                generator.set_stop_conditions([tokenizer.eos_token_id])
                generator.begin_stream(input_ids, settings)

                while output_tokens < max_output_tokens:
                    chunk, eos, _ = generator.stream()
                    ts = now_ms()
                    if ttft_ms is None:
                        ttft_ms = ts - start
                    else:
                        decode_times.append(ts - last)
                    last = ts
                    if chunk:
                        output_pieces.append(chunk)
                    output_tokens += 1
                    if eos:
                        break
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="".join(output_pieces),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                decode_token_times_ms=decode_times,
                wall_ms=0.0,
                error=f"exllamav2 streaming failed: {exc}",
            )

        return ChatRunOutcome(
            success=True,
            output_text="".join(output_pieces),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            decode_token_times_ms=decode_times,
            wall_ms=wall[0],
            backend_extras={"streaming": True},
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _load(
        self, exl2: Any, generator_mod: Any, identifier: str
    ) -> tuple[Any, Any, Any, Any]:
        if identifier in self._loaded:
            return self._loaded[identifier]

        config_cls = exl2.ExLlamaV2Config
        model_cls = exl2.ExLlamaV2
        cache_cls = exl2.ExLlamaV2Cache
        tokenizer_cls = exl2.ExLlamaV2Tokenizer
        streaming_gen_cls = getattr(generator_mod, "ExLlamaV2StreamingGenerator", None)
        if streaming_gen_cls is None:
            streaming_gen_cls = generator_mod.ExLlamaV2BaseGenerator

        LOG.info("exllamav2 loading %s", identifier)
        config = config_cls()
        config.model_dir = identifier
        config.prepare()

        backend_model = model_cls(config)
        cache = cache_cls(backend_model, lazy=True)
        backend_model.load_autosplit(cache)
        tokenizer = tokenizer_cls(config)
        generator = streaming_gen_cls(backend_model, cache, tokenizer)

        self._loaded[identifier] = (backend_model, cache, tokenizer, generator)
        return backend_model, cache, tokenizer, generator


register_driver(_BACKEND_NAME, lambda: Exllamav2Driver())


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    drv = Exllamav2Driver()
    det = drv.detect()
    if not det.available:
        print(f"exllamav2 not available: {det.notes}")
    else:
        print(
            f"exllamav2 available: version={det.version} runtime={det.runtime_versions}"
        )
        models = drv.list_models()
        print(f"Found {len(models)} model(s):")
        for m in models[:10]:
            print(f"  - {m.identifier}  size={m.size} quant={m.quant}")
