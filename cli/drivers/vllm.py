"""vLLM backend driver.

Two operating modes:

* HTTP mode (preferred): a vLLM OpenAI-compatible server is reachable. Speaks
  the OpenAI streaming chat-completions protocol.
* Embedded mode (fallback): import `vllm` and run `vllm.LLM(...).generate(...)`
  in-process. Streaming timings are best-effort; if not available we synthesize
  flat per-token deltas and flag the run.

Heavy imports of `vllm` are deferred to method bodies; the module imports fine
without vllm installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
from typing import Any

import httpx

from ..registry import register_driver
from ..timing import now_ms, stopwatch
from ..types import (
    BackendDetection,
    ChatRunOutcome,
    ModelRef,
)

LOG = logging.getLogger("cli.drivers.vllm")

_BACKEND_NAME = "vllm"
_DEFAULT_URL = "http://localhost:8000"
_HTTP_PROBE_TIMEOUT = 1.5
_HTTP_REQUEST_TIMEOUT = 600.0


def _server_url() -> str:
    return os.environ.get("VLLM_URL", _DEFAULT_URL).rstrip("/")


def _server_reachable(url: str) -> bool:
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=_HTTP_PROBE_TIMEOUT)
    except Exception:  # noqa: BLE001 — any network failure means not reachable
        return False
    return 200 <= resp.status_code < 500


class VllmDriver:
    name: str = _BACKEND_NAME

    def __init__(self) -> None:
        self._embedded_engine: Any | None = None
        self._embedded_model_id: str | None = None

    # ------------------------------------------------------------------
    # detection
    # ------------------------------------------------------------------
    def detect(self) -> BackendDetection:
        url = _server_url()
        http_up = _server_reachable(url)

        embedded_available = importlib.util.find_spec("vllm") is not None
        version: str | None = None
        if embedded_available:
            try:
                vllm = importlib.import_module("vllm")
                version = getattr(vllm, "__version__", None)
            except Exception as exc:  # noqa: BLE001
                LOG.debug("vllm import failed: %s", exc)
                embedded_available = False

        runtime: dict[str, str] = {}
        flags: list[str] = []
        if http_up:
            flags.append("http")
            runtime["http_url"] = url
        if embedded_available:
            flags.append("embedded")

        if not http_up and not embedded_available:
            return BackendDetection(
                available=False,
                name=_BACKEND_NAME,
                notes="vllm not installed and no server at " + url,
            )

        return BackendDetection(
            available=True,
            name=_BACKEND_NAME,
            version=version,
            build_flags=flags,
            runtime_versions=runtime,
            notes=("HTTP server reachable" if http_up else "embedded mode only"),
        )

    # ------------------------------------------------------------------
    # model discovery
    # ------------------------------------------------------------------
    def list_models(self) -> list[ModelRef]:
        url = _server_url()
        if not _server_reachable(url):
            return []
        try:
            resp = httpx.get(f"{url}/v1/models", timeout=_HTTP_PROBE_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            LOG.debug("vllm /v1/models failed: %s", exc)
            return []
        out: list[ModelRef] = []
        for entry in payload.get("data", []):
            mid = entry.get("id")
            if not isinstance(mid, str):
                continue
            out.append(
                ModelRef(
                    backend=_BACKEND_NAME,
                    identifier=mid,
                    name=mid,
                    extras={"served_by": "http", "url": url},
                )
            )
        return out

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
        url = _server_url()
        if _server_reachable(url):
            return self._run_http(url, model, prompt, max_output_tokens, stream, extras)
        if importlib.util.find_spec("vllm") is None:
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error="vllm unavailable: no HTTP server and vllm not installed",
            )
        return self._run_embedded(model, prompt, max_output_tokens, stream, extras)

    # ------------------------------------------------------------------
    # HTTP mode
    # ------------------------------------------------------------------
    def _run_http(
        self,
        url: str,
        model: ModelRef,
        prompt: str,
        max_output_tokens: int,
        stream: bool,
        extras: dict[str, Any] | None,
    ) -> ChatRunOutcome:
        headers = {"content-type": "application/json"}
        api_key = os.environ.get("VLLM_API_KEY")
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {
            "model": model.identifier,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "stream": stream,
        }
        if stream:
            # vLLM honours these options on streaming responses.
            body["stream_options"] = {"include_usage": True}
        if extras:
            for key, value in extras.items():
                body.setdefault(key, value)

        if not stream:
            return self._run_http_nonstream(url, headers, body)

        decode_times: list[float] = []
        ttft_ms: float | None = None
        output_pieces: list[str] = []
        prompt_tokens = 0
        output_tokens = 0
        prefix_cache_hit_rate: float | None = None
        backend_extras: dict[str, Any] = {"mode": "http", "url": url}

        try:
            with stopwatch() as wall:
                start = now_ms()
                last = start
                with httpx.stream(
                    "POST",
                    f"{url}/v1/chat/completions",
                    headers=headers,
                    content=json.dumps(body),
                    timeout=_HTTP_REQUEST_TIMEOUT,
                ) as resp:
                    if resp.status_code >= 400:
                        text = resp.read().decode(errors="replace")
                        return ChatRunOutcome(
                            success=False,
                            output_text="",
                            prompt_tokens=0,
                            output_tokens=0,
                            ttft_ms=None,
                            decode_token_times_ms=[],
                            wall_ms=0.0,
                            error=f"http {resp.status_code}: {text[:300]}",
                        )
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode(errors="replace")
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        # Pull token deltas from choices[0].delta.content.
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            text_piece = delta.get("content")
                            if text_piece:
                                ts = now_ms()
                                if ttft_ms is None:
                                    ttft_ms = ts - start
                                else:
                                    decode_times.append(ts - last)
                                last = ts
                                output_pieces.append(text_piece)
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = int(
                                usage.get("prompt_tokens") or prompt_tokens
                            )
                            output_tokens = int(
                                usage.get("completion_tokens") or output_tokens
                            )
                        # vLLM-specific: prefix cache stats may surface here.
                        for key in ("prefix_cache_hit_rate", "cache_hit_rate"):
                            if key in chunk:
                                try:
                                    prefix_cache_hit_rate = float(chunk[key])
                                except (TypeError, ValueError):
                                    pass
        except httpx.HTTPError as exc:
            return ChatRunOutcome(
                success=False,
                output_text="".join(output_pieces),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens
                or len(decode_times) + (1 if ttft_ms is not None else 0),
                ttft_ms=ttft_ms,
                decode_token_times_ms=decode_times,
                wall_ms=0.0,
                error=f"http error: {exc}",
            )

        if output_tokens == 0:
            output_tokens = len(decode_times) + (1 if ttft_ms is not None else 0)

        if prefix_cache_hit_rate is not None:
            backend_extras["prefix_cache_hit_rate"] = prefix_cache_hit_rate

        return ChatRunOutcome(
            success=True,
            output_text="".join(output_pieces),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            decode_token_times_ms=decode_times,
            wall_ms=wall[0],
            backend_extras=backend_extras,
        )

    def _run_http_nonstream(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> ChatRunOutcome:
        body = dict(body)
        body["stream"] = False
        try:
            with stopwatch() as wall:
                resp = httpx.post(
                    f"{url}/v1/chat/completions",
                    headers=headers,
                    content=json.dumps(body),
                    timeout=_HTTP_REQUEST_TIMEOUT,
                )
                if resp.status_code >= 400:
                    return ChatRunOutcome(
                        success=False,
                        output_text="",
                        prompt_tokens=0,
                        output_tokens=0,
                        ttft_ms=None,
                        decode_token_times_ms=[],
                        wall_ms=0.0,
                        error=f"http {resp.status_code}: {resp.text[:300]}",
                    )
                payload = resp.json()
        except httpx.HTTPError as exc:
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"http error: {exc}",
            )
        choices = payload.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        backend_extras: dict[str, Any] = {
            "mode": "http",
            "url": url,
            "streaming": False,
        }
        return ChatRunOutcome(
            success=True,
            output_text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=None,
            decode_token_times_ms=[],
            wall_ms=wall[0],
            backend_extras=backend_extras,
        )

    # ------------------------------------------------------------------
    # embedded mode
    # ------------------------------------------------------------------
    def _run_embedded(
        self,
        model: ModelRef,
        prompt: str,
        max_output_tokens: int,
        stream: bool,
        extras: dict[str, Any] | None,
    ) -> ChatRunOutcome:
        try:
            vllm = importlib.import_module("vllm")
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"vllm import failed: {exc}",
            )

        try:
            engine = self._get_or_build_engine(vllm, model.identifier, extras or {})
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"vllm.LLM init failed: {exc}",
            )

        sampling_cls = vllm.SamplingParams
        sampling = sampling_cls(max_tokens=max_output_tokens)

        flags: list[str] = []
        decode_times: list[float] = []
        ttft_ms: float | None = None
        try:
            with stopwatch() as wall:
                outputs = engine.generate([prompt], sampling)
        except Exception as exc:  # noqa: BLE001
            return ChatRunOutcome(
                success=False,
                output_text="",
                prompt_tokens=0,
                output_tokens=0,
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=0.0,
                error=f"vllm.generate failed: {exc}",
            )

        text, prompt_tokens, output_tokens = self._extract_embedded_output(outputs)

        # Synthesize decode timings from total wall time (we don't get per-token
        # timestamps from synchronous LLM.generate). Better than nothing for now.
        if output_tokens > 0:
            per_token = wall[0] / max(1, output_tokens)
            ttft_ms = per_token
            decode_times = [per_token] * max(0, output_tokens - 1)
            flags.append("no-streaming-timings")

        backend_extras: dict[str, Any] = {"mode": "embedded", "flags": flags}
        return ChatRunOutcome(
            success=True,
            output_text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            decode_token_times_ms=decode_times,
            wall_ms=wall[0],
            backend_extras=backend_extras,
        )

    def _get_or_build_engine(
        self, vllm: Any, identifier: str, extras: dict[str, Any]
    ) -> Any:
        if self._embedded_engine is not None and self._embedded_model_id == identifier:
            return self._embedded_engine
        llm_cls = vllm.LLM
        kwargs: dict[str, Any] = {"model": identifier}
        # Allow callers to forward init kwargs via a reserved key.
        init_kwargs = extras.get("vllm_init") if isinstance(extras, dict) else None
        if isinstance(init_kwargs, dict):
            kwargs.update(init_kwargs)
        LOG.info("vllm embedded init for %s", identifier)
        engine = llm_cls(**kwargs)
        self._embedded_engine = engine
        self._embedded_model_id = identifier
        return engine

    @staticmethod
    def _extract_embedded_output(outputs: Any) -> tuple[str, int, int]:
        if not outputs:
            return "", 0, 0
        first = outputs[0]
        completions = getattr(first, "outputs", None) or []
        if not completions:
            return "", 0, 0
        comp = completions[0]
        text = getattr(comp, "text", "") or ""
        token_ids = getattr(comp, "token_ids", None) or []
        output_tokens = len(token_ids) if token_ids else 0
        prompt_token_ids = getattr(first, "prompt_token_ids", None) or []
        prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0
        return text, prompt_tokens, output_tokens


register_driver(_BACKEND_NAME, lambda: VllmDriver())


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    drv = VllmDriver()
    det = drv.detect()
    if not det.available:
        print(f"vLLM not available: {det.notes}")
    else:
        print(
            f"vLLM available: version={det.version} flags={det.build_flags} "
            f"runtime={det.runtime_versions}"
        )
        models = drv.list_models()
        print(f"HTTP models: {len(models)}")
        for m in models[:10]:
            print(f"  - {m.identifier}")
