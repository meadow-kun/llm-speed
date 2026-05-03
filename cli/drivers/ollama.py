"""Ollama backend driver.

Talks to a running ollama daemon at http://localhost:11434. The daemon must be
running — we do not attempt to start it. Ollama exposes:
  - GET  /api/version       — server version
  - GET  /api/tags          — installed models
  - POST /api/chat          — chat completions, NDJSON streaming

Streamed `/api/chat` returns one JSON object per line; the final object has
`done: true` plus authoritative timing counters (eval_count, eval_duration,
prompt_eval_count, prompt_eval_duration, load_duration). We use those when
present and ALSO record per-chunk wall-clock deltas as a cross-check.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from ..registry import register_driver
from ..timing import now_ms
from ..types import (
    BackendDetection,
    ChatRunOutcome,
    ModelRef,
)

LOG = logging.getLogger("cli.drivers.ollama")

DEFAULT_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
if not DEFAULT_BASE.startswith("http"):
    DEFAULT_BASE = f"http://{DEFAULT_BASE}"

CONNECT_TIMEOUT_S = 3.0
LIST_TIMEOUT_S = 10.0
REQUEST_TIMEOUT_S = 600.0
CHUNK_READ_TIMEOUT_S = 60.0  # generation can pause on long contexts; cap stalls

_QUANT_RE = re.compile(
    r"\b(Q[0-9][^\s/]*|IQ[0-9][^\s/]*|F16|F32|BF16|FP8)\b", re.IGNORECASE
)
_SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?[bBmM])\b")


def _parse_quant(*haystacks: str) -> str | None:
    for h in haystacks:
        if not h:
            continue
        m = _QUANT_RE.search(h)
        if m:
            return m.group(1).upper()
    return None


def _parse_size(*haystacks: str) -> str | None:
    for h in haystacks:
        if not h:
            continue
        m = _SIZE_RE.search(h)
        if m:
            return m.group(1).upper()
    return None


def _name_from_tag(tag: str) -> str:
    """Strip the `:tag` suffix, return the model family. `qwen3:8b` → `qwen3`."""
    return tag.split(":", 1)[0]


class OllamaDriver:
    name = "ollama"

    def __init__(self, base_url: str = DEFAULT_BASE) -> None:
        self.base_url = base_url
        # Daemon-local; tight default timeouts. We override per-request for chat.
        self._client = httpx.Client(
            timeout=httpx.Timeout(LIST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        )

    # -------------------------------------------------------------------- detect

    def detect(self) -> BackendDetection:
        try:
            tags_r = self._client.get(f"{self.base_url}/api/tags")
        except httpx.HTTPError as exc:
            return BackendDetection(
                available=False,
                name=self.name,
                notes=f"daemon not reachable at {self.base_url}: {exc}",
            )
        if tags_r.status_code != 200:
            return BackendDetection(
                available=False,
                name=self.name,
                notes=f"daemon returned HTTP {tags_r.status_code} on /api/tags",
            )

        version: str | None = None
        try:
            v_r = self._client.get(f"{self.base_url}/api/version")
            if v_r.status_code == 200:
                version = (v_r.json() or {}).get("version")
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            LOG.debug("version probe failed: %s", exc)

        return BackendDetection(
            available=True,
            name=self.name,
            version=version,
            notes=f"daemon at {self.base_url}",
        )

    # ---------------------------------------------------------------- list_models

    def list_models(self) -> list[ModelRef]:
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=LIST_TIMEOUT_S)
        except httpx.HTTPError as exc:
            LOG.warning("list_models failed: %s", exc)
            return []
        if r.status_code != 200:
            LOG.warning("list_models HTTP %d: %s", r.status_code, r.text[:200])
            return []

        try:
            payload = r.json()
        except json.JSONDecodeError:
            LOG.warning("list_models bad JSON")
            return []

        out: list[ModelRef] = []
        for entry in payload.get("models", []) or []:
            tag = entry.get("name") or entry.get("model") or ""
            if not tag:
                continue
            details = entry.get("details") or {}
            quant = details.get("quantization_level") or _parse_quant(
                tag, details.get("format", "")
            )
            size = details.get("parameter_size") or _parse_size(
                tag, details.get("family", "")
            )
            digest = entry.get("digest")
            if digest and len(digest) > 16:
                digest = digest[:16]
            out.append(
                ModelRef(
                    backend=self.name,
                    identifier=tag,
                    name=_name_from_tag(tag),
                    size=size,
                    quant=quant.upper() if isinstance(quant, str) else quant,
                    digest=digest,
                    extras={
                        "modified_at": entry.get("modified_at"),
                        "size_bytes": entry.get("size"),
                        "details": details,
                    },
                )
            )
        return out

    # ----------------------------------------------------------------- run_chat

    def run_chat(
        self,
        model: ModelRef,
        prompt: str,
        *,
        max_output_tokens: int,
        stream: bool = True,
        extras: dict[str, Any] | None = None,
    ) -> ChatRunOutcome:
        extras = dict(extras or {})

        options: dict[str, Any] = {"num_predict": int(max_output_tokens)}
        # passthrough generation knobs
        for k in ("temperature", "top_p", "top_k", "seed", "num_ctx", "stop"):
            if k in extras:
                options[k] = extras[k]
        if "options" in extras and isinstance(extras["options"], dict):
            options.update(extras["options"])

        payload: dict[str, Any] = {
            "model": model.identifier,
            "messages": [{"role": "user", "content": prompt}],
            "stream": bool(stream),
            "options": options,
        }
        if "keep_alive" in extras:
            payload["keep_alive"] = extras["keep_alive"]

        url = f"{self.base_url}/api/chat"

        if not stream:
            return self._run_chat_nonstream(url, payload)
        return self._run_chat_stream(url, payload)

    # ---------------------------------------------------------- streaming impl

    def _run_chat_stream(self, url: str, payload: dict[str, Any]) -> ChatRunOutcome:
        out_text_parts: list[str] = []
        token_times: list[float] = []
        ttft_ms: float | None = None
        prompt_tokens = 0
        output_tokens = 0
        backend_extras: dict[str, Any] = {}

        t_start = now_ms()
        last_chunk_ms = t_start

        try:
            with self._client.stream(
                "POST",
                url,
                json=payload,
                timeout=httpx.Timeout(
                    REQUEST_TIMEOUT_S,
                    read=CHUNK_READ_TIMEOUT_S,
                    connect=CONNECT_TIMEOUT_S,
                ),
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    return _failed_outcome(f"HTTP {resp.status_code}: {body[:400]}")

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        LOG.debug("bad NDJSON line: %r", line[:200])
                        continue

                    now = now_ms()
                    msg = evt.get("message") or {}
                    chunk_text = msg.get("content") or ""
                    if chunk_text:
                        if ttft_ms is None:
                            ttft_ms = now - t_start
                        else:
                            token_times.append(now - last_chunk_ms)
                        last_chunk_ms = now
                        out_text_parts.append(chunk_text)

                    if evt.get("done"):
                        # Final chunk carries the authoritative counters.
                        if "eval_count" in evt:
                            output_tokens = int(evt["eval_count"])
                        if "prompt_eval_count" in evt:
                            prompt_tokens = int(evt["prompt_eval_count"])
                        backend_extras["eval_duration_ns"] = evt.get("eval_duration")
                        backend_extras["prompt_eval_duration_ns"] = evt.get(
                            "prompt_eval_duration"
                        )
                        backend_extras["load_duration_ns"] = evt.get("load_duration")
                        backend_extras["total_duration_ns"] = evt.get("total_duration")
                        backend_extras["done_reason"] = evt.get("done_reason")
                        # Cross-check tps from authoritative timings.
                        ed = evt.get("eval_duration")
                        ec = evt.get("eval_count")
                        if ed and ec and ed > 0:
                            backend_extras["decode_tps_authoritative"] = ec / (ed / 1e9)
                        break

        except httpx.HTTPError as exc:
            return _failed_outcome(f"streaming request failed: {exc}")

        wall_ms = now_ms() - t_start
        if not output_tokens:
            output_tokens = (len(token_times) + 1) if ttft_ms is not None else 0

        return ChatRunOutcome(
            success=True,
            output_text="".join(out_text_parts),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            decode_token_times_ms=token_times,
            wall_ms=wall_ms,
            backend_extras=backend_extras,
        )

    # ------------------------------------------------------- non-streaming impl

    def _run_chat_nonstream(self, url: str, payload: dict[str, Any]) -> ChatRunOutcome:
        t_start = now_ms()
        try:
            resp = self._client.post(url, json=payload, timeout=REQUEST_TIMEOUT_S)
        except httpx.HTTPError as exc:
            return _failed_outcome(f"request failed: {exc}")

        wall_ms = now_ms() - t_start
        if resp.status_code != 200:
            return _failed_outcome(f"HTTP {resp.status_code}: {resp.text[:400]}")

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            return _failed_outcome(f"bad JSON: {exc}")

        msg = data.get("message") or {}
        text = msg.get("content") or ""
        backend_extras = {
            "eval_duration_ns": data.get("eval_duration"),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
            "load_duration_ns": data.get("load_duration"),
            "total_duration_ns": data.get("total_duration"),
            "done_reason": data.get("done_reason"),
        }
        return ChatRunOutcome(
            success=True,
            output_text=text,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            ttft_ms=None,
            decode_token_times_ms=[],
            wall_ms=wall_ms,
            backend_extras=backend_extras,
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


def _failed_outcome(error: str) -> ChatRunOutcome:
    return ChatRunOutcome(
        success=False,
        output_text="",
        prompt_tokens=0,
        output_tokens=0,
        ttft_ms=None,
        decode_token_times_ms=[],
        wall_ms=0.0,
        error=error,
    )


register_driver("ollama", lambda: OllamaDriver())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    drv = OllamaDriver()
    det = drv.detect()
    print("detect:", det)
    if not det.available:
        print("ollama daemon not reachable; start with `ollama serve`")
        raise SystemExit(0)

    models = drv.list_models()
    print(f"found {len(models)} models")
    for m in models[:5]:
        print(f"  - {m.identifier}  size={m.size}  quant={m.quant}  digest={m.digest}")

    if not models:
        print("no models pulled; try `ollama pull qwen2.5:0.5b`")
        raise SystemExit(0)

    chosen = next(
        (
            m
            for m in models
            if "0.5b" in m.identifier.lower() or "1b" in m.identifier.lower()
        ),
        models[0],
    )
    print(f"smoke-testing chat against {chosen.identifier}")
    outcome = drv.run_chat(
        chosen,
        "Reply with one word: hello.",
        max_output_tokens=10,
        stream=True,
    )
    print(
        "outcome:",
        {
            "success": outcome.success,
            "ttft_ms": outcome.ttft_ms,
            "output_tokens": outcome.output_tokens,
            "decode_count": len(outcome.decode_token_times_ms),
            "wall_ms": outcome.wall_ms,
            "error": outcome.error,
            "text": outcome.output_text[:80],
            "extras": outcome.backend_extras,
        },
    )
    drv.close()
