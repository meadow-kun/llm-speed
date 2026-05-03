"""Hosted-API backend driver.

A single driver that fans out across OpenAI-compatible providers
(OpenAI, OpenRouter, Together, Fireworks, Groq) plus a thin Anthropic adapter.

Detection is purely env-var based: at least one of {OPENAI, OPENROUTER, TOGETHER,
FIREWORKS, GROQ, ANTHROPIC}_API_KEY must be set.

ModelRef.identifier is namespaced as `<provider>/<model_id>`. The provider name
is also stored in `model.extras['provider']` so the driver can route requests
without re-parsing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from ..registry import register_driver
from ..timing import now_ms
from ..types import (
    BackendDetection,
    ChatRunOutcome,
    ModelRef,
)

LOG = logging.getLogger("cli.drivers.hosted_api")

CONNECT_TIMEOUT_S = 10.0
LIST_TIMEOUT_S = 20.0
REQUEST_TIMEOUT_S = 600.0
CHUNK_READ_TIMEOUT_S = 60.0


# Provider name → (env var, base URL, kind). kind is "openai" or "anthropic".
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "openai"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai"),
    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1", "openai"),
    "fireworks": (
        "FIREWORKS_API_KEY",
        "https://api.fireworks.ai/inference/v1",
        "openai",
    ),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai"),
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "anthropic"),
}

ANTHROPIC_VERSION = "2023-06-01"


def _enabled_providers() -> list[str]:
    return [name for name, (env, _, _) in PROVIDERS.items() if os.environ.get(env)]


def _provider_meta(provider: str) -> tuple[str, str, str]:
    return PROVIDERS[provider]


def _auth_headers(provider: str, api_key: str) -> dict[str, str]:
    _, _, kind = _provider_meta(provider)
    if kind == "anthropic":
        return {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------


class HostedApiDriver:
    name = "hosted-api"

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(LIST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        )

    # -------------------------------------------------------------------- detect

    def detect(self) -> BackendDetection:
        active = _enabled_providers()
        if not active:
            return BackendDetection(
                available=False,
                name=self.name,
                notes="no provider env vars set (OPENAI_API_KEY, OPENROUTER_API_KEY, ...)",
            )
        return BackendDetection(
            available=True,
            name=self.name,
            version=None,
            notes=f"providers configured: {', '.join(active)}",
            runtime_versions={p: PROVIDERS[p][1] for p in active},
        )

    # ---------------------------------------------------------------- list_models

    def list_models(self) -> list[ModelRef]:
        out: list[ModelRef] = []
        for provider in _enabled_providers():
            try:
                out.extend(self._list_provider_models(provider))
            except Exception as exc:  # noqa: BLE001
                LOG.warning("list_models for %s failed: %s", provider, exc)
        return out

    def _list_provider_models(self, provider: str) -> list[ModelRef]:
        env_var, base_url, kind = _provider_meta(provider)
        api_key = os.environ.get(env_var)
        if not api_key:
            return []

        if kind == "anthropic":
            url = f"{base_url}/models"
        else:
            url = f"{base_url}/models"

        try:
            r = self._client.get(
                url, headers=_auth_headers(provider, api_key), timeout=LIST_TIMEOUT_S
            )
        except httpx.HTTPError as exc:
            LOG.debug("%s /models failed: %s", provider, exc)
            return []

        if r.status_code != 200:
            LOG.debug("%s /models HTTP %d: %s", provider, r.status_code, r.text[:200])
            return []

        try:
            payload = r.json()
        except json.JSONDecodeError:
            return []

        models_raw = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models_raw, list):
            return []

        out: list[ModelRef] = []
        for entry in models_raw:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id") or entry.get("name")
            if not model_id:
                continue
            # Don't carry the upstream record around: providers return arbitrary
            # JSON in their /models response (often >10 KB of pricing, context
            # specs, license blurbs) that would (a) leak undocumented metadata
            # to our server on upload, and (b) routinely trip the privacy
            # invariant's 256-char check on long-text fields like description.
            # We keep only the two values the driver itself needs at run time.
            out.append(
                ModelRef(
                    backend=self.name,
                    identifier=f"{provider}/{model_id}",
                    name=model_id,
                    extras={
                        "provider": provider,
                        "base_url": base_url,
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

        provider, model_id = self._split_identifier(model)
        if provider is None:
            return _failed_outcome("could not determine provider for model")

        env_var, base_url, kind = _provider_meta(provider)
        api_key = os.environ.get(env_var)
        if not api_key:
            return _failed_outcome(f"no API key set for provider {provider!r}")

        try:
            if kind == "anthropic":
                return self._anthropic_chat(
                    base_url,
                    api_key,
                    model_id,
                    prompt,
                    max_output_tokens=max_output_tokens,
                    stream=stream,
                    extras=extras,
                )
            return self._openai_chat(
                provider,
                base_url,
                api_key,
                model_id,
                prompt,
                max_output_tokens=max_output_tokens,
                stream=stream,
                extras=extras,
            )
        except httpx.HTTPError as exc:
            return _failed_outcome(f"network error: {exc}")

    def _split_identifier(self, model: ModelRef) -> tuple[str | None, str]:
        provider = (model.extras or {}).get("provider")
        if provider:
            ident = model.identifier
            prefix = f"{provider}/"
            model_id = ident[len(prefix) :] if ident.startswith(prefix) else ident
            return provider, model_id
        if "/" in model.identifier:
            head, _, tail = model.identifier.partition("/")
            if head in PROVIDERS:
                return head, tail
        return None, model.identifier

    # ------------------------------------------------------ OpenAI-compatible

    def _openai_chat(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        model_id: str,
        prompt: str,
        *,
        max_output_tokens: int,
        stream: bool,
        extras: dict[str, Any],
    ) -> ChatRunOutcome:
        url = f"{base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "stream": bool(stream),
        }
        if stream:
            # OpenAI returns usage in the final chunk only when explicitly requested.
            payload["stream_options"] = {"include_usage": True}
        for k in ("temperature", "top_p", "stop", "seed"):
            if k in extras:
                payload[k] = extras[k]

        headers = _auth_headers(provider, api_key)

        if not stream:
            return self._openai_nonstream(url, headers, payload)
        return self._openai_stream(
            url, headers, payload, provider=provider, model_id=model_id
        )

    def _openai_stream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        provider: str,
        model_id: str,
    ) -> ChatRunOutcome:
        out_text_parts: list[str] = []
        token_times: list[float] = []
        ttft_ms: float | None = None
        prompt_tokens = 0
        output_tokens = 0
        backend_extras: dict[str, Any] = {"provider": provider, "model_id": model_id}

        t_start = now_ms()
        last_chunk_ms = t_start

        with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(
                REQUEST_TIMEOUT_S, read=CHUNK_READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S
            ),
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                return _failed_outcome(f"HTTP {resp.status_code}: {body[:400]}")

            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                elif line.startswith("data:"):
                    data = line[5:]
                else:
                    continue
                data = data.strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                now = now_ms()
                choices = evt.get("choices") or []
                delta_text = ""
                if choices:
                    delta = choices[0].get("delta") or {}
                    delta_text = delta.get("content") or ""
                if delta_text:
                    if ttft_ms is None:
                        ttft_ms = now - t_start
                    else:
                        token_times.append(now - last_chunk_ms)
                    last_chunk_ms = now
                    out_text_parts.append(delta_text)

                if evt.get("usage"):
                    usage = evt["usage"]
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    output_tokens = int(usage.get("completion_tokens") or output_tokens)
                    backend_extras["usage"] = usage

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

    def _openai_nonstream(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> ChatRunOutcome:
        payload = {**payload, "stream": False}
        payload.pop("stream_options", None)
        t_start = now_ms()
        resp = self._client.post(
            url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S
        )
        wall_ms = now_ms() - t_start
        if resp.status_code != 200:
            return _failed_outcome(f"HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return ChatRunOutcome(
            success=True,
            output_text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            ttft_ms=None,
            decode_token_times_ms=[],
            wall_ms=wall_ms,
            backend_extras={"usage": usage} if usage else {},
        )

    # --------------------------------------------------------------- Anthropic

    def _anthropic_chat(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        prompt: str,
        *,
        max_output_tokens: int,
        stream: bool,
        extras: dict[str, Any],
    ) -> ChatRunOutcome:
        url = f"{base_url}/messages"
        payload: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": bool(stream),
        }
        if "system" in extras:
            payload["system"] = extras["system"]
        for k in ("temperature", "top_p", "top_k", "stop_sequences"):
            if k in extras:
                payload[k] = extras[k]

        headers = _auth_headers("anthropic", api_key)

        if not stream:
            t_start = now_ms()
            resp = self._client.post(
                url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S
            )
            wall_ms = now_ms() - t_start
            if resp.status_code != 200:
                return _failed_outcome(f"HTTP {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
            text = "".join(
                blk.get("text", "")
                for blk in (data.get("content") or [])
                if isinstance(blk, dict) and blk.get("type") == "text"
            )
            usage = data.get("usage") or {}
            return ChatRunOutcome(
                success=True,
                output_text=text,
                prompt_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                ttft_ms=None,
                decode_token_times_ms=[],
                wall_ms=wall_ms,
                backend_extras={"usage": usage, "stop_reason": data.get("stop_reason")},
            )

        # Streaming: Anthropic uses named SSE events with `event:` + `data:` pairs.
        out_text_parts: list[str] = []
        token_times: list[float] = []
        ttft_ms: float | None = None
        prompt_tokens = 0
        output_tokens = 0
        backend_extras: dict[str, Any] = {"provider": "anthropic", "model_id": model_id}

        t_start = now_ms()
        last_chunk_ms = t_start

        with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(
                REQUEST_TIMEOUT_S, read=CHUNK_READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S
            ),
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                return _failed_outcome(f"HTTP {resp.status_code}: {body[:400]}")

            current_event: str | None = None
            for line in resp.iter_lines():
                if line is None:
                    continue
                if not line:
                    current_event = None
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                now = now_ms()
                etype = evt.get("type") or current_event

                if etype == "message_start":
                    usage = (evt.get("message") or {}).get("usage") or {}
                    prompt_tokens = int(usage.get("input_tokens") or prompt_tokens)
                elif etype == "content_block_delta":
                    delta = evt.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        chunk = delta.get("text") or ""
                        if chunk:
                            if ttft_ms is None:
                                ttft_ms = now - t_start
                            else:
                                token_times.append(now - last_chunk_ms)
                            last_chunk_ms = now
                            out_text_parts.append(chunk)
                elif etype == "message_delta":
                    usage = evt.get("usage") or {}
                    if usage.get("output_tokens") is not None:
                        output_tokens = int(usage["output_tokens"])
                    if (evt.get("delta") or {}).get("stop_reason"):
                        backend_extras["stop_reason"] = evt["delta"]["stop_reason"]
                elif etype == "message_stop":
                    break
                elif etype == "error":
                    return _failed_outcome(f"anthropic stream error: {evt}")

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


register_driver("hosted-api", lambda: HostedApiDriver())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    drv = HostedApiDriver()
    det = drv.detect()
    print("detect:", det)
    if not det.available:
        print("set one of OPENAI_API_KEY, OPENROUTER_API_KEY, ... to enable")
        raise SystemExit(0)

    models = drv.list_models()
    print(f"found {len(models)} models across providers")
    by_provider: dict[str, int] = {}
    for m in models:
        prov = (m.extras or {}).get("provider", "?")
        by_provider[prov] = by_provider.get(prov, 0) + 1
    print("by provider:", by_provider)

    # Pick a small/cheap default per provider for the smoke run.
    candidates = {
        "groq": "llama-3.1-8b-instant",
        "openai": "gpt-4o-mini",
        "openrouter": "openai/gpt-4o-mini",
        "together": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "fireworks": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "anthropic": "claude-3-5-haiku-latest",
    }
    enabled = _enabled_providers()
    chosen: ModelRef | None = None
    for p in enabled:
        target = candidates.get(p)
        if not target:
            continue
        match = next(
            (
                m
                for m in models
                if (m.extras or {}).get("provider") == p
                and m.name.endswith(target.split("/")[-1])
            ),
            None,
        )
        if match:
            chosen = match
            break
    if chosen is None and models:
        chosen = models[0]

    if chosen is None:
        print("no model to smoke-test")
        raise SystemExit(0)

    print(f"smoke-testing chat against {chosen.identifier}")
    outcome = drv.run_chat(
        chosen,
        "Reply with exactly one word: hello.",
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
        },
    )
    drv.close()
