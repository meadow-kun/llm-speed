"""llama.cpp backend driver.

Spawns a managed `llama-server` subprocess on a free local port, talks to it
via the OpenAI-compatible `/v1/chat/completions` endpoint, parses SSE for
per-token timing.

Detection is path-based: we look for `llama-server` and `llama-cli` on PATH and
parse `--version` output for the build flags (CUDA / Metal / BLAS / AVX2).

list_models scans well-known GGUF locations:
  - $LLAMA_CPP_MODELS_DIR (if set)
  - ~/.cache/llama.cpp/
  - ~/.cache/huggingface/hub/   (HF snapshot store; *.gguf live under blobs/)
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..registry import register_driver
from ..timing import now_ms
from ..types import (
    BackendDetection,
    ChatRunOutcome,
    ModelRef,
)

LOG = logging.getLogger("cli.drivers.llama_cpp")

DEFAULT_HOST = "127.0.0.1"
SERVER_BOOT_TIMEOUT_S = 60.0
SERVER_HEALTH_POLL_S = 0.25
REQUEST_TIMEOUT_S = 600.0  # generation can be long
CHUNK_READ_TIMEOUT_S = 30.0  # but a single SSE chunk must not stall
DIGEST_BYTES = 256 * 1024  # 256KB sha256 prefix

_BUILD_FLAG_PATTERNS = (
    ("CUDA", re.compile(r"\bCUDA\b", re.IGNORECASE)),
    ("Metal", re.compile(r"\bMetal\b", re.IGNORECASE)),
    ("BLAS", re.compile(r"\bBLAS\b")),
    ("AVX2", re.compile(r"\bAVX2\b")),
    ("AVX512", re.compile(r"\bAVX512\b")),
    ("ROCm", re.compile(r"\bROCm\b|\bHIP\b", re.IGNORECASE)),
    ("Vulkan", re.compile(r"\bVulkan\b", re.IGNORECASE)),
)

# Quant suffix in GGUF filenames: foo-Q4_K_M.gguf, foo.IQ3_XXS.gguf, etc.
_QUANT_RE = re.compile(
    r"[-._](Q[0-9][^.\-_/]*|IQ[0-9][^.\-_/]*|F16|F32|BF16|FP8)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _which(binary: str) -> str | None:
    return shutil.which(binary)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((DEFAULT_HOST, 0))
        return s.getsockname()[1]


def _parse_version_output(text: str) -> tuple[str | None, list[str]]:
    """Return (version_string, build_flags). llama-server --version usually prints
    one or two lines. Format observed:
        version: 4567 (abc1234)
        built with Apple clang ... for arm64-apple-darwin... CUDA Metal
    """
    version: str | None = None
    flags: list[str] = []

    m = re.search(r"version:\s*(\S+(?:\s*\([^)]+\))?)", text, re.IGNORECASE)
    if m:
        version = m.group(1).strip()
    else:
        m2 = re.search(r"\b(b\d{3,5}|\d+\.\d+(?:\.\d+)?)\b", text)
        if m2:
            version = m2.group(1)

    for label, pat in _BUILD_FLAG_PATTERNS:
        if pat.search(text):
            flags.append(label)
    return version, flags


def _quant_from_name(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else None


def _strip_quant(filename: str) -> str:
    """Best-effort human name: foo-Q4_K_M.gguf -> foo."""
    base = Path(filename).stem
    return _QUANT_RE.sub("", base).rstrip("-._")


def _digest_prefix(path: Path, n_bytes: int = DIGEST_BYTES) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fp:
            buf = fp.read(n_bytes)
        h.update(buf)
        return h.hexdigest()[:16]
    except OSError as exc:
        LOG.debug("digest failed for %s: %s", path, exc)
        return None


def _find_gguf_files(roots: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        try:
            # rglob is fine — these caches are bounded.
            for p in root.rglob("*.gguf"):
                if p.is_file():
                    rp = p.resolve()
                    seen[str(rp)] = rp
        except OSError as exc:
            LOG.debug("scan %s failed: %s", root, exc)
    return sorted(seen.values())


# ---------------------------------------------------------------------------
# Managed llama-server subprocess
# ---------------------------------------------------------------------------


@dataclass
class _ServerHandle:
    # proc is None when we're reusing an externally-managed llama-server
    # via LLAMA_CPP_SERVER_URL — we never spawn or terminate that process.
    proc: subprocess.Popen | None
    port: int
    base_url: str
    model_path: str


class LlamaCppDriver:
    name = "llama.cpp"

    def __init__(self) -> None:
        self._server: _ServerHandle | None = None
        self._client: httpx.Client | None = None
        # Make sure we always tear down on interpreter exit.
        atexit.register(self.close)

    # ------------------------------------------------------------------ detect

    def detect(self) -> BackendDetection:
        # If the user already has a llama-server running and pointed at it
        # via LLAMA_CPP_SERVER_URL, treat the backend as available even if
        # the binaries aren't on this host's PATH (common when the server
        # is in a sibling container or a different login shell). The bench
        # will reuse the running server in _ensure_server.
        external = os.environ.get("LLAMA_CPP_SERVER_URL", "").strip()
        server_bin = _which("llama-server")
        cli_bin = _which("llama-cli")
        if not server_bin and not cli_bin and not external:
            return BackendDetection(
                available=False,
                name=self.name,
                notes="neither llama-server nor llama-cli on PATH; set LLAMA_CPP_SERVER_URL to reuse an existing server",
            )

        version: str | None = None
        flags: list[str] = []
        for binary in (server_bin, cli_bin):
            if not binary:
                continue
            try:
                proc = subprocess.run(
                    [binary, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
                v, f = _parse_version_output(blob)
                version = version or v
                for flag in f:
                    if flag not in flags:
                        flags.append(flag)
                if version:
                    break
            except (OSError, subprocess.SubprocessError) as exc:
                LOG.debug("--version probe failed for %s: %s", binary, exc)

        notes_parts = []
        if server_bin:
            notes_parts.append(f"server={server_bin}")
        if cli_bin:
            notes_parts.append(f"cli={cli_bin}")
        return BackendDetection(
            available=True,
            name=self.name,
            version=version,
            build_flags=flags,
            notes="; ".join(notes_parts),
        )

    # -------------------------------------------------------------- list_models

    def list_models(self) -> list[ModelRef]:
        roots: list[Path] = []
        env_dir = os.environ.get("LLAMA_CPP_MODELS_DIR")
        if env_dir:
            roots.append(Path(env_dir).expanduser())
        roots.append(Path.home() / ".cache" / "llama.cpp")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")

        out: list[ModelRef] = []
        for path in _find_gguf_files(roots):
            fname = path.name
            quant = _quant_from_name(fname)
            name = _strip_quant(fname)
            digest = _digest_prefix(path)
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None
            out.append(
                ModelRef(
                    backend=self.name,
                    identifier=str(path),
                    name=name,
                    quant=quant,
                    digest=digest,
                    extras={
                        "path": str(path),
                        "size_bytes": size_bytes,
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
        try:
            self._ensure_server(model.identifier, extras)
        except Exception as exc:  # noqa: BLE001
            return _failed_outcome(f"failed to start llama-server: {exc}")

        assert self._server is not None
        assert self._client is not None

        payload: dict[str, Any] = {
            "model": Path(model.identifier).name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "stream": bool(stream),
        }
        # Bubble up generation knobs that callers may want.
        for k in ("temperature", "top_p", "top_k", "seed", "stop"):
            if k in extras:
                payload[k] = extras[k]

        url = f"{self._server.base_url}/v1/chat/completions"

        if not stream:
            return self._run_chat_nonstream(url, payload)
        return self._run_chat_stream(url, payload)

    # ------------------------------------------------------------- streaming

    def _run_chat_stream(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> ChatRunOutcome:
        assert self._client is not None
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
                timeout=httpx.Timeout(REQUEST_TIMEOUT_S, read=CHUNK_READ_TIMEOUT_S),
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
                        LOG.debug("could not parse SSE chunk: %r", data[:120])
                        continue

                    now = now_ms()
                    delta_text = ""
                    choices = evt.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        # Reasoning models (Qwen3-Thinking, DeepSeek-R1, etc.)
                        # emit hidden chain-of-thought as `reasoning_content`
                        # before the visible `content`. Both are real
                        # GPU-generated tokens; counting only one
                        # under-reports decode tps.
                        delta_text = (delta.get("content") or "") + (
                            delta.get("reasoning_content") or ""
                        )
                    if delta_text:
                        if ttft_ms is None:
                            ttft_ms = now - t_start
                        else:
                            token_times.append(now - last_chunk_ms)
                        last_chunk_ms = now
                        out_text_parts.append(delta_text)

                    if "usage" in evt and evt["usage"]:
                        usage = evt["usage"]
                        prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                        output_tokens = usage.get("completion_tokens") or output_tokens
                        backend_extras["usage"] = usage

        except httpx.HTTPError as exc:
            return _failed_outcome(f"streaming request failed: {exc}")

        wall_ms = now_ms() - t_start
        text = "".join(out_text_parts)
        if not output_tokens:
            # token_times is len-of-output minus 1 (the first token's wait is ttft).
            output_tokens = (len(token_times) + 1) if ttft_ms is not None else 0

        return ChatRunOutcome(
            success=True,
            output_text=text,
            prompt_tokens=int(prompt_tokens or 0),
            output_tokens=int(output_tokens or 0),
            ttft_ms=ttft_ms,
            decode_token_times_ms=token_times,
            wall_ms=wall_ms,
            backend_extras=backend_extras,
        )

    # ------------------------------------------------------------ non-streaming

    def _run_chat_nonstream(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> ChatRunOutcome:
        assert self._client is not None
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

    # --------------------------------------------------------- server lifecycle

    def _ensure_server(self, model_path: str, extras: dict[str, Any]) -> None:
        # External-server mode: caller pointed us at an already-running
        # llama-server with LLAMA_CPP_SERVER_URL. We don't own its lifecycle,
        # so we never spawn or terminate; just open a client to it.
        external = os.environ.get("LLAMA_CPP_SERVER_URL", "").strip()
        if external:
            base_url = external.rstrip("/")
            if self._server is not None and self._server.base_url == base_url:
                return
            LOG.info("reusing external llama-server at %s", base_url)
            self._server = _ServerHandle(
                proc=None, port=0, base_url=base_url, model_path=model_path
            )
            self._client = httpx.Client(timeout=REQUEST_TIMEOUT_S)
            return

        if self._server is not None and self._server.model_path == model_path:
            if self._server.proc is not None and self._server.proc.poll() is None:
                return
            LOG.warning("existing llama-server died; respawning")
            self.close()

        if self._server is not None and self._server.model_path != model_path:
            LOG.info("model changed; restarting llama-server")
            self.close()

        binary = _which("llama-server")
        if not binary:
            raise RuntimeError("llama-server not on PATH")
        if not Path(model_path).exists():
            raise RuntimeError(f"model not found: {model_path}")

        port = _free_port()
        base_url = f"http://{DEFAULT_HOST}:{port}"
        cmd: list[str] = [
            binary,
            "-m",
            model_path,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(port),
            "--log-disable",
        ]

        # Configurable extras → CLI flags.
        if "n_ctx" in extras:
            cmd += ["--ctx-size", str(int(extras["n_ctx"]))]
        if "n_batch" in extras:
            cmd += ["--batch-size", str(int(extras["n_batch"]))]
        if "n_gpu_layers" in extras:
            cmd += ["--n-gpu-layers", str(int(extras["n_gpu_layers"]))]
        if extras.get("flash_attn"):
            cmd += ["--flash-attn"]

        LOG.info("starting llama-server: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        try:
            self._wait_ready(base_url, proc)
        except Exception:
            try:
                proc.terminate()
            except OSError:
                pass
            raise

        self._server = _ServerHandle(
            proc=proc, port=port, base_url=base_url, model_path=model_path
        )
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT_S)

    def _wait_ready(self, base_url: str, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + SERVER_BOOT_TIMEOUT_S
        with httpx.Client(timeout=2.0) as probe:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited early (rc={proc.returncode})"
                    )
                for path in ("/health", "/v1/models"):
                    try:
                        r = probe.get(f"{base_url}{path}")
                        if r.status_code in (200, 503):
                            # 503 means "loading" on /health for some builds — keep waiting
                            if r.status_code == 200:
                                return
                    except httpx.HTTPError:
                        pass
                time.sleep(SERVER_HEALTH_POLL_S)
        raise RuntimeError("llama-server did not become ready in time")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        if self._server is not None:
            proc = self._server.proc
            self._server = None
            # External-server mode (proc is None): we don't own this process.
            if proc is None:
                return
            if proc.poll() is None:
                LOG.info("terminating llama-server (pid=%d)", proc.pid)
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                except OSError as exc:
                    LOG.debug("terminate failed: %s", exc)

    def __del__(self) -> None:  # belt-and-braces
        try:
            self.close()
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


register_driver("llama.cpp", lambda: LlamaCppDriver())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    drv = LlamaCppDriver()
    det = drv.detect()
    print("detect:", det)
    if not det.available:
        raise SystemExit(0)

    models = drv.list_models()
    print(f"found {len(models)} GGUF models")
    for m in models[:5]:
        print(f"  - {m.name} quant={m.quant} digest={m.digest} path={m.identifier}")

    if not models:
        print("no models to smoke-test against; exiting cleanly")
        raise SystemExit(0)

    chosen = models[0]
    print(f"smoke-testing chat against {chosen.identifier}")
    try:
        outcome = drv.run_chat(
            chosen,
            "Reply with the single word: hello.",
            max_output_tokens=10,
            stream=True,
            extras={"n_ctx": 1024, "n_gpu_layers": 99},
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
    finally:
        drv.close()
