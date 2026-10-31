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
import math
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

from ..config import CACHE_DIR
from ..launch_telemetry import (
    ReasoningTracker,
    label_server_args,
    model_file_label,
    parse_llama_build,
)
from ..registry import register_driver
from ..sampling import DRIVER_SUPPORT as SAMPLING_DRIVER_SUPPORT
from ..sampling import split as split_sampling
from ..timing import now_ms
from ..types import (
    BackendDetection,
    ChatRunOutcome,
    ModelRef,
)

LOG = logging.getLogger("cli.drivers.llama_cpp")

DEFAULT_HOST = "127.0.0.1"


def _boot_timeout_s() -> float:
    """Big GGUFs (25 GB+) on slow disks need minutes to load; 60 s was too short."""
    try:
        return float(os.environ.get("LLM_SPEED_LLAMA_BOOT_TIMEOUT_S", "600"))
    except ValueError:
        return 600.0


SERVER_BOOT_TIMEOUT_S = _boot_timeout_s()
SERVER_LOG_TAIL_BYTES = 16 * 1024
SERVER_HEALTH_POLL_S = 0.25
REQUEST_TIMEOUT_S = 600.0  # generation can be long
CHUNK_READ_TIMEOUT_S = 30.0  # but a single SSE chunk must not stall
# llama-server sends nothing until prefill finishes, so the first read must allow
# for the prompt: assume prefill no slower than this many tokens/s (~3.5 chars each).
MIN_PREFILL_TPS = 50.0
MAX_READ_TIMEOUT_S = 3600.0
# Local overrides for runs that read weights from disk (mmap / partial offload), where
# prefill can fall below 50 tok/s and a cold first request can take over 30 s. Read at
# request time, never uploaded. Unset, empty or invalid values keep the defaults above.
MIN_PREFILL_TPS_ENV = "LLM_SPEED_MIN_PREFILL_TPS"
READ_TIMEOUT_ENV = "LLM_SPEED_READ_TIMEOUT_S"


def _positive_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        val = float("nan")
    if not math.isfinite(val) or val <= 0:
        LOG.warning("ignoring %s=%r (want a number > 0); using %s", name, raw, default)
        return default
    return val


def min_prefill_tps() -> float:
    """Prefill floor (tokens/s) used to size the read timeout; $LLM_SPEED_MIN_PREFILL_TPS."""
    return _positive_env_float(MIN_PREFILL_TPS_ENV, MIN_PREFILL_TPS)


def base_read_timeout_s() -> float:
    """Read timeout before the prompt allowance; $LLM_SPEED_READ_TIMEOUT_S (default 30 s)."""
    return _positive_env_float(READ_TIMEOUT_ENV, CHUNK_READ_TIMEOUT_S)


def read_timeout_for_prompt(prompt_chars: int) -> float:
    """Per-read timeout that scales with prompt length (CLI_ISSUES #9): 30 s for
    short prompts, ~12 min for a 128K-token prompt, capped at an hour.
    $LLM_SPEED_MIN_PREFILL_TPS / $LLM_SPEED_READ_TIMEOUT_S loosen it for disk-bound runs."""
    est_tokens = max(0, prompt_chars) / 3.5
    return min(
        MAX_READ_TIMEOUT_S, base_read_timeout_s() + est_tokens / min_prefill_tps()
    )


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
    r"[-._]((?:UD-)?(?:Q[0-9][^.\-/]*|IQ[0-9][^.\-/]*)|F16|F32|BF16|FP8|NVFP4|MXFP4)\b",
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
        m2 = re.search(r"\b(b\d{3,}|\d+\.\d+(?:\.\d+)?)\b", text)
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
    log_path: Path | None = None


# Lines in llama-server's log that explain why it died, most specific first.
_SERVER_ERROR_HINTS = re.compile(
    r"out of memory|cudaMalloc failed|failed to allocate|unable to allocate|"
    r"alloc_buffer|unknown model architecture|failed to load model|"
    r"error loading model|exception|error:",
    re.IGNORECASE,
)


def _server_log_reason(log_path: Path | None) -> str | None:
    """Best single line from the tail of the llama-server log, or None."""
    if log_path is None:
        return None
    try:
        with log_path.open("rb") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            fp.seek(max(0, size - SERVER_LOG_TAIL_BYTES))
            tail = fp.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    hits = [ln for ln in lines if _SERVER_ERROR_HINTS.search(ln)]
    if not hits:
        return None
    # Prefer a memory / architecture line over the generic final error.
    for ln in hits:
        if re.search(r"memory|alloc|architecture", ln, re.IGNORECASE):
            return ln[:200]
    return hits[-1][:200]


class LlamaCppDriver:
    name = "llama.cpp"

    def __init__(self) -> None:
        self._server: _ServerHandle | None = None
        self._client: httpx.Client | None = None
        # Extra llama-server args from `llm-speed bench --llama-args` /
        # $LLM_SPEED_LLAMA_ARGS. Applied when we spawn the server; recorded as a
        # declared label when reusing an external server (LLAMA_CPP_SERVER_URL).
        self.server_args: list[str] = []
        # Sampling sent with every request (cli/sampling.py; set by `bench`).
        self.sampling: dict[str, Any] = {}
        self._version_text: str | None = None
        self._props_build_info: str | None = None
        self._props_n_ctx: int | None = None
        self._props_fetched = False
        self._telemetry: list[dict[str, Any]] = []
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
                if v and self._version_text is None:
                    self._version_text = blob
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
        if stream:
            payload["stream_options"] = {"include_usage": True}
        # 0.0.10: sampling requested on the bench command line / preset. llama-server's
        # OpenAI-compatible endpoint takes all five as request fields.
        sent, _ignored = split_sampling(self.sampling, self.sampling_supported(model))
        payload.update(sent)
        # Bubble up generation knobs that callers may want (per call; wins).
        for k in ("temperature", "top_p", "top_k", "min_p", "seed", "stop"):
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
        tracker = ReasoningTracker(t_start)

        try:
            with self._client.stream(
                "POST",
                url,
                json=payload,
                timeout=httpx.Timeout(
                    REQUEST_TIMEOUT_S,
                    read=read_timeout_for_prompt(
                        sum(
                            len(str(m.get("content") or ""))
                            for m in payload.get("messages") or []
                        )
                    ),
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
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        delta_text = content + reasoning
                        if delta_text:
                            tracker.feed(now, content=content, reasoning=reasoning)
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

        except httpx.ReadTimeout as exc:
            return _failed_outcome(
                f"streaming request failed: ReadTimeout: {exc} (prompt-sized read timeout; "
                f"for disk-bound runs lower ${MIN_PREFILL_TPS_ENV} or raise ${READ_TIMEOUT_ENV})"
            )
        except httpx.HTTPError as exc:
            return _failed_outcome(f"streaming request failed: {exc}")

        wall_ms = now_ms() - t_start
        text = "".join(out_text_parts)
        self._telemetry.append(tracker.summary())
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
            self._fetch_props(base_url)
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
        cmd += list(self.server_args)

        # Keep the server log (local only, never uploaded) so a load / OOM failure
        # is recorded with its reason instead of a bare exit code.
        log_path: Path | None = None
        log_fp = None
        try:
            log_dir = CACHE_DIR / "logs"
            log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            log_path = log_dir / "llama-server-last.log"
            log_fp = log_path.open("wb")
        except OSError as exc:
            LOG.debug("could not open llama-server log: %s", exc)
            log_path, log_fp = None, None

        LOG.info("starting llama-server: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fp or subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log_fp else subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        finally:
            if log_fp is not None:
                log_fp.close()
        try:
            self._wait_ready(base_url, proc)
        except Exception as exc:
            try:
                proc.terminate()
            except OSError:
                pass
            reason = _server_log_reason(log_path)
            if reason:
                raise RuntimeError(f"{exc}: {reason}") from exc
            raise

        self._server = _ServerHandle(
            proc=proc,
            port=port,
            base_url=base_url,
            model_path=model_path,
            log_path=log_path,
        )
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT_S)
        self._fetch_props(base_url)

    # ---------------------------------------------------------- labels/telemetry

    def _fetch_props(self, base_url: str) -> None:
        """`GET /props` carries `build_info` ("b11803-abc1234") for the server that
        is actually answering, which beats the local binary's --version when the
        server is external."""
        if self._props_fetched:
            return
        self._props_fetched = True
        try:
            r = httpx.get(f"{base_url}/props", timeout=5.0)
            if r.status_code == 200:
                props = r.json()
                info = props.get("build_info")
                if isinstance(info, str) and info:
                    self._props_build_info = info[:80]
                gs = props.get("default_generation_settings") or {}
                n_ctx = gs.get("n_ctx") or props.get("n_ctx")
                if isinstance(n_ctx, int) and n_ctx > 0:
                    self._props_n_ctx = n_ctx
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            LOG.debug("/props probe failed: %s", exc)

    def reset_run_telemetry(self) -> None:
        self._telemetry = []

    def pop_run_telemetry(self) -> list[dict[str, Any]]:
        out, self._telemetry = self._telemetry, []
        return out

    def sampling_supported(self, model: ModelRef | None = None) -> tuple[str, ...]:
        return SAMPLING_DRIVER_SUPPORT["llama.cpp"]

    def run_labels(self, model: ModelRef) -> dict[str, Any]:
        """Publishable labels for this run: exact build, passthrough args, GGUF file."""
        labels: dict[str, Any] = {}
        build = parse_llama_build(self._props_build_info)
        source = "server-props"
        if not build["commit"] and not build["build"]:
            build = parse_llama_build(self._version_text)
            source = "llama-server --version"
        if build["build"] or build["commit"]:
            labels["llama_cpp_build"] = build["build"]
            labels["llama_cpp_commit"] = build["commit"]
            labels["llama_cpp_build_source"] = source
        if self._props_n_ctx:
            labels["server_n_ctx"] = self._props_n_ctx
        args_label = label_server_args(self.server_args)
        if args_label:
            external = bool(os.environ.get("LLAMA_CPP_SERVER_URL", "").strip())
            labels["llama_server_args"] = (
                f"declared: {args_label}" if external else args_label
            )
        fname = model_file_label(model.identifier)
        if fname:
            labels["model_file"] = fname
            size = (model.extras or {}).get("size_bytes")
            if not isinstance(size, int):
                try:
                    size = Path(model.identifier).stat().st_size
                except OSError:
                    size = None
            if isinstance(size, int):
                labels["model_file_bytes"] = size
        return labels

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
        raise RuntimeError(
            f"llama-server did not become ready in time ({SERVER_BOOT_TIMEOUT_S:.0f}s; "
            "raise LLM_SPEED_LLAMA_BOOT_TIMEOUT_S for very large models)"
        )

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
