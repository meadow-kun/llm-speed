"""Launch telemetry added in 0.0.8: failure classification, reasoning vs answer
timing, peak accelerator memory, llama.cpp build labels and passthrough-arg labels.

Everything here is pure (no network) and small so it can be unit-tested without a
GPU. The values it produces ride ``WorkloadResult.extras`` (the API stores extras as
opaque JSON capped at 4 KB per row), so no schema migration is needed. Every key it
adds to an uploaded payload is listed in ``UPLOADED_EXTRAS_KEYS`` so privacy review
has one place to look.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import threading
from pathlib import PurePath
from typing import Any

# Keys this module adds to WorkloadResult.extras (uploaded unless --no-upload).
UPLOADED_EXTRAS_KEYS = (
    "failure_kind",
    "peak_vram_mib",
    "peak_vram_source",
    "time_to_first_answer_ms",
    "reasoning_tokens_est",
    "answer_started",
    "llama_cpp_build",
    "llama_cpp_commit",
    "llama_cpp_build_source",
    "llama_server_args",
    "server_n_ctx",
    "model_file",
    "model_file_bytes",
    # 0.0.10 (cli/sampling.py): settings sent with each request, and requested
    # keys the backend could not take. Numbers and fixed key names only.
    "sampling",
    "sampling_ignored",
)

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

FAILURE_OOM = "oom"
FAILURE_UNSUPPORTED_ARCH = "unsupported_arch"
FAILURE_LOAD = "load_failed"
FAILURE_TIMEOUT = "load_timeout"
FAILURE_SERVER_UNREACHABLE = "server_unreachable"
FAILURE_SKIPPED = "skipped_after_load_failure"
FAILURE_RUNTIME = "runtime_error"

_OOM_PATTERNS = re.compile(
    r"out of memory|\bOOM\b|cudaMalloc failed|failed to allocate|"
    r"unable to allocate|alloc_buffer|insufficient memory|"
    r"kIOGPUCommandBufferCallbackErrorOutOfMemory|"
    r"\[metal::malloc\]|greater than the maximum allowed buffer size|"
    r"std::bad_alloc|CUBLAS_STATUS_ALLOC_FAILED",
    re.IGNORECASE,
)
_ARCH_PATTERNS = re.compile(
    r"unknown model architecture|model type \S+ not supported|"
    r"unsupported model (?:type|architecture)|no model named|"
    r"cannot load model_type",
    re.IGNORECASE,
)
_LOAD_PATTERNS = re.compile(
    r"failed to start llama-server|exited early|failed to load model|"
    r"error loading model|mlx_lm\.load failed|model not found|"
    r"failed to read|invalid magic|gguf",
    re.IGNORECASE,
)
_TIMEOUT_PATTERNS = re.compile(
    r"did not become ready in time|load timed out", re.IGNORECASE
)
_UNREACHABLE_PATTERNS = re.compile(
    r"ConnectError|Connection refused|connection reset|RemoteProtocolError|"
    r"server disconnected",
    re.IGNORECASE,
)


def classify_failure(error: str | None) -> str | None:
    """Map a free-text workload error onto a small enum. None when no error."""
    if not error:
        return None
    if _OOM_PATTERNS.search(error):
        return FAILURE_OOM
    if _ARCH_PATTERNS.search(error):
        return FAILURE_UNSUPPORTED_ARCH
    if _TIMEOUT_PATTERNS.search(error):
        return FAILURE_TIMEOUT
    if _LOAD_PATTERNS.search(error):
        return FAILURE_LOAD
    if _UNREACHABLE_PATTERNS.search(error):
        return FAILURE_SERVER_UNREACHABLE
    return FAILURE_RUNTIME


def is_load_failure(kind: str | None) -> bool:
    """Failures after which every later workload would fail the same way."""
    return kind in (
        FAILURE_OOM,
        FAILURE_UNSUPPORTED_ARCH,
        FAILURE_LOAD,
        FAILURE_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Reasoning vs answer split
# ---------------------------------------------------------------------------


class ReasoningTracker:
    """Track when the first *answer* (non-reasoning) text arrives in a stream.

    Handles both shapes reasoning models use:
      - separate ``reasoning_content`` deltas (llama-server --reasoning-format auto)
      - inline ``<think>...</think>`` inside ``content`` (reasoning-format none, mlx-lm)
    ``ttft_ms`` in results stays "first token of any kind" for comparability with
    older runs; ``time_to_first_answer_ms`` is the new, user-facing number.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, t_start_ms: float) -> None:
        self.t_start_ms = t_start_ms
        self.first_answer_ms: float | None = None
        self.reasoning_chunks = 0
        self.answer_chunks = 0
        self._in_think = False
        self._seen_any_content = False
        self._buf = ""

    def feed(self, now_ms: float, *, content: str = "", reasoning: str = "") -> None:
        if reasoning:
            self.reasoning_chunks += 1
        if not content:
            return
        text = content
        if not self._seen_any_content:
            stripped = text.lstrip()
            if stripped.startswith(self.OPEN) or (
                stripped and self.OPEN.startswith(stripped)
            ):
                self._in_think = True
            self._seen_any_content = bool(stripped)
        if self._in_think:
            self._buf = (self._buf + text)[-64:]
            self.reasoning_chunks += 1
            if self.CLOSE in self._buf:
                self._in_think = False
                tail = self._buf.split(self.CLOSE, 1)[1]
                self._buf = ""
                if tail.strip():
                    self._mark_answer(now_ms)
            return
        if text.strip():
            self._mark_answer(now_ms)

    def _mark_answer(self, now_ms: float) -> None:
        self.answer_chunks += 1
        if self.first_answer_ms is None:
            self.first_answer_ms = now_ms - self.t_start_ms

    def summary(self) -> dict[str, Any]:
        return {
            "time_to_first_answer_ms": (
                round(self.first_answer_ms, 1)
                if self.first_answer_ms is not None
                else None
            ),
            "reasoning_tokens_est": self.reasoning_chunks,
            "answer_started": self.first_answer_ms is not None,
        }


# ---------------------------------------------------------------------------
# llama.cpp build label
# ---------------------------------------------------------------------------

_VERSION_LINE_RE = re.compile(r"version:\s*(\d+)\s*\(([0-9a-f]{6,40})\)", re.IGNORECASE)
# llama.cpp 0.7+ (b11804 and later): "version: 0.7.0-dev (build 11804, commit c7fc50941)"
_VERSION_BUILD_RE = re.compile(
    r"\(build\s+(\d+),\s*commit\s+([0-9a-f]{6,40})\)", re.IGNORECASE
)
_BUILD_INFO_RE = re.compile(r"\bb(\d+)-([0-9a-f]{6,40})\b", re.IGNORECASE)
_TAG_RE = re.compile(r"\bb(\d{3,})\b")
_SHALLOW_BUILD_MAX = 100  # `git clone --depth 1` builds report build number 1


def parse_llama_build(text: str | None) -> dict[str, str | None]:
    """Return {'build': 'b11803'|None, 'commit': 'abc1234'|None}.

    Accepts ``llama-server --version`` output ("version: 11803 (abc1234)", or from
    0.7 on "version: 0.7.0-dev (build 11804, commit abc1234)"),
    ``/props`` ``build_info`` ("b11803-abc1234") or a bare tag ("b11803").
    A build number below 100 comes from a shallow clone (git rev-list count is
    meaningless there), so only the commit is trusted.
    """
    out: dict[str, str | None] = {"build": None, "commit": None}
    if not text:
        return out
    number: str | None = None
    m = (
        _VERSION_LINE_RE.search(text)
        or _VERSION_BUILD_RE.search(text)
        or _BUILD_INFO_RE.search(text)
    )
    if m:
        number, out["commit"] = m.group(1), m.group(2).lower()[:12]
    else:
        t = _TAG_RE.search(text)
        if t:
            number = t.group(1)
    if number is not None and int(number) >= _SHALLOW_BUILD_MAX:
        out["build"] = f"b{int(number)}"
    return out


# ---------------------------------------------------------------------------
# llama-server passthrough args
# ---------------------------------------------------------------------------

# Flags whose values are safe and useful to publish (no paths, no secrets).
_ARG_ALLOWLIST = {
    "-c",
    "--ctx-size",
    "-ngl",
    "--n-gpu-layers",
    "--gpu-layers",
    "--n-cpu-moe",
    "-ncmoe",
    "--cpu-moe",
    "-cmoe",
    "-fa",
    "--flash-attn",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "-ctk",
    "--cache-type-k",
    "-ctv",
    "--cache-type-v",
    "-ot",
    "--override-tensor",
    "-t",
    "--threads",
    "-tb",
    "--threads-batch",
    "-np",
    "--parallel",
    "-sm",
    "--split-mode",
    "-ts",
    "--tensor-split",
    "-mg",
    "--main-gpu",
    "--no-mmap",
    "--mlock",
    "--jinja",
    "--no-kv-offload",
    "-nkvo",
    "--reasoning-format",
    "--reasoning-budget",
    "--spec-type",
    "--draft-max",
    "--draft-min",
    "-ngld",
    "--n-gpu-layers-draft",
    "--swa-full",
    "--no-warmup",
    "--cont-batching",
    "--no-cont-batching",
    "--fit",
    "--rope-scaling",
    "--rope-freq-base",
    "--yarn-orig-ctx",
}

# 0.0.10: flags whose values are checked strictly. A flag listed here is kept only
# with a value of its declared kind (anything else becomes ``<redacted>``), and a
# value-less flag never swallows the next token. Fixed flag names and
# numeric / enum values only: no paths, no free text. Names and value types follow
# llama.cpp common/arg.cpp (checked against 9725a313b, the 2026-10-22 5090 rerun).
_INT = "int"
_NUM = "num"
_NONE = "none"
# on/off/auto plus the numeric spellings llama.cpp also takes; nothing else (PRIVACY.md
# promises numbers or on/off/auto only).
_ON_OFF_AUTO = frozenset({"on", "off", "auto", "1", "0", "-1"})
# -lm/--load-mode and -lzm/--lazy-mode: their fixed words only (no numeric spellings).
_LOAD_MODE = frozenset({"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"})
_ON_OFF_AUTO_WORDS = frozenset({"on", "auto", "off"})
_STRICT_ARGS: dict[str, object] = {
    # prompt cache (llama-server; both off for comparable timings)
    "--no-cache-prompt": _NONE,
    "--cache-prompt": _NONE,
    "-cram": _INT,  # MiB, -1 = no limit, 0 = disable
    "--cache-ram": _INT,
    # sampling (server-side defaults; the CLI now also sends them per request)
    "--temp": _NUM,
    "--temperature": _NUM,
    "-s": _INT,  # seed
    "--seed": _INT,
    "--top-k": _INT,
    "--top-p": _NUM,
    "--min-p": _NUM,
    # thinking on/off (llama.cpp "-rea, --reasoning [on|off|auto]")
    "-rea": _ON_OFF_AUTO,
    "--reasoning": _ON_OFF_AUTO,
    # allowed by the preset schema's server_args pattern
    "--kv-unified": _NONE,
    # next release (Boss, 2026-10-31): weight load mode and lazy loading, fixed
    # values only (DR-005; disk-bound offload runs need them on the record)
    "-lm": _LOAD_MODE,
    "--load-mode": _LOAD_MODE,
    "-lzm": _ON_OFF_AUTO_WORDS,
    "--lazy-mode": _ON_OFF_AUTO_WORDS,
    # value-less flags that were already allowlisted: never consume the next token
    "--no-mmap": _NONE,
    "--mlock": _NONE,
    "--jinja": _NONE,
    "--no-kv-offload": _NONE,
    "-nkvo": _NONE,
    "--cpu-moe": _NONE,
    "-cmoe": _NONE,
    "--swa-full": _NONE,
    "--no-warmup": _NONE,
    "--cont-batching": _NONE,
    "--no-cont-batching": _NONE,
}
_ARG_ALLOWLIST |= set(_STRICT_ARGS)
_INT_RE = re.compile(r"-?[0-9]{1,10}")
_NUM_RE = re.compile(r"-?[0-9]{1,7}(\.[0-9]{1,6})?")
_MAX_ARGS_LABEL = 200


def parse_server_args(raw: str | list[str] | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            out += shlex.split(item)
        return out
    return shlex.split(raw)


def label_server_args(args: list[str]) -> str | None:
    """Publishable label for llama-server args: allowlisted flags keep their values,
    anything else (paths, api keys, hostnames) becomes ``<redacted>``."""
    if not args:
        return None
    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        flag, eq, val = tok.partition("=")
        if flag.startswith("-") and flag in _STRICT_ARGS:
            kind = _STRICT_ARGS[flag]
            if kind == _NONE:
                out.append(f"{flag}=<redacted>" if eq else flag)
            elif eq:
                out.append(f"{flag}={_strict_value(val, kind)}")
            else:
                out.append(flag)
                if i + 1 < len(args) and not _looks_like_flag(args[i + 1]):
                    out.append(_strict_value(args[i + 1], kind))
                    i += 1
        elif flag.startswith("-") and flag in _ARG_ALLOWLIST:
            if eq:
                out.append(f"{flag}={_safe_value(val)}")
            else:
                out.append(flag)
                if i + 1 < len(args) and not _looks_like_flag(args[i + 1]):
                    out.append(_safe_value(args[i + 1]))
                    i += 1
        elif flag.startswith("-"):
            out.append("<redacted-flag>")
            if not eq and i + 1 < len(args) and not _looks_like_flag(args[i + 1]):
                i += 1
        else:
            out.append("<redacted>")
        i += 1
    label = " ".join(out)
    if len(label) > _MAX_ARGS_LABEL:
        label = label[: _MAX_ARGS_LABEL - 1] + "…"
    return label


def _looks_like_flag(tok: str) -> bool:
    return tok.startswith("-") and not re.fullmatch(r"-\d+(\.\d+)?", tok)


def _strict_value(val: str, kind: object) -> str:
    if kind == _INT and _INT_RE.fullmatch(val):
        return val
    if kind == _NUM and _NUM_RE.fullmatch(val):
        return val
    if isinstance(kind, frozenset) and val in kind:
        return val
    return "<redacted>"


def _safe_value(val: str) -> str:
    # Override-tensor patterns like "blk\.(1[0-9])\.ffn_.*_exps=CPU" are fine;
    # anything path-shaped is not.
    if "/" in val or re.match(r"^[A-Za-z]:\\", val):
        return "<redacted>"
    if re.fullmatch(r"[A-Za-z0-9_.,:=+*|()\[\]\\^$?{}-]{1,80}", val):
        return val
    return "<redacted>"


def model_file_label(identifier: str | None) -> str | None:
    """Basename of a model file (the CLI already uploads it as model.identifier)."""
    if not identifier:
        return None
    try:
        base = PurePath(identifier).name
    except (TypeError, ValueError):
        return None
    if not base.lower().endswith(".gguf"):
        return None
    return base[:200]


# ---------------------------------------------------------------------------
# Peak VRAM (NVIDIA via nvidia-smi sampling)
# ---------------------------------------------------------------------------


def _query_nvidia_used_mib() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    total = 0
    seen = False
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += int(float(line))
            seen = True
        except ValueError:
            continue
    return total if seen else None


class NvidiaVramSampler:
    """Background sampler of device-wide used VRAM (sum over GPUs), in MiB.

    Device-wide means it includes every process on the GPU (that's what
    ``nvidia-smi`` reports, and it's the only view that works when the CLI talks
    to an external llama-server). Source label says so.
    """

    SOURCE = "nvidia-smi-device-used"

    def __init__(self, interval_s: float = 0.5, query=_query_nvidia_used_mib) -> None:
        self.interval_s = interval_s
        self._query = query
        self._peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def available() -> bool:
        return shutil.which("nvidia-smi") is not None

    def _sample(self) -> None:
        v = self._query()
        if v is not None and (self._peak is None or v > self._peak):
            self._peak = v

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def start(self) -> NvidiaVramSampler:
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._sample()
        return self._peak


def round_mib(v: float | int | None) -> int | None:
    """Round to 16 MiB: enough precision for 'does it fit', less host detail."""
    if v is None:
        return None
    return int(round(float(v) / 16.0)) * 16
