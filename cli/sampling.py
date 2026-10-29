"""Sampling settings the CLI sends with every benchmark request (0.0.10).

Before 0.0.10 the CLI sent no temperature or seed, so runs used whatever the server
defaulted to (Launch Lab had to pin ``--temp 0 --seed 42`` on llama-server itself).
Now ``llm-speed bench --temperature/--seed/--top-k/--top-p/--min-p`` (and a preset's
``sampling`` field) are validated here, sent by each driver in each request where the
backend supports the knob, and recorded in the run record as:

* ``extras.sampling``: the settings actually sent with each request (numbers only).
* ``extras.sampling_ignored``: requested keys this backend cannot take (key names only).

Both keys are absent when no sampling was requested, so a plain ``llm-speed bench``
sends and records exactly what 0.0.9 did. Pure and offline: no network here.
"""

from __future__ import annotations

import math
from typing import Any

# Order is the order they are printed and recorded in.
SAMPLING_KEYS: tuple[str, ...] = ("temperature", "seed", "top_k", "top_p", "min_p")

# Keys this module adds to WorkloadResult.extras (uploaded unless --no-upload).
UPLOADED_EXTRAS_KEYS: tuple[str, ...] = ("sampling", "sampling_ignored")

# key -> (python type, minimum, maximum). Same bounds as $defs.sampling in
# cli/schemas/llm-speed-preset-1.schema.json (a test keeps them in step).
BOUNDS: dict[str, tuple[type, float, float]] = {
    "temperature": (float, 0.0, 5.0),
    "seed": (int, 0, 4294967295),
    "top_k": (int, 0, 1000),
    "top_p": (float, 0.0, 1.0),
    "min_p": (float, 0.0, 1.0),
}

# What each driver can send per request. Unknown drivers get nothing (fail closed:
# a key is recorded as sent only when the driver really puts it on the wire).
DRIVER_SUPPORT: dict[str, tuple[str, ...]] = {
    # llama-server /v1/chat/completions accepts all five as request fields.
    "llama.cpp": SAMPLING_KEYS,
    # vLLM's OpenAI server takes top_k/min_p as extra sampling params; embedded
    # mode maps them onto SamplingParams.
    "vllm": SAMPLING_KEYS,
    # Ollama /api/chat "options".
    "ollama": SAMPLING_KEYS,
    # mlx_lm make_sampler(temp, top_p, min_p, top_k) + mx.random.seed.
    "mlx": SAMPLING_KEYS,
    # ExLlamaV2Sampler.Settings has no per-request seed.
    "exllamav2": ("temperature", "top_k", "top_p", "min_p"),
}
# hosted-api depends on the provider API (never uploaded: the Worker rejects it).
HOSTED_SUPPORT: dict[str, tuple[str, ...]] = {
    "openai": ("temperature", "seed", "top_p"),
    "anthropic": ("temperature", "top_k", "top_p"),
}


class SamplingError(ValueError):
    pass


def _coerce(key: str, value: Any) -> int | float:
    typ, lo, hi = BOUNDS[key]
    if isinstance(value, bool):
        raise SamplingError(f"{key}: expected a number, got {value!r}")
    if isinstance(value, str):
        try:
            value = int(value) if typ is int else float(value)
        except ValueError as exc:
            raise SamplingError(f"{key}: expected a number, got {value!r}") from exc
    if typ is int:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise SamplingError(f"{key}: expected a whole number, got {value!r}")
    else:
        if not isinstance(value, (int, float)):
            raise SamplingError(f"{key}: expected a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise SamplingError(f"{key}: must be finite")
        value = round(value, 6)
    if value < lo or value > hi:
        raise SamplingError(f"{key}: {value!r} is outside [{lo:g}, {hi:g}]")
    return value


def normalize(raw: dict[str, Any] | None) -> dict[str, int | float]:
    """Validate a sampling dict (preset field or parsed flags). Unknown keys and
    out-of-range values raise; None values are dropped."""
    out: dict[str, int | float] = {}
    for key, value in (raw or {}).items():
        if key not in BOUNDS:
            raise SamplingError(
                f"unknown sampling key {key!r} (allowed: {', '.join(SAMPLING_KEYS)})"
            )
        if value is None:
            continue
        out[key] = _coerce(key, value)
    return {k: out[k] for k in SAMPLING_KEYS if k in out}


def from_args(args: Any) -> dict[str, int | float]:
    """Sampling requested on the bench command line (--temperature, --seed, ...)."""
    return normalize({k: getattr(args, k, None) for k in SAMPLING_KEYS})


def merge(*layers: dict[str, Any] | None) -> dict[str, int | float]:
    """Later layers win key by key (preset-wide, then per-model, then flags)."""
    out: dict[str, Any] = {}
    for layer in layers:
        out.update({k: v for k, v in (layer or {}).items() if v is not None})
    return normalize(out)


def split(
    requested: dict[str, Any] | None, supported: tuple[str, ...] | list[str]
) -> tuple[dict[str, int | float], list[str]]:
    """(sent, ignored): what a driver puts on the wire and what it cannot."""
    req = requested or {}
    sent = {k: req[k] for k in SAMPLING_KEYS if k in req and k in supported}
    ignored = [k for k in SAMPLING_KEYS if k in req and k not in supported]
    return sent, ignored


def supported_for(driver: Any, model: Any = None) -> tuple[str, ...]:
    fn = getattr(driver, "sampling_supported", None)
    if callable(fn):
        try:
            return tuple(fn(model))
        except Exception:  # noqa: BLE001
            return ()
    return DRIVER_SUPPORT.get(getattr(driver, "name", ""), ())


def record(driver: Any, model: Any, requested: dict[str, Any] | None) -> dict[str, Any]:
    """Extras for the run record. Empty when nothing was requested."""
    if not requested:
        return {}
    sent, ignored = split(requested, supported_for(driver, model))
    out: dict[str, Any] = {}
    if sent:
        out["sampling"] = sent
    if ignored:
        out["sampling_ignored"] = ignored
    return out


def to_cli_flags(sampling: dict[str, Any] | None) -> list[str]:
    """Bench flags for a child `llm-speed bench` (preset runs)."""
    out: list[str] = []
    for key in SAMPLING_KEYS:
        if sampling and key in sampling:
            out += [f"--{key.replace('_', '-')}", repr(sampling[key])]
    return out


def describe(sampling: dict[str, Any] | None) -> str:
    if not sampling:
        return "backend defaults"
    return ", ".join(f"{k}={sampling[k]}" for k in SAMPLING_KEYS if k in sampling)
