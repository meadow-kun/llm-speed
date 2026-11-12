"""Shared timing & percentile helpers. Drivers and workloads use these to
compute consistent metrics from per-token latency arrays."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple


@contextmanager
def stopwatch() -> Iterator[list[float]]:
    """Yields a single-element list whose [0] is the elapsed milliseconds at exit."""
    box: list[float] = [0.0]
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box[0] = (time.perf_counter() - t0) * 1000.0


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    k = (len(sorted_v) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(sorted_v) - 1)
    if f == c:
        return sorted_v[f]
    return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)


def tps_from_decode_times(decode_token_times_ms: list[float]) -> float | None:
    """Decode tok/s computed from per-token wall-clock deltas (excludes prefill)."""
    if not decode_token_times_ms:
        return None
    total_s = sum(decode_token_times_ms) / 1000.0
    if total_s <= 0:
        return None
    return len(decode_token_times_ms) / total_s


# llama.cpp fecdc81 (build 12157), tools/server/server-common.h:454-455.
# t_gen_us() clamps t_gen_last - t_prompt_last to 1 microsecond, so a clock
# that has not moved is reported as 0.001 ms. Dividing by that is not a rate
# (one token becomes about 1e6 tok/s).
_PREDICTED_MS_FLOOR = 0.001

DECODE_TPS_SERVER_TIMINGS = "server_timings"
DECODE_TPS_SERVER_N_MS = "server_predicted_n_ms"
DECODE_TPS_STREAM_CHUNKS = "stream_chunks"
_SERVER_DECODE_SOURCES = frozenset({DECODE_TPS_SERVER_TIMINGS, DECODE_TPS_SERVER_N_MS})


class ServerDecodeRate(NamedTuple):
    """A llama-server decode rate and which timings field produced it."""

    tps: float
    source: str


def _timing_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def server_decode_tps(timings: object) -> ServerDecodeRate | None:
    """llama-server decode tok/s, and which timings field it came from.

    llama.cpp fecdc81 (build 12157) does not put the first generated token
    inside `predicted_ms`. `t_prompt_last` is taken when `n_gen` becomes 1
    (`tools/server/server-context.cpp:4902-4910`), then `update_gen_last`
    runs. `t_gen_us()` is `t_gen_last - t_prompt_last`, clamped to 1µs
    (`tools/server/server-common.h:450-458`). The comment at
    `server-common.h:461-465` says that token is free (it comes from the
    last prompt batch) and `n_gen_steps()` is `n_gen - 1`.
    `predicted_per_second` is `n_gen_tps()` (`server-common.cpp:150-153`):
    `(n - 1) / (predicted_ms / 1000)` when that duration is above 0, else 0.

    Prefer a positive `predicted_per_second` (`server_timings`). When it is
    0 or missing, use `(n - 1) / (ms / 1000)` only for `n >= 2` and `ms`
    above the 1µs floor (`server_predicted_n_ms`). One token, a missing
    timings object, or a duration at that floor returns None.
    """
    if not isinstance(timings, dict):
        return None
    n = _timing_number(timings.get("predicted_n"))
    ms = _timing_number(timings.get("predicted_ms"))
    per = _timing_number(timings.get("predicted_per_second"))
    # n == 1 has no decode step (n_gen_steps is 0). A positive
    # predicted_per_second beside it is not an inter-token rate.
    if n is not None and n < 2:
        return None
    if ms is not None and ms <= _PREDICTED_MS_FLOOR:
        return None
    if per is not None and per > 0:
        return ServerDecodeRate(per, DECODE_TPS_SERVER_TIMINGS)
    if n is not None and n >= 2 and ms is not None and ms > _PREDICTED_MS_FLOOR:
        return ServerDecodeRate((n - 1.0) / (ms / 1000.0), DECODE_TPS_SERVER_N_MS)
    return None


def select_decode_tps(
    chunk_tps: float | None,
    backend_extras: dict | None,
) -> tuple[float | None, str | None]:
    """The decode rate a workload should publish, and `decode_tps_source`.

    A llama.cpp server rate wins when the driver labeled it. Otherwise the
    chunk-gap rate is `stream_chunks`. No rate leaves both values unset.
    """
    extras = backend_extras or {}
    raw = extras.get("decode_tps")
    source = extras.get("decode_tps_source")
    if (
        isinstance(raw, (int, float))
        and not isinstance(raw, bool)
        and raw > 0
        and source in _SERVER_DECODE_SOURCES
    ):
        return float(raw), str(source)
    if (
        isinstance(chunk_tps, (int, float))
        and not isinstance(chunk_tps, bool)
        and chunk_tps > 0
    ):
        return float(chunk_tps), DECODE_TPS_STREAM_CHUNKS
    return None, None


def combine_decode_tps_sources(sources: list[str]) -> str | None:
    """One point's source. A batch that mixed methods lists each one."""
    order = (
        DECODE_TPS_SERVER_TIMINGS,
        DECODE_TPS_SERVER_N_MS,
        DECODE_TPS_STREAM_CHUNKS,
    )
    present = [name for name in order if name in sources]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return "+".join(present)


def summarize_decode(decode_token_times_ms: list[float]) -> dict:
    """Return {decode_tps, p50, p95, mean} from per-token deltas."""
    if not decode_token_times_ms:
        return {"decode_tps": None, "p50": None, "p95": None, "mean_ms": None}
    return {
        "decode_tps": tps_from_decode_times(decode_token_times_ms),
        "p50": percentile(decode_token_times_ms, 0.50),
        "p95": percentile(decode_token_times_ms, 0.95),
        "mean_ms": statistics.fmean(decode_token_times_ms),
    }
