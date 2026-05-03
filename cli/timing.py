"""Shared timing & percentile helpers. Drivers and workloads use these to
compute consistent metrics from per-token latency arrays."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager


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
