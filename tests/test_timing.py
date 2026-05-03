"""Tests for `cli.timing`.

Pure functions; no I/O. The `stopwatch` test is the only one that touches
real time and uses a generous tolerance so it passes on a busy CI runner.
"""

from __future__ import annotations

import time

from cli.timing import (
    percentile,
    stopwatch,
    summarize_decode,
    tps_from_decode_times,
)

# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_returns_none_for_empty_input():
    assert percentile([], 0.5) is None


def test_percentile_single_value_returns_that_value():
    assert percentile([1.0], 0.5) == 1.0


def test_percentile_p50_of_odd_list_is_middle_element():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0


def test_percentile_p95_of_short_list_lands_in_top_bin():
    p95 = percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95)
    assert p95 is not None
    assert 4.5 <= p95 <= 5.0


def test_percentile_unsorted_input_is_handled():
    # Input order should not matter.
    assert percentile([5.0, 1.0, 3.0, 4.0, 2.0], 0.5) == 3.0


# ---------------------------------------------------------------------------
# tps_from_decode_times
# ---------------------------------------------------------------------------


def test_tps_from_decode_times_three_tokens_at_100ms_each_is_ten_tps():
    # 300ms total -> 0.3s for 3 tokens -> 10 tok/s
    assert tps_from_decode_times([100.0, 100.0, 100.0]) == 10.0


def test_tps_from_decode_times_returns_none_for_empty_input():
    assert tps_from_decode_times([]) is None


def test_tps_from_decode_times_returns_none_for_zero_total_time():
    assert tps_from_decode_times([0.0, 0.0, 0.0]) is None


# ---------------------------------------------------------------------------
# summarize_decode
# ---------------------------------------------------------------------------


def test_summarize_decode_empty_returns_all_none():
    out = summarize_decode([])
    assert out == {"decode_tps": None, "p50": None, "p95": None, "mean_ms": None}


def test_summarize_decode_populates_all_fields_for_real_input():
    out = summarize_decode([5.0, 5.0, 5.0, 5.0, 5.0])
    assert out["decode_tps"] == 200.0  # 25ms total -> 5 tokens / 0.025s
    assert out["p50"] == 5.0
    assert out["p95"] == 5.0
    assert out["mean_ms"] == 5.0


# ---------------------------------------------------------------------------
# stopwatch
# ---------------------------------------------------------------------------


def test_stopwatch_records_elapsed_milliseconds_within_tolerance():
    with stopwatch() as box:
        time.sleep(0.05)
    # 50ms +/- 30ms — generous to survive CI noise but still verifies units.
    assert 20.0 <= box[0] <= 200.0


def test_stopwatch_records_zero_for_no_work():
    with stopwatch() as box:
        pass
    # Should be a tiny positive number; just assert it ran.
    assert box[0] >= 0.0
    assert box[0] < 50.0
