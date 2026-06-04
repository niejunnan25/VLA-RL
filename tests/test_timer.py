from __future__ import annotations

import time

from vla_rl.runtime.timer import Timer


def test_timer_context_records_average_and_resets() -> None:
    timer = Timer()
    with timer.context("work"):
        time.sleep(0.001)

    metrics = timer.get_average_times(prefix="time/", suffix="_sec")

    assert metrics["time/work_sec"] > 0.0
    assert timer.get_average_times() == {}


def test_timer_total_times_can_be_read_without_reset() -> None:
    timer = Timer()
    with timer.context("a"):
        time.sleep(0.001)
    first = timer.get_total_times(reset=False)
    second = timer.get_total_times(reset=False)

    assert first["a"] > 0.0
    assert second == first
