from __future__ import annotations

import json
import threading

import pytest

from daguandan_bridge.live.pipeline_timing import PipelineTiming


def test_timing_uses_bounded_cardinality_and_samples():
    timing = PipelineTiming()
    for index in range(1000):
        timing.observe("analysis", index)
        timing.observe(f"untrusted-stage-{index}", index)
        timing.increment(f"untrusted-counter-{index}")
    result = timing.snapshot()
    assert len(result["stages"]) == PipelineTiming.MAX_STAGES
    assert len(result["counters"]) == PipelineTiming.MAX_COUNTERS
    assert result["stages"]["analysis"]["count"] == 1000
    assert result["stages"]["analysis"]["recent_count"] == 128
    assert len(json.dumps(result)) < 20_000
    assert result["percentile_scope"] == "last_128_samples_per_stage_not_all_session"


def test_bad_duration_cannot_create_non_finite_json_or_negative_latency():
    timing = PipelineTiming()
    for value in (-1, float("inf"), float("nan")):
        timing.observe("bad", value)
    assert timing.snapshot()["counters"]["invalid_timing"] == 3
    assert not timing.snapshot()["stages"]
    json.dumps(timing.snapshot(), allow_nan=False)


def test_real_clock_elapsed_does_not_read_capture_timestamp(monkeypatch):
    timing = PipelineTiming()
    monkeypatch.setattr(timing, "now_ns", lambda: 2_500_000_000)
    timing.elapsed("analysis", 2_000_000_000)
    assert timing.snapshot()["stages"]["analysis"]["max_ms"] == 500


def test_concurrent_no_result_and_failure_counts_are_retained():
    timing = PipelineTiming()
    def run():
        for _ in range(1000):
            timing.increment("no_result")
            timing.observe("analysis", 2)
    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert timing.snapshot()["counters"]["no_result"] == 4000
    assert timing.snapshot()["stages"]["analysis"]["count"] == 4000


@pytest.mark.parametrize("operation_seconds, expected_wait", [(0.04, 0.06), (0.26, 0.04)])
def test_capture_absolute_schedule_compensates_work_without_backlog(
    monkeypatch, operation_seconds, expected_wait
):
    from daguandan_bridge.gui import workers
    now = [0.0]
    waits = []
    class StopEvent:
        def is_set(self):
            return False
        def wait(self, delay):
            waits.append(delay)
            return True
    def operation():
        now[0] += operation_seconds
        return None
    monkeypatch.setattr(workers.time, "monotonic", lambda: now[0])
    worker = workers.CaptureWorker(operation, .1)
    worker._stop_event = StopEvent()
    worker.run()
    assert waits == pytest.approx([expected_wait])


def test_capture_schedule_adapts_to_lobby_interval(monkeypatch):
    from daguandan_bridge.gui import workers
    now, waits = [0.0], []
    class StopEvent:
        def is_set(self):
            return False
        def wait(self, delay):
            waits.append(delay)
            return True
    def operation():
        now[0] += .04
        worker.interval_sec = 1.0
    monkeypatch.setattr(workers.time, "monotonic", lambda: now[0])
    worker = workers.CaptureWorker(operation, .2)
    worker._stop_event = StopEvent()
    worker.run()
    assert waits == pytest.approx([1.0])
