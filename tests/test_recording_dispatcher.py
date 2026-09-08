from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import numpy as np
import pytest

from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.gui.recording_dispatcher import (
    BoundedRecordingDispatcher,
    RecordingDispatchStats,
    RecordingDrop,
    RecordingDropAccountingAdapter,
    RecordingFrame,
)


def _frame(sequence: int, image=None) -> RecordingFrame:
    return RecordingFrame(
        image=np.full((2, 3, 3), sequence, np.uint8) if image is None else image,
        captured_ms=sequence * 100,
        wall_time=f"frame-{sequence}",
        capture_sequence=sequence,
    )


def test_full_fifo_evicts_oldest_pending_and_keeps_the_latest_tail() -> None:
    entered, release = Event(), Event()
    seen: list[RecordingFrame] = []
    dropped: list[tuple[int, str]] = []

    def write(frame: RecordingFrame):
        seen.append(frame)
        if frame.capture_sequence == 1:
            entered.set()
            assert release.wait(1)

    dispatcher = BoundedRecordingDispatcher(
        write,
        capacity=2,
        on_drop=lambda frame, reason: dropped.append((frame.capture_sequence, reason)),
    )
    dispatcher.start()
    assert dispatcher.submit(_frame(1))
    assert entered.wait(1)
    assert dispatcher.submit(_frame(2))
    assert dispatcher.submit(_frame(3))
    assert not dispatcher.submit(_frame(4))
    release.set()
    assert dispatcher.wait_idle(1)
    stats = dispatcher.close()

    assert [frame.capture_sequence for frame in seen] == [1, 3, 4]
    assert [(frame.captured_ms, frame.wall_time) for frame in seen] == [
        (100, "frame-1"), (300, "frame-3"), (400, "frame-4")
    ]
    assert dropped == [(2, "capacity")]
    assert [drop.to_dict() for drop in stats.drops] == [{
        "capture_sequence": 2, "captured_ms": 200, "reason": "capacity"
    }]
    assert stats.submitted == 4 and stats.written == 3
    assert stats.dropped_capacity == 1 and not stats.running


def test_submit_owns_an_immutable_ndarray_before_producer_mutation() -> None:
    entered, release, completed = Event(), Event(), Event()
    observed: list[np.ndarray] = []

    def write(frame: RecordingFrame):
        if frame.capture_sequence == 1:
            entered.set()
            assert release.wait(1)
        else:
            observed.append(frame.image)
            completed.set()

    dispatcher = BoundedRecordingDispatcher(write, capacity=2)
    dispatcher.start()
    assert dispatcher.submit(_frame(1))
    assert entered.wait(1)
    producer_owned = np.zeros((2, 3, 3), np.uint8)
    assert dispatcher.submit(_frame(2, producer_owned))
    producer_owned[:] = 255
    release.set()
    assert completed.wait(1)
    assert dispatcher.wait_idle(1)
    dispatcher.close()

    assert np.count_nonzero(observed[0]) == 0
    assert observed[0].flags.writeable is False


def test_recording_failure_is_audited_and_later_frames_continue() -> None:
    completed = Event()
    errors: list[tuple[int, str]] = []

    def write(frame: RecordingFrame):
        if frame.capture_sequence == 1:
            raise OSError("disk unavailable")
        completed.set()

    dispatcher = BoundedRecordingDispatcher(
        write,
        on_error=lambda frame, error: errors.append((frame.capture_sequence, str(error))),
    )
    dispatcher.start()
    assert dispatcher.submit(_frame(1))
    assert dispatcher.submit(_frame(2))
    assert completed.wait(1)
    assert dispatcher.wait_idle(1)
    stats = dispatcher.close()

    assert errors == [(1, "disk unavailable")]
    assert stats.failed == 1 and stats.written == 1
    assert [(drop.capture_sequence, drop.reason) for drop in stats.drops] == [
        (1, "write_failed")
    ]


def test_close_discards_pending_frames_and_joins_a_bounded_writer() -> None:
    entered, release, pending_discarded = Event(), Event(), Event()
    dropped: list[int] = []

    def write(frame: RecordingFrame):
        entered.set()
        assert release.wait(1)

    def on_drop(frame: RecordingFrame, reason: str):
        assert reason == "close"
        dropped.append(frame.capture_sequence)
        if len(dropped) == 2:
            pending_discarded.set()

    dispatcher = BoundedRecordingDispatcher(write, capacity=4, on_drop=on_drop)
    dispatcher.start()
    for sequence in (1, 2, 3):
        assert dispatcher.submit(_frame(sequence))
    assert entered.wait(1)
    results = []
    closing = Thread(
        target=lambda: results.append(dispatcher.close(drain_timeout=0, stop_timeout=1))
    )
    closing.start()
    assert pending_discarded.wait(1)
    release.set()
    closing.join(1)

    assert not closing.is_alive()
    assert dropped == [2, 3]
    assert results[0].written == 1 and results[0].dropped_close == 2
    assert not results[0].running and not results[0].stop_timed_out


def test_permanently_blocked_writer_is_daemon_and_reports_stop_timeout() -> None:
    entered, release = Event(), Event()

    def write(frame: RecordingFrame):
        entered.set()
        release.wait()

    dispatcher = BoundedRecordingDispatcher(write)
    dispatcher.start()
    assert dispatcher.submit(_frame(9))
    assert entered.wait(1)
    assert dispatcher.is_daemon
    timed_out = dispatcher.close(drain_timeout=0, stop_timeout=0)

    assert timed_out.running and timed_out.stop_timed_out
    assert [(drop.capture_sequence, drop.reason) for drop in timed_out.drops] == [
        (9, "stop_timeout")
    ]
    release.set()
    assert dispatcher.wait_idle(1)
    assert not dispatcher.close(drain_timeout=0, stop_timeout=1).running


class _Recorder:
    session_directory = Path("session")
    video_path = Path("session/video/game.avi")
    index_path = Path("session/video/frame_index.jsonl")
    frame_count = 5
    dropped_frames = 2

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        return RecordingResult(
            self.video_path, self.index_path, self.frame_count, self.dropped_frames,
            integrity={"status": "PASS", "issues": []},
        )


def _stats(*, timeout: bool = False) -> RecordingDispatchStats:
    drops = (
        RecordingDrop(2, 200, "capacity"),
        RecordingDrop(3, 300, "close"),
    )
    if timeout:
        drops += (RecordingDrop(4, 400, "stop_timeout"),)
    return RecordingDispatchStats(
        submitted=5, written=2, failed=0, dropped_capacity=1,
        dropped_close=1, pending=0, inflight=int(timeout), running=timeout,
        stop_timed_out=timeout, drops=drops,
    )


def test_accounting_adapter_adds_dispatcher_drops_to_sealed_result() -> None:
    recorder = _Recorder()
    adapter = RecordingDropAccountingAdapter(recorder)
    adapter.accept_dispatcher_stats(_stats())
    adapter.accept_recording_cadence({
        "schema": "guandan.recording-cadence/1", "target_fps": 10,
        "observed": 40, "admitted": 10, "sampled_out": 30,
    })

    result = adapter.close()

    assert recorder.close_calls == 1
    assert result.dropped_frames == 4
    assert result.integrity["status"] == "PARTIAL"
    assert result.integrity["recording_dispatcher"]["drops"] == [
        {"capture_sequence": 2, "captured_ms": 200, "reason": "capacity"},
        {"capture_sequence": 3, "captured_ms": 300, "reason": "close"},
    ]
    accounting = result.integrity["recording_dispatcher"]
    assert accounting["recorder_dropped_frames"] == 2
    assert accounting["dispatcher_dropped_frames"] == 2
    assert accounting["total_dropped_frames"] == 4
    assert result.integrity["recording_cadence"]["sampled_out"] == 30


def test_timeout_adapter_avoids_unsafe_close_and_forces_health_failure() -> None:
    recorder = _Recorder()
    adapter = RecordingDropAccountingAdapter(recorder)
    adapter.accept_dispatcher_stats(_stats(timeout=True))

    with pytest.raises(RuntimeError, match="inflight writer"):
        adapter.close()
    assert recorder.close_calls == 0
