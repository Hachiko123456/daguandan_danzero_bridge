"""Bounded FIFO sidecar for recording captured frames off the capture loop."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Condition, Thread, current_thread
from time import monotonic
from typing import Any

import numpy as np

from ..domain.recording import RecordingResult


@dataclass(frozen=True, slots=True)
class RecordingFrame:
    image: Any
    captured_ms: int
    wall_time: str
    capture_sequence: int
    context: object | None = None


@dataclass(frozen=True, slots=True)
class RecordingDispatchStats:
    submitted: int
    written: int
    failed: int
    dropped_capacity: int
    dropped_close: int
    pending: int
    inflight: int
    running: bool
    stop_timed_out: bool
    drops: tuple["RecordingDrop", ...]

    def to_dict(self) -> dict[str, object]:
        return _audit(self)


@dataclass(frozen=True, slots=True)
class RecordingDrop:
    capture_sequence: int
    captured_ms: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "capture_sequence": self.capture_sequence,
            "captured_ms": self.captured_ms,
            "reason": self.reason,
        }


class BoundedRecordingDispatcher:
    """Write admitted frames in order without blocking capture or analysis."""

    def __init__(
        self,
        operation: Callable[[RecordingFrame], object | None],
        *,
        capacity: int = 16,
        on_warning: Callable[[RecordingFrame, object], None] | None = None,
        on_error: Callable[[RecordingFrame, Exception], None] | None = None,
        on_drop: Callable[[RecordingFrame, str], None] | None = None,
        clone_image: Callable[[Any], Any] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("recording FIFO capacity must be positive")
        self._operation = operation
        self._capacity = int(capacity)
        self._on_warning = on_warning
        self._on_error = on_error
        self._on_drop = on_drop
        self._clone_image = clone_image or _owned_image
        self._condition = Condition()
        self._queue: deque[RecordingFrame] = deque()
        self._thread: Thread | None = None
        self._accepting = True
        self._stop_requested = False
        self._inflight = 0
        self._inflight_frame: RecordingFrame | None = None
        self._submitted = 0
        self._written = 0
        self._failed = 0
        self._dropped_capacity = 0
        self._dropped_close = 0
        self._drops: list[RecordingDrop] = []
        self._stop_timed_out = False

    @property
    def stats(self) -> RecordingDispatchStats:
        with self._condition:
            return self._stats_locked()

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def is_daemon(self) -> bool:
        thread = self._thread
        return bool(thread and thread.daemon)

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                if self._thread.is_alive():
                    return
                raise RuntimeError("recording dispatcher cannot be restarted")
            self._thread = Thread(
                target=self._run,
                name="live-recording-dispatcher",
                daemon=True,
            )
            self._thread.start()

    def submit(self, frame: RecordingFrame) -> bool:
        if not isinstance(frame, RecordingFrame):
            raise TypeError("frame must be RecordingFrame")
        owned = replace(frame, image=self._clone_image(frame.image))
        dropped: RecordingFrame | None = None
        with self._condition:
            if not self._accepting or self._stop_requested:
                raise RuntimeError("recording dispatcher is closing")
            self._submitted += 1
            if len(self._queue) >= self._capacity:
                dropped = self._queue.popleft()
                self._dropped_capacity += 1
                self._drops.append(_drop(dropped, "capacity"))
            self._queue.append(owned)
            self._condition.notify()
        if dropped is not None and self._on_drop is not None:
            self._notify(self._on_drop, dropped, "capacity")
        return dropped is None

    def wait_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else monotonic() + max(0.0, timeout)
        with self._condition:
            while self._queue or self._inflight:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(
        self,
        *,
        drain_timeout: float = 0.25,
        stop_timeout: float = 1.5,
    ) -> RecordingDispatchStats:
        with self._condition:
            self._accepting = False
        drained = self.wait_idle(drain_timeout)
        dropped: tuple[RecordingFrame, ...] = ()
        with self._condition:
            if not drained and self._queue:
                dropped = tuple(self._queue)
                self._queue.clear()
                self._dropped_close += len(dropped)
                self._drops.extend(_drop(frame, "close") for frame in dropped)
            self._stop_requested = True
            self._condition.notify_all()
        if self._on_drop is not None:
            for frame in dropped:
                self._notify(self._on_drop, frame, "close")
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join(max(0.0, stop_timeout))
        with self._condition:
            self._stop_timed_out = bool(thread and thread.is_alive())
            if self._stop_timed_out and self._inflight_frame is not None:
                timeout_drop = _drop(self._inflight_frame, "stop_timeout")
                if timeout_drop not in self._drops:
                    self._drops.append(timeout_drop)
        return self.stats

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stop_requested or self._queue)
                if self._stop_requested and not self._queue:
                    return
                frame = self._queue.popleft()
                self._inflight = 1
                self._inflight_frame = frame
            try:
                warning = self._operation(frame)
            except Exception as exc:
                with self._condition:
                    self._failed += 1
                    self._drops.append(_drop(frame, "write_failed"))
                if self._on_error is not None:
                    self._notify(self._on_error, frame, exc)
            else:
                with self._condition:
                    self._written += 1
                if warning is not None and self._on_warning is not None:
                    self._notify(self._on_warning, frame, warning)
            finally:
                with self._condition:
                    self._inflight = 0
                    self._inflight_frame = None
                    self._condition.notify_all()

    def _stats_locked(self) -> RecordingDispatchStats:
        thread = self._thread
        return RecordingDispatchStats(
            submitted=self._submitted,
            written=self._written,
            failed=self._failed,
            dropped_capacity=self._dropped_capacity,
            dropped_close=self._dropped_close,
            pending=len(self._queue),
            inflight=self._inflight,
            running=bool(thread and thread.is_alive()),
            stop_timed_out=self._stop_timed_out,
            drops=tuple(self._drops),
        )

    @staticmethod
    def _notify(callback: Callable[..., None], *values: object) -> None:
        try:
            callback(*values)
        except Exception:
            # Telemetry/UI callbacks are side effects and must never kill the
            # recording worker or turn a media failure into a capture failure.
            pass


class RecordingDropAccountingAdapter:
    """Merge dispatcher omissions into the recorder result consumed by seal()."""

    def __init__(self, recorder: object) -> None:
        self._recorder = recorder
        self._stats: RecordingDispatchStats | None = None
        self._cadence: dict[str, object] | None = None

    def accept_dispatcher_stats(
        self,
        stats: RecordingDispatchStats,
    ) -> dict[str, object]:
        if not isinstance(stats, RecordingDispatchStats):
            raise TypeError("stats must be RecordingDispatchStats")
        self._stats = stats
        return _accounting_audit(
            stats,
            recorder_drops=int(getattr(self._recorder, "dropped_frames", 0)),
        )

    def accept_recording_cadence(self, cadence: dict[str, object]) -> None:
        self._cadence = dict(cadence)

    def close(self) -> RecordingResult:
        stats = self._stats
        if stats is not None and stats.stop_timed_out:
            raise RuntimeError("recording dispatcher still has an inflight writer")
        result = self._recorder.close()
        if not isinstance(result, RecordingResult) or stats is None:
            return result
        return _merge_result(result, stats, cadence=self._cadence)

    def __getattr__(self, name: str) -> object:
        return getattr(self._recorder, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in {"_recorder", "_stats", "_cadence"} or "_recorder" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        setattr(self._recorder, name, value)

def _owned_image(image: Any) -> Any:
    if isinstance(image, np.ndarray):
        owned = np.array(image, copy=True, order="C")
        owned.setflags(write=False)
        return owned
    copier = getattr(image, "copy", None)
    return copier() if callable(copier) else image


def _drop(frame: RecordingFrame, reason: str) -> RecordingDrop:
    return RecordingDrop(frame.capture_sequence, frame.captured_ms, reason)


def _audit(stats: RecordingDispatchStats) -> dict[str, object]:
    return {
        "schema": "guandan.recording-dispatcher/1",
        "submitted": stats.submitted,
        "written": stats.written,
        "failed": stats.failed,
        "dropped_capacity": stats.dropped_capacity,
        "dropped_close": stats.dropped_close,
        "stop_timed_out": stats.stop_timed_out,
        "drops": [drop.to_dict() for drop in stats.drops],
    }


def _merge_result(
    result: RecordingResult,
    stats: RecordingDispatchStats,
    *, cadence: dict[str, object] | None = None,
) -> RecordingResult:
    integrity = dict(result.integrity)
    issues = list(integrity.get("issues", ()) or ())
    if stats.drops:
        issues.append("recording_dispatcher_frames_dropped")
        if str(integrity.get("status", "")).upper() not in {"FAIL", "NOT_RECORDED"}:
            integrity["status"] = "PARTIAL"
            integrity["recording_state"] = "partial"
    integrity["issues"] = list(dict.fromkeys(issues))
    integrity["recording_dispatcher"] = _accounting_audit(
        stats,
        recorder_drops=result.dropped_frames,
    )
    if cadence is not None:
        integrity["recording_cadence"] = dict(cadence)
    integrity["omitted_capture_frames"] = int(
        integrity.get("omitted_capture_frames", result.dropped_frames) or 0
    ) + len(stats.drops)
    return replace(
        result,
        dropped_frames=result.dropped_frames + len(stats.drops),
        integrity=integrity,
    )


def _accounting_audit(
    stats: RecordingDispatchStats,
    *,
    recorder_drops: int,
) -> dict[str, object]:
    document = _audit(stats)
    dispatcher_drops = len(stats.drops)
    document.update({
        "recorder_dropped_frames": int(recorder_drops),
        "dispatcher_dropped_frames": dispatcher_drops,
        "total_dropped_frames": int(recorder_drops) + dispatcher_drops,
    })
    return document


__all__ = [
    "BoundedRecordingDispatcher",
    "RecordingDrop",
    "RecordingDropAccountingAdapter",
    "RecordingDispatchStats",
    "RecordingFrame",
]
