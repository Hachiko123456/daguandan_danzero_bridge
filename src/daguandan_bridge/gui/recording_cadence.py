"""Capture-time cadence admission for bounded-rate session recording."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RecordingCadenceStats:
    target_fps: float
    interval_ms: float
    observed: int
    admitted: int
    sampled_out: int
    first_captured_ms: int | None
    last_captured_ms: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "guandan.recording-cadence/1",
            "policy": "capture_timestamp_absolute_cadence",
            "target_fps": self.target_fps,
            "interval_ms": self.interval_ms,
            "observed": self.observed,
            "admitted": self.admitted,
            "sampled_out": self.sampled_out,
            "first_captured_ms": self.first_captured_ms,
            "last_captured_ms": self.last_captured_ms,
        }


class RecordingCadenceGate:
    def __init__(self, target_fps: float) -> None:
        fps = float(target_fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("recording target_fps must be finite and positive")
        self._fps = fps
        self._interval = 1000.0 / fps
        self._next_due: float | None = None
        self._observed = self._admitted = self._sampled = 0
        self._first: int | None = None
        self._last: int | None = None

    def admit(self, captured_ms: int) -> bool:
        if isinstance(captured_ms, bool) or not isinstance(captured_ms, int) or captured_ms < 0:
            raise ValueError("captured_ms must be a non-negative integer")
        self._observed += 1
        self._first = captured_ms if self._first is None else self._first
        self._last = captured_ms
        if self._next_due is None:
            self._admitted += 1
            self._next_due = captured_ms + self._interval
            return True
        if captured_ms + 1e-9 < self._next_due:
            self._sampled += 1
            return False
        missed = math.floor((captured_ms - self._next_due) / self._interval)
        self._next_due += (missed + 1) * self._interval
        self._admitted += 1
        return True

    @property
    def stats(self) -> RecordingCadenceStats:
        return RecordingCadenceStats(
            self._fps, self._interval, self._observed, self._admitted,
            self._sampled, self._first, self._last,
        )


__all__ = ["RecordingCadenceGate", "RecordingCadenceStats"]
