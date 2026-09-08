"""Short, memory-only capture history for a just-created action window."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TurnFrame:
    captured_ms: int
    generation: int
    frame: np.ndarray


class TurnEvidenceCache:
    """Retain bounded original pixels, never inferred actions or disk media.

    This cache only supplies a *before* image. A consumer must still prove
    ownership, visual change and legal action semantics in the current turn.
    """

    def __init__(self, *, max_frames: int = 4, max_bytes: int = 16 * 1024 * 1024,
                 max_age_ms: int = 1_500, max_seat_frames: int = 32,
                 min_seat_frames: int = 16) -> None:
        if min(max_frames, max_bytes, max_age_ms) <= 0:
            raise ValueError("evidence bounds must be positive")
        if min(max_seat_frames, min_seat_frames) <= 0 or min_seat_frames > max_seat_frames:
            raise ValueError("seat evidence bounds must be positive and ordered")
        self.max_frames, self.max_bytes, self.max_age_ms = max_frames, max_bytes, max_age_ms
        self.max_seat_frames = max_seat_frames
        self.min_seat_frames = min_seat_frames
        self._frames: deque[TurnFrame] = deque()
        self._seat_frames: dict[str, deque[TurnFrame]] = {}
        self._bytes = 0
        self._generation: int | None = None
        self._last_ms: int | None = None
        self._consumed: dict[str, int] = {}

    @property
    def retained_bytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        self._frames.clear()
        self._seat_frames.clear()
        self._bytes = 0
        self._generation = None
        self._last_ms = None
        self._consumed.clear()

    def mark_consumed(self, seat: str, captured_ms: int) -> None:
        """Never reuse a baseline from before a seat's last committed action."""
        self._consumed[seat] = max(captured_ms, self._consumed.get(seat, captured_ms))

    def _drop_full_left(self) -> bool:
        if not self._frames:
            return False
        self._bytes -= self._frames.popleft().frame.nbytes
        return True

    def _drop_seat_left(self, seat: str) -> bool:
        frames = self._seat_frames.get(seat)
        if not frames:
            return False
        self._bytes -= frames.popleft().frame.nbytes
        if not frames:
            self._seat_frames.pop(seat, None)
        return True

    def _trim_expired(self, captured_ms: int) -> None:
        while self._frames and captured_ms - self._frames[0].captured_ms > self.max_age_ms:
            self._drop_full_left()
        for seat in tuple(self._seat_frames):
            frames = self._seat_frames.get(seat)
            while frames and captured_ms - frames[0].captured_ms > self.max_age_ms:
                self._drop_seat_left(seat)
                frames = self._seat_frames.get(seat)

    def _trim_total_bytes(self) -> None:
        # Full frames are compatibility evidence; seat ROI history is the
        # durable baseline source under full-size captures.  Prefer shedding
        # full frames before reducing per-seat history.
        while self._bytes > self.max_bytes and self._frames:
            self._drop_full_left()
        while self._bytes > self.max_bytes:
            candidates = [
                (frames[0].captured_ms, seat)
                for seat, frames in self._seat_frames.items()
                if len(frames) > self.min_seat_frames
            ]
            if not candidates:
                candidates = [
                    (frames[0].captured_ms, seat)
                    for seat, frames in self._seat_frames.items()
                    if frames
                ]
            if not candidates:
                break
            _stamp, seat = min(candidates)
            self._drop_seat_left(seat)

    def observe(
        self,
        frame: np.ndarray,
        *,
        captured_ms: int,
        generation: int,
        seat_rois: dict[str, np.ndarray] | None = None,
    ) -> None:
        if type(captured_ms) is not int or type(generation) is not int or min(captured_ms, generation) < 0:
            self.clear()
            return
        if self._generation != generation or self._last_ms is not None and captured_ms < self._last_ms:
            self.clear()
        if captured_ms == self._last_ms:
            return
        self._generation, self._last_ms = generation, captured_ms
        self._trim_expired(captured_ms)
        if frame.nbytes <= self.max_bytes:
            while self._frames and len(self._frames) >= self.max_frames:
                self._drop_full_left()
            copied = frame.copy()
            copied.flags.writeable = False
            self._frames.append(TurnFrame(captured_ms, generation, copied))
            self._bytes += copied.nbytes
        for seat, roi in (seat_rois or {}).items():
            if roi.nbytes > self.max_bytes:
                continue
            copied_roi = roi.copy()
            copied_roi.flags.writeable = False
            frames = self._seat_frames.setdefault(str(seat), deque())
            frames.append(TurnFrame(captured_ms, generation, copied_roi))
            self._bytes += copied_roi.nbytes
            while len(frames) > self.max_seat_frames:
                self._drop_seat_left(str(seat))
                frames = self._seat_frames.setdefault(str(seat), deque())
        self._trim_total_bytes()

    def before(self, captured_ms: int, *, generation: int, seat: str | None = None) -> TurnFrame | None:
        if generation != self._generation:
            return None
        # Prefer the oldest still-recent frame to retain a pre-animation
        # surface; later frames may already contain the fast player's cards.
        return next((item for item in self._frames if
                     0 < captured_ms - item.captured_ms <= self.max_age_ms
                     and item.captured_ms >= self._consumed.get(seat, -1)), None)

    def before_roi(self, captured_ms: int, *, generation: int, seat: str) -> TurnFrame | None:
        if generation != self._generation:
            return None
        return next((item for item in self._seat_frames.get(str(seat), ()) if
                     0 < captured_ms - item.captured_ms <= self.max_age_ms
                     and item.captured_ms >= self._consumed.get(str(seat), -1)), None)
