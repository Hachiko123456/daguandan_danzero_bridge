"""Bounded visual-only rule hints; never consume or mutate canonical history."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class LocalRuleHint:
    session_id: str
    capture_generation: int
    action_epoch: int
    captured_ms: int
    confirmed_ms: int
    expires_ms: int
    confidence: float
    control_box: tuple[int, int, int, int]
    confirmation_frames: int = 2
    source: str = "button_cannot_beat"
    is_pass: bool = True

    def is_current(self, *, session_id: str, capture_generation: int, now_ms: int) -> bool:
        return bool(
            session_id == self.session_id
            and capture_generation == self.capture_generation
            and self.confirmed_ms <= now_ms <= self.expires_ms
        )


class LocalRuleHintTracker:
    """Two different fresh captures within one local action-control lifecycle.

    Integrator contract: call ``observe`` on every fast result, even when the
    canonical player/history is blocked. Pass the *actual* capture generation
    and capture timestamp, plus current monotonic time after analysis. Call
    ``reset`` on pause, stop, terminal, source replacement or capture failure.
    An unchanged/duplicate capture never creates evidence or extends its TTL.
    No missing-box/unknown-active-seat compatibility path is allowed here.
    """

    def __init__(self, *, max_age_ms: int = 500, max_gap_ms: int = 500) -> None:
        if not (0 < max_age_ms <= 2_000 and 0 < max_gap_ms <= max_age_ms):
            raise ValueError("hint timing bounds must be positive and at most 2000ms")
        self.max_age_ms = int(max_age_ms)
        self.max_gap_ms = int(max_gap_ms)
        self._identity: tuple[str, int] | None = None
        self._epoch = 0
        self._last_capture_ms: int | None = None
        self._last_now_ms: int | None = None
        self._box: tuple[int, int, int, int] | None = None
        self._confidence = 0.0
        self._count = 0
        self._hint: LocalRuleHint | None = None

    def reset(self) -> None:
        self._identity = None
        self._epoch += 1
        self._last_capture_ms = None
        self._last_now_ms = None
        self._box = None
        self._confidence = 0.0
        self._count = 0
        self._hint = None

    @staticmethod
    def _control(fast: object) -> tuple[float, tuple[int, int, int, int]] | None:
        try:
            raw_score = getattr(fast, "cannot_beat_confidence", 0.0)
            if isinstance(raw_score, bool):
                return None
            score = float(raw_score)
            raw = getattr(fast, "cannot_beat_box", None)
            if not isinstance(raw, (tuple, list)) or len(raw) != 4:
                return None
            if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
                return None
            if not all(isfinite(float(item)) and float(item).is_integer() for item in raw):
                return None
            box = tuple(int(item) for item in raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if not (isfinite(score) and 0.80 <= score <= 1.0):
            return None
        if not (0 <= box[0] < 16_384 and 0 <= box[1] < 16_384 and 0 < box[2] <= 4_096 and 0 < box[3] <= 4_096):
            return None
        return score, box

    def observe(
        self, fast: object, *, session_id: str, capture_generation: int,
        captured_ms: int, now_ms: int, running: bool = True,
        unobstructed: bool = True, frame_size: tuple[int, int] | None = None,
    ) -> LocalRuleHint | None:
        identity = (str(session_id), capture_generation)
        valid_time = (
            isinstance(captured_ms, int) and not isinstance(captured_ms, bool)
            and isinstance(now_ms, int) and not isinstance(now_ms, bool)
            and 0 <= captured_ms <= now_ms
            and now_ms - captured_ms <= self.max_age_ms
        )
        control = self._control(fast)
        if control is not None and frame_size is not None:
            try:
                width, height = frame_size
                _, (x, y, box_width, box_height) = control
                if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0
                        and x + box_width <= width and y + box_height <= height):
                    control = None
            except (TypeError, ValueError):
                control = None
        safe = bool(
            running and unobstructed and session_id
            and isinstance(capture_generation, int) and not isinstance(capture_generation, bool)
            and capture_generation >= 0 and valid_time and control is not None
            and getattr(fast, "cannot_beat_visible", False)
            and getattr(fast, "active_player", None) == "self"
            and getattr(fast, "self_action_buttons_visible", False)
            and not getattr(fast, "effect_visible", False)
            and not getattr(fast, "super_double_visible", False)
            and not getattr(fast, "game_end_control", None)
        )
        if not safe:
            self.reset()
            return None
        if identity != self._identity or (self._last_now_ms is not None and now_ms < self._last_now_ms):
            self.reset()
            self._identity = identity
        self._last_now_ms = now_ms
        assert control is not None
        score, box = control
        if self._last_capture_ms is not None and captured_ms <= self._last_capture_ms:
            if captured_ms < self._last_capture_ms or (self._box is not None and max(abs(a - b) for a, b in zip(box, self._box)) > 3):
                self.reset()
                return None
            return self.current(session_id=session_id, capture_generation=capture_generation, now_ms=now_ms)
        same_control = bool(
            self._last_capture_ms is not None
            and captured_ms - self._last_capture_ms <= self.max_gap_ms
            and self._box is not None
            and max(abs(a - b) for a, b in zip(box, self._box)) <= 3
        )
        if not same_control:
            self._epoch += 1
            self._count = 0
            self._hint = None
            self._confidence = score
        self._last_capture_ms = captured_ms
        self._box = box
        self._confidence = min(self._confidence, score)
        self._count = min(2, self._count + 1)
        if self._count == 2:
            self._hint = LocalRuleHint(
                session_id=str(session_id), capture_generation=capture_generation,
                action_epoch=self._epoch, captured_ms=captured_ms, confirmed_ms=now_ms,
                expires_ms=captured_ms + self.max_age_ms,
                confidence=self._confidence, control_box=box,
            )
        return self._hint

    def current(self, *, session_id: str, capture_generation: int, now_ms: int) -> LocalRuleHint | None:
        if not isinstance(now_ms, int) or isinstance(now_ms, bool) or (self._last_now_ms is not None and now_ms < self._last_now_ms):
            self.reset()
            return None
        self._last_now_ms = now_ms
        if self._hint is not None and self._hint.is_current(
            session_id=session_id, capture_generation=capture_generation, now_ms=now_ms,
        ):
            return self._hint
        return None
