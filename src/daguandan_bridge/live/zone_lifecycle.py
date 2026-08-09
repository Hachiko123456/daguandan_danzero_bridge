from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..danzero.state import Seat


class ZonePhase(StrEnum):
    WAIT_ACTION = "wait_action"
    SETTLING = "settling"
    BURST_READ = "burst_read"


@dataclass(frozen=True)
class ZoneFrameMetrics:
    monotonic_ms: int
    occupied: bool
    motion_score: float
    pass_visible: bool
    effect_visible: bool
    content_changed: bool = False


@dataclass(frozen=True)
class ZoneDecision:
    phase: ZonePhase
    collect_sample: bool = False
    discard_burst: bool = False
    reason: str = ""
    timed_out: bool = False


class ZoneLifecycle:
    """Gate only the currently expected player's action region.

    A new turn activates one player's ROI.  The gate does not wait for that
    region to become empty and never inspects other seats.  A static region is
    ignored until its content changes, a pass marker appears, or a clear
    movement starts the action.  The action is then sampled after one fixed
    settle delay.  Visible effects restart that delay so cards are never read
    from an in-flight animation frame.
    """

    _ACTION_START_MOTION = 0.060

    def __init__(
        self,
        *,
        expected_player: Seat,
        activated_at_ms: int,
        settle_ms: int = 1_000,
        stable_ms: int = 0,
        action_timeout_ms: int = 22_000,
    ) -> None:
        if settle_ms < 0 or stable_ms < 0 or action_timeout_ms <= 0:
            raise ValueError("settle_ms and action_timeout_ms must be valid")
        self.expected_player = expected_player
        self.phase = ZonePhase.WAIT_ACTION
        self.activated_at_ms = int(activated_at_ms)
        self.settle_ms = int(settle_ms)
        self.stable_ms = int(stable_ms)
        self.action_timeout_ms = int(action_timeout_ms)
        self._settle_started_ms: int | None = None
        self._stable_since_ms: int | None = None

    def observe(self, metrics: ZoneFrameMetrics) -> ZoneDecision:
        now = int(metrics.monotonic_ms)
        if now < self.activated_at_ms:
            raise ValueError("frame timestamp cannot precede zone activation")
        if now - self.activated_at_ms >= self.action_timeout_ms:
            return ZoneDecision(
                self.phase,
                discard_burst=True,
                reason="action_timeout",
                timed_out=True,
            )

        action_visible = bool(metrics.occupied or metrics.pass_visible)
        changed = bool(
            metrics.content_changed
            or metrics.pass_visible
            or metrics.motion_score >= self._ACTION_START_MOTION
        )

        if self.phase == ZonePhase.WAIT_ACTION:
            if not changed:
                return ZoneDecision(self.phase)
            self.phase = ZonePhase.SETTLING
            self._settle_started_ms = now
            self._stable_since_ms = now

        if self.phase == ZonePhase.SETTLING:
            if not action_visible:
                self.phase = ZonePhase.WAIT_ACTION
                self._settle_started_ms = None
                self._stable_since_ms = None
                return ZoneDecision(
                    self.phase,
                    discard_burst=True,
                    reason="action_disappeared",
                )
            if self._settle_started_ms is None:
                self._settle_started_ms = now
            if metrics.effect_visible:
                self._settle_started_ms = now
                self._stable_since_ms = now
                return ZoneDecision(self.phase, reason="effect_visible")
            if metrics.content_changed or metrics.motion_score >= self._ACTION_START_MOTION:
                self._stable_since_ms = now
            if self._stable_since_ms is None:
                self._stable_since_ms = now
            if (
                now - self._settle_started_ms >= self.settle_ms
                and now - self._stable_since_ms >= self.stable_ms
            ):
                self.phase = ZonePhase.BURST_READ
                return ZoneDecision(
                    self.phase,
                    collect_sample=True,
                    reason="zone_settled",
                )
            return ZoneDecision(self.phase)

        if self.phase == ZonePhase.BURST_READ:
            if not action_visible:
                self.phase = ZonePhase.WAIT_ACTION
                self._settle_started_ms = None
                self._stable_since_ms = None
                return ZoneDecision(
                    self.phase,
                    discard_burst=True,
                    reason="burst_invalidated",
                )
            if metrics.effect_visible:
                self.phase = ZonePhase.SETTLING
                self._settle_started_ms = now
                self._stable_since_ms = now
                return ZoneDecision(
                    self.phase,
                    discard_burst=True,
                    reason="effect_visible",
                )
            return ZoneDecision(self.phase, collect_sample=True)

        return ZoneDecision(self.phase)
