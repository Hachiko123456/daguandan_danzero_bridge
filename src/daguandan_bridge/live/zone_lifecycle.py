from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..danzero.state import Seat


class ZonePhase(StrEnum):
    WAIT_CLEAR = "wait_clear"
    WAIT_ACTION = "wait_action"
    CHANGING = "changing"
    SETTLING = "settling"
    BURST_READ = "burst_read"
    VALIDATE = "validate"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class ZoneFrameMetrics:
    monotonic_ms: int
    occupied: bool
    motion_score: float
    pass_visible: bool
    effect_visible: bool


@dataclass(frozen=True)
class ZoneDecision:
    phase: ZonePhase
    collect_sample: bool = False
    discard_burst: bool = False
    reason: str = ""


class ZoneLifecycle:
    """Gate one expected player's ROI using timestamps and motion hysteresis."""

    def __init__(
        self,
        *,
        expected_player: Seat,
        started_with_clear_zone: bool,
        activated_at_ms: int,
        settle_ms: int = 400,
        action_timeout_ms: int = 15_000,
        low_motion_threshold: float = 0.015,
        high_motion_threshold: float = 0.060,
    ) -> None:
        if settle_ms < 0 or action_timeout_ms <= 0:
            raise ValueError("沉降与超时参数无效")
        if not 0 <= low_motion_threshold < high_motion_threshold:
            raise ValueError("运动迟滞阈值必须满足 0 <= low < high")
        self.expected_player = expected_player
        self.phase = (
            ZonePhase.WAIT_ACTION
            if started_with_clear_zone
            else ZonePhase.WAIT_CLEAR
        )
        self.activated_at_ms = int(activated_at_ms)
        self.settle_ms = int(settle_ms)
        self.action_timeout_ms = int(action_timeout_ms)
        self.low_motion_threshold = float(low_motion_threshold)
        self.high_motion_threshold = float(high_motion_threshold)
        self._settle_started_ms: int | None = None

    def observe(self, metrics: ZoneFrameMetrics) -> ZoneDecision:
        now = int(metrics.monotonic_ms)
        if now < self.activated_at_ms:
            raise ValueError("帧时间不能早于区域激活时间")
        if self.phase not in {ZonePhase.VALIDATE, ZonePhase.REVIEW_REQUIRED} and (
            now - self.activated_at_ms >= self.action_timeout_ms
        ):
            self.phase = ZonePhase.REVIEW_REQUIRED
            return ZoneDecision(self.phase, discard_burst=True, reason="action_timeout")

        action_visible = bool(metrics.occupied or metrics.pass_visible)
        high_or_effect = bool(
            metrics.effect_visible
            or metrics.motion_score >= self.high_motion_threshold
        )
        low_motion = metrics.motion_score <= self.low_motion_threshold

        if self.phase == ZonePhase.WAIT_CLEAR:
            if not action_visible and not metrics.effect_visible:
                self.phase = ZonePhase.WAIT_ACTION
                return ZoneDecision(self.phase, reason="previous_content_cleared")
            return ZoneDecision(self.phase)

        if self.phase == ZonePhase.WAIT_ACTION:
            if action_visible or high_or_effect:
                self.phase = ZonePhase.CHANGING
                self._settle_started_ms = None
                return ZoneDecision(self.phase, reason="action_zone_changed")
            return ZoneDecision(self.phase)

        if self.phase == ZonePhase.CHANGING:
            if high_or_effect:
                self._settle_started_ms = None
                return ZoneDecision(self.phase)
            if not action_visible:
                self.phase = ZonePhase.WAIT_ACTION
                self._settle_started_ms = None
                return ZoneDecision(self.phase, discard_burst=True, reason="action_disappeared")
            if low_motion:
                self.phase = ZonePhase.SETTLING
                self._settle_started_ms = now
                return ZoneDecision(self.phase, reason="low_motion_started")
            return ZoneDecision(self.phase)

        if self.phase == ZonePhase.SETTLING:
            if high_or_effect:
                self.phase = ZonePhase.CHANGING
                self._settle_started_ms = None
                return ZoneDecision(
                    self.phase,
                    discard_burst=True,
                    reason="effect_or_high_motion",
                )
            if not action_visible:
                self.phase = ZonePhase.WAIT_ACTION
                self._settle_started_ms = None
                return ZoneDecision(self.phase, discard_burst=True, reason="action_disappeared")
            if not low_motion:
                self.phase = ZonePhase.CHANGING
                self._settle_started_ms = None
                return ZoneDecision(
                    self.phase,
                    discard_burst=True,
                    reason="settling_interrupted",
                )
            if self._settle_started_ms is None:
                self._settle_started_ms = now
            if now - self._settle_started_ms >= self.settle_ms:
                self.phase = ZonePhase.BURST_READ
                return ZoneDecision(
                    self.phase,
                    collect_sample=True,
                    reason="zone_settled",
                )
            return ZoneDecision(self.phase)

        if self.phase == ZonePhase.BURST_READ:
            if high_or_effect or not action_visible:
                self.phase = ZonePhase.CHANGING
                self._settle_started_ms = None
                return ZoneDecision(
                    self.phase,
                    discard_burst=True,
                    reason="burst_invalidated",
                )
            return ZoneDecision(self.phase, collect_sample=low_motion)

        return ZoneDecision(self.phase)

    def begin_validation(self) -> ZoneDecision:
        if self.phase != ZonePhase.BURST_READ:
            raise RuntimeError("只有突发读取阶段可以进入校验")
        self.phase = ZonePhase.VALIDATE
        return ZoneDecision(self.phase, reason="burst_complete")

    def require_review(self, reason: str) -> ZoneDecision:
        self.phase = ZonePhase.REVIEW_REQUIRED
        return ZoneDecision(self.phase, discard_burst=True, reason=str(reason))
