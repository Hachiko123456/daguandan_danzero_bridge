from __future__ import annotations

from daguandan_bridge.live.zone_lifecycle import (
    ZoneFrameMetrics,
    ZoneLifecycle,
    ZonePhase,
)


def _metrics(
    monotonic_ms: int,
    *,
    occupied: bool,
    motion: float = 0.0,
    passed: bool = False,
    effect: bool = False,
) -> ZoneFrameMetrics:
    return ZoneFrameMetrics(
        monotonic_ms=monotonic_ms,
        occupied=occupied,
        motion_score=motion,
        pass_visible=passed,
        effect_visible=effect,
    )


def test_zone_waits_for_previous_content_to_clear():
    zone = ZoneLifecycle(
        expected_player="right",
        started_with_clear_zone=False,
        activated_at_ms=0,
    )

    assert zone.observe(_metrics(0, occupied=True)).phase == ZonePhase.WAIT_CLEAR
    assert zone.observe(_metrics(100, occupied=False)).phase == ZonePhase.WAIT_ACTION


def test_zone_enters_burst_only_after_continuous_dynamic_settling():
    zone = ZoneLifecycle(
        expected_player="right",
        started_with_clear_zone=True,
        activated_at_ms=0,
        settle_ms=300,
    )

    assert zone.observe(_metrics(100, occupied=True, motion=0.20)).phase == ZonePhase.CHANGING
    assert zone.observe(_metrics(200, occupied=True, motion=0.005)).phase == ZonePhase.SETTLING
    assert zone.observe(_metrics(450, occupied=True, motion=0.005)).phase == ZonePhase.SETTLING
    decision = zone.observe(_metrics(500, occupied=True, motion=0.005))
    assert decision.phase == ZonePhase.BURST_READ
    assert decision.collect_sample


def test_effect_restarts_dynamic_settling_and_discards_burst():
    zone = ZoneLifecycle(
        expected_player="right",
        started_with_clear_zone=True,
        activated_at_ms=0,
        settle_ms=100,
    )
    zone.observe(_metrics(10, occupied=True, motion=0.20))
    zone.observe(_metrics(20, occupied=True, motion=0.001))
    assert zone.observe(_metrics(120, occupied=True, motion=0.001)).phase == ZonePhase.BURST_READ

    decision = zone.observe(
        _metrics(140, occupied=True, motion=0.001, effect=True)
    )

    assert decision.phase == ZonePhase.CHANGING
    assert decision.discard_burst


def test_action_timeout_requires_review_instead_of_inventing_pass():
    zone = ZoneLifecycle(
        expected_player="left",
        started_with_clear_zone=True,
        activated_at_ms=1_000,
        action_timeout_ms=15_000,
    )

    decision = zone.observe(_metrics(16_000, occupied=False))

    assert decision.phase == ZonePhase.REVIEW_REQUIRED
    assert decision.reason == "action_timeout"
