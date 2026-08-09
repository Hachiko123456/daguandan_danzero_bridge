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
    content_changed: bool = False,
) -> ZoneFrameMetrics:
    return ZoneFrameMetrics(
        monotonic_ms=monotonic_ms,
        occupied=occupied,
        motion_score=motion,
        pass_visible=passed,
        effect_visible=effect,
        content_changed=content_changed,
    )


def test_zone_starts_waiting_without_clearing_previous_content():
    zone = ZoneLifecycle(
        expected_player="right",
        activated_at_ms=0,
    )

    # The expected player's stale cards may still be visible. They are not a
    # new action until this ROI changes.
    assert zone.observe(_metrics(0, occupied=True)).phase == ZonePhase.WAIT_ACTION
    assert zone.observe(
        _metrics(100, occupied=True, content_changed=True)
    ).phase == ZonePhase.SETTLING


def test_pass_marker_enters_action_window_without_waiting_for_clear():
    zone = ZoneLifecycle(
        expected_player="opposite",
        activated_at_ms=0,
        settle_ms=100,
    )

    decision = zone.observe(_metrics(100, occupied=True, passed=True))

    assert decision.phase == ZonePhase.SETTLING


def test_effect_visibility_restarts_the_settle_delay():
    zone = ZoneLifecycle(
        expected_player="right",
        activated_at_ms=0,
        settle_ms=1_000,
    )

    assert zone.observe(
        _metrics(100, occupied=True, motion=0.20, content_changed=True)
    ).phase == ZonePhase.SETTLING
    assert zone.observe(_metrics(1_100, occupied=True, effect=True)).phase == ZonePhase.SETTLING
    assert zone.observe(_metrics(1_900, occupied=True)).phase == ZonePhase.SETTLING
    decision = zone.observe(_metrics(2_100, occupied=True))
    assert decision.phase == ZonePhase.BURST_READ
    assert decision.collect_sample


def test_action_disappearance_discards_burst_and_waits_for_next_change():
    zone = ZoneLifecycle(
        expected_player="right",
        activated_at_ms=0,
        settle_ms=100,
    )
    zone.observe(_metrics(10, occupied=True, content_changed=True))
    assert zone.observe(_metrics(110, occupied=True)).phase == ZonePhase.BURST_READ

    decision = zone.observe(_metrics(140, occupied=False, effect=True))

    assert decision.phase == ZonePhase.WAIT_ACTION
    assert decision.discard_burst


def test_action_timeout_requests_retry_without_entering_review():
    zone = ZoneLifecycle(
        expected_player="left",
        activated_at_ms=1_000,
        action_timeout_ms=15_000,
    )

    decision = zone.observe(_metrics(16_000, occupied=False))

    assert decision.phase == ZonePhase.WAIT_ACTION
    assert decision.reason == "action_timeout"
    assert decision.timed_out


def test_default_action_window_allows_the_full_twenty_second_game_timer():
    zone = ZoneLifecycle(
        expected_player="left",
        activated_at_ms=0,
    )

    decision = zone.observe(_metrics(20_000, occupied=False))

    assert not decision.timed_out


def test_zone_samples_only_after_content_settles():
    zone = ZoneLifecycle(
        expected_player="right",
        activated_at_ms=0,
        settle_ms=300,
    )

    # 内容未变化：一直等待，且没有超时
    assert zone.observe(_metrics(5_000, occupied=False)).phase == ZonePhase.WAIT_ACTION
    assert zone.observe(_metrics(10_000, occupied=False)).phase == ZonePhase.WAIT_ACTION

    # 内容变化 -> CHANGING
    assert zone.observe(
        _metrics(10_100, occupied=True, content_changed=True)
    ).phase == ZonePhase.SETTLING

    # 变化期间保持 CHANGING（不采样）
    assert zone.observe(
        _metrics(10_200, occupied=True, content_changed=True)
    ).phase == ZonePhase.SETTLING

    # 内容稳定 settle_ms -> BURST_READ 采样
    decision = zone.observe(_metrics(10_400, occupied=True, content_changed=True))
    assert decision.phase == ZonePhase.BURST_READ
    assert decision.collect_sample

    # 突发读取期间内容再变 -> 回到 CHANGING 并丢弃突发
    decision = zone.observe(_metrics(10_700, occupied=True, content_changed=True))
    assert decision.phase == ZonePhase.BURST_READ
    assert decision.collect_sample


def test_idle_zone_reports_timeout_without_changing_phase():
    zone = ZoneLifecycle(
        expected_player="left",
        activated_at_ms=0,
        action_timeout_ms=15_000,
    )

    decision = zone.observe(_metrics(60_000, occupied=False))

    assert decision.phase == ZonePhase.WAIT_ACTION
    assert decision.timed_out
