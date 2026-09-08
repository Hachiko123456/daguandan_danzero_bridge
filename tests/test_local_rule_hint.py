from dataclasses import replace

import pytest

from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.live.local_rule_hint import LocalRuleHintTracker


def fast(**overrides):
    value = FastSignalResult(
        expected_player="right", active_player="self", pass_visible=False,
        self_action_buttons_visible=True, effect_visible=False,
        cannot_beat_visible=True, cannot_beat_confidence=.98,
        cannot_beat_box=(400, 600, 80, 30),
    )
    return replace(value, **overrides)


def observe(tracker, captured_ms=100, now_ms=None, **kwargs):
    return tracker.observe(
        kwargs.pop("fast", fast()), session_id=kwargs.pop("session_id", "s1"),
        capture_generation=kwargs.pop("capture_generation", 1),
        captured_ms=captured_ms, now_ms=captured_ms if now_ms is None else now_ms,
        **kwargs,
    )


def test_two_fresh_local_captures_ignore_canonical_expected_actor():
    tracker = LocalRuleHintTracker()
    assert observe(tracker) is None
    hint = observe(tracker, 200)
    assert hint.is_pass and hint.source == "button_cannot_beat"
    assert hint.confirmation_frames == 2 and hint.expires_ms == 700
    assert hint.is_current(session_id="s1", capture_generation=1, now_ms=220)
    assert not hint.is_current(session_id="other", capture_generation=1, now_ms=220)
    assert not hint.is_current(session_id="s1", capture_generation=2, now_ms=220)
    assert not hint.is_current(session_id="s1", capture_generation=1, now_ms=701)


@pytest.mark.parametrize("changes", [
    {"active_player": None}, {"active_player": "left"},
    {"self_action_buttons_visible": False}, {"cannot_beat_visible": False},
    {"effect_visible": True}, {"super_double_visible": True},
    {"game_end_control": "continue_game"}, {"cannot_beat_box": None},
    {"cannot_beat_confidence": .79}, {"cannot_beat_confidence": float("nan")},
    {"cannot_beat_confidence": float("inf")}, {"cannot_beat_confidence": 1.01},
    {"cannot_beat_confidence": True},
    {"cannot_beat_box": (400, 600, 0, 30)}, {"cannot_beat_box": (-1, 600, 80, 30)},
    {"cannot_beat_box": (400.5, 600, 80, 30)}, {"cannot_beat_box": (True, 600, 80, 30)},
])
def test_unsafe_visual_evidence_revokes_and_does_not_confirm(changes):
    tracker = LocalRuleHintTracker()
    observe(tracker)
    assert observe(tracker, 200) is not None
    assert observe(tracker, 300, fast=fast(**changes)) is None
    assert observe(tracker, 400) is None
    assert observe(tracker, 500) is not None


@pytest.mark.parametrize("option", [{"running": False}, {"unobstructed": False}])
def test_paused_or_occluded_resets(option):
    tracker = LocalRuleHintTracker()
    observe(tracker)
    assert observe(tracker, 200, **option) is None
    assert observe(tracker, 300) is None


def test_duplicate_captures_never_confirm_or_extend_ttl():
    tracker = LocalRuleHintTracker()
    assert observe(tracker) is None
    assert observe(tracker, 100, 120) is None
    hint = observe(tracker, 200)
    assert observe(tracker, 200, 250) == hint
    assert observe(tracker, 200, 701) is None
    assert observe(tracker, 800) is None


def test_old_future_backwards_and_large_gap_need_new_two_frame_epoch():
    tracker = LocalRuleHintTracker()
    assert observe(tracker, 100, 601) is None
    assert observe(tracker, 700, 699) is None
    assert observe(tracker, 800) is None
    assert observe(tracker, 1400) is None
    assert observe(tracker, 1500) is not None
    assert observe(tracker, 1450, 1550) is None
    assert observe(tracker, 1600) is None


@pytest.mark.parametrize("change", [{"session_id": "new"}, {"capture_generation": 2}])
def test_session_or_generation_change_never_reuses_first_capture(change):
    tracker = LocalRuleHintTracker()
    observe(tracker)
    assert observe(tracker, 200, **change) is None
    assert observe(tracker, 300, **change) is not None


def test_disappearance_and_control_motion_open_a_new_bounded_epoch():
    tracker = LocalRuleHintTracker()
    observe(tracker)
    first = observe(tracker, 200)
    assert observe(tracker, 300, fast=fast(cannot_beat_visible=False)) is None
    assert observe(tracker, 400) is None
    second = observe(tracker, 500)
    assert second.action_epoch > first.action_epoch
    assert observe(tracker, 600, fast=fast(cannot_beat_box=(450, 600, 80, 30))) is None
    assert observe(tracker, 700, fast=fast(cannot_beat_box=(451, 600, 80, 30))) is not None
    for timestamp in range(800, 10_800, 100):
        observe(tracker, timestamp, fast=fast(cannot_beat_box=(450, 600, 80, 30)))
    assert tracker._count == 2  # Constant-size evidence; no growing frame list.
    assert len(vars(tracker)) <= 12


def test_same_capture_with_conflicting_box_revokes_without_extending_old_hint():
    tracker = LocalRuleHintTracker()
    observe(tracker)
    assert observe(tracker, 200) is not None
    assert observe(tracker, 200, 250, fast=fast(cannot_beat_box=(500, 600, 80, 30))) is None
    assert tracker.current(session_id="s1", capture_generation=1, now_ms=260) is None
    assert observe(tracker, 300) is None


def test_control_must_fit_actual_frame_and_old_clock_cannot_resurrect_expired_hint():
    tracker = LocalRuleHintTracker()
    assert observe(tracker, frame_size=(450, 720)) is None
    assert observe(tracker, 200, frame_size=(1280, 610)) is None
    assert observe(tracker, 300, frame_size=(1280, 720)) is None
    assert observe(tracker, 400, frame_size=(1280, 720)) is not None
    assert tracker.current(session_id="s1", capture_generation=1, now_ms=901) is None
    assert tracker.current(session_id="s1", capture_generation=1, now_ms=600) is None
