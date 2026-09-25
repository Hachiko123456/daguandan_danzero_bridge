from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import test_live_v2_session_runtime as _support  # noqa: E402

runtime = _support.runtime


def _close(live) -> None:
    if live.status != "sealed":
        live.finish()


def test_action_commit_replaces_visible_advice_with_current_turn_state() -> None:
    live, _start, _store, _recorder, _clock, _visions, _advisers = runtime(
        lead="self", advice_immediate=True,
    )
    try:
        old = live.latest_advice
        assert old is not None and old.visible
        assert live.recovery_state == "RUNNING"
        assert live.advice_state.phase == "ready"

        update = live.commit_trusted_action(
            actor="self", cards=("3S",), is_pass=False, monotonic_ms=100,
        )

        assert update.snapshot.revision > old.key.state_revision
        assert update.snapshot.turn_id != old.key.turn_id
        assert live.recovery_state == "RUNNING"
        assert update.advice is not old
        assert update.advice is not None and not update.advice.visible
        assert update.advice.key != old.key
        assert live.advice_state.advice is update.advice
    finally:
        _close(live)


def test_recovery_has_explicit_lifecycle_and_rejects_old_advice() -> None:
    live, _start, _store, _recorder, clock, visions, advisers = runtime(
        lead="self", advice_immediate=False,
    )
    try:
        assert live.latest_advice is not None
        assert live.latest_advice.visible is False
        assert live.recovery_state == "RUNNING"

        rejected = live.commit_trusted_action(
            actor="right", cards=("3D",), is_pass=False, monotonic_ms=100,
        )
        assert rejected.block_reason == "rule_rejection"
        assert live.recovery_state == "RESYNCING"
        assert rejected.advice is not None and not rejected.advice.visible
        assert live.advice_state.phase != "ready"

        advisers[0].release_on_drain = True
        drained = live.poll_deadlines()
        assert drained.advice is not None and not drained.advice.visible
        assert live.recovery_state == "RESYNCING"

        visions[0].empty()
        clock.value = 8_200
        blocked = live.analyze_frame(object(), monotonic_ms=8_200)
        assert blocked.status == "review_required"
        assert blocked.block_reason == "recovery_budget_exceeded"
        assert live.recovery_state == "BLOCKED"
        assert blocked.advice is not None and not blocked.advice.visible
    finally:
        _close(live)



def test_rule_rejection_withholds_visible_advice_instead_of_reusing_it() -> None:
    live, _start, _store, _recorder, _clock, _visions, _advisers = runtime(
        lead="self", advice_immediate=True,
    )
    try:
        old = live.latest_advice
        assert old is not None and old.visible

        rejected = live.commit_trusted_action(
            actor="right", cards=("3D",), is_pass=False, monotonic_ms=100,
        )

        assert rejected.block_reason == "rule_rejection"
        assert live.recovery_state == "RESYNCING"
        assert rejected.advice is not old
        assert rejected.advice is not None
        assert rejected.advice.visible is False
        assert rejected.advice.advice is None
        assert live.latest_advice is rejected.advice
    finally:
        _close(live)

def _cannot_beat_control():
    return _support.FastSignalResult(
        "self", "self", False, True, False,
        cannot_beat_visible=True, cannot_beat_confidence=0.96,
        cannot_beat_box=(100, 100, 80, 30),
    )


def test_cannot_beat_during_recovery_gap_does_not_confirm_local_pass() -> None:
    live, _start, store, _recorder, clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=200,
    )
    try:
        rejected = live.commit_trusted_action(
            actor="right", cards=("3D",), is_pass=False, monotonic_ms=100,
        )
        assert rejected.block_reason == "rule_rejection"
        assert live.recovery_state == "RESYNCING"
        assert rejected.advice is not None and rejected.advice.visible is False

        control = _cannot_beat_control()
        clock.value = 150
        assert live.preview_controls(
            control, captured_ms=150, capture_generation=1, frame_size=(1280, 720)
        ) is None
        clock.value = 200
        assert live.preview_controls(
            control, captured_ms=200, capture_generation=1, frame_size=(1280, 720)
        ) is None

        assert live.recovery_state == "RESYNCING"
        assert live.latest_advice is not None and live.latest_advice.visible is False
        assert live.latest_advice.status == "withheld"
        assert [row["status"] for row in store.advice] == [
            "requested", "worker_started", "cancelled",
        ]
        assert not any(row["status"] == "local_pass" for row in store.advice)
        assert len(advisers[0].submissions) == 1
    finally:
        _close(live)


def test_cannot_beat_in_ready_self_opportunity_confirms_local_pass() -> None:
    live, _start, store, _recorder, clock, _visions, advisers = runtime(
        lead="self", advice_immediate=False, local_hint_window_ms=200,
    )
    try:
        control = _cannot_beat_control()
        clock.value = 100
        assert live.preview_controls(
            control, captured_ms=100, capture_generation=1, frame_size=(1280, 720)
        ) is None
        clock.value = 200
        update = live.preview_controls(
            control, captured_ms=200, capture_generation=1, frame_size=(1280, 720)
        )

        assert live.recovery_state == "RUNNING"
        assert update is not None
        assert update.advice is not None and update.advice.visible
        assert update.advice.advice is not None and update.advice.advice.is_pass
        assert live.snapshot.play_history == ()
        assert [row["status"] for row in store.advice] == [
            "requested", "worker_started", "local_pass",
        ]
        assert len(advisers[0].submissions) == 1
    finally:
        _close(live)
