from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from test_live_orchestrator import (
    FakeRecognitionService,
    FakeSelfLeadRecognitionService,
    SuccessfulAdviceService,
    _orchestrator,
    _play,
)


def _frame_update(orchestrator, timestamp: int, *, occupied: bool = True):
    return orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=timestamp,
        wall_time=f"first-action-{timestamp}",
        metrics=ZoneFrameMetrics(timestamp, occupied, 0.2 if occupied else 0.0, False, False),
    )


def test_start_without_opening_action_creates_pending_first_action_gate(tmp_path):
    advisor = SuccessfulAdviceService()
    recognition = FakeSelfLeadRecognitionService(cards=("7S",))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        advisor=advisor,
        settle_ms=0,
    )
    try:
        assert orchestrator.status == "running"
        assert orchestrator.snapshot.lead_player == "self"
        assert orchestrator.snapshot.current_player == "self"
        assert orchestrator.first_action_pending is True
        assert orchestrator.first_action_gate == {
            "pending": True,
            "reason": "awaiting_first_action",
        }
        assert orchestrator.needs_first_action_frames is True
        assert orchestrator.latest_advice is None
        assert not any(event.event_type == "advice_requested" for event in orchestrator.events)
        assert orchestrator.start_self_advice() is None
        assert advisor.calls == 0
    finally:
        orchestrator.finish()


def test_pending_first_action_does_not_emit_event_or_advice_without_action(tmp_path):
    advisor = SuccessfulAdviceService()
    recognition = FakeSelfLeadRecognitionService(cards=("7S",))
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="self",
        advisor=advisor,
        settle_ms=0,
    )
    try:
        # First establish that the local controls have been seen, then clear
        # them without providing a card read.  This is the real waiting state.
        _frame_update(orchestrator, 100)
        recognition.controls_visible = False
        for timestamp in (200, 300, 400):
            update = _frame_update(orchestrator, timestamp, occupied=False)
        assert update.event is None
        assert orchestrator.first_action_pending is True
        assert orchestrator.snapshot.play_history == ()
        assert orchestrator.latest_advice is None
        assert advisor.calls == 0
        assert not any(
            event.event_type in {"player_played", "player_passed", "advice_requested"}
            for event in orchestrator.events
        )
    finally:
        orchestrator.finish()


def test_real_first_action_commits_once_and_opens_normal_turn(tmp_path):
    recognition = FakeRecognitionService([_play("7S"), _play("7S")])
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    try:
        first = _frame_update(orchestrator, 100)
        second = _frame_update(orchestrator, 200)
        actions = [
            event for event in orchestrator.events
            if event.event_type in {"player_played", "player_passed"}
        ]
        assert first.event is None
        assert second.event is not None
        assert len(actions) == 1
        assert actions[0].event_type == "player_played"
        assert actions[0].actor == "right"
        assert actions[0].payload["cards"] == ["7S"]
        assert orchestrator.first_action_pending is False
        assert orchestrator.first_action_gate["reason"] == "first_action_confirmed"
        assert orchestrator.snapshot.current_player == "opposite"
    finally:
        orchestrator.finish()


def test_illegal_or_low_confidence_first_action_does_not_advance(tmp_path):
    illegal = _play("7S", "7S", "7S")
    low_confidence = _play("7S")
    low_confidence = low_confidence.__class__(
        player=low_confidence.player,
        cards=low_confidence.cards,
        is_pass=False,
        confidence=0.60,
        diagnostics=low_confidence.diagnostics,
        annotations=low_confidence.annotations,
        source="low-confidence-opening",
        post_hand=low_confidence.post_hand,
        post_hand_confidence=low_confidence.post_hand_confidence,
        suit_options=low_confidence.suit_options,
    )
    recognition = FakeRecognitionService([illegal, illegal, low_confidence, low_confidence])
    orchestrator = _orchestrator(
        tmp_path,
        [],
        recognition=recognition,
        lead_player="right",
        settle_ms=0,
        recognition_strategy="two_valid_streak",
    )
    try:
        for timestamp in (100, 200):
            update = _frame_update(orchestrator, timestamp)
        assert update.event is None
        assert orchestrator.first_action_pending is True
        assert orchestrator.snapshot.play_history == ()

        for timestamp in (300, 400):
            update = _frame_update(orchestrator, timestamp)
        assert update.event is None
        assert orchestrator.first_action_pending is True
        assert orchestrator.first_action_gate["reason"] == "opening_action_low_confidence"
        assert orchestrator.snapshot.play_history == ()
    finally:
        orchestrator.finish()


def test_start_with_trusted_opening_action_advances_consistent_snapshot(tmp_path):
    opening_action = SimpleNamespace(
        actor="right",
        cards=("7S",),
        next_player="opposite",
        confidence=0.95,
        source="trusted-opening-seed",
        suit_options=(),
    )
    orchestrator = _orchestrator(
        tmp_path,
        [],
        lead_player="right",
        opening_action=opening_action,
        settle_ms=0,
    )
    try:
        assert orchestrator.status == "running"
        assert orchestrator.first_action_pending is False
        assert orchestrator.first_action_gate == {
            "pending": False,
            "reason": "opening_action_confirmed",
        }
        actions = [
            event for event in orchestrator.events
            if event.event_type == "player_played"
        ]
        assert len(actions) == 1
        assert actions[0].actor == "right"
        assert actions[0].payload["cards"] == ["7S"]
        assert orchestrator.snapshot.current_player == "opposite"
    finally:
        orchestrator.finish()

