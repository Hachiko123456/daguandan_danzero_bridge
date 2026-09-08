"""Primary-review acceptance checks for the September 5 remote incident.

These tests intentionally exercise independently reconstructed failure cases.
They do not need the user's files, videos, network, or a running game client.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from time import monotonic_ns

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from daguandan_bridge.domain.advice import LocalAdvice
from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
from daguandan_bridge.gui.live_controller import LiveAssistantController, _AnalysisDelivery
from daguandan_bridge.live.consensus import ConsensusResult, RecognitionSample
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.live.local_rule_hint import LocalRuleHintTracker
from daguandan_bridge.live.orchestrator import (
    AdviceRequestKey, LiveAdvice, LiveOrchestrator, LiveUpdate, _TurnOwnershipWindow,
)


class AcceptanceRuntime(QObject):
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    listening_status = Signal(object)
    log_delivery_status = Signal(object)


@pytest.fixture
def compact_window():
    app = QApplication.instance() or QApplication([])
    runtime = AcceptanceRuntime()
    runtime.orchestrator = object()
    window = RecommendationFloatWindow(runtime)
    yield app, runtime, window
    window.close()
    window.deleteLater()
    app.processEvents()


def _snapshot(session_id="remote-session", turn=22, revision=23):
    return SimpleNamespace(
        session_id=session_id, turn_id=turn, revision=revision,
        current_player="self", finished_seats=frozenset(),
    )


def _ready(session_id="remote-session", turn=22, revision=23):
    key = AdviceRequestKey(session_id, turn, revision)
    return LiveAdvice(
        key=key, status="ready", visible=True,
        advice=LocalAdvice(
            strategy="fabledan", cards=("4S",), play_type="SINGLE",
            is_pass=False, state_revision=revision, elapsed_ms=300,
            request_id=key.request_id,
        ),
    )


def test_primary_same_request_recovery_repaints_visible_cards(compact_window):
    app, runtime, window = compact_window
    advice = _ready()
    update = LiveUpdate(status="running", snapshot=_snapshot(), advice=advice)
    runtime.update_ready.emit(update)
    app.processEvents()
    original = window.suggestion_label.text()
    assert window._card_badges
    runtime.update_ready.emit(replace(update, advice=replace(
        advice, status="withheld", visible=False,
        withhold_reason="turn_recovery_pending",
        error="缺少左家动作，请先手动出牌",
    )))
    app.processEvents()
    assert not window._card_badges
    runtime.update_ready.emit(update)
    app.processEvents()
    assert window.suggestion_label.text() == original
    assert window._card_badges


def test_primary_old_session_advice_is_not_displayed_even_with_same_numeric_key(compact_window):
    app, runtime, window = compact_window
    runtime.update_ready.emit(LiveUpdate(status="running", snapshot=_snapshot(), advice=_ready()))
    app.processEvents()
    assert window._card_badges
    runtime.update_ready.emit(LiveUpdate(
        status="running", snapshot=_snapshot(session_id="next-session"), advice=_ready(),
    ))
    app.processEvents()
    assert not window._card_badges


@pytest.mark.parametrize("delivery_status", ["PASS", "FAIL", "running", "disabled"])
def test_primary_old_log_delivery_never_replaces_current_recommendation(compact_window, delivery_status):
    app, runtime, window = compact_window
    runtime.update_ready.emit(LiveUpdate(status="running", snapshot=_snapshot(), advice=_ready()))
    app.processEvents()
    original = window.suggestion_label.text()
    runtime.log_delivery_status.emit({
        "session_id": "previous-session", "status": delivery_status,
        "diagnostic_zip_path": "C:/old-session/log.zip", "error": "old failure",
    })
    app.processEvents()
    assert window.suggestion_label.text() == original
    assert window._card_badges
    assert "old-session" not in window.detail_label.text()


def test_primary_unstamped_samples_cannot_prove_pre_recovery_handoff():
    cards = ("10H", "10H", "10S")
    samples = [RecognitionSample(cards, False, .95, "template:cards", f"OBS-{n}") for n in (124, 125)]
    window = _TurnOwnershipWindow(
        key=("remote-session", 31, 32, "right"), expected_player="right",
        handoff_detected_ms=1308776718, turn_recovery_pending=True,
        turn_recovery_detected_ms=1308780156, handoff_samples=samples,
    )
    result = ConsensusResult("confirmed", cards, False, .95, "two_valid_streak", 2, ())
    assert not LiveOrchestrator._confirmed_handoff_precedes_turn_recovery(window, result)


@pytest.mark.parametrize("case", ["late_85_seconds", "wrong_epoch", "duplicate_time", "reversed_sequence"])
def test_primary_handoff_evidence_has_real_temporal_provenance(case):
    cards = ("10H", "10H", "10S")
    epoch = ("remote-session", 31, 32, "right", 0)
    started = 1308780156
    samples = [RecognitionSample(
        cards, False, .95, "template:cards", f"OBS-{index}",
        captured_ms=started - 400 + (index - 124) * 200,
        capture_seq=index, action_epoch=epoch,
    ) for index in (124, 125)]
    window = _TurnOwnershipWindow(
        key=("remote-session", 31, 32, "right"), expected_player="right",
        evidence_epoch=epoch, handoff_detected_ms=started - 500,
        turn_recovery_pending=True, turn_recovery_detected_ms=started,
        handoff_samples=samples,
    )
    result = ConsensusResult("confirmed", cards, False, .95, "two_valid_streak", 2, ())
    assert LiveOrchestrator._confirmed_handoff_precedes_turn_recovery(window, result)
    if case == "late_85_seconds":
        samples = [replace(samples[0], captured_ms=1308865375), replace(samples[1], captured_ms=1308865781)]
    elif case == "wrong_epoch":
        samples = [replace(sample, action_epoch=("old-session", 31, 32, "right", 0)) for sample in samples]
    elif case == "duplicate_time":
        samples = [samples[0], replace(samples[1], captured_ms=samples[0].captured_ms)]
    else:
        samples = [replace(samples[0], capture_seq=125), replace(samples[1], capture_seq=124)]
    assert not LiveOrchestrator._confirmed_handoff_precedes_turn_recovery(
        replace(window, handoff_samples=samples), result,
    )


def _visual_cannot_beat(**changes):
    return replace(FastSignalResult(
        expected_player="right", active_player="self", pass_visible=False,
        self_action_buttons_visible=True, effect_visible=False,
        cannot_beat_visible=True, cannot_beat_confidence=.95,
        cannot_beat_box=(500, 540, 90, 35),
    ), **changes)


@pytest.mark.parametrize("changes", [
    {"cannot_beat_confidence": True},
    {"cannot_beat_box": (1500, 540, 90, 35)},
    {"active_player": None},
    {"effect_visible": True},
])
def test_primary_local_pass_hint_rejects_unsafe_visual_evidence(changes):
    tracker = LocalRuleHintTracker()
    for stamp in (1000, 1200):
        assert tracker.observe(
            _visual_cannot_beat(**changes), session_id="remote-session",
            capture_generation=2, captured_ms=stamp, now_ms=stamp+20,
            frame_size=(1280, 720),
        ) is None


def test_primary_duplicate_hint_frames_do_not_keep_old_evidence_alive():
    tracker = LocalRuleHintTracker()
    fast = _visual_cannot_beat()
    for stamp in (1000, 1200):
        hint = tracker.observe(fast, session_id="remote-session", capture_generation=2,
                               captured_ms=stamp, now_ms=stamp+20, frame_size=(1280, 720))
    assert hint is not None
    assert tracker.observe(fast, session_id="remote-session", capture_generation=2,
                           captured_ms=1200, now_ms=1800, frame_size=(1280, 720)) is None
    assert tracker.current(session_id="remote-session", capture_generation=2, now_ms=1800) is None


def test_primary_hint_control_change_and_generation_change_withdraw_immediately():
    tracker = LocalRuleHintTracker()
    fast = _visual_cannot_beat()
    for stamp in (1000, 1200):
        hint = tracker.observe(fast, session_id="remote-session", capture_generation=2,
                               captured_ms=stamp, now_ms=stamp+20, frame_size=(1280, 720))
    assert hint is not None
    moved = replace(fast, cannot_beat_box=(700, 540, 90, 35))
    assert tracker.observe(moved, session_id="remote-session", capture_generation=2,
                           captured_ms=1200, now_ms=1240, frame_size=(1280, 720)) is None
    for stamp in (1300, 1400):
        hint = tracker.observe(fast, session_id="remote-session", capture_generation=2,
                               captured_ms=stamp, now_ms=stamp+20, frame_size=(1280, 720))
    assert hint is not None
    assert tracker.observe(fast, session_id="remote-session", capture_generation=3,
                           captured_ms=1500, now_ms=1520, frame_size=(1280, 720)) is None


@pytest.fixture
def delivery_window(tmp_path):
    app = QApplication.instance() or QApplication([])
    controller = LiveAssistantController(
        capture_service=SimpleNamespace(profiles_root=tmp_path),
        recognition_service=object(), advisor=object(), session_factory=object(),
    )
    controller.orchestrator = SimpleNamespace(snapshot=_snapshot())
    token = controller._activate_live_token(controller.orchestrator)
    window = RecommendationFloatWindow(controller)
    yield app, controller, token, window
    controller._invalidate_live_token()
    controller.orchestrator = None
    window.close()
    window.deleteLater()
    app.processEvents()
    controller.shutdown()


@pytest.mark.parametrize("late_kind", ["old_sequence", "no_result"])
def test_primary_gui_mailbox_preserves_new_valid_update_before_ui_drain(delivery_window, late_kind):
    app, controller, token, window = delivery_window
    ready = LiveUpdate(status="running", snapshot=_snapshot(), advice=_ready(),
                       capture_generation=token.generation, update_sequence=10)
    controller._enqueue_gui_delivery(_AnalysisDelivery(token, ready, monotonic_ns()))
    late = None if late_kind=="no_result" else replace(ready, update_sequence=9, advice=replace(
        ready.advice, status="withheld", visible=False, withhold_reason="turn_recovery_pending",
    ))
    controller._enqueue_gui_delivery(_AnalysisDelivery(token, late, monotonic_ns()))
    app.processEvents()
    assert window._card_badges, "new ready was lost before the UI sequence gate could see it"


def test_primary_explicit_old_generation_callback_cannot_be_retagged_as_current(delivery_window):
    app, controller, token, window = delivery_window
    controller._capture_generation = 3
    token = controller._activate_live_token(controller.orchestrator)
    assert token.generation > 1
    ready = LiveUpdate(status="running", snapshot=_snapshot(), advice=_ready(),
                       capture_generation=token.generation, update_sequence=10)
    controller._queue_orchestrator_update(ready)
    app.processEvents()
    assert window._card_badges
    old = replace(ready, capture_generation=token.generation-1, update_sequence=11, advice=replace(
        ready.advice, status="withheld", visible=False, withhold_reason="turn_recovery_pending",
    ))
    controller._queue_orchestrator_update(old)
    app.processEvents()
    assert window._card_badges, "controller washed an explicitly old generation into the current one"
