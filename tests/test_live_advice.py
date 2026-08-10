from __future__ import annotations

import threading
import time

import numpy as np

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.recognition_service import FastSignalResult, PlayRegionResult


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


class LeftPlayRecognition:
    def recognize_fast_signals(self, _image, expected_player):
        return FastSignalResult(
            expected_player=expected_player,
            active_player=expected_player,
            pass_visible=False,
            self_action_buttons_visible=False,
            effect_visible=False,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank):
        del wild_rank
        return PlayRegionResult(
            player=seat,
            cards=("7S", "7H"),
            is_pass=False,
            confidence=0.95,
            diagnostics=(),
            annotations=(),
            source="fake",
        )


class FakeAdvisor:
    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.calls = 0
        self.called = threading.Event()

    def recommend(self, state, *, request_id=""):
        self.calls += 1
        call = self.calls
        self.called.set()
        if call == 1 and self.gate is not None:
            assert self.gate.wait(2)
        return LocalAdvice(
            strategy="fake",
            cards=("2S",),
            play_type="Single",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.5,
            request_id=request_id,
            engine_input={"request_id": request_id, "legal_actions": [["Single"]]},
            timings={"agent_step": 1.0},
        )


class FailingAdvisor:
    def recommend(self, _state, *, request_id=""):
        raise RuntimeError(f"model failed: {request_id}")


class SuitAwareAdvisor(FakeAdvisor):
    def recommend(self, state, *, request_id=""):
        self.calls += 1
        self.called.set()
        observed = state.play_history[-1].cards
        cards = ("2S",) if observed == ("8S",) else ("2H",)
        return LocalAdvice(
            strategy="suit-aware",
            cards=cards,
            play_type="Single",
            is_pass=False,
            state_revision=state.revision,
            elapsed_ms=1.0,
            request_id=request_id,
            engine_input={"request_id": request_id},
            timings={},
        )


def _build(tmp_path, advisor, *, on_update=None):
    store = LiveSessionStore(tmp_path / "profiles", "tencent_daguandan", session_id="advice")
    store.start(
        {
            "application_version": "test",
            "configuration_hash": "config",
            "template_manifest_hash": "templates",
            "target_fps": 10,
            "codec": "MJPG",
        }
    )
    orchestrator = LiveOrchestrator(
        reducer=LiveReducer("advice"),
        store=store,
        recorder=SessionRecorder(store.directory, size=(64, 32), fps=10),
        recognition_service=LeftPlayRecognition(),
        advisor=advisor,
        settle_ms=100,
        burst_sample_interval_ms=50,
        minimum_free_bytes=0,
        on_update=on_update,
    )
    orchestrator.start(
        round_level="2",
        hand=HAND,
        lead_player="left",
        monotonic_ms=0,
    )
    return orchestrator


def _commit_left_action(orchestrator):
    frame = np.zeros((32, 64, 3), np.uint8)
    for index in range(5):
        timestamp = 100 + index * 100
        orchestrator.ingest_frame(
            frame,
            monotonic_ms=timestamp,
            wall_time=f"t{index}",
            metrics=ZoneFrameMetrics(
                monotonic_ms=timestamp,
                occupied=True,
                motion_score=0.2 if index == 0 else 0.001,
                pass_visible=False,
                effect_visible=False,
            ),
        )
        if orchestrator.snapshot.current_player != "left":
            break


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_advice_starts_when_reducer_predicts_self_before_timer(tmp_path):
    advisor = FakeAdvisor()
    orchestrator = _build(tmp_path, advisor)

    _commit_left_action(orchestrator)
    assert advisor.called.wait(2)
    _wait_until(lambda: orchestrator.latest_advice is not None and orchestrator.latest_advice.status == "ready")

    assert advisor.calls == 1
    assert orchestrator.snapshot.current_player == "self"
    assert orchestrator.latest_advice.visible is False

    orchestrator.ingest_fast_signal(active_player="self")

    assert orchestrator.latest_advice.visible is True
    orchestrator.finish()


def test_ready_advice_notifies_ui_without_waiting_for_another_capture_frame(tmp_path):
    updates = []
    orchestrator = _build(tmp_path, FakeAdvisor(), on_update=updates.append)

    _commit_left_action(orchestrator)
    _wait_until(
        lambda: any(
            update.advice is not None and update.advice.status == "ready"
            for update in updates
        )
    )

    latest = updates[-1]
    assert latest.advice is not None
    assert latest.advice.status == "ready"
    assert latest.advice.visible is False
    orchestrator.finish()


def test_trusted_action_uses_live_turn_transition_and_waits_for_advice(tmp_path):
    advisor = FakeAdvisor()
    orchestrator = _build(tmp_path, advisor)

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("7S", "7H"),
        is_pass=False,
        monotonic_ms=100,
        evidence_refs=("TRUTH-000001",),
    )

    assert update.event is not None
    assert update.event.event_type == "player_played"
    assert update.event.source == "trusted_log_replay"
    assert update.event.evidence_refs == ("TRUTH-000001",)
    assert orchestrator.snapshot.current_player == "self"
    assert update.advice is not None
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)
    assert advice is not None
    assert advice.status == "ready"
    assert advice.advice is not None
    assert advisor.calls == 1
    orchestrator.finish()


def test_advice_expands_occluded_suit_only_in_temporary_variants(tmp_path):
    advisor = SuitAwareAdvisor()
    orchestrator = _build(tmp_path, advisor)

    update = orchestrator.commit_trusted_action(
        actor="left",
        cards=("8?",),
        suit_options=(("S", "C"),),
        is_pass=False,
        monotonic_ms=100,
    )

    assert update.event is not None
    assert update.event.payload["cards"] == ["8?"]
    assert update.event.payload["suit_options"] == [["S", "C"]]
    advice = orchestrator.wait_for_advice(update.advice.key, timeout=2.0)

    assert advice is not None
    assert advice.status == "ready"
    assert advice.suit_uncertain is True
    assert advice.variant_count == 2
    assert advice.advice_agrees_across_variants is False
    assert advisor.calls == 2
    assert orchestrator.snapshot.play_history[-1].cards == ("8?",)
    orchestrator.finish()


def test_corrected_state_marks_inflight_advice_stale(tmp_path):
    gate = threading.Event()
    advisor = FakeAdvisor(gate)
    orchestrator = _build(tmp_path, advisor)
    _commit_left_action(orchestrator)
    assert advisor.called.wait(2)
    request = orchestrator.latest_advice.key

    orchestrator.correct_latest(cards=("8S",), is_pass=False)
    gate.set()

    _wait_until(
        lambda: any(
            event.event_type == "advice_stale"
            and event.payload.get("request_id") == request.request_id
            for event in orchestrator.events
        )
    )
    assert advisor.calls >= 1
    orchestrator.finish()


def test_advisor_failure_incident_contains_reproducible_engine_input(tmp_path):
    orchestrator = _build(tmp_path, FailingAdvisor())
    _commit_left_action(orchestrator)
    _wait_until(
        lambda: orchestrator.latest_advice is not None
        and orchestrator.latest_advice.status == "failed"
    )

    incident = next(orchestrator.store.incidents_directory.iterdir())
    engine_input = incident / "engine_input.json"

    assert engine_input.is_file()
    assert "project_snapshot" in engine_input.read_text("utf-8")
    orchestrator.finish()
