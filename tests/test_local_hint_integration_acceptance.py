"""Independent production-path acceptance for local visual PASS hints.

Only recognition pixels and the clock are synthetic. The orchestrator,
tracker, update publication, canonical reducer/store and compact projection
are the real implementations. No Qt application, model job, sleep or video
capture is needed.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from daguandan_bridge.domain.recognition import FastSignalResult, PlayRegionResult
from daguandan_bridge.gui.compact_view_state import CompactUpdateGate, project_compact_view
from daguandan_bridge.live.local_rule_hint import LocalRuleHintTracker
from daguandan_bridge.live.orchestrator import LiveOrchestrator, LiveUpdate
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class MustNotRunModel:
    strategy_id = "acceptance-model-must-not-run"
    display_name = "验收模型"

    def __init__(self):
        self.calls = 0

    def recommend(self, _state, *, request_id):
        self.calls += 1
        raise AssertionError(f"bad-history model admission: {request_id}")


class ControlledRecognition:
    def __init__(self):
        self.visible = True
        self.active = "self"
        self.effect = False
        self.box = (410, 620, 88, 36)
        self.targeted_calls = 0
        self.before_targeted = None

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        return FastSignalResult(
            expected_player=expected_player, active_player=self.active,
            pass_visible=False, self_action_buttons_visible=self.visible,
            effect_visible=self.effect, cannot_beat_visible=self.visible,
            cannot_beat_confidence=.98, cannot_beat_box=self.box,
        )

    def recognize_play_region(self, _image, seat, *, wild_rank, allow_pass=True):
        self.targeted_calls += 1
        if self.before_targeted is not None:
            self.before_targeted(seat)
        return PlayRegionResult(
            player=seat, cards=(), is_pass=False, confidence=0.0,
            diagnostics=("acceptance_no_card_evidence",), annotations=(), source="acceptance",
        )


@pytest.fixture
def hint_rig(tmp_path):
    clock, recognition, model, published = Clock(), ControlledRecognition(), MustNotRunModel(), []
    store = LiveSessionStore(tmp_path / "profiles", "tencent_daguandan", session_id="hint-integration", automatic_log_delivery_enabled=False)
    store.start({"application_version": "acceptance", "target_fps": 10, "codec": "MJPG"})
    recorder = SessionRecorder(store.directory, size=(1280, 720), fps=10)
    orchestrator = LiveOrchestrator(
        reducer=LiveReducer("hint-integration"), store=store, recorder=recorder,
        recognition_service=recognition, advisor=model, settle_ms=0,
        burst_sample_interval_ms=50, minimum_free_bytes=0,
        processing_clock_ms=clock, on_update=published.append,
    )
    orchestrator.start(round_level="2", hand=HAND, lead_player="right", monotonic_ms=900)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert isinstance(orchestrator._local_hint_tracker, LocalRuleHintTracker)

    class Rig:
        def step(self, captured_ms, *, generation=2, occupied=False, now_ms=None):
            clock.now = captured_ms if now_ms is None else now_ms
            return orchestrator.analyze_frame(
                frame, monotonic_ms=captured_ms,
                metrics=ZoneFrameMetrics(captured_ms, occupied, 0.0, False, recognition.effect, content_changed=occupied),
                trace_context={"capture_generation": generation, "capture_seq": captured_ms, "captured_ms": captured_ms},
            )

        def fail_history_recovery(self):
            # Put the real owner window through its real failure transition;
            # never inject a ready hint or alter the canonical actor/history.
            owner = orchestrator._ensure_turn_ownership_window()
            assert owner.expected_player == "right"
            owner.active_player = "self"
            owner.turn_recovery_pending = True
            owner.turn_recovery_detected_ms = 950
            orchestrator._fail_turn_recovery(owner, "acceptance_missing_right_lead")
            assert orchestrator.latest_advice.withhold_reason == "turn_recovery_budget_exceeded"

    rig = Rig()
    rig.clock, rig.recognition, rig.model = clock, recognition, model
    rig.orchestrator, rig.published, rig.frame = orchestrator, published, frame
    yield rig
    orchestrator.finish()


def assert_no_model_or_history_change(rig, before):
    assert rig.orchestrator.snapshot == before
    assert rig.orchestrator.snapshot.revision == before.revision
    assert rig.orchestrator.snapshot.play_history == before.play_history
    assert rig.model.calls == 0
    assert not rig.orchestrator._advice_requested_at_ms
    assert not rig.orchestrator._pending_model_advice


def test_real_failed_right_lead_history_still_publishes_local_pass_without_canonical_changes(hint_rig):
    rig = hint_rig
    rig.fail_history_recovery()
    before = rig.orchestrator.snapshot
    assert before.current_player == "right" and not before.trick_plays
    first = rig.step(1_000, generation=3)
    second = rig.step(1_100, generation=3)
    assert isinstance(first, LiveUpdate) and isinstance(second, LiveUpdate)
    assert first.local_rule_hint is None
    assert second.local_rule_hint is not None
    assert second.capture_generation == 3
    assert second.local_rule_hint.control_box == (410, 620, 88, 36)
    assert second.advice.status == "withheld" and second.advice.withhold_reason == "turn_recovery_budget_exceeded"
    assert project_compact_view(second, now_ms=rig.clock.now).title == "不出"
    assert any(item.local_rule_hint is not None for item in rig.published)
    assert rig.recognition.targeted_calls == 0  # Failed history cannot keep heavy old-action scans alive.
    assert_no_model_or_history_change(rig, before)


def test_real_frame_dimensions_reject_out_of_frame_button(hint_rig):
    rig = hint_rig
    rig.fail_history_recovery()
    before = rig.orchestrator.snapshot
    rig.recognition.box = (1_250, 620, 88, 36)  # Extends beyond actual width1280.
    assert rig.step(1_000).local_rule_hint is None
    assert rig.step(1_100).local_rule_hint is None
    assert not any(item.local_rule_hint is not None for item in rig.published)
    assert_no_model_or_history_change(rig, before)


def test_duplicate_or_stale_captures_do_not_confirm_production_hint(hint_rig):
    rig = hint_rig
    rig.fail_history_recovery()
    before = rig.orchestrator.snapshot
    assert rig.step(1_000, generation=2).local_rule_hint is None
    assert rig.step(1_000, generation=2, now_ms=1_100).local_rule_hint is None
    assert rig.step(1_100, generation=2, now_ms=1_650).local_rule_hint is None
    assert rig.step(1_700, generation=2).local_rule_hint is None
    assert rig.step(1_800, generation=2).local_rule_hint is not None
    assert_no_model_or_history_change(rig, before)


def test_production_hint_revoked_by_button_disappearance_and_generation_change(hint_rig):
    rig = hint_rig
    rig.fail_history_recovery()
    before = rig.orchestrator.snapshot
    rig.step(1_000, generation=2)
    assert rig.step(1_100, generation=2).local_rule_hint is not None
    assert rig.step(1_200, generation=3).local_rule_hint is None
    assert rig.step(1_300, generation=3).local_rule_hint is not None
    rig.recognition.visible = False
    disappeared = rig.step(1_400, generation=3)
    assert disappeared.local_rule_hint is None
    assert rig.published[-1].local_rule_hint is None
    rig.recognition.visible = True
    assert rig.step(1_500, generation=3).local_rule_hint is None
    assert rig.step(1_600, generation=3).local_rule_hint is not None
    assert_no_model_or_history_change(rig, before)


def test_production_pause_resume_and_finalization_cannot_keep_or_reuse_hint(hint_rig):
    rig = hint_rig
    rig.fail_history_recovery()
    before = rig.orchestrator.snapshot
    rig.step(1_000, generation=2)
    assert rig.step(1_100, generation=2).local_rule_hint is not None
    paused = rig.orchestrator.pause()
    assert paused.status == "paused" and paused.local_rule_hint is None
    resumed = rig.orchestrator.resume(monotonic_ms=1_150)
    assert resumed.local_rule_hint is None
    assert rig.step(1_200, generation=3).local_rule_hint is None
    assert rig.step(1_300, generation=3).local_rule_hint is not None
    assert rig.orchestrator.begin_finalizing().local_rule_hint is None
    assert_no_model_or_history_change(rig, before)


def test_fast_hint_is_published_before_slow_targeted_recognition_in_same_capture(hint_rig):
    rig = hint_rig
    # Live canonical right lead has not been read. The second frame both
    # confirms the local control and requests a card-region read; the callback
    # must already have the hint before that potentially expensive read starts.
    before = rig.orchestrator.snapshot
    rig.step(1_000, generation=4, occupied=True)
    rig.published.clear()
    observed_before_slow = []

    def before_targeted(_seat):
        current_hints = [item for item in rig.published if item.local_rule_hint is not None]
        assert current_hints, "No local hint publication before targeted card recognition"
        hint_update = current_hints[-1]
        assert hint_update.capture_generation == 4
        assert hint_update.local_rule_hint.captured_ms == 1_100
        assert project_compact_view(hint_update, now_ms=rig.clock.now).title == "不出"
        observed_before_slow.append(hint_update.update_sequence)
        rig.clock.now += 800  # Deterministic slow recognizer; no real sleeping.

    rig.recognition.before_targeted = before_targeted
    rig.step(1_100, generation=4, occupied=True)
    assert observed_before_slow, "The fixture did not exercise a targeted recognizer"
    assert_no_model_or_history_change(rig, before)


def test_real_published_sequences_reject_late_same_state_hold_and_old_hint_generation(hint_rig):
    rig = hint_rig
    rig.fail_history_recovery()
    before = rig.orchestrator.snapshot
    old_hold = rig.step(1_000, generation=5)
    new_hint = rig.step(1_100, generation=5)
    assert new_hint.local_rule_hint is not None
    assert new_hint.update_sequence > old_hold.update_sequence > 0
    assert new_hint.snapshot.turn_id == old_hold.snapshot.turn_id
    assert new_hint.snapshot.revision == old_hold.snapshot.revision
    gate = CompactUpdateGate()
    assert gate.accept(new_hint)
    assert not gate.accept(old_hold)
    revoked = rig.step(1_200, generation=6)
    assert revoked.local_rule_hint is None and gate.accept(revoked)
    assert not gate.accept(new_hint)
    published_sequences = [item.update_sequence for item in rig.published]
    assert published_sequences == sorted(set(published_sequences))
    assert all(isinstance(item, LiveUpdate) and item.capture_generation >= 2 for item in rig.published)
    assert_no_model_or_history_change(rig, before)
