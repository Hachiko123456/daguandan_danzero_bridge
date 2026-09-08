"""Public capture-generation binding acceptance with real memory-only core.

No test assigns core._capture_generation. Missing binding API is an explicit
failure, never a skip. Publication is checked through real LiveUpdate output.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import daguandan_bridge.live.orchestrator as core_module
from daguandan_bridge.domain.recognition import FastSignalResult, OpeningSignal, PlayRegionResult
from daguandan_bridge.live.orchestrator import LiveOrchestrator, LiveUpdate
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class Timer:
    def __init__(self, interval, function):
        self.interval, self.function = interval, function
        self.cancelled, self.daemon = False, False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


class Recognition:
    def __init__(self):
        self.active_player = None
        self.controls_visible = False

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        return FastSignalResult(expected_player, self.active_player, False, self.controls_visible, False,
                                cannot_beat_visible=self.controls_visible, cannot_beat_confidence=.99,
                                cannot_beat_box=(10, 10, 20, 10) if self.controls_visible else None)

    def recognize_opening_signal(self, _frame):
        return OpeningSignal(False, None, None, False)

    def recognize_play_region(self, _frame, seat, *, wild_rank, allow_pass=True):
        return PlayRegionResult(seat, (), False, 0., (), (), source="generation-acceptance")


@pytest.fixture
def cores(tmp_path, monkeypatch):
    monkeypatch.setattr(core_module, "Timer", Timer)
    created = []

    def make(*, waiting_lead=False):
        clock, notices = Clock(), []
        profile = f"case{len(created)}"
        (tmp_path / profile).mkdir()
        store = InMemoryLiveSessionStore(tmp_path, profile)
        core = LiveOrchestrator(
            reducer=LiveReducer(store.session_id), store=store,
            recorder=InMemorySessionRecorder(store.directory), recognition_service=Recognition(),
            advisor=None, minimum_free_bytes=0, settle_ms=0,
            processing_clock_ms=clock, on_update=notices.append,
        )
        core.start(round_level="2", hand=HAND, lead_player=None if waiting_lead else "right", monotonic_ms=900)
        rig = SimpleNamespace(core=core, clock=clock, notices=notices)
        created.append(rig)
        return rig

    yield make
    for rig in created:
        rig.core.finish()
    assert not list(tmp_path.rglob("*.jsonl")), "memory-only acceptance unexpectedly persisted logs"


def bind(core, generation):
    method = getattr(core, "bind_capture_generation", None)
    assert callable(method), "core must expose bind_capture_generation(generation) before activate/bind/resume is testable"
    result = method(generation)
    assert isinstance(result, LiveUpdate), "binding returns the versioned LiveUpdate for controller publication"
    assert result.capture_generation == generation
    return result


def establish_history_pending_and_hint(rig):
    core = rig.core
    bind(core, 3)
    core.commit_trusted_action(actor="right", cards=("9C",), is_pass=False, monotonic_ms=950)
    frame = np.zeros((32, 64, 3), np.uint8)
    recognition = core.recognition_service
    # A foreign recovery belongs to a NEW local-control lifecycle, not two
    # persistent copies of the previous turn's buttons. Establish absence in
    # this exact response, then supply two distinct self/control captures.
    for timestamp, visible in ((1_000, False), (1_100, False), (1_200, True), (1_300, True)):
        recognition.active_player = "self" if visible else None
        recognition.controls_visible = visible
        rig.clock.now = timestamp
        update = core.analyze_frame(frame, monotonic_ms=timestamp,
                                    trace_context={"capture_generation": 3, "capture_seq": timestamp})
        if timestamp <= 1_200:
            assert not core._turn_ownership_window.turn_recovery_pending
    assert update.local_rule_hint is not None
    assert core.latest_advice is not None and core.latest_advice.status == "withheld"
    assert core._turn_ownership_window.turn_recovery_pending
    assert core._turn_evidence.retained_bytes > 0
    return update


def test_waiting_lead_public_binding_stamps_updates_and_notifications_before_first_frame(cores):
    rig = cores(waiting_lead=True)
    assert rig.core.status == "waiting_lead"
    bind(rig.core, 3)
    update = rig.core._update()
    assert isinstance(update, LiveUpdate) and update.capture_generation == 3
    assert update.status == "waiting_lead" and update.update_sequence > 0
    rig.core._notify_update_listener()
    assert rig.notices[-1].capture_generation == 3
    assert rig.notices[-1].update_sequence > update.update_sequence
    assert rig.core._capture_sequence == 0


def test_new_generation_revokes_visual_evidence_and_deadline_but_preserves_history_hold(cores):
    rig = cores()
    previous = establish_history_pending_and_hint(rig)
    core = rig.core
    before, old_timer = core.snapshot, core._deadline_timer
    assert old_timer is not None
    bind(core, 4)
    current = core._update()
    assert current.capture_generation == 4 and current.local_rule_hint is None
    assert old_timer.cancelled
    assert core._turn_evidence.retained_bytes == 0
    owner = core._turn_ownership_window
    assert owner is None or not owner.surface_baseline_captured
    assert owner is None or not owner.handoff_samples
    assert current.snapshot == before and current.snapshot.play_history == previous.snapshot.play_history
    assert core.latest_advice is not None and core.latest_advice.status == "withheld"
    assert not core.latest_advice.visible
    assert (owner is not None and owner.turn_recovery_pending) or core._advice_suspended_reason
    assert core._request_advice_if_needed(bypass_response_preflight=True) is None
    assert not core._pending_model_advice and not core._advice_requested_at_ms


def test_same_generation_bind_is_idempotent_for_valid_hint_history_and_deadline(cores):
    rig = cores()
    prior = establish_history_pending_and_hint(rig)
    core = rig.core
    before, owner, timer = core.snapshot, core._turn_ownership_window, core._deadline_timer
    retained = core._turn_evidence.retained_bytes
    bind(core, 3)
    current = core._update()
    assert current.capture_generation == 3
    assert current.local_rule_hint == prior.local_rule_hint
    assert core.snapshot == before and core._turn_ownership_window is owner
    assert core._turn_evidence.retained_bytes == retained
    assert core._deadline_timer is timer and not timer.cancelled
    assert core.latest_advice.status == "withheld"


@pytest.mark.parametrize("invalid", [2, 0, -1, True, False, 3.0, "4", None])
def test_old_or_invalid_generation_is_explicitly_rejected_without_identity_change(cores, invalid):
    rig = cores()
    prior = establish_history_pending_and_hint(rig)
    core = rig.core
    before, owner, timer = core.snapshot, core._turn_ownership_window, core._deadline_timer
    method = getattr(core, "bind_capture_generation", None)
    assert callable(method)
    with pytest.raises(ValueError):
        method(invalid)
    current = core._update()
    assert current.capture_generation == 3 and current.local_rule_hint == prior.local_rule_hint
    assert core.snapshot == before and core._turn_ownership_window is owner
    assert core._deadline_timer is timer and not timer.cancelled


def test_activate_bind_before_resume_returns_new_generation_without_old_hint(cores):
    rig = cores()
    establish_history_pending_and_hint(rig)
    core = rig.core
    before = core.snapshot
    assert core.pause().capture_generation == 3
    bind(core, 4)
    resumed = core.resume(monotonic_ms=1_400)
    assert resumed.status == "running" and resumed.capture_generation == 4
    assert resumed.local_rule_hint is None
    assert core.snapshot == before
    assert core.latest_advice is not None and core.latest_advice.status == "withheld"
    assert not core._pending_model_advice and not core._advice_requested_at_ms
