"""A foreign15s turn is not a2s recovery window because old controls linger.

Primary frozen-bundle trace: self PASS -> expected right, stale self controls/
unknown active started generic recovery46568078, capture evidence expired at
46571109 while right still had a normal timer. The whole round then froze.
Use actual legal core actions, tiny pixel-derived right cards, and a fake
processing clock. Never sleep, change global motion thresholds or run a model.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import daguandan_bridge.live.orchestrator as core_module
from daguandan_bridge.domain.recognition import FastSignalResult, PlayRegionResult
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "9C")


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class ManualTimer:
    def __init__(self, interval, function):
        self.interval, self.function, self.daemon = interval, function, True
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


class Pixels:
    def __init__(self):
        self.active, self.controls, self.markers = "self", True, set()

    @staticmethod
    def play_roi(frame, seat):
        index = {"self": 0, "right": 1, "opposite": 2, "left": 3}[seat]
        return frame[:, index * 16:(index + 1) * 16]

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        return FastSignalResult(expected_player, self.active, expected_player in self.markers, self.controls, False,
                                pass_marker_player=expected_player if expected_player in self.markers else None,
                                pass_marker_players=tuple(self.markers))

    def recognize_play_region(self, image, seat, *, wild_rank, allow_pass=True):
        cards = ("4S",) if seat == "right" and int(image[12, 24, 0]) == 180 else ()
        is_pass = not cards and seat in self.markers and allow_pass
        return PlayRegionResult(seat, cards, is_pass, .99 if cards or is_pass else 0., (), (), source="foreign-turn-pixels")


@pytest.fixture
def foreign_rigs(tmp_path, monkeypatch):
    monkeypatch.setattr(core_module, "Timer", ManualTimer)
    made = []

    def make(*, left_handoff=False):
        profile = f"profile{len(made)}"
        (tmp_path / profile).mkdir()
        store = InMemoryLiveSessionStore(tmp_path, profile)
        clock, pixels = Clock(), Pixels()
        core = LiveOrchestrator(
            reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
            recognition_service=pixels, advisor=None, minimum_free_bytes=0, settle_ms=0,
            burst_sample_interval_ms=50, processing_clock_ms=clock,
        )
        lead = "right" if left_handoff else "left"
        core.start(round_level="6", hand=HAND, lead_player=lead, monotonic_ms=900)
        core.bind_capture_generation(3)
        core.commit_trusted_action(actor=lead, cards=("3C",), is_pass=False, monotonic_ms=910)
        if left_handoff:
            core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=920)

        class Rig:
            def step(self, timestamp, *, active, controls, right_card=False, markers=(), generation=3):
                clock.now = timestamp
                pixels.active, pixels.controls, pixels.markers = active, controls, set(markers)
                image = np.zeros((32, 64, 3), np.uint8)
                if right_card:
                    image[3:29, 18:30] = 180
                expected = core.snapshot.current_player
                return core.analyze_frame(
                    image, monotonic_ms=timestamp,
                    metrics=ZoneFrameMetrics(timestamp, right_card if expected == "right" else False, 0., expected in markers, False, content_changed=right_card),
                    trace_context={"capture_generation": generation, "capture_seq": timestamp},
                )

        rig = Rig()
        rig.core, rig.clock = core, clock
        if not left_handoff:
            rig.step(1_000, active="self", controls=True)
            rig.step(1_100, active="self", controls=True)
            core.commit_trusted_action(actor="self", is_pass=True, monotonic_ms=1_200)
            assert core.snapshot.current_player == "right"
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.core.finish()


def recovery_started(core):
    window = core._turn_ownership_window
    return bool(window is not None and (window.turn_recovery_pending or window.turn_recovery_failed))


def recovery_failure_events(core):
    return [event for event in core.events if event.event_type == "advice_recovery_failed"]


@pytest.mark.parametrize("active_pattern", [("self", None), (None,), ("self",)])
def test_seven_second_foreign_wait_with_lingering_controls_is_not_generic_recovery(foreign_rigs, active_pattern):
    rig = foreign_rigs()
    before = rig.core.snapshot
    started = []
    for index, timestamp in enumerate(range(1_300, 8_201, 300)):
        rig.step(timestamp, active=active_pattern[index % len(active_pattern)], controls=True)
        started.append(recovery_started(rig.core))
    assert not any(started), "stale local controls stole the normal right-player action window"
    assert not recovery_failure_events(rig.core)
    assert rig.core.snapshot == before
    # The foreign player is allowed to think7s and then play normally.
    for timestamp in (8_300, 8_400, 8_500, 8_600):
        rig.step(timestamp, active="right", controls=False, right_card=True)
        if rig.core.snapshot.current_player == "opposite":
            break
    actions = [event for event in rig.core.reducer.events if event.actor == "right" and event.event_type == "player_played"
               and event.state_revision_after > before.revision]
    assert len(actions) == 1 and tuple(actions[0].payload["cards"]) == ("4S",)
    assert rig.core.snapshot.current_player == "opposite"


def test_new_same_identity_hidden_visible_self_controls_really_start_bounded_recovery(foreign_rigs):
    rig = foreign_rigs()
    before = rig.core.snapshot
    rig.step(1_300, active="right", controls=False)
    rig.step(1_400, active="right", controls=False)
    rig.step(1_500, active="self", controls=True)
    assert not recovery_started(rig.core), "one new control capture is not enough to claim a new self turn"
    rig.step(1_600, active="self", controls=True)
    assert recovery_started(rig.core)
    assert rig.core._deadline_timer is not None and rig.core._deadline_timer.interval == 2.0
    rig.clock.now = 3_601
    expired = rig.core.poll_deadlines()
    assert expired.block_reason == "turn_recovery_budget_exceeded"
    assert expired.missing_player == "right"
    assert rig.core.snapshot == before


def test_new_generation_cannot_reuse_old_controls_edge(foreign_rigs):
    rig = foreign_rigs()
    rig.step(1_300, active="right", controls=False)
    rig.step(1_400, active="self", controls=True)
    rig.core.bind_capture_generation(4)
    for timestamp in (1_500, 1_600, 1_700):
        rig.step(timestamp, active="self", controls=True, generation=4)
    assert not recovery_started(rig.core), "old-generation half-edge was reused as a current self turn"


def test_new_trick_cannot_reuse_previous_table_controls_edge(foreign_rigs):
    rig = foreign_rigs()
    rig.step(1_300, active="right", controls=False)
    rig.step(1_400, active="self", controls=True)
    rig.core.commit_trusted_action(actor="right", cards=("4S",), is_pass=False, monotonic_ms=1_450)
    rig.core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=1_460)
    rig.core.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=1_470)
    rig.core.commit_trusted_action(actor="self", is_pass=True, monotonic_ms=1_480)
    before = rig.core.snapshot
    assert before.current_player == "right" and not before.trick_plays
    for timestamp in (1_500, 1_600, 1_700):
        rig.step(timestamp, active="self", controls=True)
    assert not recovery_started(rig.core), "previous trick control edge polluted the new right lead"
    assert rig.core.snapshot == before


def test_fresh_left_pass_to_self_keeps_normal_fast_handoff(foreign_rigs):
    rig = foreign_rigs(left_handoff=True)
    rig.step(1_000, active="left", controls=False)
    rig.step(1_100, active="left", controls=False)
    before = rig.core.snapshot
    rig.step(1_200, active="self", controls=True, markers=("left",))
    rig.step(1_300, active="self", controls=True, markers=("left",))
    assert rig.core.snapshot.current_player == "self"
    actions = [event for event in rig.core.reducer.events if event.state_revision_after > before.revision
               and event.event_type in {"player_played", "player_passed"}]
    assert [(event.actor, event.event_type) for event in actions] == [("left", "player_passed")]
    assert not recovery_started(rig.core) and not recovery_failure_events(rig.core)
