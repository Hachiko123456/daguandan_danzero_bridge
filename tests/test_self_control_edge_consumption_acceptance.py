"""The just-executed local turn consumes its own observed control edge.

Unlike the original foreign-turn fixture, this creates a genuine hidden->
visible two-capture control edge WHILE SELF IS CURRENT, then executes the
local action. Its stale visible controls must not be treated as the NEXT self
turn. External PASSes deliberately share response identity; local execution
is the separate consumption boundary under test.
"""
from dataclasses import dataclass

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


class Timer:
    def __init__(self, interval, function):
        self.interval, self.function, self.daemon = interval, function, True
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


class Pixels:
    def __init__(self):
        self.active, self.controls = "self", False

    @staticmethod
    def play_roi(image, seat):
        index = ("self", "right", "opposite", "left").index(seat)
        return image[:, index * 16:(index + 1) * 16]

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        return FastSignalResult(expected_player, self.active, False, self.controls, False)

    def recognize_play_region(self, image, seat, *, wild_rank, allow_pass=True):
        cards = ("5S",) if seat == "right" and int(image[12, 24, 0]) == 180 else ()
        return PlayRegionResult(seat, cards, False, .99 if cards else 0., (), (), source="edge-consumption-pixels")


@pytest.fixture
def edge_rigs(tmp_path, monkeypatch):
    monkeypatch.setattr(core_module, "Timer", Timer)
    made = []

    def make(*, self_pass=True):
        profile = f"profile{len(made)}"
        (tmp_path / profile).mkdir()
        store = InMemoryLiveSessionStore(tmp_path, profile)
        clock, pixels = Clock(), Pixels()
        core = LiveOrchestrator(
            reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
            recognition_service=pixels, advisor=None, minimum_free_bytes=0, settle_ms=0,
            burst_sample_interval_ms=50, processing_clock_ms=clock,
        )
        core.start(round_level="6", hand=HAND, lead_player="left", monotonic_ms=900)
        core.bind_capture_generation(3)
        core.commit_trusted_action(actor="left", cards=("3C",), is_pass=False, monotonic_ms=910)

        class Rig:
            def step(self, timestamp, *, active="self", controls=True, right_card=False, generation=3):
                clock.now = timestamp
                pixels.active, pixels.controls = active, controls
                image = np.zeros((32, 64, 3), np.uint8)
                if right_card:
                    image[3:29, 18:30] = 180
                return core.analyze_frame(image, monotonic_ms=timestamp,
                    metrics=ZoneFrameMetrics(timestamp, right_card and core.snapshot.current_player == "right", 0., False, False, content_changed=right_card),
                    trace_context={"capture_generation": generation, "capture_seq": timestamp})

        rig = Rig()
        rig.core, rig.clock = core, clock
        for timestamp, visible in ((1_000, False), (1_100, False), (1_200, True), (1_300, True)):
            rig.step(timestamp, controls=visible)
        assert core.snapshot.current_player == "self"
        # Read-only evidence check ensures the regression really covers a
        # previously formed edge, not the simpler always-visible fixture.
        assert core._response_controls_streak >= 2
        assert core._response_controls_edge_identity == core._response_identity()
        core.commit_trusted_action(actor="self", cards=() if self_pass else ("4C",),
                                   is_pass=self_pass, monotonic_ms=1_400)
        assert core.snapshot.current_player == "right"
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.core.finish()


def has_recovery(core):
    owner = core._turn_ownership_window
    return bool(owner and (owner.turn_recovery_pending or owner.turn_recovery_failed))


@pytest.mark.parametrize("self_pass", [True, False])
@pytest.mark.parametrize("active_pattern", [("self",), ("self", None)])
def test_executed_self_action_consumes_its_old_edge_and_right_can_think_seven_seconds(edge_rigs, self_pass, active_pattern):
    rig = edge_rigs(self_pass=self_pass)
    before = rig.core.snapshot
    observed_recovery = []
    for index, timestamp in enumerate(range(1_500, 8_501, 250)):
        rig.step(timestamp, active=active_pattern[index % len(active_pattern)])
        observed_recovery.append(has_recovery(rig.core))
    assert not any(observed_recovery), "the just-executed SELF control edge was reused as a new local turn"
    assert not any(event.event_type == "advice_recovery_failed" for event in rig.core.events)
    assert rig.core.snapshot == before
    for timestamp in (8_600, 8_700, 8_800, 8_900):
        rig.step(timestamp, active="right", controls=False, right_card=True)
        if rig.core.snapshot.current_player == "opposite":
            break
    assert rig.core.snapshot.current_player == "opposite"
    new_right = [event for event in rig.core.reducer.events if event.actor == "right"
                 and event.event_type == "player_played" and event.state_revision_after > before.revision]
    assert len(new_right) == 1 and tuple(new_right[0].payload["cards"]) == ("5S",)


@pytest.mark.parametrize("self_pass", [True, False])
def test_after_consumption_a_genuinely_new_control_lifecycle_can_still_request_recovery(edge_rigs, self_pass):
    rig = edge_rigs(self_pass=self_pass)
    before = rig.core.snapshot
    rig.step(1_500, active="self", controls=True)
    assert not has_recovery(rig.core)
    rig.step(1_600, active="right", controls=False)
    rig.step(1_700, active=None, controls=False)
    rig.step(1_800, active="self", controls=True)
    assert not has_recovery(rig.core), "first new capture alone must not start recovery"
    rig.step(1_900, active="self", controls=True)
    assert has_recovery(rig.core)
    assert rig.core._deadline_timer.interval == 2.0
    rig.clock.now = 3_901
    result = rig.core.poll_deadlines()
    assert result.block_reason == "turn_recovery_budget_exceeded"
    assert result.missing_player == "right" and rig.core.snapshot == before


def test_generation_change_after_local_execution_does_not_restore_consumed_edge(edge_rigs):
    rig = edge_rigs()
    rig.core.bind_capture_generation(4)
    before = rig.core.snapshot
    for timestamp in (1_500, 1_600, 1_700):
        rig.step(timestamp, active="self", controls=True, generation=4)
    assert not has_recovery(rig.core)
    assert rig.core.snapshot == before


def test_new_trick_never_restores_edge_belonging_to_executed_local_turn(edge_rigs):
    rig = edge_rigs()
    core = rig.core
    core.commit_trusted_action(actor="right", cards=("5S",), is_pass=False, monotonic_ms=1_450)
    core.commit_trusted_action(actor="opposite", is_pass=True, monotonic_ms=1_460)
    core.commit_trusted_action(actor="left", is_pass=True, monotonic_ms=1_470)
    core.commit_trusted_action(actor="self", is_pass=True, monotonic_ms=1_480)
    before = core.snapshot
    assert before.current_player == "right" and not before.trick_plays
    for timestamp in (1_500, 1_600, 1_700):
        rig.step(timestamp, active="self", controls=True)
    assert not has_recovery(core) and core.snapshot == before
