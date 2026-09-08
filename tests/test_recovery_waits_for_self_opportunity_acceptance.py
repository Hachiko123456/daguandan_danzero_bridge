"""Foreign thinking time is not the local2s response budget.

Primary reclassified source action30: a genuinely missed right action starts
recovery, but the old blanket3000ms capture age failed it while active players
were still right/opposite/left. This is different from stale self controls.
Every frame/ROI below is current and per-seat, while old-owner return and
generation replacement remain explicit evidence-epoch invalidations.
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


SEATS = ("self", "right", "opposite", "left")
CODES = {(): 0, ("3C",): 30, ("4S",): 80, ("5D",): 120, ("7C",): 170, ("9S",): 220}
CARDS = {code: cards for cards, code in CODES.items()}
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
        self.active, self.controls, self.effect = "right", False, False

    @staticmethod
    def play_roi(image, seat):
        index = SEATS.index(seat)
        return image[:, index * 16:(index + 1) * 16]

    @staticmethod
    def image(surfaces):
        image = np.zeros((32, 64, 3), np.uint8)
        for index, seat in enumerate(SEATS):
            image[3:29, index * 16 + 2:index * 16 + 14] = CODES[tuple(surfaces.get(seat, ()))]
        return image

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        return FastSignalResult(expected_player, self.active, False, self.controls, self.effect)

    def recognize_play_region(self, image, seat, *, wild_rank, allow_pass=True):
        cards = CARDS.get(int(self.play_roi(image, seat)[12, 8, 0]), ())
        return PlayRegionResult(seat, cards, False, .97 if cards else 0., (), (), source="current-per-seat-clock-evidence")


@pytest.fixture
def waiting_rigs(tmp_path, monkeypatch):
    monkeypatch.setattr(core_module, "Timer", ManualTimer)
    made = []

    def make():
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
        core.commit_trusted_action(actor="self", is_pass=True, monotonic_ms=920)

        class Rig:
            def step(self, timestamp, surfaces, *, active, controls=False, effect=False, generation=3):
                clock.now = timestamp
                pixels.active, pixels.controls, pixels.effect = active, controls, effect
                expected = core.snapshot.current_player
                occupied = bool(surfaces.get(expected, ()))
                return core.analyze_frame(pixels.image(surfaces), monotonic_ms=timestamp,
                    metrics=ZoneFrameMetrics(timestamp, occupied, 0., False, effect, content_changed=occupied),
                    trace_context={"capture_generation": generation, "capture_seq": timestamp})

            def wait_foreign(self, seconds):
                before = core.snapshot
                states = []
                for elapsed in range(500, seconds * 1_000 + 1, 500):
                    states.append(self.step(1_400 + elapsed, {"left": ("3C",), "right": ("4S",)}, active="left"))
                return before, states

        rig = Rig()
        rig.core, rig.clock = core, clock
        baseline = {"left": ("3C",)}
        rig.step(1_000, baseline, active="right")
        rig.step(1_100, baseline, active="right")
        # Actual visible right play is hidden from confirmation by its effect;
        # then activity crosses opposite to left without enough action reads.
        rig.step(1_200, {"left": ("3C",), "right": ("4S",)}, active="right", effect=True)
        rig.step(1_300, {"left": ("3C",), "right": ("4S",)}, active="opposite", effect=True)
        rig.step(1_400, {"left": ("3C",), "right": ("4S",)}, active="left")
        assert core.snapshot.current_player == "right"
        assert core._turn_ownership_window.turn_recovery_pending, "fixture did not enter actual crossed-action recovery"
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.core.finish()


def failures(core):
    return [event for event in core.events if event.event_type == "advice_recovery_failed"]


def complete_surfaces():
    return {"right": ("4S",), "opposite": ("5D",), "left": ("7C",)}


@pytest.mark.parametrize("seconds", [8, 12])
def test_foreign_thinking_does_not_start_user_budget_or_fail_pending_history(waiting_rigs, seconds):
    rig = waiting_rigs()
    before, updates = rig.wait_foreign(seconds)
    assert not failures(rig.core), "foreign thinking time was counted against recovery capture lifetime"
    assert rig.core.snapshot == before
    assert rig.core._turn_ownership_window.turn_recovery_processing_started_ms is None
    assert rig.core._deadline_timer is None
    assert all(not update.block_reason for update in updates), "foreign-only wait emitted user-facing withheld status"
    assert not rig.core._pending_model_advice and not rig.core._advice_requested_at_ms


def test_complete_fresh_chain_recovers_when_self_finally_arrives_after_foreign_wait(waiting_rigs):
    rig = waiting_rigs()
    before, _ = rig.wait_foreign(8)
    assert not failures(rig.core)
    first = rig.step(9_600, complete_surfaces(), active="self", controls=True)
    assert first.snapshot == before
    final = rig.step(9_700, complete_surfaces(), active="self", controls=True)
    actions = [event for event in rig.core.reducer.events if event.state_revision_after > before.revision
               and event.event_type in {"player_played", "player_passed"}]
    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in actions] == [
        ("right", ("4S",)), ("opposite", ("5D",)), ("left", ("7C",)),
    ]
    assert final.snapshot.current_player == "self" and not failures(rig.core)


def test_incomplete_chain_starts_two_second_processing_budget_only_at_self_opportunity(waiting_rigs):
    rig = waiting_rigs()
    before, _ = rig.wait_foreign(12)
    assert not failures(rig.core)
    missing = {"right": ("4S",), "left": ("7C",)}  # Opposite evidence remains absent.
    rig.step(13_600, missing, active="self", controls=True)
    rig.step(13_700, missing, active="self", controls=True)
    owner = rig.core._turn_ownership_window
    assert owner.turn_recovery_processing_started_ms is not None
    assert owner.turn_recovery_processing_started_ms >= 13_600
    assert rig.core._deadline_timer is not None and rig.core._deadline_timer.interval == 2.0
    rig.clock.now = owner.turn_recovery_processing_started_ms + 1_999
    assert rig.core.poll_deadlines().block_reason != "turn_recovery_budget_exceeded"
    rig.clock.now += 2
    blocked = rig.core.poll_deadlines()
    assert blocked.block_reason == "turn_recovery_budget_exceeded"
    assert blocked.missing_player in {"right", "opposite", "left"}
    assert rig.core.snapshot == before


def test_owner_second_turn_and_new_surface_invalidate_old_missing_action_epoch(waiting_rigs):
    rig = waiting_rigs()
    before, _ = rig.wait_foreign(8)
    # Right returns after opposite/left activity and exposes a different card.
    rig.step(9_600, {"right": ("9S",)}, active="right")
    rig.step(9_700, {"right": ("9S",)}, active="right")
    assert rig.core.snapshot == before
    assert rig.core._turn_ownership_window.turn_recovery_failed
    assert any(event.payload.get("reason") == "owner_returned_before_missing_action" for event in failures(rig.core))
    rig.step(9_800, complete_surfaces(), active="self", controls=True)
    rig.step(9_900, complete_surfaces(), active="self", controls=True)
    assert rig.core.snapshot == before, "old right4S was reused after right's new turn9S"


def test_capture_generation_change_does_not_reuse_old_chain_vote_or_baseline(waiting_rigs):
    rig = waiting_rigs()
    before, _ = rig.wait_foreign(8)
    rig.step(9_600, complete_surfaces(), active="self", controls=True)
    rig.core.bind_capture_generation(4)
    result = rig.step(9_700, complete_surfaces(), active="self", controls=True, generation=4)
    assert result.capture_generation == 4
    assert rig.core.snapshot == before
    assert not rig.core._pending_model_advice
