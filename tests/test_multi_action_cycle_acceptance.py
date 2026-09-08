"""Independent multi-non-PASS short-cycle acceptance, with real ROI freshness.

Historical evidence supplied by primary review (not copied into this repo):
frame1191/source58726953: right JS/head finish, opposite KH, left timer14;
frame1206/source58729062: opposite KH, left level6 6D, self normal controls10.
These are legal successive non-PASS plays KH -> 6D, not wind-catching PASSes.
The fixture deliberately keeps right unfinished; it does not invent a one-
card remaining hand or a finish result merely to recreate this mechanism.

Recognition decodes fixed small pixel codes FROM THE GIVEN FRAME. Old baseline
reads cannot accidentally return the new mutable observation. Each seat has a
separate16x32 ROI; production core measures real changes in these ROI pixels.
No snapshot mutation, freshness stub or user model is used.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from daguandan_bridge.domain.recognition import FastSignalResult, PlayRegionResult
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "10C")
SEATS = ("self", "right", "opposite", "left")
CODES = {(): 0, ("10C",): 30, ("JS",): 70, ("KH",): 110, ("6D",): 190, ("QH",): 150}
CARDS = {code: cards for cards, code in CODES.items()}


@dataclass
class Clock:
    now: int = 1_000

    def __call__(self):
        return self.now


class PixelRecognition:
    def __init__(self):
        self.active = "right"
        self.buttons = False
        self.calls = []

    @staticmethod
    def play_roi(frame, seat):
        index = SEATS.index(seat)
        return frame[:, index * 16:(index + 1) * 16]

    @classmethod
    def cards_from_frame(cls, frame, seat):
        roi = cls.play_roi(frame, seat)
        return CARDS.get(int(roi[12, 8, 0]), ())

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        return FastSignalResult(expected_player, self.active, False, self.buttons, False)

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_pass=True):
        cards = self.cards_from_frame(frame, seat)
        self.calls.append((seat, cards))
        return PlayRegionResult(seat, cards, False, .99 if cards else 0., (), (), source="per-frame-roi-pixel-code")

    @staticmethod
    def image(cards_by_seat):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        for index, seat in enumerate(SEATS):
            cards = tuple(cards_by_seat.get(seat, ()))
            frame[3:29, index * 16 + 2:index * 16 + 14] = CODES[cards]
        return frame


@pytest.fixture
def cycle_rig(tmp_path):
    clock, recognition = Clock(), PixelRecognition()
    store = LiveSessionStore(tmp_path, "acceptance", session_id="multi-nonpass", automatic_log_delivery_enabled=False)
    store.start({"application_version": "multi-action-acceptance"})
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
        recognition_service=recognition, advisor=None, minimum_free_bytes=0, settle_ms=0,
        burst_sample_interval_ms=50, processing_clock_ms=clock,
    )
    core.start(round_level="6", hand=HAND, lead_player="self", monotonic_ms=900)
    core.commit_trusted_action(actor="self", cards=("10C",), is_pass=False, monotonic_ms=950)

    class Rig:
        def step(self, timestamp, cards, *, active, buttons, generation=3):
            clock.now = timestamp
            recognition.active, recognition.buttons = active, buttons
            frame = recognition.image(cards)
            expected = core.snapshot.current_player
            occupied = bool(cards.get(expected, ()))
            return core.analyze_frame(
                frame, monotonic_ms=timestamp,
                metrics=ZoneFrameMetrics(timestamp, occupied, 0., False, False, content_changed=False),
                trace_context={"capture_generation": generation, "capture_seq": timestamp},
            )

        def baseline(self, *, external_count=2, old_left=(), buttons=False):
            if external_count == 2:
                core.commit_trusted_action(actor="right", cards=("JS",), is_pass=False, monotonic_ms=975)
            cards = {"self": ("10C",), "left": old_left}
            if external_count == 2:
                cards["right"] = ("JS",)
            self.step(1_000, cards, active=core.snapshot.current_player, buttons=buttons)
            return core.snapshot

    rig = Rig()
    rig.core, rig.recognition = core, recognition
    yield rig
    core.finish()


def current_surfaces(left=("6D",)):
    return {"self": ("10C",), "right": ("JS",), "opposite": ("KH",), "left": left}


def external_actions_after(core, revision):
    return [event for event in core.reducer.events
            if event.state_revision_after > revision and event.event_type in {"player_played", "player_passed"}]


@pytest.mark.parametrize("external_count", [2, 3])
def test_two_fresh_captures_recover_each_legal_nonpass_roi_up_to_three_seats(cycle_rig, external_count):
    rig = cycle_rig
    before = rig.baseline(external_count=external_count)
    assert before.current_player == ("opposite" if external_count == 2 else "right")
    first = rig.step(1_200, current_surfaces(), active="self", buttons=True)
    assert not external_actions_after(rig.core, before.revision), "one capture must not confirm the short chain"
    second = rig.step(1_300, current_surfaces(), active="self", buttons=True)
    assert first.snapshot.current_player != "self"
    assert second.snapshot.current_player == "self", "valid consecutive KH->level6 6D non-PASS chain was not recovered"
    actions = external_actions_after(rig.core, before.revision)
    expected = [("opposite", ("KH",)), ("left", ("6D",))]
    if external_count == 3:
        expected.insert(0, ("right", ("JS",)))
    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in actions] == expected
    assert all(event.event_type == "player_played" for event in actions)
    assert not second.snapshot.finished_seats  # Not a synthetic finish/wind-catch scenario.
    assert not rig.core._advice_requested_at_ms and not rig.core._pending_model_advice


def test_static_old_second_seat_high_card_does_not_count_as_new_action(cycle_rig):
    rig = cycle_rig
    before = rig.baseline(old_left=("6D",))
    for timestamp in (1_200, 1_300, 1_400):
        rig.step(timestamp, current_surfaces(), active="self", buttons=True)
    assert not any(event.actor == "left" for event in external_actions_after(rig.core, before.revision)), "unchanged old 6D ROI became a new left action"
    assert rig.core.snapshot.current_player != "self"


def test_second_nonpass_that_does_not_beat_first_is_never_committed(cycle_rig):
    rig = cycle_rig
    before = rig.baseline()
    for timestamp in (1_200, 1_300, 1_400):
        rig.step(timestamp, current_surfaces(left=("QH",)), active="self", buttons=True)
    assert not any(event.actor == "left" for event in external_actions_after(rig.core, before.revision)), "QH incorrectly accepted over KH"
    assert rig.core.snapshot.current_player != "self"


def test_duplicate_capture_is_not_two_votes_for_multi_action_cycle(cycle_rig):
    rig = cycle_rig
    before = rig.baseline()
    for _ in range(3):
        rig.step(1_200, current_surfaces(), active="self", buttons=True)
    assert not external_actions_after(rig.core, before.revision)
    assert rig.core.snapshot == before


def test_generation_change_cannot_reuse_previous_generation_first_vote_or_roi_baseline(cycle_rig):
    rig = cycle_rig
    before = rig.baseline()
    rig.step(1_200, current_surfaces(), active="self", buttons=True, generation=3)
    rig.step(1_300, current_surfaces(), active="self", buttons=True, generation=4)
    assert not external_actions_after(rig.core, before.revision), "old-generation first vote combined with new source"


def test_old_self_controls_without_current_visual_self_turn_cannot_authorize_cycle(cycle_rig):
    rig = cycle_rig
    before = rig.baseline(buttons=True)
    # Left is still acting. A stale local controls layer must not authorize
    # jumping two legal new foreign cards forward into a fake self turn.
    for timestamp in (1_200, 1_300, 1_400):
        rig.step(timestamp, current_surfaces(), active="left", buttons=True)
    assert rig.core.snapshot.current_player != "self"
    assert not any(event.actor == "left" for event in external_actions_after(rig.core, before.revision))
