"""Terminal placements cannot erase the final two actual plays.

Primary's same-source window E2E lost opposite7C7S (.895) and left10C10D
when a second-place badge was accepted first. Source1498/58773343 shows those
two pairs and self10S; source1507 shows ranking animation. Complete shadow
has84 actions/healthy conservation. No screenshot/video or legacy timeline is
used as executable truth here.

The setup deals a real108-card ledger and submits only legal core actions:
right has actually played27 cards/head, left25/remains2, current opposite.
No remaining count/snapshot/finished set is manually overwritten. Final pair
recognition decodes64x32 independent seat ROI pixels across fresh captures.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from daguandan_bridge.domain.recognition import FastSignalResult, PlacementSignal, PlayRegionResult, RecognitionAnnotation
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.session_health import audit_session_health


SEATS = ("self", "right", "opposite", "left")
CODES = {(): 0, ("7C", "7S"): 100, ("10C", "10D"): 180, ("4C", "4D"): 140}
CARDS = {code: cards for cards, code in CODES.items()}


def actual_deal():
    deck = Counter({f"{rank}{suit}": 2 for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A") for suit in "CDHS"})
    deck.update({"small_joker": 2, "big_joker": 2})

    def take(cards):
        for card in cards:
            assert deck[card] > 0
            deck[card] -= 1
        return tuple(cards)

    left_final, left_last = take(("10C", "10D")), take(("2C",))
    right_final, right_first = take(("3C", "3D")), take(("3S",))
    opposite_final, own_pair = take(("7C", "7S")), take(("5C", "5D"))
    own_fill = []
    for card in reversed(tuple(deck)):
        while deck[card] and len(own_fill) < 25:
            own_fill.extend(take((card,)))

    def singles(count):
        selected = []
        for card in deck:
            if card == "6H":  # Keep wild-card verification out of setup.
                continue
            while deck[card] and len(selected) < count:
                selected.extend(take((card,)))
        assert len(selected) == count
        return tuple(selected)

    left_pre = singles(24) + left_last
    right_pre = right_first + singles(24)
    opposite_hand = opposite_final + tuple(deck.elements())
    hands = {"self": own_pair + tuple(own_fill), "right": right_pre + right_final,
             "left": left_pre + left_final, "opposite": opposite_hand}
    assert all(len(hand) == 27 for hand in hands.values())
    assert sum((Counter(hand) for hand in hands.values()), Counter()).total() == 108
    return hands, left_pre, right_pre, right_final, own_pair


@dataclass
class Clock:
    now: int = 100

    def __call__(self):
        return self.now


class Pixels:
    def __init__(self):
        self.active = "left"
        self.placement = "left"
        self.terminal = False

    @staticmethod
    def play_roi(frame, seat):
        index = SEATS.index(seat)
        return frame[:, index * 16:(index + 1) * 16]

    @staticmethod
    def image(opposite=(), left=()):
        frame = np.zeros((32, 64, 3), np.uint8)
        frame[3:28, 34:46] = CODES[tuple(opposite)]
        frame[3:28, 50:62] = CODES[tuple(left)]
        return frame

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        placement = (PlacementSignal(self.placement, "second", .99, "terminal-pixel-acceptance"),) if self.placement else ()
        return FastSignalResult(expected_player, self.active, False, False, False,
                                placements=placement, game_end_control="continue_game" if self.terminal else None)

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_pass=True):
        roi = self.play_roi(frame, seat)
        cards = CARDS.get(int(roi[12, 8, 0]), ())
        index = SEATS.index(seat)
        confidence = .895 if seat == "opposite" else .97
        annotations = (RecognitionAnnotation("pair", (index * 16 + 2, 3, 12, 25), confidence, "play"),) if cards else ()
        return PlayRegionResult(seat, cards, False, confidence if cards else 0., (), annotations, source="terminal-seat-pixels")


@pytest.fixture
def terminal_rig(tmp_path):
    (tmp_path / "profile").mkdir()
    store = InMemoryLiveSessionStore(tmp_path, "profile")
    clock, pixels = Clock(), Pixels()
    hands, left_pre, right_pre, right_final, own_pair = actual_deal()
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
        recognition_service=pixels, advisor=None, minimum_free_bytes=0, settle_ms=0,
        burst_sample_interval_ms=50, processing_clock_ms=clock,
    )
    core.start(round_level="6", hand=hands["self"], lead_player="left", monotonic_ms=clock.now)
    core.bind_capture_generation(3)

    def act(actor, cards=()):
        assert core.snapshot.current_player == actor
        clock.now += 10
        return core.commit_trusted_action(actor=actor, cards=tuple(cards), is_pass=not cards, monotonic_ms=clock.now)

    def pass_to(actor):
        for _ in range(4):
            current = core.snapshot.current_player
            if current == actor:
                return
            act(current)
        assert core.snapshot.current_player == actor

    for card in left_pre[:-1]:
        act("left", (card,))
        pass_to("left")
    act("left", left_pre[-1:])  # 2C final setup single; left really has2 left.
    act("self")
    act("right", right_pre[:1])  # 3S beats2C and takes control.
    pass_to("right")
    for card in right_pre[1:]:
        act("right", (card,))
        pass_to("right")
    act("right", right_final)  # Final low pair; right genuinely finishes27.
    act("opposite")
    act("left")
    act("self", own_pair)  # 5-pair over right3-pair; expected opposite now.
    assert core.snapshot.current_player == "opposite"
    assert core.snapshot.finished_seats == frozenset({"right"})
    assert core.snapshot.remaining_cards == {"self": 25, "right": 0, "opposite": 27, "left": 2}
    assert audit_session_health(core.snapshot, core.events)["status"] == "PASS"

    class Rig:
        def step(self, timestamp, *, opposite=(), left=(), placement="left", active="left", generation=3, terminal=False):
            clock.now = timestamp
            pixels.active, pixels.placement, pixels.terminal = active, placement, terminal
            expected = core.snapshot.current_player
            occupied = bool(opposite) if expected == "opposite" else bool(left) if expected == "left" else False
            return core.analyze_frame(
                pixels.image(opposite, left), monotonic_ms=timestamp,
                metrics=ZoneFrameMetrics(timestamp, occupied, 0., False, False, content_changed=occupied),
                trace_context={"capture_generation": generation, "capture_seq": timestamp},
            )

    rig = Rig()
    rig.core = core
    rig.step(10_000, placement=None, active="opposite")
    yield rig
    core.finish()


def new_actions(core, revision):
    return [event for event in core.reducer.events if event.state_revision_after > revision
            and event.event_type in {"player_played", "player_passed"}]


@pytest.mark.parametrize("badge_earlier", [False, True])
def test_final_two_actions_precede_second_place_and_finish_trigger_matches_left_action(terminal_rig, badge_earlier):
    rig = terminal_rig
    before = rig.core.snapshot
    if badge_earlier:
        rig.step(10_050, placement="left")
        assert rig.core.snapshot.remaining_cards["left"] == 2
    rig.step(10_100, opposite=("7C", "7S"), left=("10C", "10D"))
    # A lone capture cannot be laundered by a pre-existing placement streak.
    assert not new_actions(rig.core, before.revision)
    rig.step(10_200, opposite=("7C", "7S"), left=("10C", "10D"))
    actions = new_actions(rig.core, before.revision)
    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in actions] == [
        ("opposite", ("7C", "7S")), ("left", ("10C", "10D")),
    ], "placement preempted the two actual terminal plays"
    finishes = [event for event in rig.core.events if event.event_type == "player_finished" and event.actor == "left"]
    assert len(finishes) == 1
    assert finishes[0].payload.get("trigger_action_event_id") == actions[-1].event_id
    assert finishes[0].seq > actions[-1].seq
    assert rig.core.snapshot.remaining_cards["left"] == 0
    assert rig.core.snapshot.remaining_cards["opposite"] == 25
    assert audit_session_health(rig.core.snapshot, rig.core.events)["status"] == "PASS"


@pytest.mark.parametrize("case", ["placement_only", "illegal_left", "duplicate_capture", "old_generation", "old_static_left", "wrong_seat", "immediate_settlement"])
def test_unproven_terminal_evidence_never_manufactures_actions_or_zero_remaining(terminal_rig, case):
    rig = terminal_rig
    before = rig.core.snapshot
    if case == "old_static_left":
        # Retire the old source and build its new baseline with the static
        # high pair already visible; it cannot count as a fresh final action.
        rig.core.bind_capture_generation(4)
        rig.step(10_050, left=("10C", "10D"), placement=None, active="opposite", generation=4)
    for index in range(2):
        timestamp = 10_100 if case == "duplicate_capture" else 10_100 + index * 100
        generation = 4 if case == "old_static_left" else 3 + index if case == "old_generation" else 3
        rig.step(
            timestamp, opposite=() if case == "placement_only" else ("7C", "7S"),
            left=() if case == "placement_only" else ("4C", "4D") if case == "illegal_left" else ("10C", "10D"),
            placement="opposite" if case == "wrong_seat" else "left", generation=generation,
            terminal=case == "immediate_settlement",
        )
    actions = new_actions(rig.core, before.revision)
    assert not any(event.actor == "left" for event in actions), f"{case} manufactured the final left action"
    assert rig.core.snapshot.remaining_cards["left"] == 2, f"{case} hid missing terminal cards by setting left remaining0"
    assert "left" not in rig.core.snapshot.finished_seats
    assert not any(event.event_type == "player_finished" and event.actor == "left" for event in rig.core.events)
    if case == "wrong_seat":
        assert rig.core.snapshot.remaining_cards["opposite"] > 0
        assert "opposite" not in rig.core.snapshot.finished_seats
