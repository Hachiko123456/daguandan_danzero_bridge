"""Seat-rotated terminal reconciliation: no fixed actor/count special case.

Reuse only the independent108-card deal generator, not a snapshot or production
helper. Every count/finish comes from legal core actions. RotationA has left
head and self->right terminal pair chain. RotationB has opposite head and
left->self chain, with the first actor having already played an extra pair:
its count after the final visible pair is23, not a hard-coded25.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from test_terminal_action_reconciliation_acceptance import actual_deal
from daguandan_bridge.domain.recognition import FastSignalResult, PlacementSignal, PlayRegionResult, RecognitionAnnotation
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.session_health import audit_session_health


SEATS = ("self", "right", "opposite", "left")
CODES = {(): 0, ("7C", "7S"): 100, ("10C", "10D"): 180}
CARDS = {value: cards for cards, value in CODES.items()}


@dataclass
class Clock:
    now: int = 100

    def __call__(self):
        return self.now


class Pixels:
    def __init__(self):
        self.active, self.placement, self.label = None, None, "second"

    @staticmethod
    def play_roi(frame, seat):
        index = SEATS.index(seat)
        return frame[:, index * 16:(index + 1) * 16]

    @staticmethod
    def image(surfaces):
        frame = np.zeros((32, 64, 3), np.uint8)
        for index, seat in enumerate(SEATS):
            frame[3:28, index * 16 + 2:index * 16 + 14] = CODES[tuple(surfaces.get(seat, ()))]
        return frame

    def recognize_fast_signals(self, _frame, expected_player, *, allow_pass=True):
        placements = (PlacementSignal(self.placement, self.label, .99, "rotated-terminal"),) if self.placement else ()
        return FastSignalResult(expected_player, self.active, False, False, False, placements=placements)

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_pass=True):
        cards = CARDS.get(int(self.play_roi(frame, seat)[12, 8, 0]), ())
        index = SEATS.index(seat)
        annotations = (RecognitionAnnotation("pair", (index * 16 + 2, 3, 12, 25), .96, "play"),) if cards else ()
        return PlayRegionResult(seat, cards, False, .96 if cards else 0., (), annotations, source="rotated-seat-pixels")


@pytest.fixture
def rotated_cores(tmp_path):
    created = []

    def make(*, rotation, first_actor_preplayed=False, final_cards_remaining=2):
        base_hands, base_left_pre, right_pre, right_final, own_pair = actual_deal()
        hands = {seat: list(hand) for seat, hand in base_hands.items()}
        left_pre = list(base_left_pre)
        total_before = sum((Counter(hand) for hand in hands.values()), Counter())
        if first_actor_preplayed:
            # Exchange real card ownership before the deal, preserving108
            # physical cards, so opposite can lead2H2H then left overcalls3H3H.
            donated = [card for card in hands["opposite"] if card not in {"7C", "7S", "6H"}][:2]
            assert len(donated) == 2 and left_pre.count("2H") >= 2 and left_pre.count("3H") >= 2
            for card in donated:
                hands["opposite"].remove(card)
                hands["opposite"].append("2H")
                hands["left"].remove("2H")
                hands["left"].append(card)
                left_pre[left_pre.index("2H")] = card
        assert sum((Counter(hand) for hand in hands.values()), Counter()) == total_before
        assert total_before.total() == 108
        if final_cards_remaining == 3:
            left_pre.pop(0)  # That real card stays in the finisher's hand.
        else:
            assert final_cards_remaining == 2
        mapped = {seat: SEATS[(SEATS.index(seat) + rotation) % 4] for seat in SEATS}
        rotated_hands = {mapped[seat]: tuple(hand) for seat, hand in hands.items()}
        profile = f"profile{len(created)}"
        (tmp_path / profile).mkdir()
        store = InMemoryLiveSessionStore(tmp_path, profile)
        clock, pixels = Clock(), Pixels()
        core = LiveOrchestrator(
            reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
            recognition_service=pixels, advisor=None, minimum_free_bytes=0, settle_ms=0,
            burst_sample_interval_ms=50, processing_clock_ms=clock,
        )
        first_lead = "opposite" if first_actor_preplayed else "left"
        core.start(round_level="6", hand=rotated_hands["self"], lead_player=mapped[first_lead], monotonic_ms=100)
        core.bind_capture_generation(3)

        def act(base_seat, cards=()):
            assert core.snapshot.current_player == mapped[base_seat]
            clock.now += 10
            return core.commit_trusted_action(actor=mapped[base_seat], cards=tuple(cards), is_pass=not cards, monotonic_ms=clock.now)

        def pass_to(base_seat):
            for _ in range(4):
                if core.snapshot.current_player == mapped[base_seat]:
                    return
                actor = next(seat for seat in SEATS if mapped[seat] == core.snapshot.current_player)
                act(actor)
            assert core.snapshot.current_player == mapped[base_seat]

        if first_actor_preplayed:
            act("opposite", ("2H", "2H"))
            act("left", ("3H", "3H"))
            left_pre.remove("3H")
            left_pre.remove("3H")
            pass_to("left")
        for card in left_pre[:-1]:
            act("left", (card,))
            pass_to("left")
        act("left", left_pre[-1:])
        act("self")
        act("right", right_pre[:1])
        pass_to("right")
        for card in right_pre[1:]:
            act("right", (card,))
            pass_to("right")
        act("right", right_final)
        act("opposite")
        act("left")
        act("self", own_pair)
        head, first_actor, finisher = mapped["right"], mapped["opposite"], mapped["left"]
        assert core.snapshot.current_player == first_actor
        assert core.snapshot.finished_seats == frozenset({head})
        assert core.snapshot.remaining_cards[finisher] == final_cards_remaining
        assert audit_session_health(core.snapshot, core.events)["status"] == "PASS"

        class Rig:
            def step(self, timestamp, surfaces, *, placement=True, label="second"):
                clock.now = timestamp
                pixels.active, pixels.placement, pixels.label = finisher, finisher if placement else None, label
                expected = core.snapshot.current_player
                return core.analyze_frame(pixels.image(surfaces), monotonic_ms=timestamp,
                    metrics=ZoneFrameMetrics(timestamp, bool(surfaces.get(expected, ())), 0., False, False,
                                             content_changed=bool(surfaces.get(expected, ()))),
                    trace_context={"capture_generation": 3, "capture_seq": timestamp})

        rig = Rig()
        rig.core, rig.head, rig.first_actor, rig.finisher = core, head, first_actor, finisher
        rig.step(10_000, {}, placement=False)
        created.append(rig)
        return rig

    yield make
    for rig in created:
        rig.core.finish()


def new_actions(core, revision):
    return [event for event in core.reducer.events if event.state_revision_after > revision
            and event.event_type in {"player_played", "player_passed"}]


@pytest.mark.parametrize("rotation,preplayed", [(2, False), (1, True)])
def test_rotated_terminal_pair_chain_is_atomic_legal_and_conserves_cards(rotated_cores, rotation, preplayed):
    rig = rotated_cores(rotation=rotation, first_actor_preplayed=preplayed)
    before = rig.core.snapshot
    surfaces = {rig.first_actor: ("7C", "7S"), rig.finisher: ("10C", "10D")}
    rig.step(10_100, surfaces)
    assert not new_actions(rig.core, before.revision)
    rig.step(10_200, surfaces)
    actions = new_actions(rig.core, before.revision)
    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in actions] == [
        (rig.first_actor, ("7C", "7S")), (rig.finisher, ("10C", "10D")),
    ], f"terminal chain still depends on original seats; head={rig.head}, first={rig.first_actor}, finisher={rig.finisher}"
    assert rig.core.snapshot.remaining_cards[rig.first_actor] == before.remaining_cards[rig.first_actor] - 2
    if preplayed:
        assert rig.core.snapshot.remaining_cards[rig.first_actor] != 25
    finishes = [event for event in rig.core.events if event.event_type == "player_finished" and event.actor == rig.finisher]
    assert len(finishes) == 1 and finishes[0].payload.get("trigger_action_event_id") == actions[-1].event_id
    assert rig.core.snapshot.remaining_cards[rig.finisher] == 0
    assert audit_session_health(rig.core.snapshot, rig.core.events)["status"] == "PASS"


@pytest.mark.parametrize("case", ["wrong_order", "finished_bridge", "remaining_nonzero"])
def test_incomplete_or_out_of_order_terminal_chain_cannot_finish_or_zero_cards(rotated_cores, case):
    rig = rotated_cores(rotation=2, final_cards_remaining=3 if case == "remaining_nonzero" else 2)
    before = rig.core.snapshot
    surfaces = {rig.finisher: ("10C", "10D")}
    surfaces[rig.head if case == "finished_bridge" else rig.first_actor] = ("7C", "7S")
    for timestamp in (10_100, 10_200):
        rig.step(timestamp, surfaces, label="third" if case == "wrong_order" else "second")
    assert rig.finisher not in rig.core.snapshot.finished_seats
    actions = new_actions(rig.core, before.revision)
    # A separately proved first action may proceed through ordinary consensus;
    # an invalid placement must not finish anyone or manufacture missing cards.
    assert not any(event.actor == rig.head for event in actions)
    actual_finisher_cards = sum(len(event.payload.get("cards", ())) for event in actions if event.actor == rig.finisher)
    assert rig.core.snapshot.remaining_cards[rig.finisher] == before.remaining_cards[rig.finisher] - actual_finisher_cards
    assert rig.core.snapshot.remaining_cards[rig.finisher] > 0
    assert rig.core.snapshot.finished_seats == before.finished_seats
