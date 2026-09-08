"""Consumed placement badges must not block post-finish action recovery."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from daguandan_bridge.domain.recognition import (
    FastSignalResult,
    PlacementSignal,
    PlayRegionResult,
)
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.session_health import audit_session_health


SEATS = ("self", "right", "opposite", "left")
CODES = {
    (): 0,
    ("2C",): 20,
    ("7D",): 50,
    ("JS",): 80,
    ("KH",): 120,
    ("6D",): 180,
}
CARDS = {code: cards for cards, code in CODES.items()}


def deal():
    available = Counter(
        {
            f"{rank}{suit}": 2
            for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
            for suit in "CDHS"
        }
    )
    available.update({"small_joker": 2, "big_joker": 2})
    whole_deck = available.copy()

    def take(cards):
        for card in cards:
            assert available[card] > 0
            available[card] -= 1
        return tuple(cards)

    own = list(take(("7D", "small_joker")))
    right_last, right_26th = take(("JS",)), take(("2C",))
    opposite = list(take(("KH",)))
    left = list(take(("6D",)))
    for card in reversed(tuple(available)):
        while available[card] and len(own) < 27:
            own.extend(take((card,)))
    right_pre = []
    for card in available:
        if card == "6H":
            continue
        while available[card] and len(right_pre) < 25:
            right_pre.extend(take((card,)))
    right_pre.extend(right_26th)
    for card in available:
        while available[card] and len(opposite) < 27:
            opposite.extend(take((card,)))
    left.extend(available.elements())
    hands = {
        "self": tuple(own),
        "right": tuple(right_pre) + right_last,
        "opposite": tuple(opposite),
        "left": tuple(left),
    }
    assert all(len(hand) == 27 for hand in hands.values())
    assert sum((Counter(hand) for hand in hands.values()), Counter()) == whole_deck
    return hands, tuple(right_pre)


@dataclass
class Clock:
    now: int = 100

    def __call__(self):
        return self.now


class Pixels:
    def __init__(self):
        self.active, self.controls = "right", False
        self.show_consumed_head = False
        self.calls = []

    @staticmethod
    def play_roi(image, seat):
        index = SEATS.index(seat)
        return image[:, index * 16 : (index + 1) * 16]

    @staticmethod
    def image(surfaces):
        image = np.zeros((32, 64, 3), np.uint8)
        for index, seat in enumerate(SEATS):
            image[3:29, index * 16 + 2 : index * 16 + 14] = CODES[
                tuple(surfaces.get(seat, ()))
            ]
        return image

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        placements = (
            (PlacementSignal("right", "head", 0.99, "stale-consumed-placement"),)
            if self.show_consumed_head
            else ()
        )
        return FastSignalResult(
            expected_player,
            self.active,
            False,
            self.controls,
            False,
            placements=placements,
        )

    def recognize_play_region(self, image, seat, *, wild_rank, allow_pass=True):
        self.calls.append(seat)
        cards = CARDS.get(int(self.play_roi(image, seat)[12, 8, 0]), ())
        return PlayRegionResult(
            seat,
            cards,
            False,
            0.97 if cards else 0.0,
            (),
            (),
            source="consumed-placement-seat-pixels",
        )


def new_actions(core, revision):
    return [
        event
        for event in core.reducer.events
        if event.state_revision_after > revision
        and event.event_type in {"player_played", "player_passed"}
    ]


def head_events(core):
    return [
        event
        for event in core.events
        if event.event_type == "player_finished" and event.actor == "right"
    ]


def test_consumed_head_badge_does_not_block_post_finish_kh_6d_recovery(tmp_path):
    hands, right_pre = deal()
    (tmp_path / "consumed_placement").mkdir()
    store = InMemoryLiveSessionStore(tmp_path, "consumed_placement")
    clock, pixels = Clock(), Pixels()
    core = LiveOrchestrator(
        reducer=LiveReducer(store.session_id),
        store=store,
        recorder=InMemorySessionRecorder(store.directory),
        recognition_service=pixels,
        advisor=None,
        minimum_free_bytes=0,
        settle_ms=0,
        burst_sample_interval_ms=50,
        processing_clock_ms=clock,
    )
    core.start(round_level="6", hand=hands["self"], lead_player="right", monotonic_ms=clock.now)
    core.bind_capture_generation(3)

    def act(seat, cards=()):
        assert core.snapshot.current_player == seat
        clock.now += 10
        return core.commit_trusted_action(
            actor=seat,
            cards=tuple(cards),
            is_pass=not cards,
            monotonic_ms=clock.now,
        )

    for card in right_pre[:-1]:
        act("right", (card,))
        for seat in ("opposite", "left", "self"):
            act(seat)
    act("right", right_pre[-1:])
    act("opposite")
    act("left")
    act("self", ("7D",))
    assert core.snapshot.current_player == "right"
    assert core.snapshot.remaining_cards["right"] == 1

    clock.now = 3_050
    final = core.commit_trusted_action(
        actor="right",
        cards=("JS",),
        is_pass=False,
        monotonic_ms=clock.now,
    )
    assert final.snapshot.current_player == "opposite"
    assert final.snapshot.finished_seats == frozenset({"right"})
    assert final.snapshot.remaining_cards["right"] == 0
    assert len(head_events(core)) == 1
    assert head_events(core)[0].payload.get("placement") == "head"

    def step(timestamp, surfaces, *, active, controls):
        clock.now = timestamp
        pixels.active, pixels.controls = active, controls
        pixels.show_consumed_head = True
        expected = core.snapshot.current_player
        occupied = bool(surfaces.get(expected, ()))
        return core.analyze_frame(
            pixels.image(surfaces),
            monotonic_ms=timestamp,
            metrics=ZoneFrameMetrics(
                timestamp,
                occupied,
                0.0,
                False,
                False,
                content_changed=False,
            ),
            trace_context={"capture_generation": 3, "capture_seq": timestamp},
        )

    stale_badge_only = {"self": ("7D",), "right": ("JS",)}
    step(3_100, stale_badge_only, active="opposite", controls=False)
    step(3_200, stale_badge_only, active="left", controls=False)
    assert len(head_events(core)) == 1
    assert not any(event.event_type == "game_end_detected" for event in core.events)
    assert not any(event.event_type == "terminal_history_gap" for event in core.events)

    before = core.snapshot
    surfaces = {**stale_badge_only, "opposite": ("KH",), "left": ("6D",)}
    assert step(3_300, surfaces, active="self", controls=True).snapshot == before
    update = step(3_400, surfaces, active="self", controls=True)

    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in new_actions(core, before.revision)] == [
        ("opposite", ("KH",)),
        ("left", ("6D",)),
    ]
    assert update.snapshot.current_player == "self"
    assert len(head_events(core)) == 1
    assert not any(event.event_type == "game_end_detected" for event in core.events)
    assert not any(event.event_type == "terminal_history_gap" for event in core.events)
    assert audit_session_health(core.snapshot, core.events)["status"] == "PASS"
    core.finish()
