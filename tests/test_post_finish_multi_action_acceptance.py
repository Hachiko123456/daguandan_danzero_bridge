"""Recover KH->level6 6D after right genuinely finishes its27th card(JS).

The primary fixed-source1x replay still stopped after55 actions at this
post-head boundary. Unlike earlier multi-action tests, right is actually out:
26 legal lead cards, self7D, right's finalJS, and native player_finished/head.
No count/finished/edge/pending state is assigned by this fixture. The108-card
ledger is real; only64x32 per-seat pixels and a monotonic test clock are used.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from daguandan_bridge.domain.recognition import FastSignalResult, PlacementSignal, PlayRegionResult
from daguandan_bridge.live.orchestrator import LiveOrchestrator
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
from daguandan_bridge.live.zone_lifecycle import ZoneFrameMetrics
from daguandan_bridge.session_health import audit_session_health


SEATS = ("self", "right", "opposite", "left")
CODES = {(): 0, ("2C",): 20, ("7D",): 50, ("JS",): 80, ("KH",): 120, ("6D",): 180, ("4D",): 150, ("QS",): 210}
CARDS = {code: cards for cards, code in CODES.items()}


def deal():
    available = Counter({f"{rank}{suit}": 2 for rank in ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A") for suit in "CDHS"})
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
    hands = {"self": tuple(own), "right": tuple(right_pre) + right_last,
             "opposite": tuple(opposite), "left": tuple(left)}
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
        self.placement = None
        self.placement_label = "head"
        self.calls = []

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

    def recognize_fast_signals(self, _image, expected_player, *, allow_pass=True):
        placements = (
            (PlacementSignal(self.placement, self.placement_label, .99, "persistent-placement"),)
            if self.placement is not None
            else ()
        )
        return FastSignalResult(
            expected_player, self.active, False, self.controls, False,
            placements=placements,
        )

    def recognize_play_region(self, image, seat, *, wild_rank, allow_pass=True):
        self.calls.append(seat)
        cards = CARDS.get(int(self.play_roi(image, seat)[12, 8, 0]), ())
        return PlayRegionResult(seat, cards, False, .97 if cards else 0., (), (), source="post-finish-seat-pixels")


@pytest.fixture
def post_head_rigs(tmp_path):
    made = []

    def make(*, new_controls_allowed=True):
        hands, right_pre = deal()
        profile = f"profile{len(made)}"
        (tmp_path / profile).mkdir()
        store = InMemoryLiveSessionStore(tmp_path, profile)
        clock, pixels = Clock(), Pixels()
        core = LiveOrchestrator(
            reducer=LiveReducer(store.session_id), store=store, recorder=InMemorySessionRecorder(store.directory),
            recognition_service=pixels, advisor=None, minimum_free_bytes=0, settle_ms=0,
            burst_sample_interval_ms=50, processing_clock_ms=clock,
        )
        core.start(round_level="6", hand=hands["self"], lead_player="right", monotonic_ms=clock.now)
        core.bind_capture_generation(3)

        def act(seat, cards=()):
            assert core.snapshot.current_player == seat
            clock.now += 10
            return core.commit_trusted_action(actor=seat, cards=tuple(cards), is_pass=not cards, monotonic_ms=clock.now)

        for card in right_pre[:-1]:
            act("right", (card,))
            for seat in ("opposite", "left", "self"):
                act(seat)
        act("right", right_pre[-1:])
        act("opposite")
        act("left")
        act("self", ("7D",))
        assert core.snapshot.current_player == "right" and core.snapshot.remaining_cards["right"] == 1

        class Rig:
            def step(self, timestamp, surfaces, *, active, controls, generation=3, placement=None, placement_label="head"):
                clock.now = timestamp
                pixels.active, pixels.controls = active, controls
                pixels.placement, pixels.placement_label = placement, placement_label
                expected = core.snapshot.current_player
                occupied = bool(surfaces.get(expected, ()))
                return core.analyze_frame(pixels.image(surfaces), monotonic_ms=timestamp,
                    metrics=ZoneFrameMetrics(timestamp, occupied, 0., False, False, content_changed=False),
                    trace_context={"capture_generation": generation, "capture_seq": timestamp})

        rig = Rig()
        rig.core, rig.clock, rig.pixels = core, clock, pixels
        rig.step(3_000, {"self": ("7D",), "right": ("2C",)}, active="right", controls=False)
        clock.now = 3_050
        final = core.commit_trusted_action(actor="right", cards=("JS",), is_pass=False, monotonic_ms=3_050)
        assert final.snapshot.remaining_cards["right"] == 0
        assert final.snapshot.finished_seats == frozenset({"right"})
        assert final.snapshot.current_player == "opposite"
        head = [event for event in core.events if event.event_type == "player_finished" and event.actor == "right"]
        assert len(head) == 1 and head[0].payload.get("placement") == "head"
        assert head[0].payload.get("trigger_action_event_id") == final.event.event_id
        baseline = {"self": ("7D",), "right": ("JS",)}
        rig.step(3_100, baseline, active="opposite", controls=not new_controls_allowed)
        rig.step(3_200, baseline, active="left", controls=not new_controls_allowed)
        assert audit_session_health(core.snapshot, core.events)["status"] == "PASS"
        assert core.snapshot.current_player == "opposite"
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.core.finish()


def surfaces(*, right=("JS",), left=("6D",)):
    return {"self": ("7D",), "right": right, "opposite": ("KH",), "left": left}


def new_actions(core, revision):
    return [event for event in core.reducer.events if event.state_revision_after > revision
            and event.event_type in {"player_played", "player_passed"}]


def test_after_natural_head_two_fresh_captures_atomically_recover_kh_level6_then_continue(post_head_rigs):
    rig = post_head_rigs()
    before = rig.core.snapshot
    right_reads_before = rig.pixels.calls.count("right")
    rig.step(3_300, surfaces(), active="self", controls=True)
    assert not new_actions(rig.core, before.revision)
    update = rig.step(3_400, surfaces(), active="self", controls=True)
    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in new_actions(rig.core, before.revision)] == [
        ("opposite", ("KH",)), ("left", ("6D",)),
    ], "post-finish KH->level6 6D chain was lost while skipped right JS remained on screen"
    assert update.snapshot.current_player == "self"
    assert rig.pixels.calls.count("right") == right_reads_before
    assert not any(event.event_type == "advice_recovery_failed" for event in rig.core.events)
    assert audit_session_health(rig.core.snapshot, rig.core.events)["status"] == "PASS"
    rig.core.commit_trusted_action(actor="self", cards=("small_joker",), is_pass=False, monotonic_ms=3_500)
    assert rig.core.snapshot.current_player == "opposite"  # Finished right stays skipped.
    assert audit_session_health(rig.core.snapshot, rig.core.events)["status"] == "PASS"


def test_post_head_crossed_handoff_entry_frame_accumulates_recovery_chain(post_head_rigs):
    rig = post_head_rigs()
    before = rig.core.snapshot
    calls_before = len(rig.pixels.calls)

    update = rig.step(3_300, surfaces(), active="self", controls=True)

    assert update.snapshot == before
    assert not new_actions(rig.core, before.revision)
    assert rig.pixels.calls[calls_before:] == ["opposite", "left"]
    window = rig.core._turn_ownership_window
    assert window is not None
    assert not window.turn_recovery_pending
    assert window.pre_recovery_chain_signature is not None
    assert window.turn_recovery_cycle_missing_player is None

    # A controller may drop an identical static screenshot; the first frame
    # that crossed into recovery must already be retained as the first proof.
    rig.step(3_300, surfaces(), active="self", controls=True)
    assert not new_actions(rig.core, before.revision)
    assert window.pre_recovery_chain_signature is not None

    calls_before = len(rig.pixels.calls)
    update = rig.step(3_400, surfaces(), active="self", controls=True)
    assert rig.pixels.calls[calls_before:] == ["opposite", "left"]
    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in new_actions(rig.core, before.revision)] == [
        ("opposite", ("KH",)),
        ("left", ("6D",)),
    ]
    assert update.snapshot.current_player == "self"
    assert audit_session_health(rig.core.snapshot, rig.core.events)["status"] == "PASS"


def test_persistent_consumed_head_badge_does_not_starve_post_head_recovery(post_head_rigs):
    rig = post_head_rigs()
    before = rig.core.snapshot
    head_events_before = [
        event for event in rig.core.events
        if event.event_type == "player_finished" and event.actor == "right"
    ]
    assert len(head_events_before) == 1

    update = None
    for timestamp in (3_300, 3_400):
        update = rig.step(
            timestamp,
            surfaces(),
            active="self",
            controls=True,
            placement="right",
            placement_label="head",
        )

    actions = new_actions(rig.core, before.revision)
    assert [(event.actor, tuple(event.payload.get("cards", ()))) for event in actions] == [
        ("opposite", ("KH",)),
        ("left", ("6D",)),
    ]
    assert update is not None and update.snapshot.current_player == "self"
    head_events_after = [
        event for event in rig.core.events
        if event.event_type == "player_finished" and event.actor == "right"
    ]
    assert head_events_after == head_events_before
    assert audit_session_health(rig.core.snapshot, rig.core.events)["status"] == "PASS"


def test_unfinished_invalid_placement_badge_still_blocks_ordinary_recovery(post_head_rigs):
    rig = post_head_rigs()
    before = rig.core.snapshot

    for timestamp in (3_300, 3_400):
        rig.step(
            timestamp,
            surfaces(),
            active="self",
            controls=True,
            placement="left",
            placement_label="head",
        )

    assert not new_actions(rig.core, before.revision)
    assert rig.core.snapshot == before
    assert "left" not in rig.core.snapshot.finished_seats
    assert not any(
        event.event_type == "player_finished" and event.actor == "left"
        for event in rig.core.events
    )


@pytest.mark.parametrize("case", ["finished_new_pixels", "wrong_active", "old_controls", "missing_left", "illegal_left", "duplicate_capture", "changed_generation"])
def test_post_head_invalid_evidence_never_invents_skipped_or_missing_actions(post_head_rigs, case):
    rig = post_head_rigs(new_controls_allowed=case != "old_controls")
    before = rig.core.snapshot
    first_surfaces = surfaces(left=() if case == "missing_left" else ("4D",) if case == "illegal_left" else ("6D",),
                              right=("QS",) if case == "finished_new_pixels" else ("JS",))
    if case == "finished_new_pixels":
        # A new finished-seat surface must not substitute for the missing
        # first live actor or enter a reconstructed action chain.
        first_surfaces["opposite"] = ()
    for index in range(2):
        timestamp = 3_300 if case == "duplicate_capture" else 3_300 + index * 100
        rig.step(timestamp, first_surfaces, active="left" if case == "wrong_active" else "self", controls=True,
                 generation=3 + index if case == "changed_generation" else 3)
    actions = new_actions(rig.core, before.revision)
    assert not actions, f"{case} committed a post-finish chain without two current complete action witnesses"
    assert rig.core.snapshot == before
    assert rig.core.snapshot.remaining_cards["right"] == 0
    assert rig.core.snapshot.finished_seats == frozenset({"right"})
