from __future__ import annotations

from collections import Counter

import pytest

from daguandan_bridge.danzero import GameStateError
from daguandan_bridge.danzero.advisor import LocalGuandanAdvisor
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.turns import TURN_ORDER, next_active_seat


INITIAL_HAND = (
    "3S",
    "3H",
    "3C",
    "3D",
    "4S",
    "4H",
    "4C",
    "4D",
    "5S",
    "5H",
    "5C",
    "5D",
    "6S",
    "6H",
    "6C",
    "6D",
    "7S",
    "7H",
    "7C",
    "7D",
    "8S",
    "8H",
    "8C",
    "8D",
    "9S",
    "small_joker",
    "big_joker",
)


def _started_reducer() -> LiveReducer:
    reducer = LiveReducer("game-test")
    reducer.confirm_initial_state(
        round_level="2",
        hand=INITIAL_HAND,
        lead_player="right",
    )
    return reducer


def test_turn_order_is_counter_clockwise_and_skips_finished_seats():
    assert TURN_ORDER == ("self", "right", "opposite", "left")
    assert next_active_seat("right", frozenset()) == "opposite"
    assert next_active_seat("right", frozenset({"opposite"})) == "left"


def test_initial_state_requires_exactly_27_cards():
    reducer = LiveReducer("game-test")

    with pytest.raises(GameStateError, match="27"):
        reducer.confirm_initial_state(
            round_level="2",
            hand=("3S",),
            lead_player="right",
        )


def test_deferred_lead_confirmation_after_initial_state():
    reducer = LiveReducer("game-lead")

    reducer.confirm_initial_state(
        round_level="2",
        hand=INITIAL_HAND,
        lead_player=None,
    )

    snapshot = reducer.snapshot()
    assert snapshot.initialized
    assert snapshot.current_player is None
    assert snapshot.lead_player is None

    event = reducer.confirm_lead_player("opposite")

    assert event.event_type == "lead_player_confirmed"
    snapshot = reducer.snapshot()
    assert snapshot.current_player == "opposite"
    assert snapshot.lead_player == "opposite"
    assert snapshot.trick_id == 1
    assert snapshot.turn_id == 1

    with pytest.raises(GameStateError, match="已经确认"):
        reducer.confirm_lead_player("self")

    reducer.record_play("opposite", ("3S", "3H"))
    assert reducer.snapshot().current_player == "left"


def test_confirmed_play_advances_turn_and_decrements_remaining_cards():
    reducer = _started_reducer()

    event = reducer.record_play("right", ("10S", "10H"))

    snapshot = reducer.snapshot()
    assert event.event_type == "player_played"
    assert snapshot.current_player == "opposite"
    assert snapshot.remaining_cards["right"] == 25
    assert snapshot.trick_plays[-1].cards == ("10H", "10S")


def test_third_finish_ends_the_round_without_creating_a_fourth_turn():
    reducer = _started_reducer()
    almost_all_cards = INITIAL_HAND[:-1]

    # Bring three non-local players to one card through the public event API;
    # the reducer deliberately rebuilds from that immutable history.
    reducer.record_play("right", almost_all_cards)
    reducer.record_play("opposite", almost_all_cards)
    reducer.record_play("left", almost_all_cards)
    reducer.record_pass("self")

    reducer.record_play("right", ("7S",))
    reducer.record_play("opposite", ("7S",))
    reducer.record_play("left", ("7S",))

    snapshot = reducer.snapshot()
    assert snapshot.finished_seats == frozenset({"left", "opposite", "right"})
    assert snapshot.current_player is None
    assert snapshot.trick_plays == ()


def test_visual_finish_badge_corrects_an_unknown_opponent_starting_count():
    reducer = LiveReducer("visual-finish")
    reducer.confirm_initial_state(
        round_level="2",
        hand=INITIAL_HAND,
        lead_player="left",
    )
    reducer.record_play("left", ("9S",))

    event = reducer.confirm_player_finished(
        "left",
        placement="head",
        confidence=0.98,
    )
    snapshot = reducer.snapshot()

    assert event.event_type == "player_finished"
    assert snapshot.remaining_cards["left"] == 0
    assert snapshot.finished_seats == frozenset({"left"})
    assert snapshot.current_player == "self"
    strategy_snapshot = reducer.to_guandan_state().local_snapshot()
    assert LocalGuandanAdvisor._remaining_counts(strategy_snapshot)[1] == 0


def test_correction_rebuild_matches_clean_history():
    corrected = _started_reducer()
    original = corrected.record_play("right", ("7S", "7H"))
    corrected.correct_event(
        original.event_id,
        cards=("8S", "8H"),
        is_pass=False,
        reason="manual_candidate",
    )

    clean = _started_reducer()
    clean.record_play("right", ("8S", "8H"))

    assert corrected.snapshot().semantic_dict() == clean.snapshot().semantic_dict()
    assert corrected.events[-1].event_type == "event_correction"
    assert corrected.events[-1].payload["target_event_id"] == original.event_id


def test_projection_is_ready_when_confirmed_history_reaches_self_turn():
    reducer = _started_reducer()
    reducer.record_play("right", ("10S",))
    reducer.record_pass("opposite")
    reducer.record_pass("left")

    state = reducer.to_guandan_state()
    snapshot = state.local_snapshot()

    assert snapshot.current_player == "self"
    assert snapshot.lead_player == "right"
    assert Counter(snapshot.my_hand) == Counter(INITIAL_HAND)
    assert [event.player for event in snapshot.trick_plays] == [
        "right",
        "opposite",
        "left",
    ]


def test_self_play_removes_exact_cards_from_confirmed_hand():
    reducer = LiveReducer("game-self")
    reducer.confirm_initial_state(
        round_level="2",
        hand=INITIAL_HAND,
        lead_player="self",
    )

    reducer.record_play("self", ("3S", "3H"))

    remaining = Counter(reducer.snapshot().my_hand)
    expected = Counter(INITIAL_HAND)
    expected.subtract(("3S", "3H"))
    assert remaining == +expected
