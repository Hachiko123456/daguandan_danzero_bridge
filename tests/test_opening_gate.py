from dataclasses import replace

import pytest

from daguandan_bridge.opening_gate import (
    OpeningActionSeed, OpeningSessionSeed, OpeningTracker,
    evaluate_opening_gate, opening_semantic_key, serialized_result,
)


HAND = tuple(f"{r}{s}" for r in ("3", "4", "5", "6", "7", "8", "9") for s in "SHCD")[:27]


def result(*, hand=HAND, lead="left", current="self", cards=("2C",), confidence=.92, source="a", level="5"):
    return serialized_result(
        round_level=level, hand=hand, lead_player=lead, current_player=current,
        events=(() if cards is None else ({"player": "left", "cards": cards,
                  "is_pass": False, "confidence": confidence, "source": source},)),
    )


def observe(tracker, item, tick, *, generation=0, identity=None):
    return tracker.observe(item, anchor_score=.95, generation=generation,
                           monotonic_ms=tick, observation_id=tick if identity is None else identity)


def test_confidence_and_template_changes_are_not_different_actions():
    tracker = OpeningTracker()
    assert not observe(tracker, result(), 100).ready
    accepted = observe(tracker, result(confidence=.92001, source="template:b"), 1500)
    assert accepted.ready
    assert accepted.seed.opening_action.cards == ("2C",)
    assert not observe(tracker, result(), 1800).ready


def test_semantic_hand_key_preserves_duplicates():
    seed = OpeningSessionSeed("5", ("2C", "2C", "3H"), "left")
    assert opening_semantic_key(seed) == opening_semantic_key(replace(seed, hand=("3H", "2C", "2C")))
    assert opening_semantic_key(seed) != opening_semantic_key(replace(seed, hand=("2C", "3H", "3H")))
    action = OpeningActionSeed("left", ("2C", "2C"), "self", .9, "a")
    seed = replace(seed, opening_action=action)
    assert opening_semantic_key(seed) != opening_semantic_key(replace(seed, opening_action=replace(action, cards=("2C",))))


def test_no_missing_hand_frame_votes_but_safe_bad_frame_can_be_skipped():
    tracker = OpeningTracker()
    assert not observe(tracker, result(), 100).ready
    assert not observe(tracker, result(hand=(), lead=None, current=None, cards=None), 700).ready
    assert observe(tracker, result(), 1200).ready


def test_lead_marker_confirmation_is_independent_from_hand_frames():
    tracker = OpeningTracker()
    assert not observe(tracker, result(hand=(), current="left", cards=None), 100).ready
    assert not observe(tracker, result(hand=(), current="left", cards=None), 200).ready
    assert not observe(tracker, result(lead=None), 300).ready
    accepted = observe(tracker, result(lead=None, confidence=.93), 400)
    assert accepted.ready
    assert accepted.seed.lead_player == "left"
    assert accepted.seed.opening_action.actor == "left"


@pytest.mark.parametrize("count", [1, 21, 26])
def test_reduced_hand_never_reuses_complete_cache(count):
    tracker = OpeningTracker()
    assert not observe(tracker, result(), 100).ready
    assert not observe(tracker, result(hand=HAND[:count]), 200).ready
    assert tracker.candidate is None
    assert not observe(tracker, result(), 300).ready


@pytest.mark.parametrize("change", ["generation", "expiry", "backward", "duplicate"])
def test_evidence_does_not_cross_scope_or_vote_twice(change):
    tracker = OpeningTracker()
    observe(tracker, result(), 1000)
    tick = {"generation": 2000, "expiry": 10000, "backward": 500, "duplicate": 1000}[change]
    assert not observe(tracker, result(), tick, generation=1 if change == "generation" else 0).ready


def test_current_player_conflict_in_bad_frame_invalidates_cached_vote():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    observe(tracker, result(hand=(), lead=None, current="right", cards=None), 200)
    assert not observe(tracker, result(), 300).ready


def test_visible_action_cannot_disappear_from_seed_history():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 200).ready
    assert not observe(tracker, empty, 300).ready


def test_expiry_does_not_forget_an_already_observed_first_action():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 10000).ready
    assert not observe(tracker, empty, 10100).ready
    assert tracker.saw_action
    assert tracker.candidate is None


def test_completed_table_phase_never_starts_twice_after_expiry_or_clock_reversal():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    assert observe(tracker, result(), 200).ready
    for tick in (10000, 10100, 50, 100):
        assert observe(tracker, result(), tick).reason == "already_started"
    assert tracker.completed


def test_fresh_complete_first_action_can_reconfirm_after_observation_expiry():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    assert not observe(tracker, result(confidence=.94), 10000).ready
    accepted = observe(tracker, result(confidence=.95), 10100)
    assert accepted.ready
    assert accepted.seed.opening_action.cards == ("2C",)


def test_ordinary_clock_reversal_and_unknown_anchor_do_not_erase_history_fact():
    tracker = OpeningTracker()
    observe(tracker, result(), 1000)
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 100).ready
    assert not tracker.observe(empty, anchor_score=.1, generation=0, monotonic_ms=200).ready
    assert not observe(tracker, empty, 300).ready
    assert not observe(tracker, empty, 400).ready


@pytest.mark.parametrize("weak", [result(cards=("?",)), result(confidence=.1), result(current="right")])
def test_single_weak_or_contradictory_event_does_not_permanently_seal_opening(weak):
    tracker = OpeningTracker()
    observe(tracker, weak, 100)
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 10000).ready
    assert observe(tracker, empty, 10100).ready


def test_explicit_generation_or_settlement_boundary_may_start_a_new_phase():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    assert observe(tracker, result(), 200).ready
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 300, generation=1).ready
    assert observe(tracker, empty, 400, generation=1).ready
    terminal = serialized_result(round_level="5", hand=(), buttons=("continue_game", "change_table"))
    assert observe(tracker, terminal, 450, generation=1).reason == "settlement_screen"
    assert not observe(tracker, empty, 500, generation=1).ready
    assert observe(tracker, empty, 600, generation=1).ready


def test_action_or_level_conflict_requires_fresh_confirmation():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    assert not observe(tracker, result(cards=("3C",)), 200).ready
    assert not observe(tracker, result(cards=("3C",), level="6"), 300).ready


def test_settlement_erases_all_stages_and_does_not_make_a_session():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    terminal = serialized_result(round_level="5", hand=HAND, buttons=("continue_game",))
    assert not observe(tracker, terminal, 200).ready
    assert not observe(tracker, result(), 300).ready


def test_visible_wrong_successor_is_not_repaired_by_marker_cache():
    tracker = OpeningTracker()
    observe(tracker, result(hand=(), current="left", cards=None), 100)
    observe(tracker, result(hand=(), current="left", cards=None), 200)
    assert not observe(tracker, result(lead=None, current="right"), 300).ready
    assert not observe(tracker, result(lead=None, current="right"), 400).ready


def test_raw_opening_gate_still_rejects_unanchored_or_late_state():
    assert not evaluate_opening_gate(result(), anchor_score=.84).ready
    assert not evaluate_opening_gate(result(hand=HAND[:21]), anchor_score=.95).ready


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -.1, 1.1])
def test_non_finite_or_impossible_confidence_is_not_a_usable_vote(confidence):
    assert not evaluate_opening_gate(result(confidence=confidence), anchor_score=.95).ready


def test_staged_marker_age_is_not_renewed_by_first_complete_hand():
    tracker = OpeningTracker()
    observe(tracker, result(hand=(), current="left", cards=None), 100)
    observe(tracker, result(hand=(), current="left", cards=None), 200)
    assert not observe(tracker, result(lead=None), 7900).ready
    assert not observe(tracker, result(lead=None), 8300).ready


def test_level_conflict_while_hand_missing_discards_staged_marker():
    tracker = OpeningTracker()
    observe(tracker, result(hand=(), current="left", cards=None), 100)
    observe(tracker, result(hand=(), current="left", cards=None), 200)
    observe(tracker, result(hand=(), lead=None, current=None, cards=None, level="6"), 300)
    assert not observe(tracker, result(lead=None, level="6"), 400).ready
    assert not observe(tracker, result(lead=None, level="6"), 500).ready
