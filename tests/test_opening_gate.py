from dataclasses import replace

import pytest

from daguandan_bridge.domain.recognition import LeadEvidence
from daguandan_bridge.opening_gate import (
    OpeningActionSeed, OpeningSessionSeed, OpeningTracker,
    evaluate_opening_gate, opening_semantic_key, serialized_result,
)


HAND = tuple(f"{r}{s}" for r in ("3", "4", "5", "6", "7", "8", "9") for s in "SHCD")[:27]


def result(*, hand=HAND, lead="left", current="self", cards=("2C",), confidence=.92, source="a", level="5", player="left"):
    return serialized_result(
        round_level=level, hand=hand, lead_player=lead, current_player=current,
        events=(() if cards is None else ({"player": player, "cards": cards,
                  "is_pass": False, "confidence": confidence, "source": source},)),
    )


def observe(tracker, item, tick, *, generation=0, identity=None):
    return tracker.observe(item, anchor_score=.95, generation=generation,
                           monotonic_ms=tick, observation_id=tick if identity is None else identity)



def test_unknown_suit_hand_never_confirms_and_fresh_exact_frames_can_recover():
    tracker = OpeningTracker()
    uncertain = list(HAND)
    uncertain[0] = uncertain[0][:-1] + "?"
    uncertain = tuple(uncertain)

    first = observe(tracker, result(hand=uncertain, level="4"), 100)
    second = observe(tracker, result(hand=uncertain, level="4"), 200)
    assert not first.ready and first.reason == "hand_unresolved"
    assert not second.ready and second.reason == "hand_unresolved"
    assert tracker.candidate is None

    assert not observe(tracker, result(level="3"), 300).ready
    accepted = observe(tracker, result(level="3"), 400)
    assert accepted.ready
    assert accepted.seed is not None
    assert accepted.seed.round_level == "3"
    assert all(not card.endswith("?") for card in accepted.seed.hand)


def test_gate_rejects_complete_count_with_unresolved_physical_card():
    uncertain = list(HAND)
    uncertain[0] = uncertain[0][:-1] + "?"
    evaluation = evaluate_opening_gate(
        result(hand=tuple(uncertain)), anchor_score=.95
    )
    assert not evaluation.ready
    assert evaluation.reason == "hand_unresolved"
    assert evaluation.normalized_hand is None

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
    waiting = observe(tracker, empty, 300)
    assert waiting.ready
    assert waiting.status == "READY_WAITING_FIRST_ACTION"
    assert waiting.seed is not None
    assert waiting.seed.opening_action is None


def test_expiry_does_not_promote_a_stale_first_action():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 10000).ready
    waiting = observe(tracker, empty, 10100)
    assert waiting.ready
    assert waiting.status == "READY_WAITING_FIRST_ACTION"
    assert waiting.seed is not None
    assert waiting.seed.opening_action is None


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


def test_ordinary_clock_reversal_and_unknown_anchor_require_fresh_confirmation():
    tracker = OpeningTracker()
    observe(tracker, result(), 1000)
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 100).ready
    assert not tracker.observe(empty, anchor_score=.1, generation=0, monotonic_ms=200).ready
    assert not observe(tracker, empty, 300).ready
    waiting = observe(tracker, empty, 400)
    assert waiting.ready
    assert waiting.status == "READY_WAITING_FIRST_ACTION"
    assert waiting.seed is not None and waiting.seed.opening_action is None


@pytest.mark.parametrize("weak", [result(cards=("?",)), result(confidence=.1), result(current="right")])
def test_single_weak_or_contradictory_event_does_not_start_as_action(weak):
    tracker = OpeningTracker()
    observe(tracker, weak, 100)
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 10000).ready
    waiting = observe(tracker, empty, 10100)
    assert waiting.ready
    assert waiting.status == "READY_WAITING_FIRST_ACTION"
    assert waiting.seed is not None and waiting.seed.opening_action is None
    assert not observe(tracker, result(), 10200).ready
    confirmed = observe(tracker, result(confidence=.95), 10300)
    assert confirmed.ready
    assert confirmed.status == "READY_ACTION_CONFIRMED"


def test_explicit_generation_or_settlement_boundary_may_start_a_new_phase():
    tracker = OpeningTracker()
    observe(tracker, result(), 100)
    assert observe(tracker, result(), 200).ready
    empty = result(lead=None, current=None, cards=None)
    assert not observe(tracker, empty, 300, generation=1).ready
    waiting = observe(tracker, empty, 400, generation=1)
    assert waiting.ready
    assert waiting.status == "READY_WAITING_FIRST_ACTION"
    assert not observe(tracker, result(), 500, generation=1).ready
    confirmed = observe(tracker, result(), 600, generation=1)
    assert confirmed.ready
    assert confirmed.status == "READY_ACTION_CONFIRMED"
    terminal = serialized_result(round_level="5", hand=(), buttons=("continue_game", "change_table"))
    assert observe(tracker, terminal, 650, generation=1).reason == "settlement_screen"
    assert not observe(tracker, empty, 700, generation=1).ready
    assert not observe(tracker, result(), 800, generation=1).ready
    assert observe(tracker, result(), 900, generation=1).ready


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


def test_two_stable_complete_frames_establish_waiting_first_action_without_events():
    tracker = OpeningTracker()
    first = observe(tracker, result(lead="self", current="self", cards=None), 100)
    second = observe(tracker, result(lead="self", current="self", cards=None), 200)

    assert first.status == "NOT_READY"
    assert second.ready is True
    assert second.status == "READY_WAITING_FIRST_ACTION"
    assert second.seed is not None
    assert second.seed.opening_action is None


def test_real_first_play_confirms_action_after_waiting_seed():
    tracker = OpeningTracker()
    observe(tracker, result(lead="self", current="self", cards=None), 100)
    waiting = observe(tracker, result(lead="self", current="self", cards=None), 200)
    assert waiting.status == "READY_WAITING_FIRST_ACTION"

    first_action = observe(
        tracker,
        result(hand=HAND[:-1], lead="self", current="right", cards=("2C",), player="self"),
        300,
    )
    assert first_action.status == "NOT_READY"
    confirmed = observe(
        tracker,
        result(hand=HAND[:-1], lead="self", current="right", cards=("2C",), player="self"),
        400,
    )
    assert confirmed.ready is True
    assert confirmed.status == "READY_ACTION_CONFIRMED"
    assert confirmed.seed is not None
    assert confirmed.seed.opening_action is not None
    assert confirmed.seed.opening_action.actor == "self"
    assert confirmed.seed.opening_action.next_player == "right"


def test_conflict_short_hand_unknown_suit_and_anchor_are_not_ready():
    tracker = OpeningTracker()
    conflict = serialized_result(
        round_level="5", hand=HAND, lead_player="self", current_player="self",
        events=(),
        lead_evidence=(
            LeadEvidence("self", first_play_score=.90, status="conflict", rejection_reason="candidate_conflict"),
            LeadEvidence("right", first_play_score=.89, status="conflict", rejection_reason="candidate_conflict"),
        ),
    )
    conflict_result = tracker.observe(
        conflict, anchor_score=.95, generation=0, monotonic_ms=100, observation_id=1
    )
    assert conflict_result.status == "CONFLICT"
    assert not conflict_result.ready

    tracker = OpeningTracker()
    anchored = serialized_result(
        round_level="5", hand=HAND, lead_player="self", current_player="self", events=()
    )
    first = tracker.observe(
        anchored, anchor_score=.95, generation=0, monotonic_ms=200, observation_id=2
    )
    assert first.status == "NOT_READY"
    assert tracker.observe(
        anchored, anchor_score=.80, generation=0, monotonic_ms=300, observation_id=3
    ).status == "NOT_READY"

    uncertain = list(HAND)
    uncertain[0] = uncertain[0][:-1] + "?"
    blocked = tracker.observe(
        serialized_result(round_level="5", hand=tuple(uncertain), lead_player="self", current_player="self", events=()),
        anchor_score=.95, generation=1, monotonic_ms=300, observation_id=3,
    )
    assert blocked.status == "BLOCKED"
    assert not blocked.ready

    short = tracker.observe(
        serialized_result(round_level="5", hand=HAND[:-1], lead_player="self", current_player="self", events=()),
        anchor_score=.95, generation=2, monotonic_ms=400, observation_id=4,
    )
    assert short.status in {"NOT_READY", "BLOCKED"}
    assert not short.ready


def test_single_frame_gate_exposes_waiting_and_action_statuses():
    waiting = evaluate_opening_gate(
        result(lead="self", current="self", cards=None), anchor_score=.95
    )
    assert waiting.ready is True
    assert waiting.status == "READY_WAITING_FIRST_ACTION"
    assert waiting.session_ready is True
    assert waiting.action_confirmed is False

    action = evaluate_opening_gate(
        result(lead="self", current="right", cards=("2C",), player="self"),
        anchor_score=.95,
    )
    assert action.ready is True
    assert action.status == "READY_ACTION_CONFIRMED"
    assert action.action_confirmed is True


def test_invalid_action_cannot_be_promoted_to_opening_action():
    invalid = evaluate_opening_gate(
        result(lead="self", current="right", cards=("ZZ",), player="self"),
        anchor_score=.95,
    )
    assert invalid.ready is False
    assert invalid.status in {"BLOCKED", "CONFLICT", "NOT_READY"}


def test_legacy_reason_mapping_is_explicit_and_stable():
    from daguandan_bridge.opening_gate import (
        legacy_reason_for_status, opening_status_for_reason,
    )

    assert opening_status_for_reason("ready") == "READY_ACTION_CONFIRMED"
    assert opening_status_for_reason("confirming_opening") == "NOT_READY"
    assert opening_status_for_reason("candidate_conflict") == "CONFLICT"
    assert legacy_reason_for_status("READY_WAITING_FIRST_ACTION") == "ready_waiting_first_action"


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
