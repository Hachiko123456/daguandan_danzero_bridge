from __future__ import annotations

from daguandan_bridge.domain.recognition import LeadEvidence
from daguandan_bridge.live.lead_evidence import rank_lead_evidence
from daguandan_bridge.opening_gate import OpeningTracker, serialized_result


HAND = tuple(f"{rank}{suit}" for rank in ("3", "4", "5", "6", "7", "8", "9") for suit in "SHCD")[:27]


def _evidence(*, seat="left", first=0.0, action=0.0, timer=0.0):
    return LeadEvidence(
        candidate_seat=seat,
        first_play_score=first,
        card_action_score=action,
        timer_score=timer,
    )


def _result(evidence, *, cards=("2C",), lead=None, current="self"):
    return serialized_result(
        round_level="5",
        hand=HAND,
        lead_player=lead,
        current_player=current,
        events=({
            "player": "left",
            "cards": cards,
            "is_pass": False,
            "confidence": 0.92,
            "source": "template:cards",
        },) if cards else (),
        lead_evidence=rank_lead_evidence(evidence),
    )


def test_low_first_play_template_keeps_stable_card_action_candidate():
    candidate = rank_lead_evidence((_evidence(first=0.42, action=0.94, timer=0.88),))[0]

    assert candidate.candidate_seat == "left"
    assert candidate.first_play_score == 0.42
    assert candidate.card_action_score == 0.94
    assert candidate.timer_score == 0.88
    assert candidate.status == "pending_confirmation"
    assert candidate.rejection_reason is None


def test_close_lead_candidates_are_explicitly_rejected_as_a_conflict():
    candidates = rank_lead_evidence((
        _evidence(seat="left", first=0.91, timer=0.90),
        _evidence(seat="right", first=0.88, timer=0.89),
    ))

    assert {item.status for item in candidates} == {"conflict"}
    assert {item.rejection_reason for item in candidates} == {"candidate_conflict"}


def test_tracker_retains_unconfirmed_candidate_but_only_confirms_after_two_frames():
    tracker = OpeningTracker()
    evidence = (_evidence(first=0.42, action=0.94, timer=0.88),)

    waiting = tracker.observe(
        _result(evidence, cards=(), current="left"), anchor_score=.95, generation=0,
        monotonic_ms=100, observation_id=1,
    )
    assert not waiting.ready
    assert waiting.reason == "confirming_hand"
    assert tracker.candidate is not None
    assert tracker.candidate_count == 1
    assert tracker.completed is False

    still_waiting = tracker.observe(
        _result(evidence, cards=(), current="left"), anchor_score=.95, generation=0,
        monotonic_ms=200, observation_id=2,
    )
    assert not still_waiting.ready
    assert still_waiting.reason == "confirming_opening"
    assert tracker.candidate is not None

    tracker = OpeningTracker()
    first = tracker.observe(
        _result(evidence), anchor_score=.95, generation=0,
        monotonic_ms=300, observation_id=3,
    )
    assert not first.ready
    assert first.reason == "confirming_hand"

    confirmed = tracker.observe(
        _result(evidence), anchor_score=.95, generation=0,
        monotonic_ms=400, observation_id=4,
    )
    assert confirmed.ready
    assert confirmed.seed is not None
    assert confirmed.seed.lead_player == "left"
    assert tracker.completed is True


def test_tracker_never_starts_from_conflicting_candidates():
    tracker = OpeningTracker()
    evidence = (
        _evidence(seat="left", first=0.91, timer=0.90),
        _evidence(seat="right", first=0.88, timer=0.89),
    )

    result = tracker.observe(
        _result(evidence), anchor_score=.95, generation=0,
        monotonic_ms=100, observation_id=1,
    )

    assert not result.ready
    assert result.reason == "candidate_conflict"
    assert tracker.completed is False
    assert tracker.candidate is None
    assert tracker.candidate_evidence
