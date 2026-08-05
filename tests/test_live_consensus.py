from __future__ import annotations

from daguandan_bridge.live.consensus import (
    BurstConsensus,
    ConsensusContext,
    RecognitionSample,
)


def _play(*cards: str, confidence: float = 0.9) -> RecognitionSample:
    return RecognitionSample(
        cards=cards,
        is_pass=False,
        confidence=confidence,
        source="targeted_template",
    )


def _context(**changes) -> ConsensusContext:
    values = {
        "level_rank": "2",
        "remaining_cards": 27,
        "allow_pass": True,
        "known_hand": (),
        "region_empty": False,
        "next_turn_evidence": False,
    }
    values.update(changes)
    return ConsensusContext(**values)


def test_three_matching_burst_samples_confirm_cards_as_multiset():
    result = BurstConsensus(min_votes=3).decide(
        [
            _play("7S", "7H"),
            _play("7H", "7S"),
            _play("7S", "7H"),
            _play("7S", "7D"),
        ],
        context=_context(),
    )

    assert result.status == "confirmed"
    assert result.cards == ("7H", "7S")
    assert result.vote_count == 3


def test_illegal_card_pattern_is_rejected_even_with_three_votes():
    samples = [_play("3S", "4H") for _ in range(3)]

    result = BurstConsensus(min_votes=3).decide(samples, context=_context())

    assert result.status == "review_required"
    assert "illegal_pattern" in result.rejected_reasons


def test_cards_over_remaining_count_are_rejected():
    samples = [_play("7S", "7H") for _ in range(3)]

    result = BurstConsensus(min_votes=3).decide(
        samples,
        context=_context(remaining_cards=1),
    )

    assert result.status == "review_required"
    assert "exceeds_remaining_cards" in result.rejected_reasons


def test_inferred_pass_is_never_auto_confirmed_in_version_one():
    result = BurstConsensus(min_votes=3).decide(
        [],
        context=_context(region_empty=True, next_turn_evidence=True),
    )

    assert result.status == "needs_confirmation"
    assert result.is_pass
    assert result.source == "inferred_pass"
