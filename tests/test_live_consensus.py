from __future__ import annotations

from daguandan_bridge.live.consensus import (
    BurstConsensus,
    ConsensusContext,
    RecognitionSample,
)


def _play(
    *cards: str,
    confidence: float = 0.9,
    post_hand: tuple[str, ...] = (),
) -> RecognitionSample:
    return RecognitionSample(
        cards=cards,
        is_pass=False,
        confidence=confidence,
        source="targeted_template",
        post_hand=post_hand,
    )


def _context(**changes) -> ConsensusContext:
    values = {
        "level_rank": "2",
        "remaining_cards": 27,
        "allow_pass": True,
        "known_hand": (),
        "table_cards": (),
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


def test_structurally_impossible_pattern_is_rejected_without_becoming_pass():
    samples = [_play("3S", "4H") for _ in range(3)]

    result = BurstConsensus(min_votes=3).decide(samples, context=_context())

    assert result.status == "review_required"
    assert not result.is_pass
    assert "illegal_pattern" in result.rejected_reasons


def test_cards_over_remaining_count_are_rejected():
    samples = [_play("7S", "7H") for _ in range(3)]

    result = BurstConsensus(min_votes=3).decide(
        samples,
        context=_context(remaining_cards=1),
    )

    assert result.status == "review_required"
    assert "exceeds_remaining_cards" in result.rejected_reasons


def test_empty_zone_and_next_turn_signal_do_not_create_a_pass_without_template():
    result = BurstConsensus(min_votes=3).decide(
        [],
        context=_context(region_empty=True, next_turn_evidence=True),
    )

    assert result.status == "review_required"
    assert not result.is_pass
    assert "no_recognition_samples" in result.rejected_reasons


def test_candidate_that_would_create_a_third_known_card_is_rejected():
    result = BurstConsensus(min_votes=3).decide(
        [_play("big_joker") for _ in range(3)],
        context=_context(
            known_cards=("big_joker", "big_joker"),
            validate_rules=False,
        ),
    )

    assert result.status == "review_required"
    assert "exceeds_double_deck_limit" in result.rejected_reasons


def test_historical_effect_suit_uncertainty_does_not_reject_a_clear_straight_flush():
    cards = ("3H", "4H", "5H", "6H", "7H")
    result = BurstConsensus(min_votes=3).decide(
        [_play(*cards) for _ in range(3)],
        context=_context(
            level_rank="10",
            remaining_cards=17,
            known_cards=("6D", "6D", "6H", "6S", "6C", "6?"),
            known_suit_options=(
                ("D",), ("D",), ("H",), ("S",), ("C",), ("H", "D"),
            ),
            table_cards=("10C", "10S", "QC", "QD", "QH"),
        ),
    )

    assert result.status == "confirmed"
    assert result.cards == cards
    assert result.integrity_warnings == ("historical_suit_constraints_relaxed",)


def test_self_candidate_already_in_current_hand_is_not_counted_twice():
    result = BurstConsensus(min_votes=3).decide(
        [_play("7S") for _ in range(3)],
        context=_context(
            known_hand=("7S", "7S"),
            known_cards=("7S", "7S"),
            candidate_already_known=True,
        ),
    )

    assert result.status == "confirmed"
    assert result.cards == ("7S",)


def test_unknown_suit_self_action_is_reconciled_to_the_confirmed_hand():
    hand = ("AS", "2H", "2H", "3C", "4C", "5S")
    cards = ("A?", "2H", "3?", "4?", "5?")
    suit_options = (("S", "C"), ("H",), ("S", "C"), ("S", "C"), ("S", "C"))
    samples = [
        RecognitionSample(
            cards=cards,
            suit_options=suit_options,
            is_pass=False,
            confidence=0.86,
            source="latest-game-regression",
        )
        for _ in range(3)
    ]

    result = BurstConsensus(min_votes=3).decide(
        samples,
        context=_context(
            level_rank="10",
            allow_pass=False,
            known_hand=hand,
            known_cards=hand,
            candidate_already_known=True,
        ),
    )

    assert result.status == "confirmed"
    assert result.cards == ("2H", "3?", "4?", "5?", "A?")
    assert result.resolved_cards == ("2H", "3C", "4C", "5S", "AS")


def test_play_that_does_not_beat_reconstructed_table_is_rejected():
    result = BurstConsensus(min_votes=3).decide(
        [_play("6S") for _ in range(3)],
        context=_context(table_cards=("7S",)),
    )

    assert result.status == "review_required"
    assert not result.is_pass
    assert "does_not_beat_table" in result.rejected_reasons


def test_fresh_trick_allows_a_legal_lead_without_beating_a_prior_table():
    result = BurstConsensus(min_votes=3).decide(
        [_play("6S") for _ in range(3)],
        context=_context(table_cards=()),
    )

    assert result.status == "confirmed"
    assert result.cards == ("6S",)


def test_stable_play_wins_over_a_pass_candidate_in_the_same_turn():
    samples = [
        _play("7S"),
        _play("7S"),
        RecognitionSample((), True, 0.99, "pass_template"),
        RecognitionSample((), True, 0.99, "pass_template"),
        RecognitionSample((), True, 0.99, "pass_template"),
    ]

    result = BurstConsensus(min_votes=2).decide(samples, context=_context())

    assert result.status == "confirmed"
    assert not result.is_pass
    assert result.cards == ("7S",)


def test_self_play_does_not_require_post_hand_reconciliation():
    old_hand = ("6S", "7S", "8S", "9S")
    result = BurstConsensus(min_votes=3).decide(
        [_play("7S", post_hand=old_hand) for _ in range(3)],
        context=_context(known_hand=old_hand),
    )

    assert result.status == "confirmed"
    assert result.cards == ("7S",)
