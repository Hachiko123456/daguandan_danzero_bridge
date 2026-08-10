from __future__ import annotations

from daguandan_bridge.live.consensus import ConsensusContext, RecognitionSample
from daguandan_bridge.live.recognition_strategy import (
    decide_recognition_strategy,
    has_exhausted_valid_candidates,
)


def _context() -> ConsensusContext:
    return ConsensusContext(
        level_rank="2",
        remaining_cards=27,
        allow_pass=True,
        known_hand=(),
        table_cards=(),
        region_empty=False,
        next_turn_evidence=False,
    )


def _sample(*cards: str) -> RecognitionSample:
    return RecognitionSample(
        cards=cards,
        is_pass=False,
        confidence=0.9,
        source="test",
    )


def _unknown_straight(*, ten: str = "10?") -> RecognitionSample:
    return RecognitionSample(
        cards=("6?", "7?", "8?", "9?", ten),
        # This mirrors the recognizer's colour-level evidence: every unknown
        # suit has at most two candidates, so legal-pattern validation can
        # exhaustively inspect the 32 physical variants.
        suit_options=(
            ("S", "C"),
            ("H", "D"),
            ("S", "C"),
            ("H", "D"),
            ("H", "D") if ten.endswith("?") else ("D",),
        ),
        is_pass=False,
        confidence=0.9,
        source="test",
    )


def test_two_valid_streak_waits_one_extra_frame_for_unknown_suit():
    samples = (_sample("7?"), _sample("7?"))

    assert (
        decide_recognition_strategy(
            "two_valid_streak", samples, context=_context()
        )
        is None
    )

    result = decide_recognition_strategy(
        "two_valid_streak",
        (*samples, _sample("7?")),
        context=_context(),
    )

    assert result is not None
    assert result.cards == ("7?",)
    assert result.vote_count == 3


def test_two_valid_streak_uses_clear_suit_as_soon_as_two_clear_frames_arrive():
    result = decide_recognition_strategy(
        "two_valid_streak",
        (_sample("7?"), _sample("7?"), _sample("7C"), _sample("7C")),
        context=_context(),
    )

    assert result is not None
    assert result.cards == ("7C",)
    assert result.vote_count == 2


def test_two_valid_streak_commits_stable_ranks_when_only_suits_jitter():
    """A visible straight must not be dropped because one suit glyph flickers."""

    samples = (_unknown_straight(), _unknown_straight(), _unknown_straight(ten="10D"))

    result = decide_recognition_strategy(
        "two_valid_streak", samples, context=_context()
    )

    assert result is not None
    # The one clear ♦10 frame is evidence, not enough to rewrite the visual
    # history.  Keep all five cards and retain suit uncertainty for later
    # DanZero branching/correction.
    assert result.cards == ("10?", "6?", "7?", "8?", "9?")
    assert result.suit_options[0] == ("H", "D")
    assert result.vote_count == 3


def test_suit_jitter_alone_does_not_exhaust_a_valid_action_window():
    samples = (
        _unknown_straight(),
        _unknown_straight(),
        _unknown_straight(ten="10D"),
        _unknown_straight(),
        _unknown_straight(ten="10D"),
    )

    assert not has_exhausted_valid_candidates(
        samples,
        context=_context(),
        limit=5,
    )
