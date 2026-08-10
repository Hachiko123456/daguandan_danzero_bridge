from __future__ import annotations

from daguandan_bridge.live.consensus import ConsensusContext, RecognitionSample
from daguandan_bridge.live.recognition_strategy import decide_recognition_strategy


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
