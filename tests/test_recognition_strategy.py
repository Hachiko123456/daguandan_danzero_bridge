from __future__ import annotations

from dataclasses import replace

from daguandan_bridge.live.consensus import ConsensusContext, RecognitionSample
from daguandan_bridge.live.recognition_strategy import (
    decide_best_effort_candidate,
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


def test_best_effort_burst_prefers_a_visible_play_over_later_pass_markers():
    samples = (
        _sample("7S"),
        RecognitionSample((), True, 0.99, "pass_template"),
        _sample("7S"),
        RecognitionSample((), True, 0.99, "pass_template"),
        RecognitionSample((), True, 0.99, "pass_template"),
    )

    result = decide_best_effort_candidate(samples, context=_context())

    assert result is not None
    assert not result.is_pass
    assert result.cards == ("7S",)
    assert result.source == "best_effort_burst"
    assert "candidate_conflict_resolved_best_effort" in result.integrity_warnings


def test_best_effort_discards_an_unbeatable_play_and_keeps_the_pass_marker():
    samples = (
        _sample("6S"),
        _sample("6S"),
        RecognitionSample((), True, 0.99, "pass_template"),
        RecognitionSample((), True, 0.99, "pass_template"),
    )

    result = decide_best_effort_candidate(
        samples,
        context=replace(_context(), table_cards=("7S",)),
    )

    assert result is not None
    assert result.is_pass
    assert result.cards == ()
    assert "does_not_beat_table" in result.rejected_reasons


def test_best_effort_never_promotes_a_single_pass_marker():
    result = decide_best_effort_candidate(
        (RecognitionSample((), True, 0.99, "pass_template"),),
        context=replace(_context(), next_turn_evidence=True),
    )

    assert result is None


def test_two_valid_streak_discards_effect_text_before_confirmed_pass():
    """A rank-like effect read must not outrank the later real pass marker."""

    effect_read = RecognitionSample(
        cards=("J?",),
        suit_options=(("H", "D"),),
        is_pass=False,
        confidence=0.65,
        source="effect-covered-play-region",
    )
    pass_read = RecognitionSample((), True, 0.99, "pass_template")

    result = decide_recognition_strategy(
        "two_valid_streak",
        (effect_read, effect_read, pass_read, pass_read),
        context=replace(
            _context(),
            level_rank="6",
            table_cards=("10C", "6C", "6H", "7C", "9C"),
        ),
    )

    assert result is not None
    assert result.is_pass
    assert result.cards == ()
    assert "does_not_beat_table" in result.rejected_reasons


def test_next_turn_evidence_never_commits_an_illegal_animation_fragment():
    context = replace(_context(), next_turn_evidence=True)

    result = decide_best_effort_candidate(
        (_sample("3S", "4H"),) * 5,
        context=context,
    )

    assert result is None


def test_latest_steel_plate_waits_past_illegal_fragments_for_complete_cards():
    fragments = (
        _sample("2H", "3D", "3C"),
        _sample("2H", "3D"),
        _sample("2H", "3S"),
        _sample("2H", "3C"),
        _sample("2H", "3S"),
    )
    complete = _sample("2H", "2H", "2C", "3D", "3C", "3S")
    context = replace(_context(), level_rank="6", next_turn_evidence=True)

    assert decide_best_effort_candidate(fragments, context=context) is None
    result = decide_recognition_strategy(
        "two_valid_streak",
        (*fragments, complete, complete),
        context=context,
    )

    assert result is not None
    assert result.cards == tuple(sorted(complete.cards))
    assert result.vote_count == 2
