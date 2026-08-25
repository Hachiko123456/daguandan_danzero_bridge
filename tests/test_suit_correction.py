from __future__ import annotations

from daguandan_bridge.live.suit_correction import (
    SuitCorrectionTracker,
    validate_suit_correction,
    validate_visual_action_compatibility,
)


def test_shared_suit_correction_requires_two_identical_safe_reads():
    tracker = SuitCorrectionTracker()

    first = tracker.observe("turn-20", ("3?", "3?", "4?"), ("3C", "3H", "4S"))
    second = tracker.observe("turn-20", ("3?", "3?", "4?"), ("3C", "3H", "4S"))

    assert first.cards == ("3C", "3H", "4S")
    assert not first.confirmed
    assert second.confirmed
    assert second.cards == ("3C", "3H", "4S")


def test_shared_suit_correction_rejects_a_different_action_and_resets_the_streak():
    tracker = SuitCorrectionTracker()

    tracker.observe("turn-20", ("5?",), ("5S",))
    invalid = tracker.observe("turn-20", ("5?",), ("6S",))
    retry = tracker.observe("turn-20", ("5?",), ("5S",))

    assert validate_suit_correction(("5?",), ("6S",)) is None
    assert invalid.cards == ()
    assert retry.confirmations == 1


def test_occluded_reread_matches_exact_double_deck_action_by_aligned_candidates():
    target = ("7C", "7D", "7D", "7H")

    compatible = validate_visual_action_compatibility(
        target,
        ("7?", "7C", "7D", "7D"),
        (("H", "D"), ("C",), ("D",), ("D",)),
    )
    incompatible = validate_visual_action_compatibility(
        target,
        ("7?", "7C", "7D", "7D"),
        (("D",), ("C",), ("D",), ("D",)),
    )
    missing_candidates = validate_visual_action_compatibility(
        target,
        ("7?", "7C", "7D", "7D"),
    )

    assert compatible == tuple(sorted(target))
    assert incompatible is None
    assert missing_candidates is None


def test_occluded_reread_requires_two_reads_and_preserves_original_exact_cards():
    tracker = SuitCorrectionTracker()
    target = ("7H", "7D", "7D", "7C")
    observed = ("7?", "7D", "7D", "7C")
    suit_options = (("H", "D"), ("D",), ("D",), ("C",))

    first = tracker.observe_visual_action(
        "bomb-7",
        target,
        observed,
        suit_options,
    )
    second = tracker.observe_visual_action(
        "bomb-7",
        target,
        observed,
        suit_options,
    )

    assert first.cards == tuple(sorted(target))
    assert first.evidence_kind == "compatible"
    assert not first.confirmed
    assert second.confirmed
    assert second.cards == tuple(sorted(target))
    assert second.evidence_kind == "compatible"
