from __future__ import annotations

from daguandan_bridge.live.suit_correction import SuitCorrectionTracker, validate_suit_correction


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
