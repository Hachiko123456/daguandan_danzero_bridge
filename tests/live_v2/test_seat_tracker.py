from __future__ import annotations

import pytest

from daguandan_bridge.live_v2.seat_tracker import SeatTracker
from daguandan_bridge.live_v2.types import (
    ActionKind,
    FrameIdentity,
    ObservationKind,
    ObservationReason,
    Seat,
    SeatObservation,
    VersionIdentity,
)


def frame(seq: int, *, generation: int = 1) -> FrameIdentity:
    return FrameIdentity(
        "game-1", generation, seq, seq * 100, "roi-1", "window-1"
    )


def version(*, revision: int = 0) -> VersionIdentity:
    return VersionIdentity("game-1", 1, revision, revision, revision)


def observation(
    seq: int,
    kind: ObservationKind,
    *,
    seat: Seat = Seat.LEFT,
    cards: tuple[str, ...] = (),
    confidence: float = 0.9,
) -> SeatObservation:
    reason = {
        ObservationKind.EMPTY: ObservationReason.STABLE_EMPTY,
        ObservationKind.PASS: ObservationReason.PASS_MARKER,
        ObservationKind.PLAY: ObservationReason.CARDS_RECOGNIZED,
        ObservationKind.ANIMATING: ObservationReason.ANIMATION_DETECTED,
        ObservationKind.UNKNOWN: ObservationReason.UNREADABLE,
    }[kind]
    return SeatObservation(
        f"e-{seat.value}-{seq}", frame(seq), seat, kind, cards, confidence,
        reason, seq * 100 + 1,
        tuple((card,) for card in cards), (f"frame={seq}",),
    )


def test_all_four_seats_use_the_same_tracker_behavior() -> None:
    for seat in Seat:
        tracker = SeatTracker(seat)
        assert tracker.ingest(observation(1, ObservationKind.PASS, seat=seat)) is None
        candidate = tracker.ingest(
            observation(2, ObservationKind.PASS, seat=seat), version=version()
        )
        assert candidate is not None
        assert candidate.seat is seat
        assert candidate.kind is ActionKind.PASS


def test_distinct_frames_required_and_confidence_is_conservative() -> None:
    tracker = SeatTracker(Seat.LEFT)
    first = observation(1, ObservationKind.PLAY, cards=("3H",), confidence=0.94)
    duplicate = SeatObservation(
        "duplicate-id", first.frame, first.seat, first.kind, first.cards, 0.99,
        first.reason, first.processing_ms,
        first.suit_options, first.diagnostics,
    )
    assert tracker.ingest(first) is None
    assert tracker.ingest(duplicate) is None
    candidate = tracker.ingest(
        observation(2, ObservationKind.PLAY, cards=("3H",), confidence=0.88),
        version=version(),
    )
    assert candidate is not None
    assert candidate.evidence_ids == ("e-left-1", "e-left-2")
    assert candidate.suit_options == (("3H",),)
    assert candidate.action_epoch == 0
    assert candidate.first_frame == first.frame
    assert candidate.last_frame == frame(2)
    assert candidate.confidence == pytest.approx(0.88)
    assert tracker.snapshot().duplicate_frames == 1


def test_same_surface_requires_new_action_cycle_before_reemission() -> None:
    tracker = SeatTracker(Seat.LEFT)
    tracker.ingest(observation(1, ObservationKind.PLAY, cards=("5D", "5C")))
    assert tracker.ingest(
        observation(2, ObservationKind.PLAY, cards=("5D", "5C")),
        version=version(),
    ) is not None
    for seq in (3, 4, 5):
        assert tracker.ingest(
            observation(seq, ObservationKind.PLAY, cards=("5D", "5C"))
        ) is None
    tracker.ingest(observation(6, ObservationKind.EMPTY))
    tracker.ingest(observation(7, ObservationKind.EMPTY))
    tracker.ingest(observation(8, ObservationKind.PLAY, cards=("5D", "5C")))
    second = tracker.ingest(
        observation(9, ObservationKind.PLAY, cards=("5D", "5C")),
        version=version(revision=1),
    )
    assert second is not None
    assert tracker.action_epoch == 1
    assert second.action_epoch == 1


def test_unknown_and_animating_do_not_clear_or_infer_pass() -> None:
    tracker = SeatTracker(Seat.RIGHT)
    tracker.ingest(observation(1, ObservationKind.PASS, seat=Seat.RIGHT))
    assert tracker.ingest(
        observation(2, ObservationKind.PASS, seat=Seat.RIGHT), version=version()
    ) is not None
    tracker.ingest(observation(3, ObservationKind.UNKNOWN, seat=Seat.RIGHT))
    tracker.ingest(observation(4, ObservationKind.ANIMATING, seat=Seat.RIGHT))
    assert tracker.ingest(
        observation(5, ObservationKind.PASS, seat=Seat.RIGHT)
    ) is None
    assert tracker.action_epoch == 0


def test_pass_marker_must_disappear_before_reemission() -> None:
    tracker = SeatTracker(Seat.RIGHT)
    tracker.ingest(observation(1, ObservationKind.PASS, seat=Seat.RIGHT), version=version())
    first = tracker.ingest(
        observation(2, ObservationKind.PASS, seat=Seat.RIGHT), version=version()
    )
    assert first is not None

    # A formal turn change alone never re-arms a still-visible PASS.
    next_turn = version(revision=1)
    assert tracker.ingest(
        observation(3, ObservationKind.PASS, seat=Seat.RIGHT), version=next_turn
    ) is None

    assert not tracker.observe_pass_marker(False)
    assert tracker.observe_pass_marker(False)
    assert tracker.ingest(
        observation(4, ObservationKind.PASS, seat=Seat.RIGHT), version=next_turn
    ) is None
    second = tracker.ingest(
        observation(5, ObservationKind.PASS, seat=Seat.RIGHT), version=next_turn
    )
    assert second is not None and second.action_epoch == 1
    assert second.evidence_ids == ("e-right-4", "e-right-5")


def test_old_turn_pass_evidence_is_ignored_until_marker_clear() -> None:
    tracker = SeatTracker(Seat.LEFT)
    tracker.ingest(observation(1, ObservationKind.PASS), version=version())
    tracker.ingest(observation(2, ObservationKind.PASS), version=version())
    current = version(revision=2)

    # Old PASS evidence cannot be reused just because the formal version changed.
    assert tracker.ingest(
        observation(3, ObservationKind.PASS), version=version(revision=1)
    ) is None
    assert tracker.ingest(
        observation(4, ObservationKind.PASS), version=current
    ) is None

    tracker.observe_pass_marker(False)
    tracker.observe_pass_marker(False)
    assert tracker.ingest(
        observation(5, ObservationKind.PASS), version=current
    ) is None
    candidate = tracker.ingest(
        observation(6, ObservationKind.PASS), version=current
    )
    assert candidate is not None
    assert candidate.evidence_ids == ("e-left-5", "e-left-6")

def test_stable_different_surface_opens_cycle_if_empty_was_missed() -> None:
    tracker = SeatTracker(Seat.OPPOSITE)
    tracker.ingest(observation(1, ObservationKind.PLAY, seat=Seat.OPPOSITE, cards=("9S",)))
    tracker.ingest(
        observation(2, ObservationKind.PLAY, seat=Seat.OPPOSITE, cards=("9S",)),
        version=version(),
    )
    tracker.ingest(observation(3, ObservationKind.PLAY, seat=Seat.OPPOSITE, cards=("10S",)))
    second = tracker.ingest(
        observation(4, ObservationKind.PLAY, seat=Seat.OPPOSITE, cards=("10S",)),
        version=version(revision=1),
    )
    assert second is not None and second.cards == ("10S",)
    assert tracker.action_epoch == 1


def test_requires_version_at_confirmation_and_rejects_wrong_seat() -> None:
    tracker = SeatTracker(Seat.LEFT)
    tracker.ingest(observation(1, ObservationKind.PLAY, cards=("AH",)))
    with pytest.raises(ValueError, match="version is required"):
        tracker.ingest(observation(2, ObservationKind.PLAY, cards=("AH",)))
    with pytest.raises(ValueError, match="does not match"):
        SeatTracker(Seat.LEFT).ingest(
            observation(3, ObservationKind.EMPTY, seat=Seat.RIGHT)
        )


def test_capture_generation_change_discards_old_pending_evidence() -> None:
    tracker = SeatTracker(Seat.LEFT)
    tracker.ingest(observation(1, ObservationKind.PLAY, cards=("AH",)))
    next_generation = SeatObservation(
        "new-generation", frame(2, generation=2), Seat.LEFT,
        ObservationKind.PLAY, ("AH",), 0.9,
        ObservationReason.CARDS_RECOGNIZED, 201,
        (("AH",),), ("new",),
    )
    assert tracker.ingest(next_generation) is None
    assert tracker.snapshot().pending_count == 1


def test_version_must_match_observation_stream() -> None:
    tracker = SeatTracker(Seat.LEFT)
    with pytest.raises(ValueError, match="same stream"):
        tracker.ingest(
            observation(1, ObservationKind.PLAY, cards=("KH",)),
            version=VersionIdentity("other", 1, 0, 0, 0),
        )


def test_delayed_older_generation_cannot_reset_new_tracker_stream() -> None:
    tracker = SeatTracker(Seat.LEFT)
    newer = SeatObservation(
        "new-1", frame(1, generation=2), Seat.LEFT, ObservationKind.PLAY,
        ("QH",), 0.9, ObservationReason.CARDS_RECOGNIZED, 101,
        (("QH",),), ("new",),
    )
    delayed = SeatObservation(
        "old-2", frame(2, generation=1), Seat.LEFT, ObservationKind.PLAY,
        ("QH",), 0.9, ObservationReason.CARDS_RECOGNIZED, 201,
        (("QH",),), ("old",),
    )
    tracker.ingest(newer)
    assert tracker.ingest(delayed) is None
    assert tracker.snapshot().pending_count == 1


def test_tracker_cannot_be_configured_for_single_frame_candidates() -> None:
    with pytest.raises(ValueError, match="at least two"):
        SeatTracker(Seat.LEFT, confirmations=1)


def test_suit_uncertainty_must_stabilize_with_cards_before_candidate() -> None:
    tracker = SeatTracker(Seat.LEFT)
    first = SeatObservation(
        "uncertain-1", frame(1), Seat.LEFT, ObservationKind.PLAY, ("5?",),
        0.9, ObservationReason.CARDS_RECOGNIZED, 101,
        (("5?", "5H"),), ("red",),
    )
    changed = SeatObservation(
        "uncertain-2", frame(2), Seat.LEFT, ObservationKind.PLAY, ("5?",),
        0.9, ObservationReason.CARDS_RECOGNIZED, 201,
        (("5?", "5S"),), ("black",),
    )
    stable = SeatObservation(
        "uncertain-3", frame(3), Seat.LEFT, ObservationKind.PLAY, ("5?",),
        0.9, ObservationReason.CARDS_RECOGNIZED, 301,
        (("5?", "5S"),), ("black",),
    )
    assert tracker.ingest(first) is None
    assert tracker.ingest(changed) is None
    candidate = tracker.ingest(stable, version=version())
    assert candidate is not None
    assert candidate.evidence_ids == ("uncertain-2", "uncertain-3")


def test_unknown_suit_then_exact_read_keeps_one_confirmation_streak() -> None:
    tracker = SeatTracker(Seat.LEFT)
    first = SeatObservation(
        "unknown-1", frame(1), Seat.LEFT, ObservationKind.PLAY, ("7?",),
        0.9, ObservationReason.CARDS_RECOGNIZED, 101,
        (("7?", "7C", "7D"),), (),
    )
    exact = SeatObservation(
        "unknown-2", frame(2), Seat.LEFT, ObservationKind.PLAY, ("7C",),
        0.9, ObservationReason.CARDS_RECOGNIZED, 201,
        (("7C",),), (),
    )
    assert tracker.ingest(first) is None
    candidate = tracker.ingest(exact, version=version())
    assert candidate is not None
    assert candidate.cards == ("7C",)
    assert candidate.suit_options == (("7C",),)
    assert candidate.evidence_ids == ("unknown-1", "unknown-2")
