from __future__ import annotations

from daguandan_bridge.live_v2.candidates import ActionKind
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat
from daguandan_bridge.live_v2.single_turn_observer import (
    ObservationDisposition,
    SeatDisplay,
    SingleTurnObserver,
)


def _frame(index: int) -> FrameIdentity:
    return FrameIdentity("observer", 1, index, 2_000 + index, "roi", "capture")


def _display(
    seat: Seat,
    index: int,
    *,
    cards: tuple[str, ...] = (),
    suits: tuple[tuple[str, ...], ...] = (),
    kind: ActionKind | None = None,
    epoch: int = 0,
) -> SeatDisplay:
    return SeatDisplay(seat, kind, cards, suits, _frame(index), 0.9, epoch)


def test_foreign_seat_is_observed_but_never_enters_pending_state() -> None:
    observer = SingleTurnObserver(current_seat=Seat.LEFT, turn_token=1)
    observer.begin_turn(current_seat=Seat.LEFT, turn_token=1, baseline=_display(Seat.LEFT, 1))
    result = observer.observe(_display(Seat.SELF, 2, cards=("4S",), suits=(("S",),), kind=ActionKind.PLAY))
    assert result.disposition is ObservationDisposition.IGNORED_FOREIGN
    assert observer.pending is None


def test_persistent_display_is_not_recommitted() -> None:
    baseline = _display(Seat.LEFT, 1, cards=("2S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=4)
    observer = SingleTurnObserver(current_seat=Seat.LEFT, turn_token=1)
    observer.begin_turn(current_seat=Seat.LEFT, turn_token=1, baseline=baseline)
    result = observer.observe(_display(Seat.LEFT, 2, cards=("2S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=4))
    assert result.disposition is ObservationDisposition.UNCHANGED
    assert observer.pending is None


def test_changed_display_needs_two_independent_frames_and_emits_one_pending_action() -> None:
    observer = SingleTurnObserver(current_seat=Seat.LEFT, turn_token=3)
    observer.begin_turn(current_seat=Seat.LEFT, turn_token=3, baseline=_display(Seat.LEFT, 1))
    first = observer.observe(_display(Seat.LEFT, 2, cards=("2S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=1))
    second = observer.observe(_display(Seat.LEFT, 3, cards=("2S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=1))
    assert first.disposition is ObservationDisposition.WAITING_CONFIRMATION
    assert first.pending is None
    assert second.disposition is ObservationDisposition.CONFIRMED
    assert second.pending is not None
    assert second.pending.seat is Seat.LEFT
    assert second.pending.first_frame.frame_sequence == 2
    assert second.pending.last_frame.frame_sequence == 3


def test_more_than_five_cards_can_confirm_with_partial_suits() -> None:
    cards = ("3?", "3S", "4?", "4H", "5?", "5C")
    suits = (("C", "D"), ("S",), ("C", "D"), ("H",), ("D", "S"), ("C",))
    observer = SingleTurnObserver(current_seat=Seat.LEFT, turn_token=2)
    observer.begin_turn(current_seat=Seat.LEFT, turn_token=2, baseline=_display(Seat.LEFT, 1))
    observer.observe(_display(Seat.LEFT, 2, cards=cards, suits=suits, kind=ActionKind.PLAY, epoch=1))
    result = observer.observe(_display(Seat.LEFT, 3, cards=cards, suits=suits, kind=ActionKind.PLAY, epoch=1))
    assert result.pending is not None
    assert result.pending.partial_suits
    assert len(result.pending.cards) == 6


def test_same_cards_in_a_new_surface_epoch_are_a_new_action() -> None:
    baseline = _display(Seat.RIGHT, 1, cards=("2S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=3)
    observer = SingleTurnObserver(current_seat=Seat.RIGHT, turn_token=9)
    observer.begin_turn(current_seat=Seat.RIGHT, turn_token=9, baseline=baseline)
    observer.observe(_display(Seat.RIGHT, 2, cards=("2S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=4))
    result = observer.observe(_display(Seat.RIGHT, 3, cards=("2S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=4))
    assert result.disposition is ObservationDisposition.CONFIRMED


def test_changed_first_sample_is_replaced_instead_of_queued() -> None:
    observer = SingleTurnObserver(current_seat=Seat.OPPOSITE, turn_token=5)
    observer.begin_turn(current_seat=Seat.OPPOSITE, turn_token=5, baseline=_display(Seat.OPPOSITE, 1))
    observer.observe(_display(Seat.OPPOSITE, 2, cards=("3S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=1))
    changed = observer.observe(_display(Seat.OPPOSITE, 3, cards=("4S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=1))
    assert changed.disposition is ObservationDisposition.WAITING_CONFIRMATION
    assert observer.pending is None
    result = observer.observe(_display(Seat.OPPOSITE, 4, cards=("4S",), suits=(("S",),), kind=ActionKind.PLAY, epoch=1))
    assert result.pending is not None and result.pending.cards == ("4S",)


def test_pass_requires_a_fresh_surface_epoch() -> None:
    baseline = _display(Seat.SELF, 1, kind=ActionKind.PASS, epoch=2)
    observer = SingleTurnObserver(current_seat=Seat.SELF, turn_token=7)
    observer.begin_turn(current_seat=Seat.SELF, turn_token=7, baseline=baseline)
    stale = observer.observe(_display(Seat.SELF, 2, kind=ActionKind.PASS, epoch=2))
    assert stale.disposition is ObservationDisposition.UNCHANGED
    observer.observe(_display(Seat.SELF, 3, kind=ActionKind.PASS, epoch=3))
    fresh = observer.observe(_display(Seat.SELF, 4, kind=ActionKind.PASS, epoch=3))
    assert fresh.disposition is ObservationDisposition.CONFIRMED


def test_unknown_suit_and_later_exact_suit_are_one_action_with_narrowed_options() -> None:
    observer = SingleTurnObserver(current_seat=Seat.LEFT, turn_token=3)
    observer.begin_turn(current_seat=Seat.LEFT, turn_token=3, baseline=_display(Seat.LEFT, 1))
    observer.observe(
        _display(
            Seat.LEFT, 2, cards=("7?",), suits=(("C", "D"),),
            kind=ActionKind.PLAY, epoch=1,
        )
    )
    result = observer.observe(
        _display(
            Seat.LEFT, 3, cards=("7C",), suits=(("C",),),
            kind=ActionKind.PLAY, epoch=1,
        )
    )
    assert result.disposition is ObservationDisposition.CONFIRMED
    assert result.pending is not None
    assert result.pending.cards == ("7C",)
    assert result.pending.suit_options == (("C",),)
