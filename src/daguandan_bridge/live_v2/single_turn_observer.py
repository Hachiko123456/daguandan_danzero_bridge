"""One-seat-at-a-time visual action confirmation.

The observer deliberately does not know game rules or models.  It compares the
current seat's visible zone with its last stable baseline and exposes at most
one pending action for the turn core to commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .candidates import ActionKind
from .identity import FrameIdentity, Seat
from .turn_core import PendingAction


class ObservationDisposition(str, Enum):
    IGNORED_FOREIGN = "ignored_foreign"
    IGNORED_STALE = "ignored_stale"
    UNCHANGED = "unchanged"
    WAITING_CONFIRMATION = "waiting_confirmation"
    CONFIRMED = "confirmed"


@dataclass(frozen=True, slots=True)
class SeatDisplay:
    """A normalized read of one seat's visible play/PASS zone."""

    seat: Seat
    kind: ActionKind | None
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    frame: FrameIdentity
    confidence: float = 0.0
    surface_epoch: int = 0

    def __post_init__(self) -> None:
        if self.kind is None:
            if self.cards or self.suit_options:
                raise ValueError("unknown display cannot contain cards")
        elif self.kind is ActionKind.PASS:
            if self.cards or self.suit_options:
                raise ValueError("pass display cannot contain cards")
        elif self.kind is ActionKind.PLAY:
            if not self.cards or len(self.cards) != len(self.suit_options):
                raise ValueError("play display cards and suits must align")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("display confidence must be between zero and one")
        if isinstance(self.surface_epoch, bool) or self.surface_epoch < 0:
            raise ValueError("surface_epoch must be non-negative")

    @property
    def signature(self) -> tuple[ActionKind | None, tuple[str, ...], tuple[tuple[str, ...], ...]]:
        return self.kind, self.cards, self.suit_options


@dataclass(frozen=True, slots=True)
class ObservationResult:
    disposition: ObservationDisposition
    pending: PendingAction | None = None
    reason: str = ""


class SingleTurnObserver:
    """Compare only the authoritative current seat against one baseline."""

    def __init__(self, *, current_seat: Seat, turn_token: int) -> None:
        self._current_seat = current_seat
        self._turn_token = int(turn_token)
        self._baseline: SeatDisplay | None = None
        self._first_changed: SeatDisplay | None = None
        self._confirmed_display: SeatDisplay | None = None
        self._pending: PendingAction | None = None

    @property
    def current_seat(self) -> Seat:
        return self._current_seat

    @property
    def turn_token(self) -> int:
        return self._turn_token

    @property
    def baseline(self) -> SeatDisplay | None:
        return self._baseline

    @property
    def pending(self) -> PendingAction | None:
        return self._pending

    def begin_turn(self, *, current_seat: Seat, turn_token: int, baseline: SeatDisplay) -> None:
        if baseline.seat is not current_seat:
            raise ValueError("baseline must belong to the current seat")
        self._current_seat = current_seat
        self._turn_token = int(turn_token)
        self._baseline = baseline
        self._first_changed = None
        self._confirmed_display = None
        self._pending = None

    def observe(self, display: SeatDisplay) -> ObservationResult:
        if display.seat is not self._current_seat:
            return ObservationResult(
                ObservationDisposition.IGNORED_FOREIGN,
                reason="foreign_seat",
            )
        if self._baseline is not None and not _strictly_after(
            display.frame, self._baseline.frame
        ):
            return ObservationResult(
                ObservationDisposition.IGNORED_STALE,
                reason="frame_not_after_baseline",
            )
        if display.kind is None:
            return ObservationResult(
                ObservationDisposition.UNCHANGED,
                reason="unknown_or_animating",
            )
        if (
            self._baseline is not None
            and display.signature == self._baseline.signature
            and display.surface_epoch == self._baseline.surface_epoch
        ):
            return ObservationResult(
                ObservationDisposition.UNCHANGED,
                reason="display_matches_baseline",
            )
        if self._pending is None and self._first_changed is None:
            self._first_changed = display
            return ObservationResult(
                ObservationDisposition.WAITING_CONFIRMATION,
                reason="first_changed_sample",
            )
        reference = self._pending or self._first_changed
        if reference is None:
            raise RuntimeError("observer lost pending display reference")
        reference_signature = (
            reference.signature
            if isinstance(reference, SeatDisplay)
            else (reference.kind, reference.cards, reference.suit_options)
        )
        if display.signature != reference_signature:
            self._first_changed = display
            self._pending = None
            return ObservationResult(
                ObservationDisposition.WAITING_CONFIRMATION,
                reason="changed_before_confirmation",
            )
        if self._pending is None:
            assert self._first_changed is not None
            self._pending = PendingAction(
                seat=display.seat,
                turn_token=self._turn_token,
                kind=display.kind,
                cards=display.cards,
                suit_options=display.suit_options,
                first_frame=self._first_changed.frame,
                last_frame=display.frame,
                confidence=max(self._first_changed.confidence, display.confidence),
            )
            self._confirmed_display = display
        else:
            self._pending = self._pending.with_later_evidence(
                last_frame=display.frame,
                confidence=max(self._pending.confidence, display.confidence),
            )
            self._confirmed_display = display
        return ObservationResult(
            ObservationDisposition.CONFIRMED,
            pending=self._pending,
            reason="independent_second_sample",
        )

    def accept_committed(self, action: PendingAction | object) -> None:
        """Advance the display baseline after the turn core commits one action."""

        if self._pending is None:
            raise ValueError("no pending display to accept")
        if getattr(action, "seat", None) is not self._current_seat:
            raise ValueError("committed action does not belong to current seat")
        if self._confirmed_display is None:
            raise ValueError("pending action has not reached independent confirmation")
        self._baseline = self._confirmed_display
        self._first_changed = None
        self._confirmed_display = None
        self._pending = None


def _strictly_after(later: FrameIdentity, earlier: FrameIdentity) -> bool:
    return bool(
        later.session_id == earlier.session_id
        and later.capture_generation == earlier.capture_generation
        and later.roi_version == earlier.roi_version
        and later.source_id == earlier.source_id
        and later.frame_sequence > earlier.frame_sequence
        and later.captured_ms > earlier.captured_ms
    )


__all__ = [
    "ObservationDisposition",
    "ObservationResult",
    "SeatDisplay",
    "SingleTurnObserver",
]
