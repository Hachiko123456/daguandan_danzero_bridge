"""Seat-local visual observations, before action confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .identity import (
    FrameIdentity,
    Seat,
    require_enum,
    require_instance,
    require_non_negative_int,
    require_probability,
    require_text,
    require_tuple,
    require_unique_text_tuple,
)


class ObservationKind(str, Enum):
    EMPTY = "empty"
    PASS = "pass"
    PLAY = "play"
    ANIMATING = "animating"
    UNKNOWN = "unknown"


class ObservationReason(str, Enum):
    STABLE_EMPTY = "stable_empty"
    PASS_MARKER = "pass_marker"
    CARDS_RECOGNIZED = "cards_recognized"
    ANIMATION_DETECTED = "animation_detected"
    UNREADABLE = "unreadable"
    CONFLICTING_SIGNALS = "conflicting_signals"
    STALE_CAPTURE = "stale_capture"
    DETECTOR_FAILURE = "detector_failure"


def require_cards(cards: tuple[str, ...], *, allow_empty: bool) -> None:
    require_tuple(cards, "cards")
    if not allow_empty and not cards:
        raise ValueError("cards must not be empty for a play")
    for card in cards:
        require_text(card, "card")


def require_suit_options(
    cards: tuple[str, ...],
    suit_options: tuple[tuple[str, ...], ...],
    *,
    require_selected_card: bool = True,
) -> None:
    require_tuple(suit_options, "suit_options")
    if len(suit_options) != len(cards):
        raise ValueError("PLAY suit_options must align one-to-one with cards")
    for card, options in zip(cards, suit_options, strict=True):
        require_unique_text_tuple(options, "suit option")
        if not options:
            raise ValueError("each PLAY card requires at least one physical option")
        if require_selected_card and card not in options:
            raise ValueError("each PLAY card must occur in its physical options")


@dataclass(frozen=True, slots=True)
class SeatObservation:
    """One seat's interpretation from one immutable captured frame.

    ``suit_options`` is aligned with physical card entities, not unique card
    labels. Thus duplicate cards remain distinct entries. Diagnostics are
    normalized text facts only; decisions use the enum ``reason``.
    """

    observation_id: str
    frame: FrameIdentity
    seat: Seat
    kind: ObservationKind
    cards: tuple[str, ...]
    confidence: float
    reason: ObservationReason
    processing_ms: int
    suit_options: tuple[tuple[str, ...], ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.observation_id, "observation_id")
        require_instance(self.frame, FrameIdentity, "frame")
        require_enum(self.seat, Seat, "seat")
        require_enum(self.kind, ObservationKind, "kind")
        require_enum(self.reason, ObservationReason, "reason")
        require_non_negative_int(self.processing_ms, "processing_ms")
        if self.processing_ms < self.frame.captured_ms:
            raise ValueError("processing_ms must not precede captured_ms")
        require_probability(self.confidence, "confidence")
        require_tuple(self.diagnostics, "diagnostics")
        for diagnostic in self.diagnostics:
            require_text(diagnostic, "diagnostic")
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(self.diagnostics)))
        is_play = self.kind is ObservationKind.PLAY
        require_cards(self.cards, allow_empty=not is_play)
        if is_play:
            require_suit_options(self.cards, self.suit_options)
        else:
            if self.cards:
                raise ValueError(f"{self.kind.value} observation cannot contain cards")
            require_tuple(self.suit_options, "suit_options")
            if self.suit_options:
                raise ValueError("non-PLAY observation cannot contain suit_options")
        expected_reason = {
            ObservationKind.EMPTY: ObservationReason.STABLE_EMPTY,
            ObservationKind.PASS: ObservationReason.PASS_MARKER,
            ObservationKind.PLAY: ObservationReason.CARDS_RECOGNIZED,
            ObservationKind.ANIMATING: ObservationReason.ANIMATION_DETECTED,
        }.get(self.kind)
        if expected_reason is not None and self.reason is not expected_reason:
            raise ValueError(
                f"{self.kind.value} observation requires reason {expected_reason.value}"
            )
