"""Explicit, auditable corrections to the latest confirmed action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .action_semantics import ActionSemantics
from .candidates import ActionKind, EvidenceOrigin
from .identity import (
    Seat,
    StateVersion,
    VersionIdentity,
    require_enum,
    require_instance,
    require_non_negative_int,
    require_probability,
    require_text,
)
from .observations import require_cards, require_suit_options


class CorrectionReason(str, Enum):
    """Why an authoritative operator or trusted subsystem corrected an action."""

    MANUAL_REVIEW = "manual_review"
    TRUSTED_REPLAY = "trusted_replay"
    ADJACENT_ACTION_REVIEW = "adjacent_action_review"


@dataclass(frozen=True, slots=True)
class CorrectionCommand:
    correction_id: str
    expected_version: VersionIdentity
    target_action_id: str
    kind: ActionKind
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    reason: CorrectionReason
    evidence_id: str
    evidence_origin: EvidenceOrigin
    confidence: float
    corrected_ms: int
    requested_semantics: ActionSemantics | None = None

    def __post_init__(self) -> None:
        require_text(self.correction_id, "correction_id")
        require_instance(self.expected_version, VersionIdentity, "expected_version")
        require_text(self.target_action_id, "target_action_id")
        require_enum(self.kind, ActionKind, "kind")
        require_enum(self.reason, CorrectionReason, "reason")
        require_text(self.evidence_id, "evidence_id")
        require_enum(self.evidence_origin, EvidenceOrigin, "evidence_origin")
        if self.evidence_origin not in {EvidenceOrigin.MANUAL, EvidenceOrigin.TRUSTED}:
            raise ValueError("correction evidence must be MANUAL or TRUSTED")
        require_probability(self.confidence, "confidence")
        require_non_negative_int(self.corrected_ms, "corrected_ms")
        if self.requested_semantics is not None and not isinstance(
            self.requested_semantics, ActionSemantics
        ):
            raise TypeError("requested_semantics must be ActionSemantics or None")
        _validate_action(self.kind, self.cards, self.suit_options)


@dataclass(frozen=True, slots=True)
class ConfirmedCorrection:
    correction_id: str
    target_action_id: str
    version_before: StateVersion
    version_after: StateVersion
    seat: Seat
    previous_kind: ActionKind
    previous_cards: tuple[str, ...]
    previous_suit_options: tuple[tuple[str, ...], ...]
    corrected_kind: ActionKind
    corrected_cards: tuple[str, ...]
    corrected_suit_options: tuple[tuple[str, ...], ...]
    reason: CorrectionReason
    evidence_id: str
    evidence_origin: EvidenceOrigin
    confidence: float
    corrected_ms: int
    previous_semantics: ActionSemantics | None = None
    corrected_semantics: ActionSemantics | None = None

    def __post_init__(self) -> None:
        require_text(self.correction_id, "correction_id")
        require_text(self.target_action_id, "target_action_id")
        require_instance(self.version_before, StateVersion, "version_before")
        require_instance(self.version_after, StateVersion, "version_after")
        require_enum(self.seat, Seat, "seat")
        require_enum(self.previous_kind, ActionKind, "previous_kind")
        require_enum(self.corrected_kind, ActionKind, "corrected_kind")
        require_enum(self.reason, CorrectionReason, "reason")
        require_text(self.evidence_id, "evidence_id")
        require_enum(self.evidence_origin, EvidenceOrigin, "evidence_origin")
        if self.evidence_origin not in {EvidenceOrigin.MANUAL, EvidenceOrigin.TRUSTED}:
            raise ValueError("correction evidence must be MANUAL or TRUSTED")
        require_probability(self.confidence, "confidence")
        require_non_negative_int(self.corrected_ms, "corrected_ms")
        for name, semantics in (
            ("previous_semantics", self.previous_semantics),
            ("corrected_semantics", self.corrected_semantics),
        ):
            if semantics is not None and not isinstance(semantics, ActionSemantics):
                raise TypeError(f"{name} must be ActionSemantics or None")
        if self.corrected_kind is ActionKind.PASS and self.corrected_semantics is not None:
            raise ValueError("corrected PASS cannot contain action semantics")
        _validate_action(self.previous_kind, self.previous_cards, self.previous_suit_options)
        _validate_action(self.corrected_kind, self.corrected_cards, self.corrected_suit_options)
        before, after = self.version_before, self.version_after
        if before.session_id != after.session_id:
            raise ValueError("correction cannot cross session")
        if after.state_revision != before.state_revision + 1:
            raise ValueError("correction must advance state_revision exactly once")
        if after.turn_index != before.turn_index:
            raise ValueError("correction must not advance turn_index")
        if (
            self.previous_kind is self.corrected_kind
            and self.previous_cards == self.corrected_cards
            and self.previous_suit_options == self.corrected_suit_options
            and self.previous_semantics == self.corrected_semantics
        ):
            raise ValueError("correction must change the effective action")


def _validate_action(
    kind: ActionKind,
    cards: tuple[str, ...],
    suit_options: tuple[tuple[str, ...], ...],
) -> None:
    require_cards(cards, allow_empty=kind is ActionKind.PASS)
    if kind is ActionKind.PLAY:
        require_suit_options(cards, suit_options)
    elif cards or suit_options:
        raise ValueError("pass correction cannot contain cards or suit_options")


__all__ = ["ConfirmedCorrection", "CorrectionCommand", "CorrectionReason"]
