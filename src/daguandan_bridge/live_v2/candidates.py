"""Immutable observed-candidate and confirmed-action contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .action_semantics import ActionSemantics
from .identity import (
    FrameIdentity,
    Seat,
    StateVersion,
    VersionIdentity,
    require_enum,
    require_instance,
    require_non_negative_int,
    require_probability,
    require_text,
    require_tuple,
    require_unique_text_tuple,
)
from .observations import require_cards, require_suit_options


class ActionKind(str, Enum):
    PASS = "pass"
    PLAY = "play"


class CandidateReason(str, Enum):
    STABLE_PLAY = "stable_play"
    FRESH_PASS_EDGE = "fresh_pass_edge"
    CROSS_SOURCE_PASS = "cross_source_pass"
    LOCAL_ACTION_CONFIRMED = "local_action_confirmed"
    OPENING_ACTION_CONFIRMED = "opening_action_confirmed"
    RECONCILIATION_EVIDENCE = "reconciliation_evidence"
    VISUAL_CORRECTION = "visual_correction"


class EvidenceOrigin(str, Enum):
    """Auditable origin category for candidate evidence."""

    VISUAL = "visual"
    MANUAL = "manual"
    OPENING = "opening"
    TRUSTED = "trusted"


class ConfirmationReason(str, Enum):
    ORDERED_EVIDENCE = "ordered_evidence"
    RULE_VALIDATED = "rule_validated"
    ATOMIC_RECONCILIATION = "atomic_reconciliation"
    LOCAL_ACTION_COMMITTED = "local_action_committed"


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    candidate_id: str
    version: VersionIdentity
    seat: Seat
    kind: ActionKind
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    evidence_ids: tuple[str, ...]
    action_epoch: int
    first_frame: FrameIdentity
    last_frame: FrameIdentity
    processing_ms: int
    confidence: float
    reason: CandidateReason
    evidence_origin: EvidenceOrigin = EvidenceOrigin.VISUAL
    audit_evidence_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    requested_semantics: ActionSemantics | None = None

    def __post_init__(self) -> None:
        require_text(self.candidate_id, "candidate_id")
        require_enum(self.seat, Seat, "seat")
        require_enum(self.kind, ActionKind, "kind")
        require_enum(self.reason, CandidateReason, "reason")
        require_enum(self.evidence_origin, EvidenceOrigin, "evidence_origin")
        require_instance(self.version, VersionIdentity, "version")
        require_instance(self.first_frame, FrameIdentity, "first_frame")
        require_instance(self.last_frame, FrameIdentity, "last_frame")
        require_unique_text_tuple(self.evidence_ids, "evidence_ids")
        require_unique_text_tuple(self.audit_evidence_ids, "audit_evidence_ids")
        require_unique_text_tuple(self.diagnostics, "diagnostics")
        if self.requested_semantics is not None and not isinstance(
            self.requested_semantics, ActionSemantics
        ):
            raise TypeError("requested_semantics must be ActionSemantics or None")
        require_non_negative_int(self.action_epoch, "action_epoch")
        first_stream = (
            self.first_frame.session_id,
            self.first_frame.capture_generation,
            self.first_frame.roi_version,
            self.first_frame.source_id,
        )
        last_stream = (
            self.last_frame.session_id,
            self.last_frame.capture_generation,
            self.last_frame.roi_version,
            self.last_frame.source_id,
        )
        if first_stream != last_stream:
            raise ValueError("candidate evidence frames must belong to one stream")
        if self.version.session_id != self.first_frame.session_id or (
            self.version.capture_generation != self.first_frame.capture_generation
        ):
            raise ValueError("candidate version must match its evidence stream")
        ordered_visual_reasons = {
            CandidateReason.STABLE_PLAY,
            CandidateReason.FRESH_PASS_EDGE,
            CandidateReason.RECONCILIATION_EVIDENCE,
        }
        if self.reason in ordered_visual_reasons:
            if self.evidence_origin is not EvidenceOrigin.VISUAL:
                raise ValueError("visual candidate reason requires VISUAL evidence origin")
            if len(self.evidence_ids) < 2:
                raise ValueError("visual candidates require at least two observations")
            if self.audit_evidence_ids:
                raise ValueError("visual candidates cannot claim audit evidence")
            if self.last_frame.frame_sequence <= self.first_frame.frame_sequence:
                raise ValueError("candidate evidence frame_sequence must strictly increase")
            if self.last_frame.captured_ms <= self.first_frame.captured_ms:
                raise ValueError("candidate evidence captured_ms must strictly increase")
        elif self.reason is CandidateReason.CROSS_SOURCE_PASS:
            if self.evidence_origin is not EvidenceOrigin.VISUAL:
                raise ValueError("cross-source pass requires VISUAL evidence origin")
            if self.kind is not ActionKind.PASS:
                raise ValueError("cross-source pass reason requires a PASS candidate")
            if len(self.evidence_ids) != 2:
                raise ValueError("cross-source pass requires exactly two evidence IDs")
            if self.audit_evidence_ids:
                raise ValueError("cross-source pass cannot claim audit evidence")
            if self.first_frame != self.last_frame:
                raise ValueError("cross-source pass evidence must use one frame")
        elif self.reason is CandidateReason.VISUAL_CORRECTION:
            if self.evidence_origin is not EvidenceOrigin.VISUAL:
                raise ValueError("visual correction requires VISUAL evidence origin")
            if len(self.evidence_ids) < 1:
                raise ValueError("visual correction requires evidence")
            if self.audit_evidence_ids:
                raise ValueError("visual correction cannot claim audit evidence")
        else:
            allowed_origins = (
                {EvidenceOrigin.MANUAL, EvidenceOrigin.TRUSTED}
                if self.reason is CandidateReason.LOCAL_ACTION_CONFIRMED
                else {EvidenceOrigin.OPENING, EvidenceOrigin.TRUSTED}
            )
            if self.evidence_origin not in allowed_origins:
                raise ValueError("trusted candidate reason has an invalid evidence origin")
            if self.audit_evidence_ids:
                if len(self.audit_evidence_ids) != 1 or not self.evidence_ids:
                    raise ValueError("reviewed candidates require one audit and source evidence")
            elif len(self.evidence_ids) != 1:
                raise ValueError("trusted candidates require exactly one evidence ID")
            if self.first_frame != self.last_frame:
                raise ValueError("trusted single evidence must use one identical frame")
            prefix, separator, audit_id = self.first_frame.source_id.partition(":")
            if (
                not separator
                or prefix != self.evidence_origin.value
                or not audit_id.strip()
            ):
                raise ValueError(
                    "trusted evidence source_id must contain its origin and audit ID"
                )
            if self.audit_evidence_ids and (
                self.first_frame.source_id != self.event_evidence_ids[0]
            ):
                raise ValueError("trusted frame source_id must match its event audit evidence")
        require_non_negative_int(self.processing_ms, "processing_ms")
        if self.last_frame.captured_ms > self.processing_ms:
            raise ValueError("processing_ms must not precede captured evidence")
        require_probability(self.confidence, "confidence")
        require_cards(self.cards, allow_empty=self.kind is ActionKind.PASS)
        if self.kind is ActionKind.PLAY:
            require_suit_options(self.cards, self.suit_options)
        else:
            require_tuple(self.suit_options, "suit_options")
            if self.cards or self.suit_options:
                raise ValueError("pass candidate cannot contain cards or suit_options")

    @property
    def first_captured_ms(self) -> int:
        return self.first_frame.captured_ms

    @property
    def last_captured_ms(self) -> int:
        return self.last_frame.captured_ms

    @property
    def event_evidence_ids(self) -> tuple[str, ...]:
        """Evidence written on the formal action event."""

        return self.audit_evidence_ids or self.evidence_ids

    @property
    def all_evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.evidence_ids + self.audit_evidence_ids))


@dataclass(frozen=True, slots=True)
class ConfirmedAction:
    """A formal event that embeds its immutable source candidate."""

    action_id: str
    source_candidate: ActionCandidate
    version_before: StateVersion
    version_after: StateVersion
    cards: tuple[str, ...]
    suit_options: tuple[tuple[str, ...], ...]
    action_epoch: int
    processing_ms: int
    reason: ConfirmationReason
    semantics: ActionSemantics | None = None

    @classmethod
    def from_candidate(
        cls,
        *,
        action_id: str,
        candidate: ActionCandidate,
        version_before: StateVersion,
        version_after: StateVersion,
        processing_ms: int,
        reason: ConfirmationReason,
        cards: tuple[str, ...] | None = None,
        semantics: ActionSemantics | None = None,
    ) -> ConfirmedAction:
        """Create a formal event without dropping candidate semantics."""

        return cls(
            action_id=action_id,
            source_candidate=candidate,
            version_before=version_before,
            version_after=version_after,
            cards=candidate.cards if cards is None else cards,
            suit_options=candidate.suit_options,
            action_epoch=candidate.action_epoch,
            processing_ms=processing_ms,
            reason=reason,
            semantics=semantics,
        )

    def __post_init__(self) -> None:
        require_text(self.action_id, "action_id")
        require_enum(self.reason, ConfirmationReason, "reason")
        require_instance(self.source_candidate, ActionCandidate, "source_candidate")
        require_instance(self.version_before, StateVersion, "version_before")
        require_instance(self.version_after, StateVersion, "version_after")
        before, after = self.version_before, self.version_after
        if before.session_id != after.session_id:
            raise ValueError("confirmed action cannot cross session")
        if after.state_revision != before.state_revision + 1:
            raise ValueError("confirmed action must advance state_revision exactly once")
        if after.turn_index != before.turn_index + 1:
            raise ValueError("confirmed action must advance turn_index exactly once")
        if before.session_id != self.source_candidate.version.session_id:
            raise ValueError("confirmed action state must match its evidence session")
        require_non_negative_int(self.action_epoch, "action_epoch")
        if self.action_epoch != self.source_candidate.action_epoch:
            raise ValueError("confirmed action_epoch must match source candidate")
        require_non_negative_int(self.processing_ms, "processing_ms")
        if self.semantics is not None and not isinstance(
            self.semantics, ActionSemantics
        ):
            raise TypeError("semantics must be ActionSemantics or None")
        if self.processing_ms < self.last_frame.captured_ms:
            raise ValueError("processing_ms must not precede captured evidence")
        require_cards(self.cards, allow_empty=self.kind is ActionKind.PASS)
        if self.kind is ActionKind.PLAY:
            require_suit_options(
                self.cards, self.suit_options, require_selected_card=False
            )
            if self.suit_options != self.source_candidate.suit_options:
                raise ValueError("confirmed suit_options must match source candidate")
            if len(self.cards) != len(self.source_candidate.cards):
                raise ValueError("confirmed cards must align with source candidate entities")
        else:
            require_tuple(self.suit_options, "suit_options")
            if self.cards or self.suit_options:
                raise ValueError("confirmed pass cannot contain cards or suit_options")
            if self.semantics is not None:
                raise ValueError("confirmed pass cannot contain action semantics")

    @property
    def partial_suits(self) -> bool:
        """Whether any confirmed physical card still has multiple suit options."""
        return self.kind is ActionKind.PLAY and any(
            len(options) != 1 for options in self.suit_options
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return (self.source_candidate.candidate_id,)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.source_candidate.evidence_ids

    @property
    def event_evidence_ids(self) -> tuple[str, ...]:
        return self.source_candidate.event_evidence_ids

    @property
    def audit_evidence_ids(self) -> tuple[str, ...]:
        return self.source_candidate.audit_evidence_ids

    @property
    def evidence_origin(self) -> EvidenceOrigin:
        return self.source_candidate.evidence_origin

    @property
    def seat(self) -> Seat:
        return self.source_candidate.seat

    @property
    def kind(self) -> ActionKind:
        return self.source_candidate.kind

    @property
    def first_frame(self) -> FrameIdentity:
        return self.source_candidate.first_frame

    @property
    def last_frame(self) -> FrameIdentity:
        return self.source_candidate.last_frame

    @property
    def captured_ms(self) -> int:
        return self.last_frame.captured_ms
