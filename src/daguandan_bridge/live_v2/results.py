"""Immutable gap, opportunity, update, and transaction result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .candidates import ActionCandidate, ConfirmedAction
from .identity import (
    Seat,
    VersionIdentity,
    require_enum,
    require_non_negative_int,
    require_text,
    require_tuple,
    require_unique_text_tuple,
)
from .observations import SeatObservation


class GapPhase(str, Enum):
    CLEAR = "clear"
    OBSERVING = "observing"
    RECOVERABLE = "recoverable"
    BLOCKING = "blocking"
    EXPIRED = "expired"


class GapReason(str, Enum):
    NONE = "none"
    MISSING_EXPECTED_ACTION = "missing_expected_action"
    OUT_OF_ORDER_EVIDENCE = "out_of_order_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    STALE_EVIDENCE = "stale_evidence"
    RULE_REJECTION = "rule_rejection"
    CAPTURE_DISCONTINUITY = "capture_discontinuity"
    OBSERVATION_FAILURE = "observation_failure"
    RECOVERY_BUDGET_EXCEEDED = "recovery_budget_exceeded"


class OpportunityStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    CLOSED = "closed"


class OpportunityReason(str, Enum):
    TRUSTED_STATE = "trusted_state"
    HISTORY_GAP = "history_gap"
    NOT_LOCAL_TURN = "not_local_turn"
    OBSERVATION_UNCERTAIN = "observation_uncertain"
    TERMINAL_STATE = "terminal_state"
    SUPERSEDED = "superseded"


class EngineUpdateReason(str, Enum):
    OBSERVATION_RECEIVED = "observation_received"
    CANDIDATE_CREATED = "candidate_created"
    ACTION_COMMITTED = "action_committed"
    GAP_CHANGED = "gap_changed"
    OPPORTUNITY_CHANGED = "opportunity_changed"
    CAPTURE_GENERATION_CHANGED = "capture_generation_changed"


class ProjectionReason(str, Enum):
    ACCEPTED = "accepted"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    OUT_OF_ORDER = "out_of_order"
    RULE_REJECTED = "rule_rejected"
    VERSION_MISMATCH = "version_mismatch"


class CommitReason(str, Enum):
    COMMITTED = "committed"
    VERSION_CONFLICT = "version_conflict"
    TRANSACTION_REJECTED = "transaction_rejected"
    PERSISTENCE_FAILED = "persistence_failed"


class EvidenceDropReason(str, Enum):
    AGE_BUDGET = "age_budget"
    BYTE_BUDGET = "byte_budget"
    CONSUMED_BOUNDARY = "consumed_boundary"
    COUNT_BUDGET = "count_budget"
    EXPIRED_ON_ARRIVAL = "expired_on_arrival"
    OVERSIZE = "oversize"


class SchedulingDropReason(str, Enum):
    RAW_EXPIRED = "raw_expired"
    RAW_REPLACED = "raw_replaced"
    STALE_RAW = "stale_raw"
    STREAM_MISMATCH = "stream_mismatch"
    STREAM_REBOUND = "stream_rebound"
    CANDIDATE_CAPACITY = "candidate_capacity"
    CANDIDATE_DUPLICATE = "candidate_duplicate"
    CANDIDATE_EXPIRED = "candidate_expired"


class ScheduledItemKind(str, Enum):
    RAW = "raw"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class GapState:
    version: VersionIdentity
    phase: GapPhase
    reason: GapReason
    expected_seats: tuple[Seat, ...]
    evidence_ids: tuple[str, ...]
    opened_captured_ms: int
    processing_ms: int

    def __post_init__(self) -> None:
        require_enum(self.phase, GapPhase, "phase")
        require_enum(self.reason, GapReason, "reason")
        require_tuple(self.expected_seats, "expected_seats")
        for seat in self.expected_seats:
            require_enum(seat, Seat, "expected_seat")
        if len(set(self.expected_seats)) != len(self.expected_seats):
            raise ValueError("expected_seats must not contain duplicates")
        require_unique_text_tuple(self.evidence_ids, "evidence_ids")
        require_non_negative_int(self.opened_captured_ms, "opened_captured_ms")
        require_non_negative_int(self.processing_ms, "processing_ms")
        if self.processing_ms < self.opened_captured_ms:
            raise ValueError("processing_ms must not precede opened_captured_ms")
        if self.phase is GapPhase.CLEAR:
            if self.reason is not GapReason.NONE:
                raise ValueError("clear gap state requires reason NONE")
            if self.expected_seats or self.evidence_ids:
                raise ValueError("clear gap state cannot retain recovery evidence")
        elif self.reason is GapReason.NONE:
            raise ValueError("non-clear gap state requires a concrete GapReason")


@dataclass(frozen=True, slots=True)
class AdviceOpportunity:
    opportunity_id: str
    version: VersionIdentity
    seat: Seat
    status: OpportunityStatus
    reason: OpportunityReason
    captured_ms: int
    processing_ms: int

    def __post_init__(self) -> None:
        require_text(self.opportunity_id, "opportunity_id")
        require_enum(self.seat, Seat, "seat")
        require_enum(self.status, OpportunityStatus, "status")
        require_enum(self.reason, OpportunityReason, "reason")
        require_non_negative_int(self.captured_ms, "captured_ms")
        require_non_negative_int(self.processing_ms, "processing_ms")
        if self.processing_ms < self.captured_ms:
            raise ValueError("processing_ms must not precede captured_ms")
        if self.status is OpportunityStatus.READY:
            if self.reason is not OpportunityReason.TRUSTED_STATE:
                raise ValueError("ready opportunity requires TRUSTED_STATE")
            if self.seat is not Seat.SELF:
                raise ValueError("only the local seat can have a ready opportunity")
        elif self.reason is OpportunityReason.TRUSTED_STATE:
            raise ValueError("non-ready opportunity requires a blocking reason")


@dataclass(frozen=True, slots=True)
class EngineUpdate:
    version: VersionIdentity
    reason: EngineUpdateReason
    captured_ms: int
    processing_ms: int
    observations: tuple[SeatObservation, ...] = ()
    candidates: tuple[ActionCandidate, ...] = ()
    confirmed_actions: tuple[ConfirmedAction, ...] = ()
    gap: GapState | None = None
    advice_opportunity: AdviceOpportunity | None = None

    def __post_init__(self) -> None:
        require_enum(self.reason, EngineUpdateReason, "reason")
        for field_name in ("observations", "candidates", "confirmed_actions"):
            require_tuple(getattr(self, field_name), field_name)
        require_non_negative_int(self.captured_ms, "captured_ms")
        require_non_negative_int(self.processing_ms, "processing_ms")
        if self.processing_ms < self.captured_ms:
            raise ValueError("processing_ms must not precede captured_ms")
        identity = (self.version.session_id, self.version.capture_generation)
        for observation in self.observations:
            if (observation.frame.session_id, observation.frame.capture_generation) != identity:
                raise ValueError("observation belongs to another session or generation")
        for candidate in self.candidates:
            if (candidate.version.session_id, candidate.version.capture_generation) != identity:
                raise ValueError("candidate belongs to another session or generation")
        for action in self.confirmed_actions:
            after = action.version_after
            if after.session_id != self.version.session_id:
                raise ValueError("confirmed action belongs to another session")
            if after.state_revision > self.version.state_revision:
                raise ValueError("update version cannot precede a confirmed action")
        for versioned in (self.gap, self.advice_opportunity):
            if versioned is not None and versioned.version != self.version:
                raise ValueError("nested state must use the exact update version")


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    base_version: VersionIdentity
    actions: tuple[ConfirmedAction, ...]
    rejected_candidate_ids: tuple[str, ...]
    reason: ProjectionReason

    def __post_init__(self) -> None:
        require_enum(self.reason, ProjectionReason, "reason")
        require_tuple(self.actions, "actions")
        require_unique_text_tuple(self.rejected_candidate_ids, "rejected_candidate_ids")
        if self.reason is ProjectionReason.ACCEPTED:
            if not self.actions:
                raise ValueError("accepted projection must contain an action")
            if self.rejected_candidate_ids:
                raise ValueError("accepted projection cannot contain rejected candidates")
        elif self.actions:
            raise ValueError("rejected projection cannot contain confirmed actions")
        expected = self.base_version.state_version
        for action in self.actions:
            if action.version_before != expected:
                raise ValueError("projected actions must form one contiguous version chain")
            expected = action.version_after


@dataclass(frozen=True, slots=True)
class CommitResult:
    expected_version: VersionIdentity
    resulting_version: VersionIdentity
    committed_actions: tuple[ConfirmedAction, ...]
    reason: CommitReason

    def __post_init__(self) -> None:
        require_enum(self.reason, CommitReason, "reason")
        require_tuple(self.committed_actions, "committed_actions")
        same_capture_stream = (
            self.expected_version.session_id,
            self.expected_version.capture_generation,
        ) == (
            self.resulting_version.session_id,
            self.resulting_version.capture_generation,
        )
        if not same_capture_stream:
            raise ValueError("commit result cannot cross session or generation")
        if self.reason is CommitReason.COMMITTED:
            if not self.committed_actions:
                raise ValueError("successful commit must contain actions")
            expected = self.expected_version.state_version
            for action in self.committed_actions:
                if action.version_before != expected:
                    raise ValueError("committed actions must form one contiguous version chain")
                expected = action.version_after
            if expected != self.resulting_version.state_version:
                raise ValueError(
                    "resulting_version state must equal the end of the action chain"
                )
            if self.resulting_version.update_sequence < self.expected_version.update_sequence:
                raise ValueError("successful commit cannot move update_sequence backwards")
        else:
            if self.committed_actions:
                raise ValueError("failed commit cannot report committed actions")
            if self.resulting_version != self.expected_version:
                raise ValueError("failed commit cannot advance the version")
