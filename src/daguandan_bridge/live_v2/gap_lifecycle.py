"""Pure lifecycle rules for missing or contradictory action history."""

from __future__ import annotations

from dataclasses import dataclass

from .types import (
    ActionCandidate,
    CommitReason,
    GapPhase,
    GapReason,
    GapState,
    ProjectionReason,
    ProjectionResult,
    Seat,
    VersionIdentity,
)


def _evidence_ids(candidates: tuple[ActionCandidate, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            evidence_id
            for candidate in candidates
            for evidence_id in candidate.evidence_ids
        )
    )


def _expected_seats(candidates: tuple[ActionCandidate, ...]) -> tuple[Seat, ...]:
    return tuple(dict.fromkeys(candidate.seat for candidate in candidates))


@dataclass(frozen=True, slots=True)
class GapLifecycle:
    """Advance a gap without ever disabling future observation ingestion."""

    recovery_window_ms: int = 8_000

    def __post_init__(self) -> None:
        if isinstance(self.recovery_window_ms, bool) or self.recovery_window_ms <= 0:
            raise ValueError("recovery_window_ms must be a positive integer")

    def clear(
        self,
        *,
        version: VersionIdentity,
        captured_ms: int,
        processing_ms: int,
    ) -> GapState:
        return GapState(
            version=version,
            phase=GapPhase.CLEAR,
            reason=GapReason.NONE,
            expected_seats=(),
            evidence_ids=(),
            opened_captured_ms=captured_ms,
            processing_ms=processing_ms,
        )

    def from_projection(
        self,
        *,
        current: GapState,
        version: VersionIdentity,
        projection: ProjectionResult,
        candidates: tuple[ActionCandidate, ...],
        captured_ms: int,
        processing_ms: int,
    ) -> GapState:
        if projection.reason is ProjectionReason.ACCEPTED:
            return self.clear(
                version=version,
                captured_ms=captured_ms,
                processing_ms=processing_ms,
            )
        phase, reason = {
            ProjectionReason.NEED_MORE_EVIDENCE: (
                GapPhase.OBSERVING,
                GapReason.MISSING_EXPECTED_ACTION,
            ),
            ProjectionReason.OUT_OF_ORDER: (
                GapPhase.RECOVERABLE,
                GapReason.OUT_OF_ORDER_EVIDENCE,
            ),
            ProjectionReason.RULE_REJECTED: (
                GapPhase.BLOCKING,
                GapReason.RULE_REJECTION,
            ),
            ProjectionReason.VERSION_MISMATCH: (
                GapPhase.BLOCKING,
                GapReason.STALE_EVIDENCE,
            ),
        }[projection.reason]
        return self._open_or_continue(
            current=current,
            version=version,
            phase=phase,
            reason=reason,
            candidates=candidates,
            captured_ms=captured_ms,
            processing_ms=processing_ms,
        )

    def from_commit_failure(
        self,
        *,
        current: GapState,
        version: VersionIdentity,
        reason: CommitReason,
        candidates: tuple[ActionCandidate, ...],
        captured_ms: int,
        processing_ms: int,
    ) -> GapState:
        gap_reason = (
            GapReason.OUT_OF_ORDER_EVIDENCE
            if reason is CommitReason.VERSION_CONFLICT
            else GapReason.RULE_REJECTION
        )
        return self._open_or_continue(
            current=current,
            version=version,
            phase=GapPhase.BLOCKING,
            reason=gap_reason,
            candidates=candidates,
            captured_ms=captured_ms,
            processing_ms=processing_ms,
        )

    def refresh(
        self,
        current: GapState,
        *,
        version: VersionIdentity,
        captured_ms: int,
        processing_ms: int,
    ) -> GapState:
        if current.phase is GapPhase.CLEAR:
            return self.clear(
                version=version,
                captured_ms=captured_ms,
                processing_ms=processing_ms,
            )
        phase, reason = current.phase, current.reason
        if captured_ms - current.opened_captured_ms >= self.recovery_window_ms:
            phase, reason = GapPhase.EXPIRED, GapReason.RECOVERY_BUDGET_EXCEEDED
        return GapState(
            version=version,
            phase=phase,
            reason=reason,
            expected_seats=current.expected_seats,
            evidence_ids=current.evidence_ids,
            opened_captured_ms=current.opened_captured_ms,
            processing_ms=processing_ms,
        )

    def _open_or_continue(
        self,
        *,
        current: GapState,
        version: VersionIdentity,
        phase: GapPhase,
        reason: GapReason,
        candidates: tuple[ActionCandidate, ...],
        captured_ms: int,
        processing_ms: int,
    ) -> GapState:
        same_gap = current.phase not in (GapPhase.CLEAR, GapPhase.EXPIRED)
        opened_ms = current.opened_captured_ms if same_gap else captured_ms
        prior_evidence = current.evidence_ids if same_gap else ()
        prior_expected = current.expected_seats if same_gap else ()
        evidence = tuple(dict.fromkeys(prior_evidence + _evidence_ids(candidates)))
        expected = tuple(dict.fromkeys(prior_expected + _expected_seats(candidates)))
        state = GapState(
            version=version,
            phase=phase,
            reason=reason,
            expected_seats=expected,
            evidence_ids=evidence,
            opened_captured_ms=opened_ms,
            processing_ms=processing_ms,
        )
        return self.refresh(
            state,
            version=version,
            captured_ms=captured_ms,
            processing_ms=processing_ms,
        )
