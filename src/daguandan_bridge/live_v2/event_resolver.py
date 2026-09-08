"""One resolution path for ordinary and reconciliation candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .protocols import RuleProjector, TransactionalActionCommitter
from .types import (
    ActionCandidate,
    CommitReason,
    CommitResult,
    GapPhase,
    GapReason,
    GapState,
    ProjectionReason,
    ProjectionResult,
    Seat,
    VersionIdentity,
)


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Explicit projection and gap outcome; ambiguity is never implicit."""

    projection: ProjectionResult
    gap: GapState
    commit: CommitResult | None = None

    @property
    def accepted(self) -> bool:
        projected = self.projection.reason is ProjectionReason.ACCEPTED
        return projected and (
            self.commit is None or self.commit.reason is CommitReason.COMMITTED
        )


class UnifiedActionResolver:
    """Apply the same projector to normal and recovery evidence."""

    def __init__(
        self,
        projector: RuleProjector,
        committer: TransactionalActionCommitter | None = None,
    ) -> None:
        self._projector = projector
        self._committer = committer

    def resolve(
        self,
        *,
        base_version: VersionIdentity,
        candidates: tuple[ActionCandidate, ...],
        processing_ms: int,
        expected_seats: tuple[Seat, ...] = (),
        opened_captured_ms: int | None = None,
        commit: bool = False,
    ) -> ResolutionResult:
        if not isinstance(candidates, tuple):
            raise TypeError("candidates must be a tuple")
        opened = (
            min((item.first_captured_ms for item in candidates), default=processing_ms)
            if opened_captured_ms is None
            else opened_captured_ms
        )
        evidence_ids = tuple(
            evidence
            for candidate in candidates
            for evidence in candidate.evidence_ids
        )
        candidate_ids = tuple(item.candidate_id for item in candidates)
        has_conflict = (
            len(candidate_ids) != len(set(candidate_ids))
            or len(evidence_ids) != len(set(evidence_ids))
        )
        projection = self._projector.project(
            base_version=base_version,
            candidates=candidates,
            processing_ms=processing_ms,
        )
        gap = self._gap_for(
            projection,
            expected_seats=expected_seats,
            candidate_seats=tuple(item.seat for item in candidates),
            evidence_ids=tuple(
                dict.fromkeys(
                    evidence
                    for candidate in candidates
                    for evidence in candidate.evidence_ids
                )
            ),
            opened_captured_ms=opened,
            processing_ms=processing_ms,
        )
        if has_conflict:
            gap = GapState(
                base_version,
                GapPhase.BLOCKING,
                GapReason.CONFLICTING_EVIDENCE,
                expected_seats,
                tuple(dict.fromkeys(evidence_ids)),
                opened,
                processing_ms,
            )
        committed: CommitResult | None = None
        if commit and projection.reason is ProjectionReason.ACCEPTED:
            if self._committer is None:
                raise RuntimeError("commit requested without a committer")
            committed = self._committer.commit(
                expected_version=base_version,
                actions=projection.actions,
            )
            if committed.reason is CommitReason.COMMITTED:
                gap = GapState(
                    committed.resulting_version,
                    GapPhase.CLEAR,
                    GapReason.NONE,
                    (),
                    (),
                    opened,
                    processing_ms,
                )
            else:
                gap = GapState(
                    base_version,
                    GapPhase.BLOCKING,
                    GapReason.CONFLICTING_EVIDENCE,
                    expected_seats,
                    tuple(
                        dict.fromkeys(
                            evidence
                            for candidate in candidates
                            for evidence in candidate.evidence_ids
                        )
                    ),
                    opened,
                    processing_ms,
                )
        return ResolutionResult(projection, gap, committed)

    @staticmethod
    def _gap_for(
        projection: ProjectionResult,
        *,
        expected_seats: tuple[Seat, ...],
        candidate_seats: tuple[Seat, ...],
        evidence_ids: tuple[str, ...],
        opened_captured_ms: int,
        processing_ms: int,
    ) -> GapState:
        reason = projection.reason
        if reason is ProjectionReason.ACCEPTED:
            return GapState(
                projection.base_version,
                GapPhase.CLEAR,
                GapReason.NONE,
                (),
                (),
                opened_captured_ms,
                processing_ms,
            )
        phase, gap_reason = {
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
                GapPhase.EXPIRED,
                GapReason.CAPTURE_DISCONTINUITY,
            ),
        }[reason]
        if (
            reason is ProjectionReason.RULE_REJECTED
            and expected_seats
            and expected_seats[0] not in candidate_seats
        ):
            phase = GapPhase.RECOVERABLE
            gap_reason = GapReason.MISSING_EXPECTED_ACTION
        return GapState(
            projection.base_version,
            phase,
            gap_reason,
            expected_seats,
            evidence_ids,
            opened_captured_ms,
            processing_ms,
        )


EventResolver = UnifiedActionResolver
