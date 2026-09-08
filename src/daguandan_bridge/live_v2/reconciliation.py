"""Bounded gap recovery that delegates all semantics to the normal resolver."""

from __future__ import annotations

from dataclasses import dataclass

from .event_resolver import ResolutionResult, UnifiedActionResolver
from .types import (
    ActionCandidate,
    GapPhase,
    GapReason,
    GapState,
    ProjectionReason,
    ProjectionResult,
    Seat,
    VersionIdentity,
)


@dataclass(frozen=True, slots=True)
class ReconciliationAttempt:
    """One recovery attempt and the resulting explicit gap state."""

    gap: GapState
    resolution: ResolutionResult | None = None

    @property
    def recovered(self) -> bool:
        return (
            self.resolution is not None
            and self.resolution.projection.reason is ProjectionReason.ACCEPTED
            and self.resolution.commit is not None
            and bool(self.resolution.commit.committed_actions)
        )


class BoundedGapReconciler:
    """Recover at most three actions from fresh evidence in one capture stream."""

    def __init__(
        self,
        resolver: UnifiedActionResolver,
        *,
        max_actions: int = 3,
        max_age_ms: int = 1_500,
        max_turn_span: int = 4,
    ) -> None:
        if not 1 <= max_actions <= 3:
            raise ValueError("max_actions must be between one and three")
        if max_age_ms <= 0 or max_turn_span <= 0:
            raise ValueError("reconciliation budgets must be positive")
        self._resolver = resolver
        self.max_actions = max_actions
        self.max_age_ms = max_age_ms
        self.max_turn_span = max_turn_span
        self._gap: GapState | None = None

    @property
    def gap(self) -> GapState | None:
        return self._gap

    def open(
        self,
        *,
        version: VersionIdentity,
        expected_seats: tuple[Seat, ...],
        opened_captured_ms: int,
        processing_ms: int,
        evidence_ids: tuple[str, ...] = (),
    ) -> GapState:
        self._gap = GapState(
            version,
            GapPhase.RECOVERABLE,
            GapReason.MISSING_EXPECTED_ACTION,
            expected_seats,
            evidence_ids,
            opened_captured_ms,
            processing_ms,
        )
        return self._gap

    def close(self) -> None:
        self._gap = None

    def attempt(
        self,
        *,
        candidates: tuple[ActionCandidate, ...],
        current_version: VersionIdentity,
        processing_ms: int,
    ) -> ReconciliationAttempt:
        if self._gap is None:
            raise RuntimeError("reconciliation window is not open")
        if self._gap.phase not in {GapPhase.OBSERVING, GapPhase.RECOVERABLE}:
            return ReconciliationAttempt(self._gap)
        invalid = self._validate_window(candidates, current_version, processing_ms)
        if invalid is not None:
            self._gap = invalid
            return ReconciliationAttempt(invalid)
        if len(candidates) > self.max_actions:
            self._gap = self._expired(
                current_version,
                processing_ms,
                GapReason.RECOVERY_BUDGET_EXCEEDED,
                candidates,
            )
            return ReconciliationAttempt(self._gap)

        resolution = self._resolver.resolve(
            base_version=current_version,
            candidates=candidates,
            processing_ms=processing_ms,
            expected_seats=self._gap.expected_seats,
            opened_captured_ms=self._gap.opened_captured_ms,
            commit=True,
        )
        self._gap = resolution.gap
        if resolution.commit is not None and not resolution.commit.committed_actions:
            self._gap = GapState(
                current_version,
                GapPhase.BLOCKING,
                GapReason.CONFLICTING_EVIDENCE,
                self._gap.expected_seats,
                self._gap.evidence_ids,
                self._gap.opened_captured_ms,
                processing_ms,
            )
        if self._gap.phase is GapPhase.CLEAR:
            self.close()
        return ReconciliationAttempt(self._gap_or_clear(resolution), resolution)

    def _validate_window(
        self,
        candidates: tuple[ActionCandidate, ...],
        version: VersionIdentity,
        processing_ms: int,
    ) -> GapState | None:
        assert self._gap is not None
        gap_version = self._gap.version
        same_generation = (
            version.session_id,
            version.capture_generation,
        ) == (
            gap_version.session_id,
            gap_version.capture_generation,
        )
        if not same_generation or any(
            (item.version.session_id, item.version.capture_generation)
            != (gap_version.session_id, gap_version.capture_generation)
            for item in candidates
        ):
            return self._expired(
                version, processing_ms, GapReason.CAPTURE_DISCONTINUITY, candidates
            )
        if version.turn_index - gap_version.turn_index > self.max_turn_span:
            return self._expired(
                version,
                processing_ms,
                GapReason.RECOVERY_BUDGET_EXCEEDED,
                candidates,
            )
        if processing_ms - self._gap.opened_captured_ms > self.max_age_ms or any(
            item.first_captured_ms < self._gap.opened_captured_ms
            or processing_ms - item.last_captured_ms > self.max_age_ms
            for item in candidates
        ):
            return self._expired(
                version, processing_ms, GapReason.STALE_EVIDENCE, candidates
            )
        return None

    def _expired(
        self,
        version: VersionIdentity,
        processing_ms: int,
        reason: GapReason,
        candidates: tuple[ActionCandidate, ...],
    ) -> GapState:
        assert self._gap is not None
        return GapState(
            version,
            GapPhase.EXPIRED,
            reason,
            self._gap.expected_seats,
            tuple(
                dict.fromkeys(
                    evidence
                    for candidate in candidates
                    for evidence in candidate.evidence_ids
                )
            ),
            self._gap.opened_captured_ms,
            processing_ms,
        )

    def _gap_or_clear(self, resolution: ResolutionResult) -> GapState:
        if self._gap is not None:
            return self._gap
        return resolution.gap


GapReconciler = BoundedGapReconciler
