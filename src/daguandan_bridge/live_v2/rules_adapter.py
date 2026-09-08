"""Atomic RuleProjector/Committer backed by the reviewed legacy reducer."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from .candidate_projection import (
    CandidateFingerprint,
    candidate_fingerprint,
    candidate_orders,
)
from .reducer_transaction import (
    DurableCommitFatalError,
    ReducerTransaction,
    ReducerTransactionBackend,
    TrustedSnapshot,
)
from .types import (
    ActionCandidate,
    CommitReason,
    CommitResult,
    ConfirmedAction,
    ProjectionReason,
    ProjectionResult,
    VersionIdentity,
)


@dataclass(slots=True)
class _PendingProjection:
    transaction: ReducerTransaction
    fingerprints: frozenset[CandidateFingerprint]
    evidence_ids: frozenset[str]


class LiveReducerRuleAdapter:
    """Project on clones and atomically adopt exactly one validated chain."""

    def __init__(
        self,
        backend: ReducerTransactionBackend,
        version: VersionIdentity,
        *,
        max_projection_actions: int = 4,
    ) -> None:
        if max_projection_actions < 1:
            raise ValueError("max_projection_actions must be positive")
        if not backend.matches(version):
            raise ValueError("backend and version must describe the same state")
        self._backend = backend
        self._commit_lock = RLock()
        self._version = version
        self._max_projection_actions = max_projection_actions
        self._pending: dict[tuple[str, ...], _PendingProjection] = {}
        self._consumed: set[CandidateFingerprint] = set()
        self._consumed_evidence: set[str] = set()

    @property
    def version(self) -> VersionIdentity:
        with self._commit_lock:
            return self._version

    def snapshot_for(
        self, *, version: VersionIdentity, captured_ms: int
    ) -> TrustedSnapshot:
        with self._commit_lock:
            return self._backend.snapshot_for(
                version=version, captured_ms=captured_ms
            )

    def project(
        self,
        *,
        base_version: VersionIdentity,
        candidates: tuple[ActionCandidate, ...],
        processing_ms: int,
    ) -> ProjectionResult:
        with self._commit_lock:
            return self._project_unlocked(
                base_version=base_version,
                candidates=candidates,
                processing_ms=processing_ms,
            )

    def _project_unlocked(
        self,
        *,
        base_version: VersionIdentity,
        candidates: tuple[ActionCandidate, ...],
        processing_ms: int,
    ) -> ProjectionResult:
        self._pending.clear()
        rejected = tuple(dict.fromkeys(item.candidate_id for item in candidates))
        if base_version != self._version:
            return ProjectionResult(
                base_version, (), rejected, ProjectionReason.VERSION_MISMATCH
            )
        orders, rejection = candidate_orders(
            base_version=base_version,
            candidates=candidates,
            consumed=frozenset(self._consumed),
            consumed_evidence=frozenset(self._consumed_evidence),
            max_candidates=self._max_projection_actions,
        )
        if rejection is not None:
            return ProjectionResult(base_version, (), rejected, rejection)

        staged: list[ReducerTransaction] = []
        needs_more = False
        for ordered in orders:
            result = self._backend.stage(
                ordered=ordered,
                base_version=base_version,
                processing_ms=processing_ms,
            )
            needs_more = needs_more or result.needs_more_evidence
            if result.transaction is not None:
                staged.append(result.transaction)
        if staged:
            maximum = max(len(item.actions) for item in staged)
            maximal = (item for item in staged if len(item.actions) == maximum)
            unique = {_transaction_signature(item): item for item in maximal}
        else:
            unique = {}
        if len(unique) != 1:
            reason = (
                ProjectionReason.OUT_OF_ORDER
                if unique
                else ProjectionReason.NEED_MORE_EVIDENCE
                if needs_more
                else ProjectionReason.RULE_REJECTED
            )
            return ProjectionResult(base_version, (), rejected, reason)

        transaction = next(iter(unique.values()))
        actions = transaction.actions
        key = tuple(action.action_id for action in actions)
        source_candidates = tuple(action.source_candidate for action in actions)
        self._pending[key] = _PendingProjection(
            transaction,
            frozenset(candidate_fingerprint(item) for item in source_candidates),
            frozenset(
                evidence
                for candidate in source_candidates
                for evidence in candidate.all_evidence_ids
            ),
        )
        return ProjectionResult(
            base_version, actions, (), ProjectionReason.ACCEPTED
        )

    def commit(
        self,
        *,
        expected_version: VersionIdentity,
        actions: tuple[ConfirmedAction, ...],
    ) -> CommitResult:
        with self._commit_lock:
            return self._commit_unlocked(
                expected_version=expected_version, actions=actions
            )

    def _commit_unlocked(
        self,
        *,
        expected_version: VersionIdentity,
        actions: tuple[ConfirmedAction, ...],
    ) -> CommitResult:
        if expected_version != self._version or (
            not self._backend.matches(self._version)
        ):
            return CommitResult(
                expected_version, expected_version, (), CommitReason.VERSION_CONFLICT
            )
        pending = self._pending.get(tuple(action.action_id for action in actions))
        if pending is None or pending.transaction.actions != actions:
            return CommitResult(
                expected_version,
                expected_version,
                (),
                CommitReason.TRANSACTION_REJECTED,
            )
        if pending.fingerprints & self._consumed or (
            pending.evidence_ids & self._consumed_evidence
        ):
            return CommitResult(
                expected_version,
                expected_version,
                (),
                CommitReason.TRANSACTION_REJECTED,
            )

        resulting_version = expected_version.with_state(
            actions[-1].version_after,
            update_sequence=expected_version.update_sequence + len(actions),
        )
        adoption = self._backend.adopt(pending.transaction, resulting_version)
        if adoption is not CommitReason.COMMITTED:
            return CommitResult(
                expected_version,
                expected_version,
                (),
                adoption,
            )
        try:
            self._version = resulting_version
            self._consumed.update(pending.fingerprints)
            self._consumed_evidence.update(pending.evidence_ids)
            self._pending.clear()
        except BaseException as exc:
            raise DurableCommitFatalError(
                "durable rule adoption succeeded but adapter publication failed"
            ) from exc
        return CommitResult(
            expected_version, self._version, actions, CommitReason.COMMITTED
        )


ReducerRuleAdapter = LiveReducerRuleAdapter


def _transaction_signature(transaction: ReducerTransaction) -> tuple[object, ...]:
    """Keep distinct maximal rule outcomes distinct; never pick one arbitrarily."""

    return tuple(
        (
            candidate_fingerprint(action.source_candidate),
            action.source_candidate.candidate_id,
            action.source_candidate.all_evidence_ids,
            action.seat,
            action.kind,
            action.cards,
            action.suit_options,
            action.version_before,
            action.version_after,
        )
        for action in transaction.actions
    )
