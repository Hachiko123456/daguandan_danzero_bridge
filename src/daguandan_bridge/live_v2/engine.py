"""Small, deterministic coordinator for the live-v2 action pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .event_resolver import UnifiedActionResolver
from .game_state import TrustedGameSnapshot
from .gap_lifecycle import GapLifecycle
from .input_lifecycle import (
    CandidateReceipt,
    EngineInput,
    EvidenceLifecycle,
    InputRejection,
    InputRejectionReason,
    choose_update_reason,
    next_flow_version,
    processing_watermark,
    validate_version_sync,
)
from .opportunity import (
    OpportunityLifecycle,
    OpportunityState,
    SideEffectDispatcher,
    SideEffectFailure,
    SideEffectFailureKind,
    load_trusted_snapshot,
    validate_snapshot_transition,
)
from .protocols import (
    AdviceConsumer,
    EventJournal,
    ProcessingClock,
    RuleProjector,
    RuleStateProvider,
    TransactionalActionCommitter,
)
from .types import (
    ActionCandidate,
    CommitReason,
    ConfirmedAction,
    EngineUpdate,
    GapPhase,
    GapState,
    ProjectionReason,
    Seat,
    SeatObservation,
    StateVersion,
    VersionIdentity,
)

_SEATS = tuple(Seat)


@dataclass(frozen=True, slots=True)
class EngineState:
    version: VersionIdentity
    committed_state: StateVersion
    committed_update_sequence: int
    observations: tuple[SeatObservation | None, ...]
    candidates: tuple[ActionCandidate, ...]
    gap: GapState
    snapshot: TrustedGameSnapshot
    opportunity: OpportunityState
    capture_watermark_ms: int
    receipts: tuple[CandidateReceipt, ...] = ()

    def __post_init__(self) -> None:
        if len(self.observations) != len(_SEATS):
            raise ValueError("observations must contain exactly one slot per seat")
        if self.version.state_version != self.committed_state:
            raise ValueError("current flow version must expose committed rule state")
        if self.committed_update_sequence > self.version.update_sequence:
            raise ValueError("committed update sequence cannot lead flow updates")

    def observation_for(self, seat: Seat) -> SeatObservation | None:
        return self.observations[_SEATS.index(seat)]

    @property
    def commit_version(self) -> VersionIdentity:
        return VersionIdentity.from_state(
            self.committed_state,
            capture_generation=self.version.capture_generation,
            update_sequence=self.committed_update_sequence,
        )


@dataclass(frozen=True, slots=True)
class EngineResult:
    update: EngineUpdate | None
    rejections: tuple[InputRejection, ...] = ()
    side_effect_failures: tuple[SideEffectFailure, ...] = ()


class LiveEngine:
    """Own authoritative pipeline lifecycle, never visual or rule details."""

    def __init__(
        self,
        *,
        initial_version: VersionIdentity,
        projector: RuleProjector,
        committer: TransactionalActionCommitter,
        state_provider: RuleStateProvider,
        clock: ProcessingClock,
        journal: EventJournal | None = None,
        advice_consumer: AdviceConsumer | None = None,
        initial_captured_ms: int = 0,
        evidence_max_age_ms: int = 1_500,
        receipt_capacity: int = 256,
        gap_lifecycle: GapLifecycle | None = None,
        opportunity_lifecycle: OpportunityLifecycle | None = None,
    ) -> None:
        if isinstance(evidence_max_age_ms, bool) or evidence_max_age_ms <= 0:
            raise ValueError("evidence_max_age_ms must be a positive integer")
        if isinstance(receipt_capacity, bool) or receipt_capacity <= 0:
            raise ValueError("receipt_capacity must be a positive integer")
        if isinstance(initial_captured_ms, bool) or initial_captured_ms < 0:
            raise ValueError("initial_captured_ms must be a non-negative integer")
        now = clock.processing_ms()
        if now < initial_captured_ms:
            raise ValueError("processing clock cannot precede initial capture")
        self._resolver = UnifiedActionResolver(projector, committer)
        self._state_provider = state_provider
        self._clock = clock
        self._side_effects = SideEffectDispatcher(journal, advice_consumer)
        self._evidence = EvidenceLifecycle(evidence_max_age_ms, receipt_capacity)
        self._gaps = gap_lifecycle or GapLifecycle()
        self._opportunities = opportunity_lifecycle or OpportunityLifecycle()
        gap = self._gaps.clear(version=initial_version, captured_ms=initial_captured_ms,
                               processing_ms=now)
        snapshot = load_trusted_snapshot(
            state_provider, version=initial_version,
            captured_ms=initial_captured_ms, processing_ms=now,
        )
        self._state = EngineState(
            version=initial_version,
            committed_state=initial_version.state_version,
            committed_update_sequence=initial_version.update_sequence,
            observations=(None,) * len(_SEATS),
            candidates=(),
            gap=gap,
            snapshot=snapshot,
            opportunity=OpportunityState(),
            capture_watermark_ms=initial_captured_ms,
        )

    @property
    def state(self) -> EngineState:
        return self._state

    def process(self, incoming: EngineInput) -> EngineResult:
        processing_ms = processing_watermark(
            self._clock.processing_ms(), self._state.gap.processing_ms,
            incoming.observations, incoming.candidates,
        )
        rebind_rejection = validate_version_sync(
            self._state.version, incoming.rebind_version)
        if rebind_rejection is not None:
            return EngineResult(None, (rebind_rejection,))
        self._apply_version_sync(incoming.rebind_version)
        retained_snapshot = self._state.snapshot
        captured_now = self._evidence.watermark(
            self._state.capture_watermark_ms, self._state.version,
            incoming.observations, incoming.candidates, retained_snapshot.version,
            retained_snapshot.captured_ms, incoming.captured_watermark_ms,
        )
        if captured_now > processing_ms:
            raise ValueError("processing clock cannot precede capture watermark")
        observations, observation_rejections = self._evidence.observations(
            incoming.observations,
            self._state.observations,
            self._state.version,
            captured_now,
        )
        pending = self._evidence.pending(
            self._state.candidates, self._state.version, captured_now)
        candidates, candidate_rejections = self._evidence.candidates(
            incoming.candidates, pending, self._state.receipts,
            self._state.version, captured_now,
        )
        pending += candidates
        rejections = observation_rejections + candidate_rejections

        projection = None
        committed: tuple[ConfirmedAction, ...] = ()
        commit_reason: CommitReason | None = None
        rule_version = self._state.commit_version
        resulting_state = self._state.committed_state
        resulting_sequence = self._state.committed_update_sequence
        if candidates:
            projected_candidates = tuple(
                replace(item, version=rule_version)
                for item in pending
            )
            resolution = self._resolver.resolve(
                base_version=rule_version,
                candidates=projected_candidates,
                processing_ms=processing_ms,
                expected_seats=self._state.gap.expected_seats,
                opened_captured_ms=(self._state.gap.opened_captured_ms
                                    if self._state.gap.phase is not GapPhase.CLEAR else None),
                commit=True,
            )
            projection = resolution.projection
            if projection.base_version != rule_version:
                raise ValueError("projector returned a result for another base version")
            commit = resolution.commit
            if commit is not None:
                if commit.expected_version != rule_version:
                    raise ValueError("committer returned another expected version")
                commit_reason = commit.reason
                if commit.reason is CommitReason.COMMITTED:
                    if commit.committed_actions != projection.actions:
                        raise ValueError("committer changed the projected action chain")
                    committed = commit.committed_actions
                    resulting_state = commit.resulting_version.state_version
                    resulting_sequence = commit.resulting_version.update_sequence

        committed_view = VersionIdentity.from_state(
            resulting_state,
            capture_generation=self._state.version.capture_generation,
            update_sequence=resulting_sequence,
        )
        version = next_flow_version(
            committed_view, self._state.version.update_sequence)
        captured_ms = captured_now
        snapshot = load_trusted_snapshot(
            self._state_provider, version=version, captured_ms=captured_ms,
            processing_ms=processing_ms,
        )
        validate_snapshot_transition(self._state.snapshot, snapshot, committed)
        gap = self._next_gap(
            projection=projection,
            commit_reason=commit_reason,
            pending=pending,
            version=version,
            captured_ms=captured_ms,
            processing_ms=processing_ms,
        )
        if committed:
            receipts = self._evidence.receipts(
                self._state.receipts, captured_now, pending
            )
            pending = ()
        else:
            receipts = self._evidence.receipts(self._state.receipts, captured_now)

        transition = self._opportunities.advance(
            self._state.opportunity,
            snapshot=snapshot,
            gap=gap,
            version=version,
            processing_ms=processing_ms,
        )
        latest = self._evidence.merge_observations(
            self._state.observations, observations)
        reason = choose_update_reason(
            observations=observations,
            candidates=candidates,
            committed=committed,
            old_gap=self._state.gap,
            gap=gap,
            opportunity_changed=bool(transition.publications),
        )
        update = EngineUpdate(
            version=version,
            reason=reason,
            captured_ms=captured_ms,
            processing_ms=processing_ms,
            observations=observations,
            candidates=candidates,
            confirmed_actions=committed,
            gap=gap,
            advice_opportunity=transition.state.current,
        )
        self._state = EngineState(
            version=version,
            committed_state=resulting_state,
            committed_update_sequence=resulting_sequence,
            observations=latest,
            candidates=pending,
            gap=gap,
            snapshot=snapshot,
            opportunity=transition.state,
            capture_watermark_ms=captured_now,
            receipts=receipts,
        )
        failures = self._side_effects.publish(update, transition.publications)
        return EngineResult(update, rejections, failures)

    def _apply_version_sync(self, requested: VersionIdentity | None) -> None:
        if requested is None:
            return
        current = self._state.version
        if requested == current:
            return
        gap = replace(self._state.gap, version=requested)
        opportunity = self._state.opportunity
        if opportunity.current is not None:
            opportunity = replace(
                opportunity,
                current=replace(opportunity.current, version=requested),
            )
        self._state = replace(
            self._state, version=requested, gap=gap,
            opportunity=opportunity)

    def _next_gap(self, **values: object) -> GapState:
        projection = values["projection"]
        version = values["version"]
        if projection is None:
            return self._gaps.refresh(
                self._state.gap,
                version=version,
                captured_ms=values["captured_ms"],
                processing_ms=values["processing_ms"],
            )
        accepted_but_uncommitted = (
            projection.reason is ProjectionReason.ACCEPTED
            and values["commit_reason"] is not CommitReason.COMMITTED
        )
        if accepted_but_uncommitted:
            return self._gaps.from_commit_failure(
                current=self._state.gap,
                version=version,
                reason=values["commit_reason"],
                candidates=values["pending"],
                captured_ms=values["captured_ms"],
                processing_ms=values["processing_ms"],
            )
        return self._gaps.from_projection(
            current=self._state.gap,
            version=version,
            projection=projection,
            candidates=values["pending"],
            captured_ms=values["captured_ms"],
            processing_ms=values["processing_ms"],
        )
