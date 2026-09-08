"""Advice opportunity lifecycle derived from versioned game snapshots."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .game_state import GameAction, TrustedGameSnapshot
from .protocols import AdviceConsumer, EventJournal, RuleStateProvider

from .types import (
    AdviceOpportunity,
    ConfirmedAction,
    GapPhase,
    GapState,
    EngineUpdate,
    OpportunityReason,
    OpportunityStatus,
    Seat,
    VersionIdentity,
)


def load_trusted_snapshot(
    provider: RuleStateProvider,
    *,
    version: VersionIdentity,
    captured_ms: int,
    processing_ms: int,
) -> TrustedGameSnapshot:
    snapshot = provider.snapshot_for(version=version, captured_ms=captured_ms)
    if not isinstance(snapshot, TrustedGameSnapshot):
        raise TypeError("state provider must return a TrustedGameSnapshot")
    if snapshot.version != version:
        raise ValueError("state provider returned a snapshot for another version")
    if not snapshot.trusted:
        raise ValueError("state provider must return a trusted rule snapshot")
    if snapshot.captured_ms != captured_ms:
        raise ValueError("state provider must preserve requested capture watermark")
    if snapshot.captured_ms > processing_ms:
        raise ValueError("state snapshot cannot be newer than processing time")
    return snapshot


def validate_snapshot_transition(
    previous: TrustedGameSnapshot,
    current: TrustedGameSnapshot,
    committed: tuple[ConfirmedAction, ...],
) -> None:
    if current.trick_index < previous.trick_index:
        raise ValueError("state provider moved trick_index backwards")
    if current.play_history[:len(previous.play_history)] != previous.play_history:
        raise ValueError("state provider rewrote confirmed play history")
    added = current.play_history[len(previous.play_history):]
    expected = tuple(GameAction.from_confirmed(action) for action in committed)
    if added != expected:
        raise ValueError("state provider did not expose the committed action chain")


class SideEffectFailureKind(str, Enum):
    JOURNAL_APPEND_FAILED = "journal_append_failed"
    ADVICE_PUBLISH_FAILED = "advice_publish_failed"


@dataclass(frozen=True, slots=True)
class SideEffectFailure:
    kind: SideEffectFailureKind
    opportunity_id: str | None = None


@dataclass(frozen=True, slots=True)
class SideEffectDispatcher:
    journal: EventJournal | None = None
    advice_consumer: AdviceConsumer | None = None

    def publish(
        self,
        update: EngineUpdate,
        opportunities: tuple[AdviceOpportunity, ...],
    ) -> tuple[SideEffectFailure, ...]:
        failures: list[SideEffectFailure] = []
        if self.journal is not None:
            try:
                self.journal.append(update)
            except Exception:
                failures.append(SideEffectFailure(SideEffectFailureKind.JOURNAL_APPEND_FAILED))
        if self.advice_consumer is not None:
            for opportunity in opportunities:
                try:
                    self.advice_consumer.publish(opportunity)
                except Exception:
                    failures.append(SideEffectFailure(
                        SideEffectFailureKind.ADVICE_PUBLISH_FAILED,
                        opportunity.opportunity_id,
                    ))
        return tuple(failures)


@dataclass(frozen=True, slots=True)
class OpportunityState:
    current: AdviceOpportunity | None = None
    closed_keys: tuple[tuple[str, int, int], ...] = ()
    opened_processing_ms: int | None = None


@dataclass(frozen=True, slots=True)
class OpportunityTransition:
    state: OpportunityState
    publications: tuple[AdviceOpportunity, ...]


def _key(version: VersionIdentity) -> tuple[str, int, int]:
    return (version.session_id, version.capture_generation, version.turn_index)


def _opportunity_id(version: VersionIdentity) -> str:
    return f"advice:{version.session_id}:{version.capture_generation}:{version.turn_index}"


@dataclass(frozen=True, slots=True)
class OpportunityLifecycle:
    """Keep the two-second user response deadline separate from gap recovery."""

    response_deadline_ms: int = 2_000
    closed_key_capacity: int = 32

    def __post_init__(self) -> None:
        if isinstance(self.response_deadline_ms, bool) or self.response_deadline_ms <= 0:
            raise ValueError("response_deadline_ms must be a positive integer")
        if isinstance(self.closed_key_capacity, bool) or self.closed_key_capacity <= 0:
            raise ValueError("closed_key_capacity must be a positive integer")

    def advance(
        self,
        state: OpportunityState,
        *,
        snapshot: TrustedGameSnapshot | None,
        gap: GapState,
        version: VersionIdentity,
        processing_ms: int,
    ) -> OpportunityTransition:
        publications: list[AdviceOpportunity] = []
        current = state.current
        desired_key = _key(snapshot.version) if snapshot is not None else None
        current_key = _key(current.version) if current is not None else None

        if current is not None and current_key != desired_key:
            closed = self._make(
                version=version,
                seat=current.seat,
                status=OpportunityStatus.CLOSED,
                reason=OpportunityReason.SUPERSEDED,
                captured_ms=current.captured_ms,
                processing_ms=processing_ms,
                opportunity_id=current.opportunity_id,
            )
            if current.status is not OpportunityStatus.CLOSED:
                publications.append(closed)
            state = self._remember_closed(state, current_key)
            current = None

        if snapshot is None:
            return OpportunityTransition(
                OpportunityState(
                    current=None,
                    closed_keys=state.closed_keys,
                    opened_processing_ms=None,
                ),
                tuple(publications),
            )

        if snapshot.version != version:
            raise ValueError("opportunity snapshot must match the exact engine version")
        if snapshot.current_seat is None:
            if current is not None:
                reason = (
                    OpportunityReason.TERMINAL_STATE
                    if snapshot.terminal else OpportunityReason.SUPERSEDED
                )
                closed = self._make(
                    version=version,
                    seat=current.seat,
                    status=OpportunityStatus.CLOSED,
                    reason=reason,
                    captured_ms=snapshot.captured_ms,
                    processing_ms=processing_ms,
                    opportunity_id=current.opportunity_id,
                )
                if current.status is not OpportunityStatus.CLOSED:
                    publications.append(closed)
                state = self._remember_closed(state, desired_key)
                return OpportunityTransition(
                    OpportunityState(closed, state.closed_keys, None),
                    tuple(publications),
                )
            return OpportunityTransition(
                OpportunityState(None, state.closed_keys, None),
                tuple(publications),
            )

        desired_key = _key(snapshot.version)
        if desired_key in state.closed_keys:
            if current is not None and current_key == desired_key:
                closed = replace(current, version=version, processing_ms=processing_ms)
            else:
                closed = self._make(
                    version=version,
                    seat=snapshot.current_seat,
                    status=OpportunityStatus.CLOSED,
                    reason=OpportunityReason.SUPERSEDED,
                    captured_ms=snapshot.captured_ms,
                    processing_ms=processing_ms,
                )
            return OpportunityTransition(
                OpportunityState(
                    closed,
                    state.closed_keys,
                    (state.opened_processing_ms if current is not None
                     and current_key == desired_key else processing_ms),
                ),
                tuple(publications),
            )

        status, reason = self._status(snapshot, gap)
        opened_processing_ms = (
            state.opened_processing_ms
            if current is not None and current_key == desired_key
            else processing_ms
        )
        already_ready = (
            current is not None
            and current_key == desired_key
            and current.status is OpportunityStatus.READY
        )
        if not already_ready and snapshot.current_seat is Seat.SELF and (
            processing_ms - opened_processing_ms >= self.response_deadline_ms
        ):
            status, reason = OpportunityStatus.CLOSED, OpportunityReason.SUPERSEDED
        desired = self._make(
            version=version,
            seat=snapshot.current_seat,
            status=status,
            reason=reason,
            captured_ms=snapshot.captured_ms,
            processing_ms=processing_ms,
        )
        if current is None or self._meaning(current) != self._meaning(desired):
            publications.append(desired)
        if desired.status is OpportunityStatus.CLOSED:
            state = self._remember_closed(state, desired_key)
        return OpportunityTransition(
            OpportunityState(desired, state.closed_keys, opened_processing_ms),
            tuple(publications),
        )

    @staticmethod
    def _status(
        snapshot: TrustedGameSnapshot,
        gap: GapState,
    ) -> tuple[OpportunityStatus, OpportunityReason]:
        if snapshot.terminal:
            return OpportunityStatus.CLOSED, OpportunityReason.TERMINAL_STATE
        if snapshot.current_seat is not Seat.SELF:
            return OpportunityStatus.BLOCKED, OpportunityReason.NOT_LOCAL_TURN
        if not snapshot.trusted:
            return OpportunityStatus.BLOCKED, OpportunityReason.OBSERVATION_UNCERTAIN
        if gap.phase is not GapPhase.CLEAR:
            return OpportunityStatus.BLOCKED, OpportunityReason.HISTORY_GAP
        return OpportunityStatus.READY, OpportunityReason.TRUSTED_STATE

    @staticmethod
    def _meaning(opportunity: AdviceOpportunity) -> tuple:
        return (
            opportunity.opportunity_id,
            opportunity.seat,
            opportunity.status,
            opportunity.reason,
            opportunity.captured_ms,
        )

    @staticmethod
    def _make(
        *,
        version: VersionIdentity,
        seat: Seat,
        status: OpportunityStatus,
        reason: OpportunityReason,
        captured_ms: int,
        processing_ms: int,
        opportunity_id: str | None = None,
    ) -> AdviceOpportunity:
        return AdviceOpportunity(
            opportunity_id=opportunity_id or _opportunity_id(version),
            version=version,
            seat=seat,
            status=status,
            reason=reason,
            captured_ms=captured_ms,
            processing_ms=processing_ms,
        )

    def _remember_closed(
        self,
        state: OpportunityState,
        key: tuple[str, int, int] | None,
    ) -> OpportunityState:
        if key is None:
            return state
        keys = tuple(dict.fromkeys(state.closed_keys + (key,)))[-self.closed_key_capacity :]
        return replace(state, closed_keys=keys)
