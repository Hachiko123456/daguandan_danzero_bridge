"""Narrow dependency-inversion ports for the live-v2 core."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .game_state import TrustedGameSnapshot
from .types import (
    ActionCandidate,
    AdviceOpportunity,
    CommitResult,
    ConfirmedAction,
    EngineUpdate,
    ProjectionResult,
    SeatObservation,
    VersionIdentity,
)


@runtime_checkable
class RuleProjector(Protocol):
    """Project stream-bound candidates into StateVersion actions."""

    def project(
        self,
        *,
        base_version: VersionIdentity,
        candidates: tuple[ActionCandidate, ...],
        processing_ms: int,
    ) -> ProjectionResult: ...


@runtime_checkable
class TransactionalActionCommitter(Protocol):
    """Commit StateVersion actions and return the current stream identity."""

    def commit(
        self,
        *,
        expected_version: VersionIdentity,
        actions: tuple[ConfirmedAction, ...],
    ) -> CommitResult: ...


@runtime_checkable
class ProcessingClock(Protocol):
    """Supply processing time; capture timestamps come from observations."""

    def processing_ms(self) -> int: ...


@runtime_checkable
class RuleStateProvider(Protocol):
    """Build a fresh trusted snapshot for exactly the requested rule version."""

    def snapshot_for(
        self,
        *,
        version: VersionIdentity,
        captured_ms: int,
    ) -> TrustedGameSnapshot: ...


@runtime_checkable
class ObservationInput(Protocol):
    """Yield already-extracted observations, never raw GUI/image objects."""

    def receive(self) -> tuple[SeatObservation, ...]: ...


@runtime_checkable
class EventJournal(Protocol):
    """Receive immutable engine facts as an asynchronous side effect."""

    def append(self, update: EngineUpdate) -> None: ...


@runtime_checkable
class AdviceConsumer(Protocol):
    """Receive readiness changes without deciding authoritative history."""

    def publish(self, opportunity: AdviceOpportunity) -> None: ...
