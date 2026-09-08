"""Pure transaction contracts for rule projection and durable adoption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..domain.live import LiveEvent
from .corrections import ConfirmedCorrection, CorrectionCommand
from .game_state import TrustedGameSnapshot
from .types import (
    ActionCandidate,
    CommitReason,
    ConfirmedAction,
    VersionIdentity,
)

TrustedSnapshot = TrustedGameSnapshot


@runtime_checkable
class ReducerEventSink(Protocol):
    def append_event_batch(self, events: tuple[LiveEvent, ...]) -> None: ...


@runtime_checkable
class DurableActionSink(Protocol):
    def adopt(
        self,
        transaction: "ReducerTransaction",
        resulting_version: VersionIdentity,
    ) -> CommitReason: ...


class DurableCommitFatalError(RuntimeError):
    """Durability succeeded but authoritative in-memory adoption failed."""


@dataclass(frozen=True, slots=True)
class ReducerTransaction:
    actions: tuple[ConfirmedAction, ...]
    token: object


@dataclass(frozen=True, slots=True)
class StageResult:
    transaction: ReducerTransaction | None
    needs_more_evidence: bool = False


@runtime_checkable
class ReducerTransactionBackend(DurableActionSink, Protocol):
    @property
    def version(self) -> VersionIdentity: ...

    @property
    def confirmed_actions(self) -> tuple[ConfirmedAction, ...]: ...

    @property
    def correction_history(self) -> tuple[ConfirmedCorrection, ...]: ...

    def events_for_actions(
        self, actions: tuple[ConfirmedAction, ...]
    ) -> tuple[LiveEvent, ...]: ...

    def correct_latest(
        self, command: CorrectionCommand
    ) -> tuple[CommitReason, ConfirmedCorrection | None, VersionIdentity]: ...

    def matches(self, version: VersionIdentity) -> bool: ...

    def snapshot_for(
        self,
        *,
        version: VersionIdentity,
        captured_ms: int,
    ) -> TrustedGameSnapshot: ...

    def stage(
        self,
        *,
        ordered: tuple[ActionCandidate, ...],
        base_version: VersionIdentity,
        processing_ms: int,
    ) -> StageResult: ...


__all__ = [
    "DurableActionSink",
    "DurableCommitFatalError",
    "ReducerEventSink",
    "ReducerTransaction",
    "ReducerTransactionBackend",
    "StageResult",
    "TrustedSnapshot",
]
