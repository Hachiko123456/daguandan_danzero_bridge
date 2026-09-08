"""Application contract for the sole production live-v2 rule boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from ..domain.live import LiveEvent
from ..live_v2.corrections import ConfirmedCorrection, CorrectionCommand
from ..live_v2.game_state import TrustedGameSnapshot
from ..live_v2.protocols import (
    RuleProjector,
    RuleStateProvider,
    TransactionalActionCommitter,
)
from ..live_v2.types import ActionCandidate, ConfirmedAction, Seat, VersionIdentity


@runtime_checkable
class RuleAdapter(
    RuleProjector,
    RuleStateProvider,
    TransactionalActionCommitter,
    Protocol,
):
    @property
    def version(self) -> VersionIdentity: ...


@dataclass(frozen=True, slots=True)
class RuleBinding:
    version: VersionIdentity
    adapter: RuleAdapter

    def __post_init__(self) -> None:
        if not isinstance(self.version, VersionIdentity):
            raise TypeError("version must be a VersionIdentity")
        if not isinstance(self.adapter, RuleAdapter):
            raise TypeError("adapter must implement RuleAdapter")
        if self.adapter.version != self.version:
            raise ValueError("adapter and binding versions differ")


@dataclass(frozen=True, slots=True)
class OpeningActionCommit:
    binding: RuleBinding
    action: ConfirmedAction
    events: tuple[LiveEvent, ...]

    def __post_init__(self) -> None:
        if len(self.events) != 2:
            raise ValueError("opening commit requires lead and action events")
        if tuple(event.event_type for event in self.events) != (
            "lead_player_confirmed", "player_played",
        ):
            raise ValueError("opening commit events are out of order")


class RuleSessionError(RuntimeError):
    """Base error raised without silently mutating authoritative rule state."""


class RuleSessionRejected(RuleSessionError):
    """The requested lead or correction is invalid for the current state."""


class RuleSessionPersistenceError(RuleSessionError):
    """Durable publication failed, so in-memory state was not adopted."""


class RuleSessionFatalError(RuleSessionError):
    """Durability succeeded but authoritative in-memory publication failed."""


@runtime_checkable
class RuleSession(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def version(self) -> VersionIdentity: ...

    @property
    def confirmed_actions(self) -> tuple[ConfirmedAction, ...]: ...

    @property
    def correction_history(self) -> tuple[ConfirmedCorrection, ...]: ...

    def initialize(
        self,
        *,
        round_level: str,
        hand: tuple[str, ...],
        lead_player: Seat | None,
        monotonic_ms: int,
        wall_time: str | None = None,
        capture_generation: int = 0,
        evidence_id: str = "initial-state",
    ) -> RuleBinding: ...

    def bind_generation(self, capture_generation: int) -> RuleBinding: ...

    def snapshot(self, *, captured_ms: int) -> TrustedGameSnapshot: ...

    def events_for_actions(
        self, actions: tuple[ConfirmedAction, ...]
    ) -> tuple[LiveEvent, ...]: ...

    def confirm_lead(
        self,
        lead_player: Seat,
        *,
        monotonic_ms: int,
        evidence_id: str,
    ) -> RuleBinding: ...

    def confirm_opening_action(
        self, candidate: ActionCandidate, *, processing_ms: int
    ) -> OpeningActionCommit: ...

    def correct_latest(self, command: CorrectionCommand) -> ConfirmedCorrection: ...

    def health(
        self,
        *,
        additional_events: tuple[LiveEvent, ...] = (),
        recording_integrity: Mapping[str, object] | None = None,
    ) -> dict[str, object]: ...


__all__ = [
    "RuleAdapter",
    "RuleBinding",
    "OpeningActionCommit",
    "RuleSession",
    "RuleSessionError",
    "RuleSessionFatalError",
    "RuleSessionPersistenceError",
    "RuleSessionRejected",
]
