"""Production anti-corruption boundary around the legacy rule reducer."""

from __future__ import annotations

from threading import RLock
from typing import Mapping

from ..application.live_v2_event_sink import SessionStoreEventSink
from ..application.live_v2_rule_session_protocol import (
    OpeningActionCommit,
    RuleBinding,
    RuleSessionFatalError,
    RuleSessionPersistenceError,
    RuleSessionRejected,
)
from ..application.ports import SessionPersistencePort
from ..domain.live import LiveEvent
from ..live_v2.corrections import ConfirmedCorrection, CorrectionCommand
from ..live_v2.game_state import TrustedGameSnapshot
from ..live_v2.identity import Seat, VersionIdentity
from ..live_v2.reducer_transaction import (
    DurableCommitFatalError,
    ReducerTransactionBackend,
)
from ..live_v2.results import CommitReason
from ..live_v2.rules_adapter import LiveReducerRuleAdapter
from ..live_v2.types import ActionCandidate, ConfirmedAction
from ..session_health import audit_session_health
from .live_v2_legacy_gateway import (
    adopt_reducer,
    confirm_lead_reducer,
    create_legacy_reducer,
    initialize_reducer,
    legacy_events,
    legacy_identity,
    legacy_snapshot,
)
from .live_v2_rule_backend import create_reducer_backend
from .live_v2_opening_transaction import stage_opening_action


class ProductionRuleSession:
    """Own one reducer, durable sink, ledger and capture-generation binding."""

    def __init__(self, store: SessionPersistencePort) -> None:
        if not store.persistence_enabled:
            raise ValueError("production RuleSession requires durable persistence")
        self._store = store
        self._sink = SessionStoreEventSink(store)
        self._reducer = create_legacy_reducer(store.session_id)
        self._backend: ReducerTransactionBackend | None = None
        self._adapter: LiveReducerRuleAdapter | None = None
        self._version: VersionIdentity | None = None
        self._lock = RLock()

    @property
    def session_id(self) -> str:
        return self._store.session_id

    @property
    def version(self) -> VersionIdentity:
        with self._lock:
            return self._current_version()

    @property
    def confirmed_actions(self) -> tuple[ConfirmedAction, ...]:
        with self._lock:
            return self._require_backend().confirmed_actions

    @property
    def correction_history(self) -> tuple[ConfirmedCorrection, ...]:
        with self._lock:
            return self._require_backend().correction_history

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
    ) -> RuleBinding:
        with self._lock:
            if self._version is not None:
                raise RuleSessionRejected("RuleSession is already initialized")
            try:
                staged, event = initialize_reducer(
                    self._reducer,
                    round_level=round_level,
                    hand=hand,
                    lead_player=lead_player,
                    monotonic_ms=monotonic_ms,
                    wall_time=wall_time,
                    evidence_id=evidence_id,
                )
                version = self._version_from(
                    staged, capture_generation=capture_generation, update_sequence=0
                )
                create_reducer_backend(staged, version=version, in_memory=True)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RuleSessionRejected(str(exc)) from exc
            self._persist_before_adopt((event,))
            try:
                adopt_reducer(self._reducer, staged)
                self._install_backend(version, (), ())
            except BaseException as exc:
                raise RuleSessionFatalError(
                    "initial state is durable but could not be adopted"
                ) from exc
            return self._binding()

    def bind_generation(self, capture_generation: int) -> RuleBinding:
        with self._lock:
            current = self._current_version()
            if isinstance(capture_generation, bool) or capture_generation < 0:
                raise RuleSessionRejected("capture_generation must be non-negative")
            if capture_generation < current.capture_generation:
                raise RuleSessionRejected("capture generation cannot move backwards")
            if capture_generation == current.capture_generation:
                return self._binding()
            rebound = VersionIdentity.from_state(
                current.state_version,
                capture_generation=capture_generation,
                update_sequence=0,
            )
            backend = self._require_backend()
            self._install_backend(
                rebound,
                backend.confirmed_actions,
                backend.correction_history,
            )
            return self._binding()

    def snapshot(self, *, captured_ms: int) -> TrustedGameSnapshot:
        with self._lock:
            current = self._current_version()
            return self._require_backend().snapshot_for(
                version=current,
                captured_ms=captured_ms,
            )

    def events_for_actions(
        self, actions: tuple[ConfirmedAction, ...]
    ) -> tuple[LiveEvent, ...]:
        with self._lock:
            return self._require_backend().events_for_actions(actions)

    def confirm_lead(
        self,
        lead_player: Seat,
        *,
        monotonic_ms: int,
        evidence_id: str,
    ) -> RuleBinding:
        with self._lock:
            current = self._current_version()
            backend = self._require_backend()
            try:
                staged, event = confirm_lead_reducer(
                    self._reducer,
                    lead_player,
                    monotonic_ms=monotonic_ms,
                    evidence_id=evidence_id,
                )
                version = self._version_from(
                    staged,
                    capture_generation=current.capture_generation,
                    update_sequence=current.update_sequence + 1,
                )
                create_reducer_backend(
                    staged,
                    version=version,
                    seed_actions=backend.confirmed_actions,
                    seed_corrections=backend.correction_history,
                    in_memory=True,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RuleSessionRejected(str(exc)) from exc
            self._persist_before_adopt((event,))
            try:
                adopt_reducer(self._reducer, staged)
                self._install_backend(
                    version,
                    backend.confirmed_actions,
                    backend.correction_history,
                )
            except BaseException as exc:
                raise RuleSessionFatalError(
                    "lead confirmation is durable but could not be adopted"
                ) from exc
            return self._binding()

    def confirm_opening_action(
        self, candidate: ActionCandidate, *, processing_ms: int
    ) -> OpeningActionCommit:
        with self._lock:
            current = self._current_version()
            backend = self._require_backend()
            try:
                staged = stage_opening_action(
                    self._reducer,
                    current=current,
                    seed_actions=backend.confirmed_actions,
                    seed_corrections=backend.correction_history,
                    candidate=candidate,
                    processing_ms=processing_ms,
                )
                actions = backend.confirmed_actions + (staged.action,)
                create_reducer_backend(
                    staged.reducer, version=staged.version,
                    seed_actions=actions,
                    seed_corrections=backend.correction_history,
                    in_memory=True,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RuleSessionRejected(str(exc)) from exc
            self._persist_before_adopt(staged.events)
            try:
                adopt_reducer(self._reducer, staged.reducer)
                self._install_backend(
                    staged.version, actions, backend.correction_history
                )
            except BaseException as exc:
                raise RuleSessionFatalError(
                    "opening action is durable but could not be adopted"
                ) from exc
            return OpeningActionCommit(
                self._binding(), staged.action, staged.events
            )

    def correct_latest(self, command: CorrectionCommand) -> ConfirmedCorrection:
        with self._lock:
            backend = self._require_backend()
            try:
                reason, correction, resulting = backend.correct_latest(command)
            except DurableCommitFatalError as exc:
                raise RuleSessionFatalError(str(exc)) from exc
            if reason is CommitReason.VERSION_CONFLICT:
                raise RuleSessionRejected("correction version is stale")
            if reason is CommitReason.TRANSACTION_REJECTED:
                raise RuleSessionRejected("correction target or action is invalid")
            if reason is CommitReason.PERSISTENCE_FAILED:
                raise RuleSessionPersistenceError("correction was not persisted")
            if reason is not CommitReason.COMMITTED or correction is None:
                raise RuleSessionFatalError("unexpected correction result")
            self._version = resulting
            self._adapter = LiveReducerRuleAdapter(backend, resulting)
            return correction

    def health(
        self,
        *,
        additional_events: tuple[LiveEvent, ...] = (),
        recording_integrity: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        with self._lock:
            if any(event.session_id != self.session_id for event in additional_events):
                raise RuleSessionRejected("health event belongs to another session")
            return audit_session_health(
                legacy_snapshot(self._reducer),
                legacy_events(self._reducer) + additional_events,
                recording_integrity=recording_integrity,
            )

    def _persist_before_adopt(self, events: tuple[LiveEvent, ...]) -> None:
        try:
            self._sink.append_event_batch(events)
        except Exception as exc:
            raise RuleSessionPersistenceError("rule event was not persisted") from exc

    def _install_backend(
        self,
        version: VersionIdentity,
        actions: tuple[ConfirmedAction, ...],
        corrections: tuple[ConfirmedCorrection, ...],
    ) -> None:
        backend = create_reducer_backend(
            self._reducer,
            version=version,
            seed_actions=actions,
            seed_corrections=corrections,
            event_sink=self._sink,
        )
        self._backend = backend
        self._adapter = LiveReducerRuleAdapter(backend, version)
        self._version = version

    def _binding(self) -> RuleBinding:
        adapter = self._adapter
        if adapter is None:
            raise RuleSessionRejected("RuleSession is not initialized")
        self._version = adapter.version
        return RuleBinding(self._version, adapter)

    def _current_version(self) -> VersionIdentity:
        if self._adapter is not None:
            self._version = self._adapter.version
        if self._version is None:
            raise RuleSessionRejected("RuleSession is not initialized")
        return self._version

    def _require_backend(self) -> ReducerTransactionBackend:
        if self._backend is None:
            raise RuleSessionRejected("RuleSession is not initialized")
        return self._backend

    @staticmethod
    def _version_from(
        reducer: object,
        *,
        capture_generation: int,
        update_sequence: int,
    ) -> VersionIdentity:
        session_id, revision, turn_index = legacy_identity(reducer)
        return VersionIdentity(
            session_id,
            capture_generation,
            revision,
            update_sequence,
            turn_index,
        )


RuleSession = ProductionRuleSession

__all__ = ["ProductionRuleSession", "RuleSession"]
