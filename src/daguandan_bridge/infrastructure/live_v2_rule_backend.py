"""Durable transaction coordinator over the single legacy gateway."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from ..domain.live import LiveEvent
from ..live_v2.corrections import ConfirmedCorrection, CorrectionCommand
from ..live_v2.reducer_snapshot import TrustedSnapshotLedger
from ..live_v2.reducer_transaction import (
    DurableCommitFatalError,
    ReducerEventSink,
    ReducerTransaction,
    ReducerTransactionBackend,
    StageResult,
)
from ..live_v2.types import (
    ActionCandidate,
    CommitReason,
    ConfirmedAction,
    VersionIdentity,
)
from .live_v2_legacy_gateway import (
    adopt_reducer,
    is_legacy_reducer,
    legacy_events,
    legacy_identity,
    stage_latest_correction,
    stage_reducer,
)
from .live_v2_legacy_projection import (
    event_matches_action,
    extract_reduced_state,
    validate_reducer_seed,
)


@dataclass(frozen=True, slots=True)
class _ReducerBaseline:
    session_id: str
    revision: int
    turn_index: int
    event_count: int
    last_event_id: str | None


@dataclass(frozen=True, slots=True)
class _TransactionToken:
    staged: object
    events: tuple[LiveEvent, ...]
    baseline: _ReducerBaseline
    base_version: VersionIdentity


class _InMemoryEventSink:
    def append_event_batch(self, events: tuple[LiveEvent, ...]) -> None:
        del events


class LegacyReducerTransactionBackend:
    def __init__(
        self,
        reducer: object,
        version: VersionIdentity,
        seed_actions: tuple[ConfirmedAction, ...],
        seed_corrections: tuple[ConfirmedCorrection, ...],
        event_sink: ReducerEventSink,
    ) -> None:
        if not is_legacy_reducer(reducer):
            raise TypeError("backend requires the legacy gateway reducer token")
        if not isinstance(event_sink, ReducerEventSink):
            raise TypeError("event_sink must implement ReducerEventSink")
        self._reducer = reducer
        self._ledger = TrustedSnapshotLedger(
            version, seed_actions, seed_corrections
        )
        self._event_sink = event_sink
        self._lock = RLock()
        validate_reducer_seed(reducer, version, seed_actions, seed_corrections)

    @property
    def version(self) -> VersionIdentity:
        with self._lock:
            return self._ledger.version

    @property
    def confirmed_actions(self) -> tuple[ConfirmedAction, ...]:
        with self._lock:
            return self._ledger.actions

    @property
    def correction_history(self) -> tuple[ConfirmedCorrection, ...]:
        with self._lock:
            return self._ledger.corrections

    def events_for_actions(
        self, actions: tuple[ConfirmedAction, ...]
    ) -> tuple[LiveEvent, ...]:
        with self._lock:
            available = tuple(
                event
                for event in legacy_events(self._reducer)
                if event.event_type in {
                    "player_played",
                    "player_passed",
                    "manual_confirmed_event",
                }
            )
            result: list[LiveEvent] = []
            for action in actions:
                match = next(
                    (event for event in available if event_matches_action(event, action)),
                    None,
                )
                if match is None:
                    raise ValueError("confirmed action has no reducer event")
                result.append(match)
            return tuple(result)

    def matches(self, version: VersionIdentity) -> bool:
        with self._lock:
            session_id, revision, turn_index = legacy_identity(self._reducer)
            return (
                self._ledger.matches(version)
                and session_id == version.session_id
                and revision == version.state_revision
                and turn_index == version.turn_index
            )

    def snapshot_for(self, *, version: VersionIdentity, captured_ms: int):
        with self._lock:
            if not self.matches(version):
                raise ValueError("snapshot version does not match reducer backend")
            return self._ledger.snapshot(
                extract_reduced_state(self._reducer), captured_ms, version=version
            )

    def stage(
        self,
        *,
        ordered: tuple[ActionCandidate, ...],
        base_version: VersionIdentity,
        processing_ms: int,
    ) -> StageResult:
        with self._lock:
            if not self.matches(base_version):
                return StageResult(None)
            baseline = _baseline(self._reducer)
            result = stage_reducer(
                reducer=self._reducer,
                ordered=ordered,
                base_version=base_version,
                processing_ms=processing_ms,
            )
            if result.reducer is None:
                return StageResult(None, result.needs_more_evidence)
            token = _TransactionToken(
                result.reducer, result.events, baseline, base_version
            )
            return StageResult(ReducerTransaction(result.actions, token))

    def adopt(
        self,
        transaction: ReducerTransaction,
        resulting_version: VersionIdentity,
    ) -> CommitReason:
        with self._lock:
            token = transaction.token
            if not isinstance(token, _TransactionToken):
                return CommitReason.TRANSACTION_REJECTED
            if not self.matches(token.base_version) or token.baseline != _baseline(
                self._reducer
            ):
                return CommitReason.VERSION_CONFLICT
            if not self._valid(transaction, token, resulting_version):
                return CommitReason.TRANSACTION_REJECTED
            try:
                self._event_sink.append_event_batch(token.events)
            except Exception:
                return CommitReason.PERSISTENCE_FAILED
            if not self.matches(token.base_version) or token.baseline != _baseline(
                self._reducer
            ):
                raise DurableCommitFatalError(
                    "formal events are durable but the reducer baseline changed"
                )
            try:
                adopt_reducer(self._reducer, token.staged)
                self._ledger.append(transaction.actions, resulting_version)
            except BaseException as exc:
                raise DurableCommitFatalError(
                    "formal events are durable but adoption failed"
                ) from exc
            return CommitReason.COMMITTED

    def correct_latest(
        self, command: CorrectionCommand
    ) -> tuple[CommitReason, ConfirmedCorrection | None, VersionIdentity]:
        with self._lock:
            current = self._ledger.version
            actions = self._ledger.actions
            if command.expected_version != current:
                return CommitReason.VERSION_CONFLICT, None, current
            if not actions or actions[-1].action_id != command.target_action_id:
                return CommitReason.TRANSACTION_REJECTED, None, current
            baseline = _baseline(self._reducer)
            try:
                staged = stage_latest_correction(
                    reducer=self._reducer,
                    target_action=actions[-1],
                    target_event=self.events_for_actions((actions[-1],))[0],
                    command=command,
                )
                resulting = current.with_state(
                    staged.correction.version_after,
                    update_sequence=current.update_sequence + 1,
                )
                self._ledger.preview_correction(
                    extract_reduced_state(staged.reducer),
                    staged.correction,
                    resulting,
                    command.corrected_ms,
                )
            except (TypeError, ValueError):
                return CommitReason.TRANSACTION_REJECTED, None, current
            try:
                self._event_sink.append_event_batch((staged.event,))
            except Exception:
                return CommitReason.PERSISTENCE_FAILED, None, current
            if baseline != _baseline(self._reducer):
                raise DurableCommitFatalError(
                    "correction is durable but the reducer baseline changed"
                )
            try:
                adopt_reducer(self._reducer, staged.reducer)
                self._ledger.correct(staged.correction, resulting)
            except BaseException as exc:
                raise DurableCommitFatalError(
                    "correction is durable but adoption failed"
                ) from exc
            return CommitReason.COMMITTED, staged.correction, resulting

    def _valid(
        self,
        transaction: ReducerTransaction,
        token: _TransactionToken,
        resulting_version: VersionIdentity,
    ) -> bool:
        if not is_legacy_reducer(token.staged) or not transaction.actions:
            return False
        events = legacy_events(token.staged)
        current_events = legacy_events(self._reducer)
        if events[: token.baseline.event_count] != current_events:
            return False
        if events[token.baseline.event_count :] != token.events:
            return False
        if len(token.events) != len(transaction.actions):
            return False
        if not all(
            event_matches_action(event, action)
            for event, action in zip(token.events, transaction.actions, strict=True)
        ):
            return False
        try:
            self._ledger.preview_append(
                extract_reduced_state(token.staged),
                transaction.actions,
                resulting_version,
                transaction.actions[-1].captured_ms,
            )
        except (TypeError, ValueError):
            return False
        return True


def create_reducer_backend(
    reducer: object,
    *,
    version: VersionIdentity,
    seed_actions: tuple[ConfirmedAction, ...] = (),
    seed_corrections: tuple[ConfirmedCorrection, ...] = (),
    event_sink: ReducerEventSink | None = None,
    in_memory: bool = False,
) -> ReducerTransactionBackend:
    if not isinstance(seed_actions, tuple):
        raise TypeError("seed_actions must be a tuple")
    if not isinstance(seed_corrections, tuple):
        raise TypeError("seed_corrections must be a tuple")
    if event_sink is None and not in_memory:
        raise ValueError("production reducer backend requires an event_sink")
    if event_sink is not None and in_memory:
        raise ValueError("event_sink and in_memory=True are mutually exclusive")
    sink = _InMemoryEventSink() if event_sink is None else event_sink
    return LegacyReducerTransactionBackend(
        reducer, version, seed_actions, seed_corrections, sink
    )


def _baseline(reducer: object) -> _ReducerBaseline:
    session_id, revision, turn_index = legacy_identity(reducer)
    events = legacy_events(reducer)
    return _ReducerBaseline(
        session_id,
        revision,
        turn_index,
        len(events),
        events[-1].event_id if events else None,
    )


__all__ = ["LegacyReducerTransactionBackend", "create_reducer_backend"]
