"""Production durability adapter for formal live-v2 reducer events."""

from __future__ import annotations

from ..domain.live import LiveEvent
from ..live_v2.reducer_transaction import ReducerEventSink
from .ports import SessionPersistencePort


class SessionStoreEventSink:
    """Forward one already-validated reducer batch to the session store."""

    def __init__(self, store: SessionPersistencePort) -> None:
        if not store.persistence_enabled:
            raise ValueError("live-v2 production commits require durable persistence")
        self._store = store

    @property
    def session_id(self) -> str:
        return self._store.session_id

    def append_event_batch(self, events: tuple[LiveEvent, ...]) -> None:
        if not isinstance(events, tuple):
            raise TypeError("events must be a tuple")
        if not events:
            raise ValueError("formal reducer event batch must not be empty")
        if any(event.session_id != self._store.session_id for event in events):
            raise ValueError("event batch belongs to another session")
        self._store.append_event_batch(events)

__all__ = ["SessionStoreEventSink"]
