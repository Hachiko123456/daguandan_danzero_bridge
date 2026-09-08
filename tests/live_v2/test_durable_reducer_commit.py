from __future__ import annotations

import os

import pytest

from daguandan_bridge.application.live_v2_event_sink import SessionStoreEventSink
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live_v2.reducer_transaction import (
    DurableActionSink,
    DurableCommitFatalError,
    ReducerEventSink,
)
from daguandan_bridge.infrastructure.live_v2_rule_backend import create_reducer_backend
from daguandan_bridge.live_v2.rules_adapter import LiveReducerRuleAdapter
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    CommitReason,
    FrameIdentity,
    Seat,
    VersionIdentity,
)


def _hand() -> tuple[str, ...]:
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    return tuple(f"{rank}{suit}" for suit in "SHC" for rank in ranks)[:27]


def _candidate(
    name: str,
    version: VersionIdentity,
    seat: Seat,
    cards: tuple[str, ...],
    captured_ms: int,
) -> ActionCandidate:
    first = FrameIdentity(
        "s", 1, captured_ms, captured_ms, "roi", "window"
    )
    last = FrameIdentity(
        "s", 1, captured_ms + 1, captured_ms + 10, "roi", "window"
    )
    return ActionCandidate(
        candidate_id=name,
        version=version,
        seat=seat,
        kind=ActionKind.PLAY,
        cards=cards,
        suit_options=tuple((card,) for card in cards),
        evidence_ids=(f"{name}-a", f"{name}-b"),
        action_epoch=captured_ms,
        first_frame=first,
        last_frame=last,
        processing_ms=captured_ms + 11,
        confidence=0.95,
        reason=CandidateReason.STABLE_PLAY,
    )


def _system(tmp_path):
    store = LiveSessionStore(tmp_path, "profile", session_id="s")
    store.start({})
    reducer = LiveReducer("s")
    reducer.confirm_initial_state(
        round_level="2",
        hand=_hand(),
        lead_player=Seat.RIGHT.value,
        monotonic_ms=1,
    )
    version = VersionIdentity("s", 1, 1, 0, 0)
    sink = SessionStoreEventSink(store)
    backend = create_reducer_backend(
        reducer, version=version, event_sink=sink
    )
    return LiveReducerRuleAdapter(backend, version), reducer, store, backend


def test_production_backend_requires_explicit_durable_sink() -> None:
    reducer = LiveReducer("s")
    reducer.confirm_initial_state(
        round_level="2", hand=_hand(), lead_player="right", monotonic_ms=1
    )
    version = VersionIdentity("s", 1, 1, 0, 0)
    with pytest.raises(ValueError, match="requires an event_sink"):
        create_reducer_backend(reducer, version=version)
    backend = create_reducer_backend(reducer, version=version, in_memory=True)
    assert isinstance(backend, DurableActionSink)


def test_success_persists_ordered_formal_batch_before_rule_adoption(tmp_path) -> None:
    adapter, reducer, store, backend = _system(tmp_path)
    assert isinstance(SessionStoreEventSink(store), ReducerEventSink)
    assert isinstance(backend, DurableActionSink)
    right = _candidate("right-3", adapter.version, Seat.RIGHT, ("3D",), 100)
    opposite = _candidate(
        "opposite-4", adapter.version, Seat.OPPOSITE, ("4D",), 200
    )
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(opposite, right),
        processing_ms=300,
    )

    committed = adapter.commit(
        expected_version=adapter.version, actions=projected.actions
    )

    assert committed.reason is CommitReason.COMMITTED
    records = read_json_lines(store.timeline_path)
    assert [record["event_id"] for record in records] == [
        "EVT-000002",
        "EVT-000003",
    ]
    assert [record["actor"] for record in records] == ["right", "opposite"]
    assert [record["evidence_refs"] for record in records] == [
        ["right-3-a", "right-3-b"],
        ["opposite-4-a", "opposite-4-b"],
    ]
    assert [(record["state_revision_before"], record["state_revision_after"])
            for record in records] == [(1, 2), (2, 3)]
    assert [event.cards for event in reducer.snapshot().play_history] == [
        ("3D",),
        ("4D",),
    ]


def test_selected_current_subset_is_the_only_action_persisted(tmp_path) -> None:
    adapter, reducer, store, _backend = _system(tmp_path)
    current = _candidate("right-8", adapter.version, Seat.RIGHT, ("8D",), 100)
    stale_future = _candidate(
        "opposite-7", adapter.version, Seat.OPPOSITE, ("7D",), 200
    )
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(current, stale_future),
        processing_ms=300,
    )

    committed = adapter.commit(
        expected_version=adapter.version,
        actions=projected.actions,
    )

    assert committed.reason is CommitReason.COMMITTED
    assert tuple(action.source_candidate for action in committed.committed_actions) == (
        current,
    )
    records = read_json_lines(store.timeline_path)
    assert [record["actor"] for record in records] == ["right"]
    assert [record["evidence_refs"] for record in records] == [
        ["right-8-a", "right-8-b"]
    ]
    assert [event.cards for event in reducer.snapshot().play_history] == [("8D",)]


def test_persistence_failure_keeps_disk_reducer_ledger_and_version_unchanged(
    tmp_path, monkeypatch
) -> None:
    adapter, reducer, store, _backend = _system(tmp_path)
    candidate = _candidate("right-3", adapter.version, Seat.RIGHT, ("3D",), 100)
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=200,
    )
    initial_version = adapter.version
    real_replace = os.replace
    attempts = 0

    def fail_once(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected durable replace failure")
        return real_replace(source, target)

    monkeypatch.setattr("daguandan_bridge.live.session_store.os.replace", fail_once)
    failed = adapter.commit(
        expected_version=initial_version, actions=projected.actions
    )

    assert failed.reason is CommitReason.PERSISTENCE_FAILED
    assert read_json_lines(store.timeline_path) == []
    assert reducer.snapshot().revision == 1
    assert reducer.snapshot().play_history == ()
    assert adapter.version == initial_version
    assert adapter.snapshot_for(version=initial_version, captured_ms=200).play_history == ()
    assert not tuple(store.directory.glob(".timeline.jsonl.*.tmp"))

    reprojected = adapter.project(
        base_version=initial_version,
        candidates=(candidate,),
        processing_ms=250,
    )
    assert tuple(action.action_id for action in reprojected.actions) == tuple(
        action.action_id for action in projected.actions
    )
    retried = adapter.commit(
        expected_version=initial_version, actions=reprojected.actions
    )
    assert retried.reason is CommitReason.COMMITTED
    assert [record["event_id"] for record in read_json_lines(store.timeline_path)] == [
        "EVT-000002"
    ]
    duplicate = adapter.commit(
        expected_version=initial_version, actions=projected.actions
    )
    assert duplicate.reason is CommitReason.VERSION_CONFLICT
    assert len(read_json_lines(store.timeline_path)) == 1


def test_staged_baseline_conflict_never_reaches_durable_sink(tmp_path) -> None:
    adapter, reducer, store, _backend = _system(tmp_path)
    candidate = _candidate("right-3", adapter.version, Seat.RIGHT, ("3D",), 100)
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=200,
    )
    reducer.record_play("right", ("4D",), monotonic_ms=150)

    conflict = adapter.commit(
        expected_version=adapter.version, actions=projected.actions
    )

    assert conflict.reason is CommitReason.VERSION_CONFLICT
    assert read_json_lines(store.timeline_path) == []
    assert reducer.snapshot().play_history[-1].cards == ("4D",)


def test_post_durability_adoption_failure_is_fatal_not_normal_rejection(
    tmp_path, monkeypatch
) -> None:
    adapter, reducer, store, _backend = _system(tmp_path)
    candidate = _candidate("right-3", adapter.version, Seat.RIGHT, ("3D",), 100)
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=200,
    )

    def fail_adoption(_staged) -> None:
        raise RuntimeError("injected adoption failure")

    monkeypatch.setattr(reducer, "adopt_staged", fail_adoption)
    with pytest.raises(DurableCommitFatalError, match="events are durable"):
        adapter.commit(expected_version=adapter.version, actions=projected.actions)
    assert [record["event_id"] for record in read_json_lines(store.timeline_path)] == [
        "EVT-000002"
    ]
    assert reducer.snapshot().revision == 1


def test_reducer_change_during_durable_write_is_detected_before_adoption(
    tmp_path, monkeypatch
) -> None:
    adapter, reducer, store, _backend = _system(tmp_path)
    candidate = _candidate("right-3", adapter.version, Seat.RIGHT, ("3D",), 100)
    projected = adapter.project(
        base_version=adapter.version,
        candidates=(candidate,),
        processing_ms=200,
    )
    original_append = store.append_event_batch

    def persist_then_mutate(events) -> None:
        original_append(events)
        reducer.record_play("right", ("4D",), monotonic_ms=150)

    monkeypatch.setattr(store, "append_event_batch", persist_then_mutate)
    with pytest.raises(DurableCommitFatalError, match="baseline changed"):
        adapter.commit(expected_version=adapter.version, actions=projected.actions)

    assert [record["event_id"] for record in read_json_lines(store.timeline_path)] == [
        "EVT-000002"
    ]
    assert reducer.snapshot().play_history[-1].cards == ("4D",)
    assert adapter.version == VersionIdentity("s", 1, 1, 0, 0)
