from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from daguandan_bridge.application.live_v2_rule_session_protocol import (
    RuleSessionPersistenceError,
    RuleSessionRejected,
)
from daguandan_bridge.application.live_v2_runtime_updates import trusted_to_live_snapshot
from daguandan_bridge.domain.live import LiveEvent
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live_v2.corrections import CorrectionCommand, CorrectionReason
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    EvidenceOrigin,
    FrameIdentity,
    Seat,
)


def _hand() -> tuple[str, ...]:
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    return tuple(f"{rank}{suit}" for suit in "SHC" for rank in ranks)[:27]


def _store(tmp_path: Path) -> LiveSessionStore:
    store = LiveSessionStore(tmp_path, "profile", session_id="s")
    store.start({})
    return store


def _candidate(session: ProductionRuleSession, name: str, seat: Seat, card: str, ms: int):
    version = session.version
    first = FrameIdentity("s", version.capture_generation, ms, ms, "roi", "window")
    last = FrameIdentity("s", version.capture_generation, ms + 1, ms + 10, "roi", "window")
    return ActionCandidate(
        candidate_id=name,
        version=version,
        seat=seat,
        kind=ActionKind.PLAY,
        cards=(card,),
        suit_options=((card,),),
        evidence_ids=(f"{name}-1", f"{name}-2"),
        action_epoch=ms,
        first_frame=first,
        last_frame=last,
        processing_ms=ms + 20,
        confidence=0.99,
        reason=CandidateReason.STABLE_PLAY,
    )


def _commit(session: ProductionRuleSession, candidate: ActionCandidate):
    binding = session.bind_generation(session.version.capture_generation)
    projected = binding.adapter.project(
        base_version=binding.version,
        candidates=(candidate,),
        processing_ms=candidate.processing_ms,
    )
    result = binding.adapter.commit(
        expected_version=binding.version,
        actions=projected.actions,
    )
    assert result.committed_actions
    return result.committed_actions[0]


def _correction(session: ProductionRuleSession, action_id: str, card: str, ms: int):
    return CorrectionCommand(
        correction_id=f"fix-{ms}",
        expected_version=session.version,
        target_action_id=action_id,
        kind=ActionKind.PLAY,
        cards=(card,),
        suit_options=((card,),),
        reason=CorrectionReason.MANUAL_REVIEW,
        evidence_id=f"operator-{ms}",
        evidence_origin=EvidenceOrigin.MANUAL,
        confidence=1.0,
        corrected_ms=ms,
    )


def test_initialization_is_persisted_before_authoritative_adoption(tmp_path, monkeypatch):
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    monkeypatch.setattr(store, "append_event_batch", lambda _events: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(RuleSessionPersistenceError):
        session.initialize(
            round_level="2", hand=_hand(), lead_player=Seat.RIGHT, monotonic_ms=1
        )

    with pytest.raises(RuleSessionRejected):
        _ = session.version
    assert session._reducer.snapshot().initialized is False
    assert read_json_lines(store.timeline_path) == []


def test_lead_confirmation_is_durable_and_does_not_advance_turn(tmp_path):
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    initial = session.initialize(
        round_level="2", hand=_hand(), lead_player=None, monotonic_ms=1
    )

    bound = session.confirm_lead(Seat.LEFT, monotonic_ms=10, evidence_id="lead-left")

    assert bound.version.state_revision == initial.version.state_revision + 1
    assert bound.version.turn_index == initial.version.turn_index == 0
    assert session.snapshot(captured_ms=10).current_seat is Seat.LEFT
    assert read_json_lines(store.timeline_path)[-1]["event_type"] == "lead_player_confirmed"


def test_visual_opening_atomically_persists_lead_and_original_candidate(tmp_path):
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    session.initialize(
        round_level="2", hand=_hand(), lead_player=None,
        monotonic_ms=1, capture_generation=1,
    )
    candidate = _candidate(session, "left-opening", Seat.LEFT, "3D", 100)

    committed = session.confirm_opening_action(candidate, processing_ms=130)
    snapshot = session.snapshot(captured_ms=130)

    assert [event.event_type for event in committed.events] == [
        "lead_player_confirmed", "player_played",
    ]
    assert snapshot.lead_seat is Seat.LEFT
    assert snapshot.current_seat is Seat.SELF
    assert committed.action.source_candidate.candidate_id == candidate.candidate_id
    assert committed.action.cards == candidate.cards
    assert committed.action.suit_options == candidate.suit_options
    assert committed.action.action_epoch == candidate.action_epoch
    assert committed.action.evidence_ids == candidate.evidence_ids
    assert [row["event_type"] for row in read_json_lines(store.timeline_path)] == [
        "initial_state_confirmed", "lead_player_confirmed", "player_played",
    ]


def test_visual_opening_persistence_failure_has_no_half_lead(tmp_path, monkeypatch):
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    session.initialize(
        round_level="2", hand=_hand(), lead_player=None,
        monotonic_ms=1, capture_generation=1,
    )
    before = session.version
    monkeypatch.setattr(
        store, "append_event_batch",
        lambda _events: (_ for _ in ()).throw(OSError("disk")),
    )
    with pytest.raises(RuleSessionPersistenceError):
        session.confirm_opening_action(
            _candidate(session, "left-opening", Seat.LEFT, "3D", 100),
            processing_ms=130,
        )
    snapshot = session.snapshot(captured_ms=130)
    assert session.version == before
    assert snapshot.lead_seat is None and snapshot.current_seat is None
    assert not snapshot.play_history and not session.confirmed_actions


def test_pass_and_old_generation_cannot_confirm_opening_lead(tmp_path):
    session = ProductionRuleSession(_store(tmp_path))
    session.initialize(
        round_level="2", hand=_hand(), lead_player=None,
        monotonic_ms=1, capture_generation=1,
    )
    play = _candidate(session, "left-opening", Seat.LEFT, "3D", 100)
    passed = replace(
        play, kind=ActionKind.PASS, cards=(), suit_options=(),
        reason=CandidateReason.FRESH_PASS_EDGE,
    )
    with pytest.raises(RuleSessionRejected, match="cannot be PASS"):
        session.confirm_opening_action(passed, processing_ms=130)

    session.bind_generation(2)
    with pytest.raises(RuleSessionRejected, match="another state or generation"):
        session.confirm_opening_action(play, processing_ms=140)
    snapshot = session.snapshot(captured_ms=140)
    assert snapshot.lead_seat is None and not snapshot.play_history


def test_normal_commit_and_explicit_correction_keep_separate_audit_histories(tmp_path):
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    session.initialize(round_level="2", hand=_hand(), lead_player=Seat.RIGHT, monotonic_ms=1)
    action = _commit(session, _candidate(session, "right-3", Seat.RIGHT, "3D", 100))
    before = session.version

    correction = session.correct_latest(_correction(session, action.action_id, "4D", 200))
    snapshot = session.snapshot(captured_ms=200)

    assert correction.target_action_id == action.action_id
    assert session.version.state_revision == before.state_revision + 1
    assert session.version.turn_index == before.turn_index
    assert session.confirmed_actions[0].cards == ("3D",)
    assert snapshot.play_history[0].cards == ("4D",)
    assert snapshot.play_history[0].correction_id == correction.correction_id
    assert snapshot.correction_history == (correction,)
    record = read_json_lines(store.timeline_path)[-1]
    assert record["event_type"] == "event_correction"
    assert record["payload"]["target_action_id"] == action.action_id
    assert record["evidence_refs"] == ["operator-200"]


def test_correction_persistence_failure_does_not_advance_any_authority(tmp_path, monkeypatch):
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    session.initialize(round_level="2", hand=_hand(), lead_player=Seat.RIGHT, monotonic_ms=1)
    action = _commit(session, _candidate(session, "right-3", Seat.RIGHT, "3D", 100))
    version = session.version
    records = tuple(read_json_lines(store.timeline_path))
    monkeypatch.setattr(store, "append_event_batch", lambda _events: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(RuleSessionPersistenceError):
        session.correct_latest(_correction(session, action.action_id, "4D", 200))

    assert session.version == version
    assert session.correction_history == ()
    assert session.snapshot(captured_ms=200).play_history[0].cards == ("3D",)
    assert tuple(read_json_lines(store.timeline_path)) == records


def test_correction_rejects_nonlatest_and_cross_session_targets(tmp_path):
    session = ProductionRuleSession(_store(tmp_path))
    session.initialize(round_level="2", hand=_hand(), lead_player=Seat.RIGHT, monotonic_ms=1)
    first = _commit(session, _candidate(session, "right-3", Seat.RIGHT, "3D", 100))
    _commit(session, _candidate(session, "opposite-4", Seat.OPPOSITE, "4D", 130))

    with pytest.raises(RuleSessionRejected):
        session.correct_latest(_correction(session, first.action_id, "5D", 200))
    command = _correction(session, session.confirmed_actions[-1].action_id, "5D", 210)
    foreign = replace(
        command,
        expected_version=command.expected_version.__class__(
            "other", command.expected_version.capture_generation,
            command.expected_version.state_revision, command.expected_version.update_sequence,
            command.expected_version.turn_index,
        ),
    )
    with pytest.raises(RuleSessionRejected):
        session.correct_latest(foreign)


def test_new_capture_generation_seeds_actions_and_corrections(tmp_path):
    session = ProductionRuleSession(_store(tmp_path))
    session.initialize(round_level="2", hand=_hand(), lead_player=Seat.RIGHT, monotonic_ms=1, capture_generation=1)
    action = _commit(session, _candidate(session, "right-3", Seat.RIGHT, "3D", 100))
    session.correct_latest(_correction(session, action.action_id, "4D", 200))

    rebound = session.bind_generation(2)

    assert rebound.version.capture_generation == 2
    assert rebound.version.turn_index == 1
    assert session.snapshot(captured_ms=200).play_history[0].cards == ("4D",)
    next_action = _commit(session, _candidate(session, "opposite-5", Seat.OPPOSITE, "5D", 250))
    assert next_action.version_before.state_revision == rebound.version.state_revision
    assert [item.cards for item in session.snapshot(captured_ms=260).play_history] == [
        ("4D",),
        ("5D",),
    ]


def test_repeated_explicit_corrections_remain_auditable_and_advice_sees_latest(tmp_path):
    session = ProductionRuleSession(_store(tmp_path))
    session.initialize(round_level="2", hand=_hand(), lead_player=Seat.RIGHT, monotonic_ms=1)
    action = _commit(session, _candidate(session, "right-3", Seat.RIGHT, "3D", 100))
    first = session.correct_latest(_correction(session, action.action_id, "4D", 200))
    second = session.correct_latest(_correction(session, action.action_id, "5D", 300))

    trusted = session.snapshot(captured_ms=300)
    advice_snapshot = trusted_to_live_snapshot(trusted)

    assert trusted.correction_history == (first, second)
    assert trusted.play_history[0].cards == ("5D",)
    assert advice_snapshot.play_history[0].cards == ("5D",)
    assert [item["payload"]["correction_id"] for item in read_json_lines(
        session._store.timeline_path
    ) if item["event_type"] == "event_correction"] == ["fix-200", "fix-300"]


def test_health_uses_real_audit_and_reports_premature_terminal(tmp_path):
    session = ProductionRuleSession(_store(tmp_path))
    session.initialize(round_level="2", hand=_hand(), lead_player=Seat.RIGHT, monotonic_ms=1)
    version = session.version
    terminal = LiveEvent(
        event_id="terminal", event_type="game_end_detected", session_id="s", seq=99,
        monotonic_ms=100, wall_time="2026-09-06T12:00:00+08:00", trick_id=1,
        turn_id=1, actor=None, payload={}, confidence=1.0, source="visual",
        state_revision_before=version.state_revision,
        state_revision_after=version.state_revision, evidence_refs=(),
    )

    report = session.health(additional_events=(terminal,))

    assert report["status"] == "FAIL"
    assert report["issues"][0]["code"] == "HEALTH-PREMATURE-GAME-END"
    assert not hasattr(session, "confirm_visual_placement")
