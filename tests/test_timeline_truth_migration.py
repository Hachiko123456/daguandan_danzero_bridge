from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from daguandan_bridge.application.timeline_truth_migration import (
    MIGRATION_SOURCE,
    TimelineTruthMigrationService,
)
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.reducer import LiveReducer
from daguandan_bridge.live.truth_log import load_truth_log
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _write_timeline(session: Path, events: list[LiveEvent]) -> Path:
    session.mkdir(parents=True)
    path = session / "timeline.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {**event.to_dict(), "schema_version": 1},
                ensure_ascii=False,
            )
            + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    return path


def _valid_events(session_id: str) -> list[LiveEvent]:
    reducer = LiveReducer(session_id)
    initial = reducer.confirm_initial_state(
        round_level="8", hand=HAND, lead_player="self"
    )
    played = reducer.record_play("self", ("2S",))
    passed = reducer.record_pass("right")
    return [initial, played, passed]


def test_valid_timeline_migrates_to_draft_truth_with_audit_and_is_idempotent(
    tmp_path: Path,
):
    session = tmp_path / "sessions" / "game"
    timeline = _write_timeline(session, _valid_events("game"))
    before = timeline.read_bytes()

    service = TimelineTruthMigrationService()
    result = service.migrate_session(session)

    assert result.status == "migrated"
    assert timeline.read_bytes() == before
    truth = load_truth_log(session / "truth_log.json", session_id="game")
    assert truth.source_session_id == "game"
    assert truth.initial_state.round_level == "8"
    assert truth.initial_state.lead_player == "self"
    assert truth.initial_state.my_hand == tuple(sorted(HAND))
    assert [(turn.actor, turn.is_pass, turn.cards) for turn in truth.turns] == [
        ("self", False, ("2S",)),
        ("right", True, ()),
    ]
    assert truth.label_status == "draft"
    assert truth.provenance.source == MIGRATION_SOURCE
    assert all(turn.provenance.source == MIGRATION_SOURCE for turn in truth.turns)
    assert all(not turn.evidence.frame_indices for turn in truth.turns)
    receipt = json.loads(
        (session / "truth_log.migration.json").read_text(encoding="utf-8")
    )
    assert receipt["source_timeline"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt["source_timeline"]["event_count"] == 3
    assert receipt["source_timeline"]["action_count"] == 2
    assert receipt["output_truth_log"]["turn_count"] == 2
    truth_before = (session / "truth_log.json").read_bytes()

    repeated = service.migrate_session(session)

    assert repeated.status == "skipped"
    assert repeated.code == "existing_truth_log"
    assert (session / "truth_log.json").read_bytes() == truth_before


def test_action_sequence_mismatch_blocks_without_truth_output(tmp_path: Path):
    session = tmp_path / "sessions" / "bad-sequence"
    events = _valid_events("bad-sequence")
    events[-1] = replace(events[-1], actor="left")
    _write_timeline(session, events)

    result = TimelineTruthMigrationService().migrate_session(session)

    assert result.status == "blocked"
    assert result.code == "source_reducer_replay_failed"
    assert not (session / "truth_log.json").exists()
    assert not (session / "truth_log.migration.json").exists()


def test_missing_lead_without_confirmation_blocks(tmp_path: Path):
    session = tmp_path / "sessions" / "missing-lead"
    reducer = LiveReducer("missing-lead")
    initial = reducer.confirm_initial_state(
        round_level="8", hand=HAND, lead_player=None
    )
    _write_timeline(session, [initial])

    result = TimelineTruthMigrationService().migrate_session(session)

    assert result.status == "blocked"
    assert result.code == "lead_confirmation_count"
    assert not (session / "truth_log.json").exists()


def test_existing_repair_blocks_timeline_integrity_warning_without_writes(
    tmp_path: Path,
):
    session = tmp_path / "sessions" / "warned"
    events = _valid_events("warned")
    warned = replace(
        events[1],
        payload={**events[1].payload, "integrity_warnings": ["observed_table_mismatch"], "beats_table": False},
    )
    _write_timeline(session, [events[0], warned, events[2]])
    legacy = TruthLog(
        "warned",
        TruthInitialState("8", "self", HAND),
        (
            TruthTurn(1, "self", False, ("2S",), trick_id=1),
            TruthTurn(2, "right", True, (), trick_id=1),
        ),
    ).to_dict()
    legacy.pop("schema")
    legacy["schema_version"] = 2
    truth_path = session / "truth_log.json"
    truth_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    before = truth_path.read_bytes()

    result = TimelineTruthMigrationService().repair_existing_truth(session)

    assert result.status == "blocked"
    assert result.code == "unsafe_timeline_integrity"
    assert truth_path.read_bytes() == before
    assert not (session / "truth_log.repair.json").exists()
    assert not list(session.glob("truth_log.repair.*.json"))
