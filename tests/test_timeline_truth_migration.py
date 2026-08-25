from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from daguandan_bridge.application.timeline_truth_migration import (
    LEGACY_TURN_PROJECTION_POLICY,
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


def test_timeline_migration_preserves_realtime_move_semantics(tmp_path: Path):
    session = tmp_path / "sessions" / "semantic"
    events = _valid_events("semantic")
    events[1] = replace(
        events[1],
        payload={
            **events[1].payload,
            "play_type": "Single",
            "logical_rank": "2",
            "logical_label": "单张2",
            "interpretation_ambiguous": False,
            "wildcard_substitutions": [],
            "candidate_interpretations": [
                {"move_type": "Single", "key": "2"}
            ],
            "selected_interpretation": {
                "move_type": "Single",
                "key": "2",
                "wildcard_assignments": [],
            },
            "selection_source": "realtime_semantics",
        },
    )
    _write_timeline(session, events)

    result = TimelineTruthMigrationService().migrate_session(session)
    truth = load_truth_log(session / "truth_log.json", session_id="semantic")

    assert result.status == "migrated"
    semantics = truth.turns[0].move_semantics
    assert semantics["move_type"] == "Single"
    assert semantics["selection_source"] == "realtime_semantics"
    assert semantics["selected_interpretation"]["key"] == "2"


def test_timeline_migration_accepts_only_the_adjacent_visual_expansion_scope(
    tmp_path: Path,
):
    session = tmp_path / "sessions" / "adjacent-reread"
    reducer = LiveReducer("adjacent-reread")
    initial = reducer.confirm_initial_state(
        round_level="8", hand=HAND, lead_player="self"
    )
    played = reducer.record_play("self", ("2H",))
    followed = reducer.record_pass("right")
    correction = reducer.correct_previous_action_after_followup(
        played.event_id,
        expected_followup_actor="right",
        followup_event_id=followed.event_id,
        cards=("2H", "2D", "2C", "3H", "3C", "3S"),
        reason="two_distinct_adjacent_action_rereads",
    )
    _write_timeline(session, [initial, played, followed, correction])

    result = TimelineTruthMigrationService().migrate_session(session)
    truth = load_truth_log(session / "truth_log.json", session_id="adjacent-reread")

    assert result.status == "migrated"
    assert truth.turns[0].cards == ("2C", "2D", "2H", "3C", "3H", "3S")
    assert truth.turns[1].actor == "right"
    assert truth.turns[1].is_pass


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


def test_existing_repair_reindexes_only_stale_trick_ids(tmp_path: Path):
    session = tmp_path / "sessions" / "stale-trick"
    session.mkdir(parents=True)
    (session / "timeline.jsonl").write_text("", encoding="utf-8")
    truth = TruthLog(
        "stale-trick",
        TruthInitialState("8", "self", HAND),
        (
            TruthTurn(1, "self", False, ("3S",), trick_id=1),
            TruthTurn(2, "right", True, (), trick_id=1),
            TruthTurn(3, "opposite", True, (), trick_id=1),
            TruthTurn(4, "left", True, (), trick_id=1),
            # The old reducer kept the cleared trick open, so this legal lead
            # was persisted under its stale id instead of trick 2.
            TruthTurn(5, "self", False, ("4S",), trick_id=1),
        ),
    )
    truth_path = session / "truth_log.json"
    truth_path.write_text(
        json.dumps(truth.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    before = truth_path.read_bytes()

    inspection = TimelineTruthMigrationService().inspect_existing_truth_repair(session)

    assert inspection.status == "candidate"
    assert inspection.code == "safe_to_repair_trick_context"
    assert inspection.replacement is not None
    assert inspection.replacement.turns[4].trick_id == 2
    assert inspection.context_repairs == (
        "turn 5: trick 1 -> 2 after replayed wind catch",
    )
    result = TimelineTruthMigrationService().repair_existing_truth(session)
    assert result.status == "repaired"
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == before
    receipt = json.loads((session / "truth_log.repair.json").read_text(encoding="utf-8"))
    assert receipt["validation"]["context_repairs"] == list(inspection.context_repairs)
    repaired = load_truth_log(truth_path, session_id="stale-trick")
    assert repaired.turns[4].trick_id == 2
    assert [(turn.actor, turn.cards) for turn in repaired.turns] == [
        (turn.actor, turn.cards) for turn in truth.turns
    ]


def test_source_replay_normalizes_proven_wind_catch_context(tmp_path: Path):
    session = tmp_path / "sessions" / "wind-catch"
    reducer = LiveReducer("wind-catch", wind_receiver_must_pass=False)
    initial = reducer.confirm_initial_state(
        round_level="8", hand=HAND, lead_player="left"
    )
    left_finished = reducer.record_play("left", HAND)
    self_finished = reducer.record_play("self", HAND)
    declined = reducer.record_pass("right")
    catch = reducer.record_play("opposite", ("3S",))
    stale_catch = replace(
        catch,
        trick_id=catch.trick_id - 1,
        payload={
            **catch.payload,
            "integrity_warnings": ["observed_table_mismatch"],
            "beats_table": False,
        },
    )
    _write_timeline(
        session,
        [initial, left_finished, self_finished, declined, stale_catch],
    )
    normalized, context_repairs = TimelineTruthMigrationService()._normalize_source_actions(
        "wind-catch",
        (initial, left_finished, self_finished, declined, stale_catch),
        (left_finished, self_finished, declined, stale_catch),
        wind_receiver_must_pass=False,
    )

    assert context_repairs == (
        "EVT-000005: trick 1 -> 2 after proven wind catch",
    )
    assert normalized[-1].trick_id == 2


def test_legacy_wind_timeline_requires_explicit_migration_policy(tmp_path: Path):
    session = tmp_path / "sessions" / "legacy-wind"
    producer = LiveReducer(
        "legacy-wind",
        wind_receiver_must_pass=False,
    )
    initial = producer.confirm_initial_state(
        round_level="8", hand=HAND, lead_player="right"
    )
    right_finished = producer.record_play("right", HAND)
    opposite_passed = producer.record_pass("opposite")
    self_passed = producer.record_pass("self")
    left_caught_wind = producer.record_play("left", ("4S",))
    _write_timeline(
        session,
        [
            initial,
            right_finished,
            opposite_passed,
            self_passed,
            left_caught_wind,
        ],
    )

    service = TimelineTruthMigrationService()
    default = service.inspect_session(session)

    assert default.status == "blocked"
    assert default.code == "source_reducer_replay_failed"
    assert not (session / "truth_log.json").exists()

    migrated = service.migrate_session(
        session,
        turn_projection_policy=LEGACY_TURN_PROJECTION_POLICY,
    )

    assert migrated.status == "migrated"
    truth = load_truth_log(session / "truth_log.json", session_id="legacy-wind")
    assert [turn.actor for turn in truth.turns] == [
        "right",
        "opposite",
        "self",
        "left",
    ]
    receipt = json.loads(
        (session / "truth_log.migration.json").read_text(encoding="utf-8")
    )
    assert receipt["turn_projection_policy"] == LEGACY_TURN_PROJECTION_POLICY
    assert (
        receipt["output_truth_log"]["turn_projection_policy"]
        == LEGACY_TURN_PROJECTION_POLICY
    )
