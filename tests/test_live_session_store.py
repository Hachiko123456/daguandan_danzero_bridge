from __future__ import annotations

import gzip
import json

from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines


def _manifest() -> dict[str, object]:
    return {
        "application_version": "0.1.0",
        "configuration_hash": "config-hash",
        "template_manifest_hash": "template-hash",
        "target_fps": 10,
        "codec": "MJPG",
    }


def _event(
    session_id: str,
    *,
    event_type: str = "player_played",
    actor: str | None = "right",
    payload: dict[str, object] | None = None,
) -> LiveEvent:
    return LiveEvent(
        event_id="EVT-000017",
        event_type=event_type,
        session_id=session_id,
        seq=17,
        monotonic_ms=18_342,
        wall_time="2026-08-06T19:30:33+08:00",
        trick_id=2,
        turn_id=6,
        actor=actor,  # type: ignore[arg-type]
        payload=payload or {"cards": ["7S", "7H"], "is_pass": False},
        confidence=0.94,
        source="multi_frame_consensus",
        state_revision_before=11,
        state_revision_after=12,
        evidence_refs=("OBS-0183", "frame:344"),
    )


def _started_store(tmp_path, session_id: str = "game-a") -> LiveSessionStore:
    store = LiveSessionStore(
        tmp_path,
        "tencent_daguandan",
        session_id=session_id,
    )
    store.start(_manifest())
    return store


def test_sessions_are_physically_isolated(tmp_path):
    first = _started_store(tmp_path, "game-a")
    second = _started_store(tmp_path, "game-b")

    first.append_event(_event("game-a"))

    assert first.directory != second.directory
    assert len(read_json_lines(first.timeline_path)) == 1
    assert read_json_lines(second.timeline_path) == []


def test_timeline_markdown_is_llm_readable(tmp_path):
    store = _started_store(tmp_path)

    store.append_event(_event("game-a"))

    text = store.timeline_markdown_path.read_text(encoding="utf-8")
    assert "右家出牌：[7S, 7H]" in text
    assert "置信度=94%" in text
    assert "证据=OBS-0183, frame:344" in text


def test_advice_log_keeps_full_engine_input(tmp_path):
    store = _started_store(tmp_path)
    engine_input = {"request_id": "ADV-4", "legal_actions": [["PASS"]]}

    store.append_advice(
        {
            "request_id": "ADV-4",
            "status": "ready",
            "engine_input": engine_input,
            "timings": {"agent_step": 12.5},
        }
    )

    record = read_json_lines(store.advice_path)[0]
    assert record["engine_input"] == engine_input
    assert record["timings"]["agent_step"] == 12.5


def test_incident_contains_state_observations_and_llm_report(tmp_path):
    store = _started_store(tmp_path)

    path = store.create_incident(
        reason="candidate_conflict",
        state_before={"revision": 4, "current_player": "right"},
        state_after={"revision": 4, "current_player": "right"},
        observations=[{"id": "OBS-4", "cards": ["7S"]}],
        engine_input={"request_id": "ADV-3"},
    )

    assert (path / "incident.json").is_file()
    assert json.loads((path / "state_before.json").read_text("utf-8"))["revision"] == 4
    assert json.loads((path / "engine_input.json").read_text("utf-8"))["request_id"] == "ADV-3"
    report = (path / "llm_report.md").read_text("utf-8")
    assert "candidate_conflict" in report
    assert "OBS-4" in report


def test_reader_ignores_only_a_truncated_last_json_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"seq": 1}\n{"seq": 2', encoding="utf-8")

    assert read_json_lines(path) == [{"seq": 1}]


def test_seal_compresses_observations_and_updates_manifest(tmp_path):
    store = _started_store(tmp_path)
    store.append_observation({"id": "OBS-1", "phase": "settling"})

    store.seal(frame_count=12, dropped_frames=1)

    manifest = json.loads(store.manifest_path.read_text("utf-8"))
    assert manifest["status"] == "sealed"
    assert manifest["frame_count"] == 12
    assert manifest["dropped_frames"] == 1
    assert not store.observations_part_path.exists()
    with gzip.open(store.observations_gzip_path, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline())["id"] == "OBS-1"


def test_recovery_marks_unsealed_previous_process_session_aborted(tmp_path):
    store = _started_store(tmp_path, "crashed-game")
    store.append_observation({"id": "OBS-before-crash"})

    recovered = LiveSessionStore.recover_incomplete_sessions(
        tmp_path,
        "tencent_daguandan",
    )

    manifest = json.loads(store.manifest_path.read_text("utf-8"))
    assert recovered == (store.directory,)
    assert manifest["status"] == "aborted"
    assert manifest["recovery_reason"] == "previous_process_did_not_seal"
    assert store.observations_part_path.is_file()
