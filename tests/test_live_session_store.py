from __future__ import annotations

import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor

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
    assert "[第6手] 右家出牌：7♠ 7♥" in text
    assert "置信度=94%" in text
    assert "证据=OBS-0183, frame:344" in text


def test_event_keeps_canonical_fsync_but_defers_derived_markdown(tmp_path, monkeypatch):
    from daguandan_bridge.live import session_store as module
    store = _started_store(tmp_path)
    synced = []
    original = module.os.fsync
    monkeypatch.setattr(module.os, "fsync", lambda fd: (synced.append(fd), original(fd))[1])
    store.append_event(_event("game-a"))
    assert len(synced) == 1
    assert len(read_json_lines(store.timeline_path)) == 1
    assert "右家出牌" not in store._timeline_markdown_path.read_text("utf-8")
    assert "右家出牌" in store.timeline_markdown_path.read_text("utf-8")


def test_derived_write_and_cleanup_failures_do_not_abort_seal(tmp_path, monkeypatch):
    from pathlib import Path
    store = _started_store(tmp_path)
    store.append_event(_event("game-a"))
    replace, unlink = Path.replace, Path.unlink
    def denied_replace(path, target):
        if ".timeline.md." in path.name:
            raise PermissionError("derived view locked")
        return replace(path, target)
    def denied_unlink(path, *args, **kwargs):
        if ".timeline.md." in path.name:
            raise PermissionError("derived temp locked")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "replace", denied_replace)
    monkeypatch.setattr(Path, "unlink", denied_unlink)
    store.seal(frame_count=0, dropped_frames=0)
    manifest = json.loads(store.manifest_path.read_text("utf-8"))
    assert manifest["status"] == "sealed"
    counters = manifest["pipeline_timing"]["counters"]
    assert counters["markdown_write_failed"] == 1
    assert counters["markdown_cleanup_failed"] == 1
    assert len(read_json_lines(store.timeline_path)) == 1


def test_pipeline_summary_is_bounded_exported_and_throttled(tmp_path, monkeypatch):
    store = _started_store(tmp_path)
    store.pipeline_timing.increment("analysis_without_result", 7)
    store.pipeline_timing.observe("analysis", 17)
    writes = []
    original = store._update_manifest
    monkeypatch.setattr(store, "_update_manifest", lambda changes: (writes.append(changes), original(changes))[1])
    store.flush_pipeline_timing()
    assert not writes
    store.flush_pipeline_timing(force=True)
    store.flush_pipeline_timing()
    assert len(writes) == 1
    store.seal(frame_count=0, dropped_frames=0)
    manifest = json.loads(store.manifest_path.read_text("utf-8"))
    assert manifest["pipeline_timing"]["counters"]["analysis_without_result"] == 7
    assert manifest["pipeline_timing"]["clock"] == "processing_monotonic_ns"


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


def test_runtime_identity_updates_merge_without_replacing_first_run_fields(tmp_path):
    store = LiveSessionStore(tmp_path, "profile", session_id="identity-merge")
    store.start(
        {
            "runtime_identity": {
                "schema": "guandan.runtime-identity/1",
                "run_id": "RUN-FIRST",
                "implementation_fingerprint": "initial-fingerprint",
                "executable_path": "DaguandanAssistant.exe",
                "build": {"status": "identified", "build_id": "BUILD-1"},
            }
        }
    )

    store.update_runtime_identity(
        {
            "implementation_fingerprint": "later-fingerprint",
            "executable_path": "C:/Users/private/python.exe",
            "orchestrator_probe": "available",
            "build": {"extra": "preserved"},
        }
    )

    identity = json.loads(store.manifest_path.read_text(encoding="utf-8"))[
        "runtime_identity"
    ]
    assert identity["run_id"] == "RUN-FIRST"
    assert identity["implementation_fingerprint"] == "initial-fingerprint"
    assert identity["executable_path"] == "DaguandanAssistant.exe"
    assert identity["orchestrator_probe"] == "available"
    assert identity["build"] == {
        "status": "identified",
        "build_id": "BUILD-1",
        "extra": "preserved",
    }


def test_decision_upsert_keeps_one_correlated_record(tmp_path):
    store = LiveSessionStore(tmp_path, "profile", session_id="decision-test")
    store.start({})
    store.upsert_decision({"decision_id": "D-1", "state_before": {"revision": 1}})
    store.upsert_decision({"decision_id": "D-1", "actual_action_event_id": "EV-1"})

    records = read_json_lines(store.decisions_path)

    assert len(records) == 1
    assert records[0]["state_before"] == {"revision": 1}
    assert records[0]["actual_action_event_id"] == "EV-1"


def test_decision_upsert_uses_unique_temp_files_and_retries_transient_windows_lock(
    tmp_path,
    monkeypatch,
):
    store = LiveSessionStore(tmp_path, "profile", session_id="decision-retry")
    store.start({})
    real_replace = os.replace
    attempts = []

    def transient_replace(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError(5, "transient lock", str(target))
        return real_replace(source, target)

    monkeypatch.setattr("daguandan_bridge.live.session_store.os.replace", transient_replace)
    store.upsert_decision({"decision_id": "D-1", "status": "ready"})

    assert len(attempts) == 3
    assert read_json_lines(store.decisions_path)[0]["status"] == "ready"
    assert not tuple(store.directory.glob(".decisions.jsonl.*.tmp"))


def test_decision_upserts_are_serialized_without_lost_records(tmp_path):
    store = LiveSessionStore(tmp_path, "profile", session_id="decision-concurrent")
    store.start({})

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                store.upsert_decision,
                {"decision_id": f"D-{index:02d}", "state_revision": index},
            )
            for index in range(24)
        ]
        for future in futures:
            future.result()

    records = read_json_lines(store.decisions_path)
    assert len(records) == 24
    assert {record["decision_id"] for record in records} == {
        f"D-{index:02d}" for index in range(24)
    }
    assert not tuple(store.directory.glob(".decisions.jsonl.*.tmp"))


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
    assert "候选动作相互冲突" in report
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


def test_recovery_does_not_abort_session_owned_by_a_live_process(tmp_path):
    store = LiveSessionStore(
        tmp_path,
        "tencent_daguandan",
        session_id="active-game",
    )
    store.start({"owner_pid": os.getpid()})

    recovered = LiveSessionStore.recover_incomplete_sessions(
        tmp_path,
        "tencent_daguandan",
    )

    manifest = json.loads(store.manifest_path.read_text("utf-8"))
    assert recovered == ()
    assert manifest["status"] == "running"


def test_recovery_preserves_crashed_preopening_without_creating_a_game(tmp_path):
    store = LiveSessionStore.for_opening_evidence(
        tmp_path,
        "tencent_daguandan",
    )
    store.start(
        {
            "owner_pid": 987654321,
            "schema": "guandan.opening-evidence/1",
            "recording_phase": "listening",
            "initial_state_status": "unconfirmed",
        }
    )
    video = store.directory / "video"
    video.mkdir()
    partial = video / "game.avi.part"
    partial.write_bytes(b"recoverable-opening-video")

    recovered = LiveSessionStore.recover_incomplete_sessions(
        tmp_path,
        "tencent_daguandan",
    )

    assert recovered == (store.directory,)
    assert store.directory.parent.name == ".preopening"
    assert store.directory.name.startswith("opening_")
    assert partial.read_bytes() == b"recoverable-opening-video"
    assert not tuple((tmp_path / "tencent_daguandan" / "sessions").glob("game_*"))
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert manifest["recording_phase"] == "aborted_before_initial_state"
    assert manifest["initial_state_status"] == "unconfirmed"
    assert manifest["termination_reason"] == "previous_process_did_not_seal"
