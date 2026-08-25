from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from scripts import run_full_flow_validation as launcher


def _session(root: Path, name: str, *, mtime: int) -> Path:
    session = root / name
    video = session / "video"
    video.mkdir(parents=True)
    manifest = session / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    (session / "timeline.jsonl").write_text("", encoding="utf-8")
    (video / "game.avi").write_bytes(b"video")
    (video / "frame_index.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(manifest, (mtime, mtime))
    return session.resolve()


def test_launcher_discovers_newest_first_and_supports_single_or_all_prompt(tmp_path):
    older = _session(tmp_path / "profiles", "game_old", mtime=1_700_000_100)
    newer = _session(tmp_path / "profiles", "game_new", mtime=1_700_000_200)

    discovered = launcher.discover_sessions((tmp_path / "profiles",))

    assert discovered == (newer, older)
    assert launcher.select_interactively(discovered, input_fn=lambda _prompt: "2") == (
        older,
    )
    assert launcher.select_interactively(discovered, input_fn=lambda _prompt: "a") == discovered


def test_launcher_runs_unit_gate_then_selected_replays_and_writes_aggregate(tmp_path):
    sessions = (
        _session(tmp_path / "profiles" / "sessions", "game_new", mtime=200),
        _session(tmp_path / "profiles" / "sessions", "game_old", mtime=100),
    )
    calls: list[object] = []

    class FakeRunner:
        def run(self, config):
            calls.append(config)
            run_directory = Path(config.output) / str(config.run_id)
            run_directory.mkdir()
            runtime_directory = run_directory / "runtime"
            runtime_directory.mkdir()
            (runtime_directory / "timeline.jsonl").write_text("", encoding="utf-8")
            summary_path = run_directory / "summary.json"
            summary_path.write_text("{}\n", encoding="utf-8")
            return SimpleNamespace(
                execution_ok=True,
                run_directory=run_directory,
                summary_path=summary_path,
                summary={"runtime_directory": str(runtime_directory)},
            )

    unit_commands: list[tuple[object, object]] = []

    def fake_subprocess(command, *, cwd):
        unit_commands.append((command, cwd))
        return SimpleNamespace(returncode=0)

    code, report = launcher.run_validation(
        sessions,
        output=tmp_path / "reports",
        run_id="test-run",
        time_scale=4.0,
        skip_unit=False,
        subprocess_run=fake_subprocess,
        replay_runner_factory=FakeRunner,
    )

    summary = json.loads(report.read_text(encoding="utf-8"))
    assert code == 0
    assert unit_commands[0][0][-2:] == ["-m", "pytest"]
    assert len(calls) == 2
    assert all(config.time_scale == 4.0 for config in calls)
    assert summary["execution_ok"] is True
    assert summary["unit_tests"]["status"] == "passed"
    assert [row["session_name"] for row in summary["replays"]] == [
        "game_new",
        "game_old",
    ]
    assert report.parent != sessions[0].parent
    assert (report.parent / "full_flow_summary.md").is_file()


def test_launcher_stops_before_replay_when_full_pytest_fails(tmp_path):
    session = _session(tmp_path / "profiles" / "sessions", "game", mtime=100)

    class ForbiddenRunner:
        def __init__(self):
            raise AssertionError("unit gate failure must stop replay")

    code, report = launcher.run_validation(
        (session,),
        output=tmp_path / "reports",
        run_id="unit-failure",
        time_scale=1.0,
        skip_unit=False,
        subprocess_run=lambda *_args, **_kwargs: SimpleNamespace(returncode=5),
        replay_runner_factory=ForbiddenRunner,
    )

    summary = json.loads(report.read_text(encoding="utf-8"))
    assert code == 1
    assert summary["unit_tests"]["status"] == "failed"
    assert summary["replays"] == []
    assert summary["execution_ok"] is False


def test_launcher_timeline_gate_rejects_next_action_as_reread_closure(tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    rows = [
        {
            "event_id": "AUX-1",
            "event_type": "advice_withheld",
            "monotonic_ms": 100,
            "payload": {
                "reason": "previous_action_reread_pending",
                "target_event_id": "EVT-1",
                "followup_event_id": "EVT-2",
            },
        },
        {
            "event_id": "EVT-3",
            "event_type": "player_passed",
            # Even inside 1200 ms, a user action does not prove that the
            # previous-action verification reached a terminal lifecycle.
            "monotonic_ms": 1_000,
            "payload": {"is_pass": True},
        },
    ]
    timeline.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    audit = launcher.audit_previous_action_verifications(timeline)

    assert audit["execution_ok"] is False
    assert audit["targets"] == 1
    assert audit["unresolved"][0]["target_event_id"] == "EVT-1"
    assert audit["unresolved"][0]["closure"] is None


def test_launcher_timeline_gate_accepts_explicit_timeout_within_1200ms(tmp_path):
    timeline = tmp_path / "timeline.jsonl"
    rows = [
        {
            "event_id": "AUX-1",
            "event_type": "advice_withheld",
            "monotonic_ms": 100,
            "payload": {
                "reason": "previous_action_reread_pending",
                "target_event_id": "EVT-1",
                "followup_event_id": "EVT-2",
            },
        },
        {
            "event_id": "AUX-2",
            "event_type": "previous_action_verification_expired",
            "monotonic_ms": 1_300,
            "payload": {"target_event_id": "EVT-1"},
        },
    ]
    timeline.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    audit = launcher.audit_previous_action_verifications(timeline)

    assert audit["execution_ok"] is True
    assert audit["targets"] == 1
    assert audit["closed"] == 1


def _played(event_id: str, actor: str, *cards: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_type": "player_played",
        "actor": actor,
        "payload": {"cards": list(cards), "is_pass": False},
    }


def _passed(event_id: str, actor: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_type": "player_passed",
        "actor": actor,
        "payload": {"cards": [], "is_pass": True},
    }


def test_business_coverage_rejects_truncated_runtime():
    source = (
        _played("source-1", "right", "7H", "7D"),
        _passed("source-2", "opposite"),
        _played("source-3", "left", "8C"),
    )
    runtime = (
        _played("runtime-9", "right", "7D", "7H"),
        _passed("runtime-10", "opposite"),
    )

    audit = launcher.audit_business_coverage(source, runtime)

    assert audit["execution_ok"] is False
    assert audit["source_action_count"] == 3
    assert audit["runtime_action_count"] == 2
    assert audit["first_divergence"]["action_number"] == 3
    assert audit["first_divergence"]["runtime"] is None


def test_business_coverage_accepts_identical_actions_with_different_event_ids():
    source = (
        _played("source-1", "right", "7H", "7D"),
        _passed("source-2", "opposite"),
    )
    runtime = (
        _played("runtime-101", "right", "7D", "7H"),
        _passed("runtime-102", "opposite"),
    )

    audit = launcher.audit_business_coverage(source, runtime)

    assert audit["execution_ok"] is True
    assert audit["counts_match"] is True
    assert audit["first_divergence"] is None


def test_business_coverage_applies_corrections_on_both_timelines():
    source = (
        _played("source-1", "left", "2H"),
        {
            "event_id": "source-correction",
            "event_type": "event_correction",
            "actor": "left",
            "payload": {
                "target_event_id": "source-1",
                "cards": ["2C", "2D", "2H"],
                "is_pass": False,
            },
        },
    )
    runtime = (
        _played("runtime-1", "left", "2H"),
        {
            "event_id": "runtime-correction",
            "event_type": "event_correction",
            "actor": "left",
            "payload": {
                "target_event_id": "runtime-1",
                "cards": ["2H", "2D", "2C"],
                "is_pass": False,
            },
        },
    )

    audit = launcher.audit_business_coverage(source, runtime)

    assert audit["execution_ok"] is True
    assert audit["source_action_count"] == 1
    assert audit["runtime_action_count"] == 1


def test_run_validation_requires_business_coverage_even_when_shadow_succeeds(tmp_path):
    session = _session(
        tmp_path / "profiles" / "sessions",
        "game_truncated",
        mtime=1_700_000_100,
    )
    (session / "timeline.jsonl").write_text(
        json.dumps(_played("source-1", "right", "7H")) + "\n",
        encoding="utf-8",
    )

    class TruncatedRunner:
        def run(self, config):
            run_directory = Path(config.output) / str(config.run_id)
            runtime_directory = run_directory / "runtime"
            runtime_directory.mkdir(parents=True)
            (runtime_directory / "timeline.jsonl").write_text("", encoding="utf-8")
            summary_path = run_directory / "summary.json"
            summary_path.write_text("{}\n", encoding="utf-8")
            return SimpleNamespace(
                execution_ok=True,
                run_directory=run_directory,
                summary_path=summary_path,
                summary={"runtime_directory": str(runtime_directory)},
            )

    code, report = launcher.run_validation(
        (session,),
        output=tmp_path / "reports",
        run_id="truncated-runtime",
        time_scale=1.0,
        skip_unit=True,
        replay_runner_factory=TruncatedRunner,
    )

    summary = json.loads(report.read_text(encoding="utf-8"))
    replay = summary["replays"][0]
    assert code == 1
    assert summary["execution_ok"] is False
    assert replay["execution_ok"] is False
    assert replay["business_coverage_audit"]["source_action_count"] == 1
    assert replay["business_coverage_audit"]["runtime_action_count"] == 0
