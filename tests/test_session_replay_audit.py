from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

import daguandan_bridge.application.session_replay_audit as audit_module

from daguandan_bridge.application.live_v2_recorded_replay import _opening_recognition_fields

from daguandan_bridge.application.session_replay_audit import (
    SessionReplayAuditService,
    _first_divergence,
    _listener_completion,
    _row_quality_failures,
    _visual_advice_summary,
    compare_truth_visual_fields,
    resolve_truth_audit_reference,
    summarize_visual_events,
)
from daguandan_bridge.live.replay import (
    ReplayComparison,
    TrustedAdviceReplayResult,
    VisualPipelineReplayResult,
)
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    save_truth_log,
)


def _replay_result(tmp_path: Path, *, status_reason: str, current_player=None, actions=1):
    output = tmp_path / "visual.jsonl"
    output.write_text(
        json.dumps({"current_player": current_player}) + "\n",
        encoding="utf-8",
    )
    return VisualPipelineReplayResult(
        output_path=output,
        comparison_path=tmp_path / "comparison.json",
        frame_count=1,
        warnings=(),
        comparison=ReplayComparison((), (), (), (), ()),
        status="complete",
        status_reason=status_reason,
        runtime_identity={
            "runtime": "live_v2",
            "listener_terminal": current_player is None,
            "current_player": current_player,
        },
    ), {"actions": {"count": actions}}


def test_audit_rejects_more_than_three_workers(tmp_path: Path):
    with pytest.raises(ValueError, match="between 1 and 3"):
        SessionReplayAuditService().audit([], output=tmp_path / "reports", max_workers=4)


def test_parallel_audit_keeps_report_order_and_serializes_progress(
    tmp_path: Path, monkeypatch
):
    sessions = []
    for name in ("game-a", "game-b", "game-c"):
        session = tmp_path / "sessions" / name
        session.mkdir(parents=True)
        (session / "manifest.json").write_text(
            json.dumps({"session_id": name}), encoding="utf-8"
        )
        sessions.append(session)

    service = SessionReplayAuditService()
    progress = []

    def fake_audit(item, output, scan_run_id, inventory, *, on_progress=None):
        del scan_run_id, inventory
        if on_progress is not None:
            on_progress("visual", item.session_id, 1, 2, "帧 1")
        # Finish in reverse order to prove that report order does not depend on
        # executor completion order.
        time.sleep({"game-a": 0.03, "game-b": 0.02, "game-c": 0.01}[item.session_id])
        if on_progress is not None:
            on_progress("fabledan_truth", item.session_id, 1, 1, "推荐 1/1")
        row = {
            "source": str(item.source),
            "sessions_root": str(item.root),
            "store_id": item.store_id,
            "session_id": item.session_id,
            "status": "completed",
            "execution_status": "completed",
            "frame_replay_status": "complete",
            "opening_status": "recognized",
            "listener_status": "complete",
            "comparison_status": "diagnostic",
            "truth_quality": "diagnostic",
            "visual_quality": "passed",
            "fabledan_quality": "passed",
            "frames_processed": 1,
            "indexed_frames": 1,
            "truth_log": {"kind": "none"},
            "fabledan": {},
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(row), encoding="utf-8"
        )
        return row

    monkeypatch.setattr(service, "_audit_session", fake_audit)
    run = service.audit(
        [],
        output=tmp_path / "reports",
        run_id="parallel",
        session_paths=sessions,
        max_workers=3,
        on_progress=lambda *values: progress.append(values),
    )

    assert [row["session_id"] for row in run.sessions] == [
        "game-a", "game-b", "game-c"
    ]
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert [row["session_id"] for row in summary["sessions"]] == [
        "game-a", "game-b", "game-c"
    ]
    assert summary["parameters"]["max_workers"] == 3
    assert summary["parameters"]["report_order"] == (
        "original session selection order"
    )
    assert len([row for row in progress if row[0] == "session_done"]) == 3
    assert (run.run_directory / "sessions.csv").is_file()
    assert (run.run_directory / "verification.json").is_file()


def test_listener_completion_does_not_equate_sealed_or_empty_listener(tmp_path: Path):
    result, visual = _replay_result(
        tmp_path, status_reason="runtime_sealed_terminal", actions=0
    )

    status, reason = _listener_completion(result, visual, True)

    assert status == "incomplete"
    assert reason == "no_listener_actions"


def test_listener_completion_requires_terminal_reason_and_no_current_player(tmp_path: Path):
    result, visual = _replay_result(
        tmp_path, status_reason="listener_not_terminal", current_player="left"
    )

    status, reason = _listener_completion(result, visual, True)

    assert status == "incomplete"
    assert reason == "current_player=left"

    result, visual = _replay_result(tmp_path, status_reason="runtime_sealed_terminal")
    status, reason = _listener_completion(result, visual, True)
    assert status == "incomplete"
    assert reason == "non_terminal_status_reason=runtime_sealed_terminal"


def test_listener_completion_prefers_runtime_identity_current_player(tmp_path: Path):
    result, visual = _replay_result(tmp_path, status_reason="listener_terminal", current_player="left")
    result = VisualPipelineReplayResult(
        output_path=result.output_path,
        comparison_path=result.comparison_path,
        frame_count=result.frame_count,
        warnings=result.warnings,
        comparison=result.comparison,
        status="complete",
        status_reason="listener_terminal",
        runtime_identity={
            "runtime": "live_v2",
            "listener_status": "complete",
            "listener_terminal": True,
            "current_player": None,
        },
    )

    status, reason = _listener_completion(result, visual, True)

    assert status == "complete"
    assert reason == "listener_terminal"


def test_listener_completion_honors_live_v2_identity_listener_status(tmp_path: Path):
    result, visual = _replay_result(tmp_path, status_reason="listener_terminal")
    result = VisualPipelineReplayResult(
        output_path=result.output_path,
        comparison_path=result.comparison_path,
        frame_count=result.frame_count,
        warnings=result.warnings,
        comparison=result.comparison,
        status=result.status,
        status_reason=result.status_reason,
        runtime_identity={
            "runtime": "live_v2",
            "listener_status": "incomplete",
            "listener_terminal": True,
            "current_player": None,
        },
    )

    status, reason = _listener_completion(result, visual, True)

    assert status == "incomplete"
    assert reason == "runtime_listener_status=incomplete"


def _visual_result(tmp_path: Path, **kwargs) -> VisualPipelineReplayResult:
    output = tmp_path / "visual.jsonl"
    output.write_text("{}\n", encoding="utf-8")
    values = {
        "advice_requested": 0,
        "advice_ready": 0,
        "advice_failed": 0,
        "advice_stale": 0,
        "advice_timeouts": 0,
        "advice_withheld": 0,
        "advice_statuses": {},
        "completed": True,
        "status": "complete",
        "runtime_identity": {"runtime": "live_v2", "listener_status": "complete"},
    }
    values.update(kwargs)
    return VisualPipelineReplayResult(
        output_path=output,
        comparison_path=tmp_path / "comparison.json",
        frame_count=1,
        warnings=(),
        comparison=ReplayComparison((), (), (), (), ()),
        **values,
    )


def test_visual_advice_summary_does_not_call_empty_completed_run_passed(tmp_path: Path):
    result = _visual_result(tmp_path)

    summary = _visual_advice_summary(result, _AuditAdvisor(), listener_status="complete")

    assert summary["status"] == "not_exercised"
    assert summary["status"] != "passed"


def test_visual_advice_summary_reports_listener_gap_separately(tmp_path: Path):
    result = _visual_result(tmp_path, advice_withheld=1)

    summary = _visual_advice_summary(result, _AuditAdvisor(), listener_status="incomplete")

    assert summary["status"] == "withheld_due_listener_gap"


def test_visual_advice_summary_uses_scoped_listener_completion_over_tail_runtime_status(tmp_path: Path):
    result = _visual_result(
        tmp_path, advice_requested=2, advice_ready=2,
        status="review_required", completed=False,
    )

    summary = _visual_advice_summary(
        result, _AuditAdvisor(), listener_status="complete"
    )

    assert summary["status"] == "passed"
    assert summary["completed"] is True


def test_visual_advice_summary_reports_passed_when_all_requests_succeed(tmp_path: Path):
    result = _visual_result(tmp_path, advice_requested=2, advice_ready=2)

    summary = _visual_advice_summary(result, _AuditAdvisor(), listener_status="complete")

    assert summary["status"] == "passed"


def test_visual_advice_summary_reports_failed_when_a_request_fails(tmp_path: Path):
    result = _visual_result(tmp_path, advice_requested=2, advice_ready=1, advice_failed=1)

    summary = _visual_advice_summary(result, _AuditAdvisor(), listener_status="complete")

    assert summary["status"] == "failed"


def test_opening_read_fields_preserve_recognized_level_and_full_hand():
    level, hand = _opening_recognition_fields(
        type("Recognized", (), {
            "round_level": "7",
            "my_hand": ("2S", "2H", "AS"),
        })()
    )

    assert level == "7"
    assert hand == ["2S", "2H", "AS"]


def test_truth_reference_prefers_canonical_then_requested_staged_draft(tmp_path: Path):
    session = tmp_path / "game"
    session.mkdir()
    draft = session / "derived" / "truth_scan_drafts" / "batch" / "truth_log.json"
    draft.parent.mkdir(parents=True)
    save_truth_log(draft, TruthLog("game", TruthInitialState("2", "self", ("2S",)), ()))

    staged = resolve_truth_audit_reference(session, scan_run_id="batch")

    assert staged is not None and staged.kind == "staged"
    canonical = session / "truth_log.json"
    save_truth_log(canonical, TruthLog("game", TruthInitialState("2", "self", ("2S",)), ()))

    selected = resolve_truth_audit_reference(session, scan_run_id="batch")

    assert selected is not None
    assert selected.kind == "canonical"
    assert selected.path == canonical


def test_truth_metadata_records_current_truth_revision_identity(tmp_path: Path):
    session = tmp_path / "game"
    session.mkdir()
    truth_path = session / "truth_log.json"
    save_truth_log(
        truth_path,
        TruthLog("game", TruthInitialState("2", "self", ("2S",)), ()),
    )
    truth = audit_module.load_truth_log(truth_path, session_id="game")
    truth_digest = audit_module.truth_log_sha256(truth)
    (session / "truth_revision_manifest.json").write_text(
        json.dumps({
            "current_revision_id": "revision-000002",
            "current_truth_sha256": truth_digest,
            "current_semantic_sha256": "semantic-digest",
            "revisions": [{
                "revision_id": "revision-000002",
                "created_at": "2026-09-14T02:17:35+00:00",
            }],
        }),
        encoding="utf-8",
    )
    reference = resolve_truth_audit_reference(session)
    assert reference is not None
    metadata = audit_module._truth_metadata(reference, truth)

    assert metadata["revision_id"] == "revision-000002"
    assert metadata["revision_truth_sha256"] == truth_digest
    assert metadata["semantic_sha256"] == "semantic-digest"
    assert metadata["revision_matches_truth"] is True
    assert metadata["revision_created_at"] == "2026-09-14T02:17:35+00:00"
    assert metadata["revision_manifest_path"] == str(
        session / "truth_revision_manifest.json"
    )


def test_truth_reference_never_selects_staged_without_explicit_scan_id(tmp_path: Path):
    session = tmp_path / "game"
    draft = session / "derived" / "truth_scan_drafts" / "newest" / "truth_log.json"
    draft.parent.mkdir(parents=True)
    save_truth_log(draft, TruthLog("game", TruthInitialState("2", "self", ("2S",)), ()))

    assert resolve_truth_audit_reference(session) is None
    selected = resolve_truth_audit_reference(
        session,
        scan_run_id=("not-in-this-store", "newest"),
    )
    assert selected is not None and selected.path == draft


def test_visual_event_summary_covers_order_wind_rankings_gaps_and_evidence():
    summary = summarize_visual_events(
        (
            {
                "event_id": "lead",
                "event_type": "lead_player_confirmed",
                "actor": "opposite",
                "payload": {"lead_player": "opposite"},
            },
            {
                "event_id": "turn-1",
                "event_type": "player_played",
                "actor": "opposite",
                "payload": {"turn_id": 1, "cards": ["AS"], "evidence": {"frame": 4}},
            },
            {
                "event_id": "turn-3",
                "event_type": "player_passed",
                "actor": "left",
                "payload": {"turn_id": 3, "is_pass": True},
            },
            {"event_id": "wind", "event_type": "wind_caught", "actor": "left", "payload": {}},
            {"event_id": "finish", "event_type": "player_finished", "actor": "left", "payload": {"placement": 3}},
            {"event_id": "gap", "event_type": "terminal_history_gap", "actor": None, "payload": {}},
        )
    )

    assert summary["actions"]["count"] == 2
    assert summary["actions"]["plays"] == 1
    assert summary["actions"]["passes"] == 1
    assert summary["actions"]["rows"][0]["evidence"] == {"frame": 4}
    assert summary["lead"]["confirmations"][0]["lead_player"] == "opposite"
    assert summary["turn_order"]["contiguous_turn_ids"] is False
    assert summary["wind_catch_chain"][0]["actor"] == "left"
    assert summary["rankings"][0]["payload"]["placement"] == 3
    assert summary["visual_gaps"][0]["event_id"] == "gap"


def test_field_metrics_compare_ordered_actor_pass_and_card_multisets():
    truth = TruthLog(
        "game",
        TruthInitialState("2", "right", ("2S", "2S")),
        (
            TruthTurn(1, "right", False, ("7S",)),
            TruthTurn(2, "opposite", True, ()),
        ),
    )
    visual = summarize_visual_events(
        (
            {
                "event_id": "a",
                "event_type": "player_played",
                "turn_id": 1,
                "actor": "left",
                "payload": {"cards": ["8S"], "is_pass": False},
            },
            {
                "event_id": "b",
                "event_type": "player_played",
                "turn_id": 1,
                "actor": "opposite",
                "payload": {"cards": ["9S"], "is_pass": False},
            },
        )
    )

    metrics = compare_truth_visual_fields(
        truth,
        {
            "reads": [
                {"round_level": "2", "hand": ["2S", "2S"]},
            ]
        },
        visual,
    )

    assert metrics["level"]["accuracy"] == 1.0
    assert metrics["hand_multiset"]["exact"] is True
    assert metrics["actions"]["actor"]["errors"] == 1
    assert metrics["actions"]["pass"]["errors"] == 1
    assert metrics["actions"]["cards"]["errors"] == 2
    assert metrics["actions"]["order"]["duplicate_actual_turn_ids"] == [1]


def test_recommendation_scope_ignores_divergence_after_self_finishes():
    truth = TruthLog(
        "game",
        TruthInitialState("2", "self", ("2S",)),
        (
            TruthTurn(1, "self", False, ("2S",)),
            TruthTurn(2, "right", False, ("3S",)),
            TruthTurn(3, "opposite", True, ()),
        ),
    )
    visual = summarize_visual_events((
        {
            "event_id": "lead",
            "event_type": "lead_player_confirmed",
            "actor": "self",
            "payload": {"lead_player": "self"},
        },
        {
            "event_id": "self-finish",
            "event_type": "player_played",
            "turn_id": 1,
            "actor": "self",
            "payload": {"cards": ["2S"], "is_pass": False},
        },
        {
            "event_id": "post-self-wrong",
            "event_type": "player_played",
            "turn_id": 2,
            "actor": "right",
            "payload": {"cards": ["AS"], "is_pass": False},
        },
    ))
    opening = {
        "reads": [{
            "round_level": "2",
            "hand": ["2S"],
            "lead_player": "self",
            "candidate_ready": True,
        }]
    }

    metrics = compare_truth_visual_fields(truth, opening, visual)

    assert metrics["recommendation_scope"]["self_finish_turn_id"] == 1
    assert metrics["actions"]["changed"] == 0
    assert metrics["actions"]["missing"] == 0
    assert metrics["actions"]["added"] == 0
    assert metrics["recommendation_scope"]["post_self_expected_count"] == 2
    assert _first_divergence(truth, opening, visual) is None


def test_visual_advice_summary_ignores_timeout_after_self_finish_scope(tmp_path: Path):
    advice = tmp_path / "advice.jsonl"
    advice.write_text(
        "\n".join((
            json.dumps({"turn_id": 1, "status": "requested"}),
            json.dumps({"turn_id": 1, "status": "ready"}),
            json.dumps({"turn_id": 2, "status": "requested"}),
            json.dumps({"turn_id": 2, "status": "timeout"}),
        )) + "\n",
        encoding="utf-8",
    )
    result = _visual_result(
        tmp_path,
        advice_requested=2, advice_ready=1, advice_timeouts=1,
        artifact_paths={"advice.jsonl": advice},
    )

    summary = _visual_advice_summary(
        result, _AuditAdvisor(), listener_status="complete", scope_end_turn=1
    )

    assert summary["status"] == "passed"
    assert summary["advice"]["requested"] == 1
    assert summary["advice"]["ready"] == 1
    assert summary["advice"]["timeout"] == 0


def test_inventory_truth_metadata_is_presence_only_and_does_not_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    session = tmp_path / "profile" / "sessions" / "game"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text('{"session_id":"game"}', encoding="utf-8")
    truth = session / "truth_log.json"
    truth.write_text('{"secret":"must-not-be-read"}', encoding="utf-8")

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("TruthLog reference must be resolved after visual replay")

    monkeypatch.setattr(audit_module, "resolve_truth_audit_reference", fail_resolve)
    item = audit_module._Session(session, session.parent, "store", "game")
    inventory = audit_module._inventory((item,), (session.parent,), None)
    files = inventory["sessions"][0]["files"]

    assert inventory["sessions"][0]["truth_kind"] == "canonical"
    assert files["truth"]["exists"] is True
    assert files["truth"]["path"] == str(truth)
    assert set(files["truth"]) == {"path", "exists", "size", "mtime_ns"}
    assert "sha256" not in files["canonical_truth"]

    snapshot = audit_module._source_snapshot((session.parent,))
    truth_snapshot = next(
        row for row in snapshot.values() if row["relative_path"] == "game/truth_log.json"
    )
    assert "sha256" not in truth_snapshot


def test_visual_advice_summary_marks_stale_results_without_marking_failure(tmp_path: Path):
    result = _visual_result(
        tmp_path,
        advice_requested=26,
        advice_ready=21,
        advice_stale=5,
    )

    summary = _visual_advice_summary(result, _AuditAdvisor(), listener_status="complete")

    assert summary["status"] == "completed_with_stale"
    assert summary["advice"]["stale"] == 5


def _strict_fabledan_quality_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "execution_status": "completed",
        "frame_replay_status": "complete",
        "listener_status": "complete",
        "opening_status": "recognized",
        "truth_quality": "passed",
        "visual_quality": "passed",
        "comparison_status": "passed",
        "fabledan_quality": "advisory",
        "fabledan": {
            "truth_driven": {
                "available": True,
                "completed": True,
                "status": "passed",
                "advice": {
                    "requested": 78,
                    "ready": 78,
                    "failed": 0,
                    "timeout": 0,
                },
            },
            "visual_driven": {
                "available": True,
                "completed": True,
                "status": "completed_with_stale",
                "advice": {
                    "requested": 26,
                    "ready": 21,
                    "failed": 0,
                    "stale": 5,
                    "timeout": 0,
                },
            },
        },
    }
    row.update(overrides)
    return row


def test_row_quality_failures_allows_advisory_for_completed_truth_and_stale_visual():
    row = _strict_fabledan_quality_row()

    assert _row_quality_failures(row) == []
    assert row["fabledan"]["visual_driven"]["advice"]["stale"] == 5


@pytest.mark.parametrize(
    "change, expected_reason",
    [
        (lambda row: row.update(fabledan_quality="failed"), "fabledan_quality='failed'"),
        (
            lambda row: row["fabledan"]["truth_driven"].update(status="incomplete", completed=False),
            "fabledan_truth_channel_incomplete",
        ),
        (
            lambda row: row["fabledan"]["truth_driven"]["advice"].update(failed=1),
            "fabledan_truth_failed=1",
        ),
        (
            lambda row: row["fabledan"]["visual_driven"]["advice"].update(timeout=1),
            "fabledan_visual_timeout=1",
        ),
    ],
)
def test_row_quality_failures_blocks_failed_or_incomplete_fabledan(
    change, expected_reason: str
):
    row = _strict_fabledan_quality_row()
    change(row)

    failures = _row_quality_failures(row)

    assert expected_reason in failures

def test_all_session_audit_writes_only_under_explicit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    profiles = tmp_path / "profiles"
    session = profiles / "profile" / "sessions" / "game-one"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text('{"session_id":"game-one"}', encoding="utf-8")
    video = session / "video"
    video.mkdir()
    (video / "frame_index.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": index,
                    "monotonic_ms": index * 100,
                    "wall_time": f"t{index}",
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    (session / "timeline.jsonl").write_text(
        json.dumps(
            {
                "event_type": "initial_state_confirmed",
                "payload": {"round_level": "2", "hand": ["2S"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    truth_path = session / "truth_log.json"
    save_truth_log(truth_path, TruthLog("game-one", TruthInitialState("2", "self", ("2S",)), ()))
    source_truth = truth_path.read_bytes()
    visual_started = False
    real_path_open = Path.open
    real_sha_file = audit_module._sha_file
    real_load_truth = audit_module._load_truth

    def guarded_path_open(path, *args, **kwargs):
        if path.name == "truth_log.json" and not visual_started:
            raise AssertionError("TruthLog must not be opened before visual replay")
        return real_path_open(path, *args, **kwargs)

    def guarded_sha_file(path):
        if Path(path).name == "truth_log.json" and not visual_started:
            raise AssertionError("TruthLog hash must not be computed before visual replay")
        return real_sha_file(path)

    def guarded_load_truth(session_path, reference):
        if not visual_started:
            raise AssertionError("TruthLog must not be loaded before visual replay")
        return real_load_truth(session_path, reference)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(audit_module, "_sha_file", guarded_sha_file)
    monkeypatch.setattr(audit_module, "_load_truth", guarded_load_truth)

    def visual(_session, _recognition, *, output_root, **kwargs):
        nonlocal visual_started
        visual_started = True
        assert "truth_log" not in kwargs
        output = Path(output_root) / "visual.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "events": [
                        {
                            "event_id": "turn-1",
                            "event_type": "player_played",
                            "actor": "self",
                            "payload": {"turn_id": 1, "cards": ["2S"]},
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return VisualPipelineReplayResult(
            output,
            Path(output_root) / "comparison.json",
            3,
            (),
            ReplayComparison((), (), (), (), ()),
        )

    def advisor_replay(_session, _advisor, *, output_root, **_kwargs):
        run = Path(output_root) / "run"
        run.mkdir(parents=True)
        summary = run / "summary.json"
        summary.write_text('{"advice_statuses":{"ready":1}}', encoding="utf-8")
        return TrustedAdviceReplayResult(
            run / "events.jsonl",
            summary,
            run,
            "game-one",
            1,
            1,
            True,
            1,
            1,
            0,
            0,
            0,
            0,
        )

    class Advisor:
        write_decision_log = True

        @staticmethod
        def audit_info():
            return {"backend": "test"}

    run = SessionReplayAuditService(
        recognition_factory=lambda _session: object(),
        opening_probe=lambda *_args: {"two_frame_matches_stored": True},
        visual_replay=visual,
        advisor_factory=lambda _root, _name: Advisor(),
        advisor_replay=advisor_replay,
    ).audit([profiles / "profile" / "sessions"], output=tmp_path / "reports", run_id="all")

    assert run.summary_path == tmp_path / "reports" / "all" / "all_session_audit.json"
    assert run.sessions[0]["status"] == "completed"
    assert run.sessions[0]["fabledan"]["truth_driven"]["advice"]["ready"] == 1
    assert truth_path.read_bytes() == source_truth
    assert not (session / "visual").exists()
    assert run.execution_ok is True
    assert (run.run_directory / "inventory.json").is_file()
    assert (run.run_directory / "sessions.csv").is_file()
    assert (run.run_directory / "failures.csv").is_file()
    assert (run.run_directory / "summary.md").is_file()
    assert (run.run_directory / "source_snapshot_before.json").is_file()
    assert (run.run_directory / "source_snapshot_after.json").is_file()
    divergence = run.sessions[0]["first_divergence"]
    assert Path(divergence["expected"]).is_file()
    assert Path(divergence["actual"]).is_file()
    assert Path(divergence["context"]).is_file()
    assert Path(divergence["description"]).is_file()
    verification = json.loads((run.run_directory / "verification.json").read_text("utf-8"))
    assert verification["passed"] is True


def test_audit_rejects_output_below_a_sessions_root(tmp_path: Path):
    sessions = tmp_path / "profile" / "sessions"
    sessions.mkdir(parents=True)

    with pytest.raises(ValueError, match="outside every sessions root"):
        SessionReplayAuditService().audit(
            [sessions],
            output=sessions / "reports",
            run_id="forbidden",
        )


def test_two_roots_with_same_session_id_keep_separate_store_artifacts(tmp_path: Path):
    roots = []
    for store_name in ("source", "release"):
        root = tmp_path / store_name / "profile" / "sessions"
        session = root / "same"
        (session / "video").mkdir(parents=True)
        (session / "manifest.json").write_text('{"session_id":"same"}', encoding="utf-8")
        (session / "timeline.jsonl").write_text(
            json.dumps(
                {
                    "event_type": "initial_state_confirmed",
                    "actor": "self",
                    "payload": {"round_level": "2", "lead_player": "self", "hand": ["2S"]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (session / "video" / "frame_index.jsonl").write_text(
            json.dumps({"frame_index": 0, "monotonic_ms": 0, "wall_time": "t0"}) + "\n",
            encoding="utf-8",
        )
        roots.append(root)

    def visual(_session, _recognition, *, output_root, **_kwargs):
        path = Path(output_root) / "visual.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("", encoding="utf-8")
        return VisualPipelineReplayResult(
            path,
            Path(output_root) / "comparison.json",
            1,
            (),
            ReplayComparison((), (), (), (), ()),
            completed=True,
        )

    run = SessionReplayAuditService(
        recognition_factory=lambda _session: object(),
        opening_probe=lambda *_args: {"reads": [], "two_frame_matches_stored": False},
        visual_replay=visual,
        advisor_factory=lambda *_args: type("Advisor", (), {})(),
    ).audit(roots, output=tmp_path / "reports", run_id="both")

    assert len(run.sessions) == 2
    assert len({row["store_id"] for row in run.sessions}) == 2
    assert all(
        (
            run.run_directory
            / "sessions"
            / str(row["store_id"])
            / "same"
            / "summary.json"
        ).is_file()
        for row in run.sessions
    )


def test_cli_returns_nonzero_for_empty_discovery_but_writes_verification(tmp_path: Path):
    empty = tmp_path / "empty-sessions"
    empty.mkdir()
    output = tmp_path / "reports"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_all_session_replays.py",
            "--sessions-root",
            str(empty),
            "--output",
            str(output),
            "--run-id",
            "empty",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    verification = json.loads((output / "empty" / "verification.json").read_text("utf-8"))
    assert verification["checks"]["nonempty_session_set"] is False


def test_missing_action_evidence_falls_back_to_nearby_visual_frame_state(tmp_path: Path):
    root = tmp_path / "profiles" / "profile" / "sessions"
    session = _make_audit_session(root, "missing-action", frame_count=4)
    truth_path = session / "truth_log.json"
    save_truth_log(
        truth_path,
        TruthLog(
            "missing-action",
            TruthInitialState("2", "right", ("2S",)),
            (
                TruthTurn(
                    1,
                    "right",
                    False,
                    ("7S",),
                    frame_index=2,
                    monotonic_ms=None,
                ),
            ),
        ),
    )

    def visual(_session, _recognition, *, output_root, **_kwargs):
        output_root = Path(output_root)
        output_root.mkdir(parents=True)
        output = output_root / "visual.jsonl"
        rows = [
            {
                "frame_index": 0,
                "monotonic_ms": 0,
                "status": "running",
                "current_player": "right",
                "state_revision": 1,
                "events": [],
            },
            {
                "frame_index": 1,
                "monotonic_ms": 100,
                "status": "running",
                "current_player": "right",
                "state_revision": 2,
                "events": [
                    {
                        "event_id": "AUX-NEAREST",
                        "event_type": "turn_started",
                        "turn_id": 1,
                        "trick_id": 1,
                        "actor": "right",
                        "monotonic_ms": 100,
                        "state_revision_before": 1,
                        "state_revision_after": 2,
                        "payload": {},
                    }
                ],
            },
            {
                "frame_index": 2,
                "monotonic_ms": 200,
                "status": "running",
                "current_player": "right",
                "state_revision": 2,
                "events": [],
            },
            {
                "frame_index": 3,
                "monotonic_ms": 300,
                "status": "running",
                "current_player": "right",
                "state_revision": 3,
                "events": [],
            },
        ]
        output.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        runtime = output_root / "runtime"
        runtime.mkdir()
        (runtime / "timeline.jsonl").write_text(
            json.dumps(rows[1]["events"][0]) + "\n",
            encoding="utf-8",
        )
        (runtime / "decisions.jsonl").write_text(
            json.dumps(
                {
                    "decision_id": "decision-1",
                    "turn_id": 1,
                    "state_revision": 2,
                    "engine_input": {"history_size": 0},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return VisualPipelineReplayResult(
            output,
            output_root / "comparison.json",
            4,
            (),
            ReplayComparison((), (), (), (), ()),
            run_directory=runtime,
            completed=True,
        )

    run = SessionReplayAuditService(
        recognition_factory=lambda _session: object(),
        opening_probe=lambda *_args: {
            "reads": [
                {
                    "frame_index": 0,
                    "monotonic_ms": 0,
                    "round_level": "2",
                    "hand": ["2S"],
                }
            ]
        },
        visual_replay=visual,
        advisor_factory=lambda *_args: _AuditAdvisor(),
        advisor_replay=_completed_advisor_replay,
    ).audit([root], output=tmp_path / "reports", run_id="fallback")

    evidence = run.sessions[0]["first_divergence"]
    context = json.loads(Path(evidence["context"]).read_text("utf-8"))
    assert context["monotonic_ms"] == 200
    assert context["event_id"] == "AUX-NEAREST"
    assert context["state_revision_before"] == 2
    assert context["state_revision_after"] == 2
    assert context["reducer_state_diff"]["before_revision"] == 2
    assert context["reducer_state_diff"]["after_revision"] == 2
    assert context["fallback"]["event_id_source"] == "nearest_actual_event_in_visual_frame_log"
    assert context["reducer_state_diff"]["scope"] == "summary_only_not_full_reducer_state"
    assert context["engine_input"] == {"history_size": 0}
    assert Path(evidence["images"]["trigger"]).is_file()
    assert Path(evidence["images"]["level_roi"]).is_file()
    assert Path(evidence["images"]["hand_roi"]).is_file()
    assert Path(evidence["images"]["play_right_roi"]).is_file()


def test_session_error_is_isolated_and_other_session_still_completes(tmp_path: Path):
    root = tmp_path / "profiles" / "profile" / "sessions"
    _make_audit_session(root, "broken", frame_count=2)
    _make_audit_session(root, "good", frame_count=2)

    def visual(session, _recognition, *, output_root, **_kwargs):
        if Path(session).name == "broken":
            raise RuntimeError("recognizer crashed")
        output = Path(output_root) / "visual.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text("", encoding="utf-8")
        return VisualPipelineReplayResult(
            output,
            Path(output_root) / "comparison.json",
            2,
            (),
            ReplayComparison((), (), (), (), ()),
            completed=True,
        )

    run = SessionReplayAuditService(
        recognition_factory=lambda _session: object(),
        opening_probe=lambda *_args: {"reads": []},
        visual_replay=visual,
        advisor_factory=lambda *_args: _AuditAdvisor(),
    ).audit([root], output=tmp_path / "reports", run_id="isolated")

    by_session = {row["session_id"]: row for row in run.sessions}
    assert by_session["broken"]["execution_status"] == "error"
    assert "recognizer crashed" in by_session["broken"]["error"]
    assert by_session["good"]["execution_status"] == "completed"
    assert by_session["good"]["frames_processed"] == 2
    verification = json.loads(run.verification_path.read_text("utf-8"))
    assert run.execution_ok is False
    assert verification["checks"]["frames_complete"] is False
    assert verification["checks"]["no_tool_errors"] is False


def test_incomplete_frame_count_fails_execution_and_verification(tmp_path: Path):
    root = tmp_path / "profiles" / "profile" / "sessions"
    _make_audit_session(root, "short", frame_count=2)

    def visual(_session, _recognition, *, output_root, **_kwargs):
        output = Path(output_root) / "visual.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text("", encoding="utf-8")
        return VisualPipelineReplayResult(
            output,
            Path(output_root) / "comparison.json",
            1,
            (),
            ReplayComparison((), (), (), (), ()),
            completed=False,
        )

    run = SessionReplayAuditService(
        recognition_factory=lambda _session: object(),
        opening_probe=lambda *_args: {"reads": []},
        visual_replay=visual,
        advisor_factory=lambda *_args: _AuditAdvisor(),
    ).audit([root], output=tmp_path / "reports", run_id="short")

    verification = json.loads(run.verification_path.read_text("utf-8"))
    assert run.sessions[0]["execution_status"] == "incomplete"
    assert run.execution_ok is False
    assert verification["checks"]["frames_complete"] is False


def test_injected_source_snapshot_change_fails_verification_without_mutating_source(
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "profiles" / "profile" / "sessions"
    session = _make_audit_session(root, "unchanged-on-disk", frame_count=1)
    manifest_before = (session / "manifest.json").read_bytes()

    snapshots = iter(
        (
            {"root::manifest.json": {"size": 1, "mtime_ns": 1}},
            {"root::manifest.json": {"size": 1, "mtime_ns": 2}},
        )
    )
    monkeypatch.setattr(audit_module, "_source_snapshot", lambda _roots: next(snapshots))

    def visual(_session, _recognition, *, output_root, **_kwargs):
        output = Path(output_root) / "visual.jsonl"
        output.parent.mkdir(parents=True)
        output.write_text("", encoding="utf-8")
        return VisualPipelineReplayResult(
            output,
            Path(output_root) / "comparison.json",
            1,
            (),
            ReplayComparison((), (), (), (), ()),
            completed=True,
        )

    run = SessionReplayAuditService(
        recognition_factory=lambda _session: object(),
        opening_probe=lambda *_args: {"reads": []},
        visual_replay=visual,
        advisor_factory=lambda *_args: _AuditAdvisor(),
    ).audit([root], output=tmp_path / "reports", run_id="changed-snapshot")

    verification = json.loads(run.verification_path.read_text("utf-8"))
    summary = json.loads(run.summary_path.read_text("utf-8"))
    assert run.execution_ok is False
    assert verification["checks"]["source_unchanged"] is False
    assert summary["source_integrity"]["unchanged"] is False
    assert (session / "manifest.json").read_bytes() == manifest_before


class _AuditAdvisor:
    write_decision_log = True

    @staticmethod
    def audit_info():
        return {"backend": "test"}


def _completed_advisor_replay(_session, _advisor, *, output_root, **_kwargs):
    run = Path(output_root) / "run"
    run.mkdir(parents=True)
    summary = run / "summary.json"
    summary.write_text('{"advice_statuses":{}}', encoding="utf-8")
    return TrustedAdviceReplayResult(
        run / "advice.jsonl",
        summary,
        run,
        Path(_session).name,
        1,
        1,
        True,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def _make_audit_session(root: Path, session_id: str, *, frame_count: int) -> Path:
    session = root / session_id
    video = session / "video"
    video.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"session_id": session_id}),
        encoding="utf-8",
    )
    (session / "timeline.jsonl").write_text(
        json.dumps(
            {
                "event_type": "initial_state_confirmed",
                "actor": "right",
                "payload": {
                    "round_level": "2",
                    "lead_player": "right",
                    "hand": ["2S"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (video / "frame_index.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "frame_index": index,
                    "monotonic_ms": index * 100,
                    "wall_time": f"t{index}",
                }
            )
            + "\n"
            for index in range(frame_count)
        ),
        encoding="utf-8",
    )
    writer = cv2.VideoWriter(
        str(video / "game.avi"),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (64, 32),
    )
    for index in range(frame_count):
        writer.write(np.full((32, 64, 3), index * 20, np.uint8))
    writer.release()
    return session


def test_is_strict_row_accepts_verified_truth_kind():
    from daguandan_bridge.application.session_replay_audit import _is_strict_row

    for kind in ("canonical", "verified"):
        for qualification in ("verified_label", "verified", "trusted_for_run"):
            assert _is_strict_row({
                "truth_log": {"kind": kind},
                "truth_qualification": qualification,
            }) is True

    assert _is_strict_row({
        "truth_log": {"kind": "verified"},
        "truth_qualification": "reference_only",
    }) is False
