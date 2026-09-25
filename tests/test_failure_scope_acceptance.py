from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_failure_scope_acceptance.py"
SESSIONS_ROOT = REPO_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions"
PROFILE_ROOT = REPO_ROOT / "data" / "profiles" / "tencent_daguandan"


def _module():
    spec = importlib.util.spec_from_file_location("run_failure_scope_acceptance", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # dataclass-heavy production imports and direct script loading both expect
    # the module to be visible while it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _hash_tree(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_cli_help_exposes_replay_screenshot_fault_and_report_contract():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "TruthLog" in completed.stdout
    assert "--sessions-root" in completed.stdout
    assert "--diagnostic-frames" in completed.stdout
    assert "--output" in completed.stdout
    assert "--seed" in completed.stdout
    assert "--skip-replay" in completed.stdout


def test_selection_is_stable_and_prioritizes_truth_initial_state_and_replayability():
    module = _module()
    first = module.select_replay_sessions(SESSIONS_ROOT, count=5, seed="acceptance-test-seed")
    second = module.select_replay_sessions(SESSIONS_ROOT, count=5, seed="acceptance-test-seed")

    assert [item.session_id for item in first] == [item.session_id for item in second]
    assert len(first) == 5
    assert len({item.session_id for item in first}) == 5
    assert all(item.truth_log and item.has_video and item.has_frame_index for item in first)
    assert all(item.truth_verified for item in first)
    assert all(item.initial_state_confirmed for item in first)
    assert all(item.path.parent == SESSIONS_ROOT for item in first)
    assert all(item.session_id != "manual_diagnostic" for item in first)


def test_current_real_frames_are_exactly_ready_waiting_first_action():
    module = _module()
    report = module.diagnose_current_frames(PROFILE_ROOT)

    assert report["status"] == "pass"
    assert report["checks"] == {
        "page_table": True,
        "hand_count_27": True,
        "level_10": True,
        "wild_rank_10": True,
        "lead_self": True,
        "current_self": True,
        "ready_waiting_first_action": True,
        "no_synthetic_opening_action": True,
        "roi_overlap_warning": True,
        "roi_not_opening_blocking": True,
    }
    assert [frame["recognition"]["hand_count"] for frame in report["frames"]] == [27, 27]
    assert [frame["opening_gate"]["reason"] for frame in report["frames"]] == [
        "ready_waiting_first_action",
        "ready_waiting_first_action",
    ]
    assert report["roi_overlap"]["opening_blocking"] is False
    assert report["roi_overlap"]["action_blocking"] is True
    assert any(
        issue["severity"] == "warning"
        for issue in report["roi_overlap"]["issues"]
    )


def test_fault_matrix_covers_local_failures_without_fabricated_actions():
    module = _module()
    diagnosis = module.diagnose_current_frames(PROFILE_ROOT)
    hand = diagnosis["frames"][0]["recognition"]["my_hand"]
    matrix = module.run_fault_injection_matrix(
        hand,
        roi_validation=diagnosis["frames"][0]["roi_validation"],
    )

    expected = {
        "roi_overlap",
        "pass_missing",
        "timer_missing",
        "button_missing",
        "single_frame_anchor_failure",
        "consecutive_page_unknown",
        "single_seat_action_uncertain",
        "duplicate_frame",
        "out_of_order_frame",
    }
    assert matrix["status"] == "pass"
    assert {row["scenario_id"] for row in matrix["scenarios"]} == expected
    for row in matrix["scenarios"]:
        assert row["status"] == "pass"
        assert row["observed"].get("actions", []) == []
        if row["scenario_id"] != "roi_overlap":
            assert row["observed"]["synthetic_actions"] == 0
            assert row["observed"]["session_reset"] is False
            assert row["observed"]["global_blocked"] is False
    controls = {
        row["scenario_id"]: row
        for row in matrix["scenarios"]
        if row["scenario_id"] in {"pass_missing", "timer_missing", "button_missing"}
    }
    assert controls["pass_missing"]["observed"]["injected_controls"]["pass"] is False
    assert controls["timer_missing"]["observed"]["injected_controls"]["timer"] is False
    assert controls["button_missing"]["observed"]["injected_controls"]["button"] is False
    assert controls["pass_missing"]["observed"]["blocked_seats"] == ["self"]
    uncertain = next(row for row in matrix["scenarios"] if row["scenario_id"] == "single_seat_action_uncertain")
    assert uncertain["observed"]["blocked_seats"] == ["right"]
    duplicate = next(row for row in matrix["scenarios"] if row["scenario_id"] == "duplicate_frame")
    out_of_order = next(row for row in matrix["scenarios"] if row["scenario_id"] == "out_of_order_frame")
    assert "duplicate_frame" in duplicate["observed"]["tracker_reasons"]
    assert out_of_order["observed"]["recovered_to_waiting_first_action"] is True


@pytest.mark.integration
def test_json_report_is_external_and_source_sessions_remain_unchanged(tmp_path):
    module = _module()
    selected = module.select_replay_sessions(SESSIONS_ROOT, count=5)
    before = {
        item.session_id: _hash_tree(item.path)
        for item in selected
    }
    report_path = tmp_path / "failure-scope-report.json"

    report = module.run_acceptance(
        sessions_root=SESSIONS_ROOT,
        profile_root=PROFILE_ROOT,
        output=report_path,
        replay=False,
    )

    assert report["acceptance_status"] == "pass"
    assert report_path.is_file()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["schema"] == "guandan.failure-scope-acceptance/v1"
    assert loaded["report_path"] == str(report_path.resolve())
    assert loaded["source_integrity"]["selected_sessions_unchanged"] is True
    assert loaded["replay"]["mode"] == "skipped"
    assert loaded["current_diagnostic_frames"]["status"] == "pass"
    assert loaded["fault_injection"]["status"] == "pass"
    after = {
        item.session_id: _hash_tree(item.path)
        for item in selected
    }
    assert after == before
