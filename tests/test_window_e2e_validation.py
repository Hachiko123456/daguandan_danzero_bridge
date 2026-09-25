from __future__ import annotations

import json
import inspect
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.application.simulated_game_window import (
    IndexedVideoPlayback,
    SimulatedGameWindow,
    SimulatedGameWindowConfig,
)
from daguandan_bridge.application.window_e2e_validation import (
    WindowE2EValidator,
    WindowE2EValidationConfig,
    build_resource_manifest,
    compare_pixels,
    perceptual_hash,
    perceptual_hash_distance,
    _rule_engine_probe,
)
from daguandan_bridge.application import window_e2e_validation as validation_module
from daguandan_bridge.capture_service import LiveCaptureInterrupted
from daguandan_bridge.models import ClientRect, TargetWindow
from daguandan_bridge.opening_gate import evaluate_opening_gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_window_e2e_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_window_e2e_validation_script",
        PROJECT_ROOT / "scripts" / "run_window_e2e_validation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _video_fixture(root: Path, *, count: int = 4) -> tuple[Path, Path]:
    video = root / "game.avi"
    index = root / "frame_index.jsonl"
    root.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (64, 32),
    )
    rows = []
    for position in range(count):
        writer.write(np.full((32, 64, 3), position * 50, np.uint8))
        rows.append(
            {
                "frame_index": position,
                "monotonic_ms": 1_000 + position * 100,
                "wall_time": f"t{position}",
                "dropped_before": 0,
            }
        )
    writer.release()
    index.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return video, index


def _sim_config(tmp_path: Path, *, count: int = 4) -> SimulatedGameWindowConfig:
    video, index = _video_fixture(tmp_path / "video", count=count)
    return SimulatedGameWindowConfig(
        video_path=video,
        frame_index_path=index,
        control_path=tmp_path / "control.json",
        status_path=tmp_path / "status.json",
        event_log_path=tmp_path / "events.jsonl",
    )


def test_indexed_playback_decodes_explicit_timestamp_order(tmp_path: Path):
    playback = IndexedVideoPlayback(_sim_config(tmp_path))

    decoded = [playback.decode(position) for position in range(playback.count)]
    playback.close()

    assert [record.frame_index for record, _frame in decoded] == [0, 1, 2, 3]
    assert [record.monotonic_ms for record, _frame in decoded] == [1000, 1100, 1200, 1300]
    assert [round(float(frame.mean()) / 50) for _record, frame in decoded] == [0, 1, 2, 3]


def test_simulated_window_constructs_paused_on_first_frame_offscreen(tmp_path: Path):
    _app()
    window = SimulatedGameWindow(_sim_config(tmp_path))

    assert window.windowTitle() == "大掼蛋（腾讯）"
    assert window._state == "paused"
    assert window._record.frame_index == 0
    assert window.viewport.pixmap() is not None

    window.seek(2)
    assert window._record.frame_index == 2
    assert window._state == "paused"
    window.reset()
    assert window._record.frame_index == 0
    window.playback.close()


def test_pixel_gate_uses_size_mae_and_perceptual_hash():
    image = np.zeros((64, 64, 3), np.uint8)
    image[:, 32:] = 255
    near = image.copy()
    near[0, 0] = (3, 3, 3)

    result = compare_pixels(image, near, mae_limit=1.0, hash_distance_limit=2)

    assert result["passed"] is True
    assert result["same_size"] is True
    assert result["mae"] > 0
    assert perceptual_hash_distance(perceptual_hash(image), perceptual_hash(near)) <= 2
    assert compare_pixels(
        image,
        np.zeros((32, 32, 3), np.uint8),
        mae_limit=1.0,
        hash_distance_limit=2,
    )["passed"] is False


def test_geometry_scenarios_use_live_controller_auto_recovery_not_manual_reopen():
    for method in (
        WindowE2EValidator._move_recovery,
        WindowE2EValidator._resize_recovery,
        WindowE2EValidator._minimize_recovery,
    ):
        source = inspect.getsource(method)
        assert "_controller_geometry_recovery" in source
        assert "open_live_source" not in source
        assert "_capture_is_interrupted" not in source

    controller_path = inspect.getsource(
        WindowE2EValidator._controller_geometry_recovery_visible
    )
    assert "_simulator_topmost_fixture" in inspect.getsource(WindowE2EValidator._controller_geometry_recovery)
    assert "LiveAssistantController" in controller_path
    assert "start_listening" in controller_path
    assert 'row.get("state") == "recovering"' in controller_path
    assert 'row.get("state") == "recovered"' in controller_path
    assert '"manual_source_reopen": False' in controller_path
    assert '"new_source_takeover"' in controller_path
    assert '"post_recovery_frame_observed"' in controller_path
    assert 'len(frames) > frame_count_at_recovered' in controller_path


def test_geometry_probe_cannot_accidentally_build_a_game_session():
    result = validation_module._RecoveryProbeRecognizer().recognize(np.zeros((720, 1280, 3), np.uint8))
    assert result.my_hand == ()
    assert not evaluate_opening_gate(result, anchor_score=1.0).ready


def test_capture_probe_preserves_native_interruption_code_and_evidence():
    class Source:
        def capture(self):
            raise LiveCaptureInterrupted("blocked by test console", code="CAPTURE-OCCLUDED", details={"blocking_hwnd": 123})
    result = validation_module._capture_probe(Source())
    assert result["captured"] is False
    assert result["error_code"] == "CAPTURE-OCCLUDED"
    assert "test console" in result["error"]
    assert result["details"] == {"blocking_hwnd": 123}


def test_native_visibility_reports_real_blockers_without_ignoring_them(monkeypatch):
    monkeypatch.setattr(validation_module, "get_client_rect_on_screen", lambda target: ClientRect(10, 20, 1280, 764))
    monkeypatch.setattr(validation_module, "find_screen_occluders", lambda target, rect: (TargetWindow(123, "external console"),))
    state = validation_module._simulator_visibility({"hwnd": 42, "window_title": "simulator"})
    assert state["visible_unoccluded"] is False
    assert state["blockers"] == [{"hwnd": 123, "title": "external console"}]


class _WindowFixtureAPI:
    def __init__(self, *, topmost=False, pid=456):
        self.topmost, self.pid, self.mutations = topmost, pid, []
    def IsWindow(self, hwnd): return hwnd == 42
    def GetWindowText(self, hwnd): return "owned simulator"
    def GetWindowThreadProcessId(self, hwnd): return (7, self.pid)
    def GetWindowLong(self, hwnd, index): return 8 if self.topmost else 0
    def SetWindowPos(self, hwnd, insert_after, x, y, width, height, flags):
        self.mutations.append((hwnd, insert_after, x, y, width, height, flags))
        self.topmost = insert_after == -1


@pytest.mark.parametrize("original_topmost", [False, True])
@pytest.mark.parametrize("body_fails", [False, True])
def test_owned_topmost_fixture_restores_original_state_even_when_scenario_fails(original_topmost, body_fails):
    api = _WindowFixtureAPI(topmost=original_topmost)
    fixture = validation_module._SimulatorTopmostFixture(
        {"hwnd": 42, "pid": 456, "window_title": "owned simulator"}, 42,
        window_api=api, process_api=api,
    )
    try:
        with fixture:
            assert api.topmost
            if body_fails:
                raise RuntimeError("scenario failure")
    except RuntimeError as exc:
        assert body_fails
        assert str(exc) == "scenario failure"
    assert api.topmost is original_topmost
    assert fixture.evidence()["original_topmost_restored"]
    assert api.mutations[-1][1] == (-1 if original_topmost else -2)
    assert all(row[0] == 42 and row[2:6] == (0, 0, 0, 0) and row[6] == 0x0013 for row in api.mutations)


def test_topmost_fixture_refuses_pid_mismatch_without_mutating_any_window():
    api = _WindowFixtureAPI(pid=999)
    with pytest.raises(RuntimeError, match="identity changed"):
        with validation_module._SimulatorTopmostFixture(
            {"hwnd": 42, "pid": 456, "window_title": "owned simulator"}, 42,
            window_api=api, process_api=api,
        ):
            pass
    assert api.mutations == []


def test_topmost_fixture_reports_restore_failure_and_preserves_primary_error():
    api = _WindowFixtureAPI()
    fixture = validation_module._SimulatorTopmostFixture(
        {"hwnd": 42, "pid": 456, "window_title": "owned simulator"}, 42,
        window_api=api, process_api=api,
    )
    with pytest.raises(validation_module._ScenarioEvidenceError) as captured:
        with fixture:
            api.pid = 999
            raise TimeoutError("original recovery timeout")
    assert "identity changed" in captured.value.summary_error
    assert "original recovery timeout" in captured.value.scenario_details["fixture_primary_error"]
    # A reused/foreign HWND is never touched while attempting restoration.
    assert len(api.mutations) == 1


def test_restore_precondition_only_restores_own_simulator_and_checks_z_order(monkeypatch):
    order = []
    validator = object.__new__(WindowE2EValidator)
    validator.control = SimpleNamespace(command=lambda command: order.append(command) or {"hwnd": 42})
    validator._assert_target = lambda status: order.append("assert_target")
    def native(status):
        order.append("native_visibility")
        return {"visible_unoccluded": True, "blockers": []}
    monkeypatch.setattr(validation_module, "_simulator_visibility", native)
    evidence = validator._restore_simulator_visibility()
    assert order == ["restore", "assert_target", "native_visibility"]
    assert evidence["native_visibility"]["visible_unoccluded"]


def test_restore_precondition_fails_closed_when_another_window_remains_on_top(monkeypatch):
    validator = object.__new__(WindowE2EValidator)
    validator.control = SimpleNamespace(command=lambda command: {"hwnd": 42})
    validator._assert_target = lambda status: None
    monkeypatch.setattr(validation_module, "_simulator_visibility", lambda status: {"visible_unoccluded": False, "blockers": [{"hwnd": 123, "title": "external"}]})
    with pytest.raises(validation_module._ScenarioEvidenceError) as captured:
        validator._restore_simulator_visibility()
    assert captured.value.scenario_details["environment_precondition"]["native_visibility"]["blockers"][0]["hwnd"] == 123


def test_scenario_failure_retains_controller_state_in_json(tmp_path):
    validator = object.__new__(WindowE2EValidator)
    validator.scenario_directory = tmp_path
    validator.scenario_results, validator.errors = {}, []
    details = {"status_transitions": [{"state": "opening"}], "controller_at_failure": {"orchestrator_present": False, "waiting_worker_running": True}}
    def fail():
        raise validation_module._ScenarioEvidenceError(TimeoutError("recovery status missing"), details)
    result = validator._record_scenario("minimize_recovery", fail)
    saved = json.loads((tmp_path / "minimize_recovery.json").read_text(encoding="utf-8"))
    assert not result["passed"]
    assert saved["controller_at_failure"]["orchestrator_present"] is False
    assert saved["status_transitions"] == [{"state": "opening"}]
    assert "TimeoutError" in saved["error"]


@pytest.mark.parametrize("interruption_code,expected_pass", [("CAPTURE-OCCLUDED", True), ("CAPTURE-BLACK-FRAME", False)])
def test_occlusion_rechecks_visibility_after_hiding_own_occluder(monkeypatch, interruption_code, expected_pass):
    state = {"occluded": False, "visible": False}
    commands = []
    validator = object.__new__(WindowE2EValidator)
    validator.config = SimpleNamespace(profile_name="test")
    validator._backend_profile = lambda backend: backend
    validator._pixel_evidence = lambda *args: {"passed": True}
    validator._assert_target = lambda status: None
    def command(name, arguments=None):
        commands.append(name)
        if name == "restore":
            state["visible"] = True
        if name == "occlude":
            state["occluded"] = arguments["visible"]
            if not state["occluded"]:
                # Model normal Windows activation of another window when
                # the owned occluder disappears. Reopen alone cannot fix it.
                state["visible"] = False
        return {"hwnd": 42}
    validator.control = SimpleNamespace(command=command)
    monkeypatch.setattr(validation_module, "_simulator_visibility", lambda status: {"visible_unoccluded": state["visible"], "blockers": []})
    class Source:
        def __init__(self, backend): self.backend = backend
        def capture(self):
            if self.backend != "printwindow" and state["occluded"]:
                raise LiveCaptureInterrupted("expected fixture blocker", code=interruption_code)
            if self.backend != "printwindow" and not state["visible"]:
                raise LiveCaptureInterrupted("external window activated", code="CAPTURE-OCCLUDED")
            return object()
        def close(self): pass
    monkeypatch.setattr(validation_module, "CaptureService", lambda backend: SimpleNamespace(open_live_source=lambda profile: Source(backend)))
    report = validator._occlusion_visible()
    assert report["passed"] is expected_pass
    assert commands == ["reset", "restore", "occlude", "occlude", "restore"]
    assert report["restored_visibility"]["native_visibility"]["visible_unoccluded"]
    for backend in ("screen", "gdi_screen"):
        assert report["backends"][backend]["occluded_capture"]["error_code"] == interruption_code
        assert report["backends"][backend]["reopened_source_captured"]


def test_resource_manifest_compares_every_template_and_both_models(tmp_path: Path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    checkpoint = tmp_path / "checkpoint.ckpt"
    for root in (reference, candidate):
        (root / "templates" / "rank").mkdir(parents=True)
        (root / "models" / "danzero").mkdir(parents=True)
        for name in ("profile.json", "regions_config.json", "templates_config.json"):
            (root / name).write_text(name, encoding="utf-8")
        (root / "templates" / "rank" / "3.png").write_bytes(b"template")
        (root / "models" / "best.npz").write_bytes(b"model")
    checkpoint.write_bytes(b"checkpoint")
    (candidate / "models" / "danzero" / "q_network.ckpt").write_bytes(b"checkpoint")

    matching = build_resource_manifest(
        reference,
        candidate,
        reference_checkpoint=checkpoint,
    )
    (candidate / "templates" / "rank" / "3.png").write_bytes(b"changed")
    different = build_resource_manifest(
        reference,
        candidate,
        reference_checkpoint=checkpoint,
    )

    assert matching["all_match"] is True
    assert {row["path"] for row in matching["files"]} == {
        "profile.json",
        "regions_config.json",
        "templates_config.json",
        "templates/rank/3.png",
        "models/best.npz",
        "models/danzero/q_network.ckpt",
    }
    assert different["all_match"] is False
    assert different["mismatches"] == ["templates/rank/3.png"]


def test_validation_config_rejects_report_below_source_sessions(tmp_path: Path):
    profile = _profile_fixture(tmp_path / "profile")
    session = _session_fixture(profile / "sessions" / "game")
    raw = {
        "session": str(session),
        "output": str(profile / "sessions" / "reports"),
        "profile_source": str(profile),
        "simulator_control_path": str(tmp_path / "control.json"),
        "simulator_status_path": str(tmp_path / "status.json"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="outside source sessions"):
        WindowE2EValidationConfig.from_path(path)


def test_acceptance_eligibility_requires_real_baseline_and_full_matrix(tmp_path: Path):
    profile = _profile_fixture(tmp_path / "profile")
    session = _session_fixture(profile / "sessions" / "game")
    baseline = tmp_path / "phase2.json"
    baseline.write_text("{}", encoding="utf-8")
    raw = {
        "session": str(session),
        "output": str(tmp_path / "report"),
        "profile_source": str(profile),
        "simulator_control_path": str(tmp_path / "control.json"),
        "simulator_status_path": str(tmp_path / "status.json"),
        "baseline_summary": str(baseline),
        "simulator_time_scale": 1.0,
        "max_frames": None,
        "scenarios": [
            "initial_capture",
            "move_recovery",
            "resize_recovery",
            "minimize_recovery",
            "occlusion",
            "dpi",
            "full_chain",
        ],
    }
    path = tmp_path / "eligible.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert WindowE2EValidationConfig.from_path(path).acceptance_eligible is True

    raw["scenarios"] = ["full_chain"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert WindowE2EValidationConfig.from_path(path).acceptance_eligible is False

    raw["baseline_summary"] = str(tmp_path / "missing.json")
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline summary does not exist"):
        WindowE2EValidationConfig.from_path(path)


def test_full_chain_business_health_fails_closed_on_bad_sealed_audit_and_incomplete_goal(tmp_path: Path):
    profile = _profile_fixture(tmp_path / "profile")
    session = _session_fixture(profile / "sessions" / "game")
    config = WindowE2EValidationConfig(
        session=session,
        output=tmp_path / "report",
        profile_source=profile,
        reference_profile=profile,
        simulator_control_path=tmp_path / "control.json",
        simulator_status_path=tmp_path / "status.json",
        scenarios=("full_chain",),
        full_chain_expected_action_count=60,
        full_chain_expected_remaining_cards={
            "self": 0,
            "right": 0,
            "opposite": 3,
            "left": 5,
        },
    )
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    health_path = runtime / "health_audit.json"
    health_path.write_text(
        json.dumps(
            {
                "schema": "guandan.session-health/1",
                "status": "FAIL",
                "issues": [{"code": "HEALTH-ACTION-CHAIN-INCONSISTENT"}],
            }
        ),
        encoding="utf-8",
    )
    actual_actions = [
        {"actor": "right", "is_pass": False, "cards": ["JS"], "turn_id": index}
        for index in range(55)
    ]

    business = validation_module._full_chain_business_health(
        config,
        runtime,
        actual_actions,
        {
            "remaining_cards": {
                "self": 1,
                "right": 0,
                "opposite": 3,
                "left": 5,
            }
        },
        {"available": False},
    )

    assert business["passed"] is False
    assert business["health_audit"]["status"] == "FAIL"
    assert business["actual_action_count"] == 55
    assert business["expected_action_count"] == 60
    assert business["action_count_ok"] is False
    assert business["remaining_cards_ok"] is False


def test_development_fragment_max_320_compares_truth_turns_1_through_11(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    actors = [
        "left", "self", "right", "opposite", "left", "self",
        "right", "opposite", "left", "self", "right", "opposite",
    ]
    frames = [83, 126, 163, 169, 175, 235, 254, 261, 267, 280, 293, 343]
    cards = [
        ["5C", "5H"], ["9C", "9H", "9H", "9S"], [], [], [],
        ["3C", "4H"], ["KD", "KS"], [], [], [], ["2H", "2S"], ["10H"],
    ]
    turns = [
        {
            "turn_id": index + 1,
            "actor": actor,
            "is_pass": not bool(cards[index]),
            "cards": cards[index],
            "evidence": {"frame_indices": [frames[index]]},
        }
        for index, actor in enumerate(actors)
    ]
    (source / "truth_log.json").write_text(json.dumps({
        "source_session_id": "fragment-source",
        "initial_state": {"my_hand": [f"C{index}" for index in range(27)]},
        "turns": turns,
    }), encoding="utf-8")
    actual = [
        {
            "actor": turn["actor"],
            "is_pass": turn["is_pass"],
            "cards": turn["cards"],
            "turn_id": turn["turn_id"],
        }
        for turn in turns[:11]
    ]
    remaining = {"self": 21, "right": 23, "opposite": 27, "left": 25}
    opportunities = {
        "available": True,
        "passed": True,
        "prefix_expected_count": 3,
        "prefix_actual_count": 3,
    }

    result = validation_module._development_fragment_acceptance(
        source, 320, actual, {"remaining_cards": remaining}, opportunities
    )

    assert result["development_fragment"] is True
    assert result["qualification_eligible"] is False
    assert result["passed"] is True
    assert result["prefix_expected_count"] == 11
    assert result["prefix_actual_count"] == 11
    assert result["expected_last_turn_id"] == 11
    assert result["first_divergence"] is None

    divergent = list(actual)
    divergent[6] = {**divergent[6], "cards": ["AD"]}
    failed = validation_module._development_fragment_acceptance(
        source, 320, divergent, {"remaining_cards": remaining}, opportunities
    )
    assert failed["passed"] is False
    assert failed["prefix_expected_count"] == 11
    assert failed["prefix_actual_count"] == 11
    assert failed["first_divergence"]["kind"] == "action"
    assert failed["first_divergence"]["position"] == 7


def test_fragment_execution_gate_and_source_entry_skip_formal_terminal_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    common_checks = {
        "simulator_eof": True,
        "frame_target_reached": True,
        "runtime_audit": True,
        "threads_and_drain": True,
        "rule_engine_probe": True,
        "opportunity_responses": True,
    }
    gate = validation_module._full_chain_execution_gate(
        development_fragment=True,
        common_checks=common_checks,
        fragment_prefix_passed=True,
        advisor_terminal=False,
        has_actions=True,
        business_health_passed=False,
        baseline_passed=False,
    )
    formal = validation_module._full_chain_execution_gate(
        development_fragment=False,
        common_checks=common_checks,
        fragment_prefix_passed=True,
        advisor_terminal=False,
        has_actions=True,
        business_health_passed=False,
        baseline_passed=False,
    )
    assert gate["passed"] is True
    assert "advisor_terminal" in gate["skipped_formal_checks"]
    assert "terminal_business_health" in gate["skipped_formal_checks"]
    assert formal["passed"] is False

    profile = _profile_fixture(tmp_path / "profile")
    session = _session_fixture(profile / "sessions" / "game")
    output = tmp_path / "source-report"
    config_path = tmp_path / "fragment.json"
    config_path.write_text(json.dumps({
        "session": str(session),
        "output": str(output),
        "profile_source": str(profile),
        "reference_profile": str(profile),
        "simulator_control_path": str(tmp_path / "control.json"),
        "simulator_status_path": str(tmp_path / "status.json"),
        "run_kind": "source",
        "scenarios": ["full_chain"],
        "max_frames": 320,
        "development_fragment": True,
    }), encoding="utf-8")
    monkeypatch.setattr(
        validation_module.SimulatorControlClient,
        "wait_ready",
        lambda self: {"hwnd": 123},
    )
    monkeypatch.setattr(
        validation_module.SimulatorControlClient,
        "status",
        lambda self: {"hwnd": 123},
    )
    monkeypatch.setattr(WindowE2EValidator, "_assert_target", lambda self, status: None)
    monkeypatch.setattr(WindowE2EValidator, "_stage_runtime_profile", lambda self: None)
    monkeypatch.setattr(
        validation_module,
        "build_resource_manifest",
        lambda *args, **kwargs: {"all_match": True, "mismatches": []},
    )
    monkeypatch.setattr(
        WindowE2EValidator,
        "_run_full_chain",
        lambda self: {
            "passed": bool(gate["passed"]),
            "business_health": {"passed": False},
            "fragment_prefix_acceptance": {
                "passed": True,
                "prefix_expected_count": 11,
                "prefix_actual_count": 11,
            },
            "execution_gate": gate,
        },
    )

    exit_code = validation_module.run_window_e2e_validation(config_path)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["execution_ok"] is True
    assert summary["acceptance_eligible"] is False
    assert summary["acceptance_passed"] is False
    assert summary["development_fragment"] is True
    assert summary["scenarios"]["full_chain"]["business_health"]["passed"] is False


def test_formal_full_chain_accepts_ready_and_local_pass_as_terminal_advice_mix():
    common_checks = {
        "simulator_eof": True,
        "frame_target_reached": True,
        "runtime_audit": True,
        "threads_and_drain": True,
        "rule_engine_probe": True,
        "opportunity_responses": True,
    }
    complete = {
        "requested": 23,
        "terminal_counts": {"ready": 14, "local_pass": 9},
    }
    missing_one = {
        "requested": 23,
        "terminal_counts": {"ready": 14, "local_pass": 8},
    }

    complete_terminal = validation_module._advice_lifecycle_terminal(complete)
    incomplete_terminal = validation_module._advice_lifecycle_terminal(missing_one)
    complete_gate = validation_module._full_chain_execution_gate(
        development_fragment=False,
        common_checks=common_checks,
        fragment_prefix_passed=False,
        advisor_terminal=complete_terminal,
        has_actions=True,
        business_health_passed=True,
        baseline_passed=True,
    )
    incomplete_gate = validation_module._full_chain_execution_gate(
        development_fragment=False,
        common_checks=common_checks,
        fragment_prefix_passed=False,
        advisor_terminal=incomplete_terminal,
        has_actions=True,
        business_health_passed=True,
        baseline_passed=True,
    )

    assert complete_terminal is True
    assert complete_gate["passed"] is True
    assert complete_gate["checks"]["advisor_terminal"] is True
    assert incomplete_terminal is False
    assert incomplete_gate["passed"] is False
    assert incomplete_gate["checks"]["advisor_terminal"] is False


def test_host_summary_rejects_full_chain_scenario_pass_without_business_health():
    script = _run_window_e2e_script_module()
    assert script._full_chain_business_ok(
        {"scenarios": {"full_chain": {"passed": True, "business_health": {"passed": False}}}}
    ) is False
    assert script._full_chain_business_ok(
        {"scenarios": {"full_chain": {"passed": True}}}
    ) is False


def test_host_full_chain_gate_requires_live_v2_and_truth_opportunity_success():
    script = _run_window_e2e_script_module()
    valid = {
        "acceptance_eligible": True,
        "scenarios": {"full_chain": {
            "passed": True,
            "business_health": {"passed": True},
            "runtime_audit": {"passed": True},
            "opportunity_acceptance": {
                "available": True,
                "passed": True,
                "denominators": {"all_opportunities": 2},
                "counts": {"valid_model_advice": 1, "valid_local_pass": 1},
                "latency_ms": {"successful_recommendations": {"p95": 900}},
            },
        }},
    }
    assert script._full_chain_acceptance(valid)["passed"] is True

    legacy = json.loads(json.dumps(valid))
    legacy["scenarios"]["full_chain"]["runtime_audit"]["passed"] = False
    assert script._full_chain_acceptance(legacy)["passed"] is False

    unavailable = json.loads(json.dumps(valid))
    unavailable["scenarios"]["full_chain"]["opportunity_acceptance"] = {
        "available": False, "passed": False,
    }
    audit = script._full_chain_acceptance(unavailable)
    assert audit["passed"] is False
    assert "opportunity_truth_unavailable" in audit["failures"]


def test_host_fragment_gate_skips_terminal_health_but_never_qualifies():
    script = _run_window_e2e_script_module()
    summary = {
        "development_fragment": True,
        "acceptance_eligible": False,
        "scenarios": {"full_chain": {
            "passed": True,
            "business_health": {"passed": False},
            "fragment_prefix_acceptance": {
                "passed": True,
                "prefix_expected_count": 11,
                "prefix_actual_count": 11,
                "first_divergence": None,
            },
            "runtime_audit": {"passed": True},
            "opportunity_acceptance": {"available": True, "passed": True},
        }},
    }

    audit = script._full_chain_acceptance(summary)

    assert audit["passed"] is True
    assert audit["development_fragment"] is True
    assert audit["qualification_eligible"] is False
    assert audit["prefix_expected_count"] == 11
    assert audit["prefix_actual_count"] == 11


def test_rule_engine_probe_confirms_a_legal_pair():
    probe = _rule_engine_probe()

    assert probe["passed"] is True
    assert probe["error"] is None
    assert any(row["play_type"] == "Pair" for row in probe["actions"])


def test_run_help_exposes_both_phase_three_routes():
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "run.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--simulated-game-window-config" in completed.stdout
    assert "--window-e2e-validation-config" in completed.stdout


def test_package_script_requires_unique_external_offline_roots_and_clean_build_env():
    script = (PROJECT_ROOT / "scripts" / "package_release.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string] $ReleaseRoot" in script
    assert "[string] $WheelhouseRoot" in script
    assert script.count("[Parameter(Mandatory = $true)]") >= 2
    assert "[switch] $AllowDirtyDevelopmentBuild" in script
    assert '"--allow-dirty"' in script
    assert "DEVELOPMENT_BUILD_NOT_FORMALLY_QUALIFIED.txt" in script
    assert "if (-not $AllowDirtyDevelopmentBuild)" in script
    assert "ReleaseRoot already exists. Use -OverwriteExisting" in script
    assert "Assert-DisjointRoots" in script
    assert 'Resolve-ManagedChildPath -Root $releaseRoot' in script
    assert ".daguandan-release-root" in script
    assert "guandan.package-release-root/2" in script
    assert "Assert-NoReparsePathChain" in script
    assert "Assert-NoReparseTree" in script
    assert "--porcelain=v1 --untracked-files=all" in script
    assert "Source tree is dirty" in script
    assert '& $Python -I -S @Arguments' in script
    assert '$bootstrapPython -I -S -c "import sys; print(sys.base_prefix)"' in script
    assert '"-m", "venv", $buildEnvPath' in script
    assert "audit_bootstrap_python.py" in script
    assert "bootstrap_python_audit.json" in script
    assert "--isolated" in script
    assert "--no-index" in script
    assert "--require-hashes" in script
    assert "--ignore-requires-python" in script
    assert "--find-links $wheelhouseRoot" in script
    assert "pip install --upgrade" not in script
    assert "Remove-Item -LiteralPath $path -Recurse" not in script
    assert "PYTHONPATH" in script
    assert "QT_PLUGIN_PATH" in script
    assert "QML2_IMPORT_PATH" in script
    assert "JAVA_HOME" in script
    assert "CONDA_PREFIX" in script
    assert "POPPLER_PATH" in script
    assert '"--collect-submodules", "rlcard"' in script
    assert '"--collect-data", "rlcard"' in script
    assert '"--noupx"' in script
    assert "audit_frozen_bundle.py" in script
    assert "native_dependency_audit.json" in script
    assert "Remove-Item -LiteralPath $file" not in script
    assert "generate_build_manifest.py" in script
    assert "build_manifest.json" in script
    assert 'Resolve-ManagedChildPath -Root $releaseRoot -Child "$archivePath.sha256"' in script
    assert "DaguandanAssistant.release.json" in script
    assert "Collect_Diagnostics.bat" in script

    launcher = (PROJECT_ROOT / "package_release.bat").read_text(encoding="utf-8")
    assert "Usage: package_release.bat [RELEASE_ROOT [WHEELHOUSE_ROOT]]" in launcher
    assert 'if /I "%~1"=="--help" goto :usage' in launcher
    assert 'if /I "%~1"=="-h" goto :usage' in launcher
    assert 'set "ReleaseRoot=%~dp0release\\current"' in launcher
    assert 'set "WheelhouseRoot=%LOCALAPPDATA%\\Daguandan\\wheelhouse"' in launcher
    assert 'set "OverwriteFlag=-OverwriteExisting"' in launcher
    assert 'set "DefaultWheelhouse=0"' in launcher
    assert 'set "NeedPrepareWheelhouse=0"' in launcher
    assert '[1/3] 检查 wheelhouse' in launcher
    assert '[2/3] 准备依赖' in launcher
    assert '[3/3] 构建发布包' in launcher
    assert 'scripts\\prepare_release_wheelhouse.ps1' in launcher
    assert 'if not exist "%WheelhouseRoot%\\.daguandan-wheelhouse-root"' in launcher
    assert 'if not exist "%WheelhouseRoot%\\wheelhouse.candidate.lock.json"' in launcher
    assert 'Explicit wheelhouse was provided; it will not be replaced or prepared by this launcher.' in launcher
    assert 'for %%I in ("%ReleaseRoot%") do set "ReleaseOutputDirectory=%%~fI"' in launcher
    assert 'echo Release output directory: "%ReleaseOutputDirectory%"' in launcher
    assert 'failed with exit code !StageExitCode!' in launcher
    assert 'EnableDelayedExpansion' in launcher
    assert '-ReleaseRoot "%ReleaseRoot%" -WheelhouseRoot "%WheelhouseRoot%"' in launcher

    help_completed = subprocess.run(
        [str(PROJECT_ROOT / "package_release.bat"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert help_completed.returncode == 0
    assert "Usage: package_release.bat [RELEASE_ROOT [WHEELHOUSE_ROOT]]" in help_completed.stdout
    assert r"%LOCALAPPDATA%\Daguandan\wheelhouse" in help_completed.stdout
    assert "-OverwriteExisting" in help_completed.stdout

    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/artifacts/" in gitignore


def _profile_fixture(path: Path) -> Path:
    (path / "templates").mkdir(parents=True)
    (path / "models").mkdir()
    (path / "profile.json").write_text("{}", encoding="utf-8")
    (path / "regions_config.json").write_text("{}", encoding="utf-8")
    (path / "templates_config.json").write_text("{}", encoding="utf-8")
    (path / "models" / "best.npz").write_bytes(b"model")
    return path


def _session_fixture(path: Path) -> Path:
    video, index = _video_fixture(path / "video", count=1)
    assert video.is_file() and index.is_file()
    (path / "timeline.jsonl").write_text("", encoding="utf-8")
    return path


@pytest.mark.skipif(sys.platform != "win32", reason="requires a real Win32 HWND")
@pytest.mark.windows_integration
def test_simulator_real_hwnd_short_integration_is_available_for_tester(tmp_path: Path):
    if os.environ.get("RUN_WINDOW_E2E_INTEGRATION") != "1":
        pytest.skip("set RUN_WINDOW_E2E_INTEGRATION=1 to open the visible HWND")
    import time

    from daguandan_bridge.storage import atomic_write_json
    from daguandan_bridge.window_capture import (
        capture_client_image_printwindow,
        find_target_window,
        get_client_rect_on_screen,
    )

    video, index = _video_fixture(tmp_path / "video", count=3)
    config = tmp_path / "simulator.json"
    control = tmp_path / "control.json"
    status = tmp_path / "status.json"
    config.write_text(
        json.dumps(
            {
                "video_path": str(video),
                "frame_index_path": str(index),
                "control_path": str(control),
                "status_path": str(status),
                "event_log_path": str(tmp_path / "events.jsonl"),
                "max_frames": 3,
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("QT_QPA_PLATFORM", None)
    process = subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "run.py"),
            "--simulated-game-window-config",
            str(config),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ready = _wait_json(
            status,
            lambda row: bool(row.get("ready")),
            timeout=20.0,
        )
        target = find_target_window(("大掼蛋（腾讯）",))
        rect = get_client_rect_on_screen(target)
        image = capture_client_image_printwindow(target, rect)
        assert target.hwnd == ready["hwnd"]
        assert (rect.width, rect.height) == (1280, 764)
        assert image.shape == (764, 1280, 3)

        atomic_write_json(
            control,
            {"request_id": "test-play", "command": "play", "arguments": {}},
        )
        eof = _wait_json(
            status,
            lambda row: row.get("state") == "eof",
            timeout=10.0,
        )
        assert eof["current_frame_index"] == 2
    finally:
        if process.poll() is None:
            atomic_write_json(
                control,
                {"request_id": "test-close", "command": "close", "arguments": {}},
            )
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5.0)


def _wait_json(path: Path, predicate, *, timeout: float) -> dict[str, object]:
    import time

    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                last = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                last = {}
        if predicate(last):
            return last
        time.sleep(0.05)
    raise TimeoutError(last)
