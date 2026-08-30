from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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
    WindowE2EValidationConfig,
    build_resource_manifest,
    compare_pixels,
    perceptual_hash,
    perceptual_hash_distance,
    _rule_engine_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_package_script_keeps_default_and_accepts_explicit_release_root():
    script = (PROJECT_ROOT / "scripts" / "package_release.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string] $ReleaseRoot" in script
    assert "[switch] $AllowDirtyDevelopmentBuild" in script
    assert '$releaseRoot = Join-Path $projectRoot "artifacts\\release"' in script
    assert 'Resolve-ManagedChildPath -Root $releaseRoot' in script
    assert ".daguandan-release-root" in script
    assert "guandan.package-release-root/1" in script
    assert "Existing non-empty ReleaseRoot is not owned" in script
    assert "Assert-NoReparsePathChain" in script
    assert "Assert-NoReparseTree" in script
    assert "--porcelain=v1 --untracked-files=all" in script
    assert "Source tree is dirty" in script
    assert script.index("$sourceTreeDirty") < script.index(
        "[System.IO.Directory]::CreateDirectory($releaseRoot)"
    )
    assert script.index("Assert-NoReparseTree -LiteralPath $path") < script.index(
        "Remove-Item -LiteralPath $path -Recurse -Force"
    )
    assert '"--collect-submodules", "rlcard"' in script
    assert '"--collect-data", "rlcard"' in script
    assert '"--noupx"' in script
    assert '"_internal\\icuuc.dll"' in script
    assert '"_internal\\icudt78.dll"' in script
    assert "generate_build_manifest.py" in script
    assert "build_manifest.json" in script
    assert 'Resolve-ManagedChildPath -Root $releaseRoot -Child "$archivePath.sha256"' in script
    assert "DaguandanAssistant.release.json" in script
    assert "Collect_Diagnostics.bat" in script

    launcher = (PROJECT_ROOT / "package_release.bat").read_text(encoding="utf-8")
    assert "artifacts\\release\\dist\\DaguandanAssistant" in launcher
    assert "artifacts\\release\\DaguandanAssistant.zip" in launcher

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
