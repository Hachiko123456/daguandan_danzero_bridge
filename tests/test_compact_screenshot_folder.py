from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui import recommendation_window as module
from daguandan_bridge.gui.main_window import DaguandanBridgeWindow
from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow

pytestmark = pytest.mark.integration


class FolderRuntime(QObject):
    diagnostic_frame_status = Signal(object)

    def __init__(self):
        super().__init__()
        self.directory = None

    def diagnostic_frame_directory(self):
        return self.directory


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def compact(app):
    runtime = FolderRuntime()
    window = RecommendationFloatWindow(runtime)
    yield window, runtime
    window.hide()
    window.deleteLater()
    app.processEvents()


@pytest.fixture
def opened(monkeypatch):
    paths = []

    def open_url(url):
        assert url.isLocalFile()
        paths.append(Path(url.toLocalFile()))
        return True

    monkeypatch.setattr(module.QDesktopServices, "openUrl", open_url)
    return paths


def saved_frame(root, name="session"):
    session = root / name
    directory = session / "diagnostic_frames"
    directory.mkdir(parents=True)
    image = directory / "000001.png"
    image.write_bytes(b"saved frame")
    return {
        "status": "SUCCESS",
        "image_path": str(image),
        "session_directory": str(session),
        "source": "live_listener_frame",
        "source_phase": "live_session",
        "count": 1,
    }


def click_folder(window):
    QTest.mouseClick(window.screenshot_folder_button, Qt.MouseButton.LeftButton)


def test_folder_opens_actual_diagnostic_frames_without_exiting_compact(compact, opened, tmp_path, app):
    window, runtime = compact
    result = saved_frame(tmp_path, "中文 对局 #1")
    runtime.directory = Path(result["image_path"]).parent
    navigations = []
    for name in (
        "open_full_assistant_requested", "open_diagnostic_requested",
        "capture_diagnostic_requested", "stop_listening_requested",
    ):
        getattr(window, name).connect(lambda name=name: navigations.append(name))
    window.show()
    app.processEvents()

    click_folder(window)

    assert opened == [runtime.directory]
    assert opened[0].name == "diagnostic_frames"
    assert navigations == []
    assert window.isVisible()
    assert "已打开截图目录" in window.capture_label.text()
    assert str(runtime.directory) not in window.capture_label.text()


@pytest.mark.parametrize("bad_path", [None, "", ".", "relative/session/diagnostic_frames", "https://example.com/diagnostic_frames"])
def test_no_frame_or_invalid_runtime_path_is_inline_and_does_not_create_paths(compact, opened, tmp_path, bad_path):
    window, runtime = compact
    runtime.directory = bad_path

    click_folder(window)

    assert opened == []
    assert "尚无截图" in window.capture_label.text()
    assert list(tmp_path.iterdir()) == []


def test_runtime_session_root_is_not_mistaken_for_frame_directory(compact, opened, tmp_path):
    window, runtime = compact
    runtime.directory = tmp_path

    click_folder(window)

    assert opened == []
    assert "尚无截图" in window.capture_label.text()


def test_runtime_image_path_is_not_opened_as_a_folder(compact, opened, tmp_path):
    window, runtime = compact
    result = saved_frame(tmp_path)
    runtime.directory = Path(result["image_path"])

    click_folder(window)

    assert opened == []
    assert "尚无截图" in window.capture_label.text()


def test_latest_runtime_directory_is_queried_on_each_click(compact, opened, tmp_path):
    window, runtime = compact
    first = saved_frame(tmp_path, "old-session")
    second = saved_frame(tmp_path, "new-session")
    runtime.diagnostic_frame_status.emit(first)
    runtime.directory = Path(first["image_path"]).parent
    click_folder(window)
    runtime.directory = Path(second["image_path"]).parent
    click_folder(window)

    assert opened == [Path(first["image_path"]).parent, Path(second["image_path"]).parent]


def test_successful_status_updates_cached_directory_when_runtime_has_no_method(compact, opened, tmp_path):
    window, runtime = compact
    runtime.diagnostic_frame_directory = None
    first = saved_frame(tmp_path, "old-session")
    second = saved_frame(tmp_path, "new-session")
    runtime.diagnostic_frame_status.emit(first)
    click_folder(window)
    runtime.diagnostic_frame_status.emit(second)
    click_folder(window)

    assert opened == [Path(first["image_path"]).parent, Path(second["image_path"]).parent]


@pytest.mark.parametrize("path_field", ["image_path", "session_directory"])
def test_successful_status_can_supply_either_canonical_path(compact, opened, tmp_path, path_field):
    window, runtime = compact
    result = saved_frame(tmp_path)
    runtime.diagnostic_frame_status.emit({"status": "SUCCESS", path_field: result[path_field]})

    click_folder(window)

    assert opened == [Path(result["image_path"]).parent]


def test_saved_image_parent_wins_over_wrong_session_path(compact, opened, tmp_path):
    window, runtime = compact
    result = saved_frame(tmp_path, "correct-session")
    wrong = saved_frame(tmp_path, "wrong-session")
    result["session_directory"] = wrong["session_directory"]
    runtime.diagnostic_frame_status.emit(result)

    click_folder(window)

    assert opened == [Path(result["image_path"]).parent]


@pytest.mark.parametrize("status", ["SAVING", "PENDING", "FAILURE", "FAIL", "ERROR", "FAILED", "MIGRATION_SKIPPED"])
def test_non_success_status_never_replaces_last_saved_directory(compact, opened, tmp_path, status):
    window, runtime = compact
    saved = saved_frame(tmp_path, "saved-session")
    other = saved_frame(tmp_path, "other-session")
    runtime.diagnostic_frame_status.emit(saved)
    runtime.diagnostic_frame_status.emit({**other, "status": status})

    click_folder(window)

    assert opened == [Path(saved["image_path"]).parent]


@pytest.mark.parametrize("action", ["copy_compact_issue", "copy_compact_summary"])
def test_clipboard_feedback_does_not_replace_folder_or_finish_save(compact, opened, tmp_path, action):
    window, runtime = compact
    saved = saved_frame(tmp_path)
    runtime.diagnostic_frame_status.emit(saved)
    runtime.diagnostic_frame_status.emit({"status": "SAVING"})
    harness = SimpleNamespace(
        recommendation_window=window,
        _compact_copy_text=lambda **_kwargs: "诊断摘要",
    )

    getattr(DaguandanBridgeWindow, action)(harness)
    assert "复制" in window.capture_label.text()
    assert window._diagnostic_frame_saving
    assert not window.capture_button.isEnabled()
    click_folder(window)

    assert opened == [Path(saved["image_path"]).parent]


def test_copy_status_with_incidental_path_still_cannot_replace_folder(compact, opened, tmp_path):
    window, runtime = compact
    saved = saved_frame(tmp_path, "saved-session")
    other = saved_frame(tmp_path, "other-session")
    runtime.diagnostic_frame_status.emit(saved)
    runtime.diagnostic_frame_status.emit({**other, "message": "诊断摘要已复制到剪贴板"})

    click_folder(window)

    assert opened == [Path(saved["image_path"]).parent]


@pytest.mark.parametrize("remove_session", [False, True])
def test_disappeared_directory_falls_back_to_nearest_existing_session_parent(compact, opened, tmp_path, remove_session):
    window, runtime = compact
    result = saved_frame(tmp_path / "sessions")
    image = Path(result["image_path"])
    runtime.diagnostic_frame_status.emit(result)
    image.unlink()
    image.parent.rmdir()
    session = image.parent.parent
    if remove_session:
        session.rmdir()

    click_folder(window)

    assert opened == [session.parent if remove_session else session]
    assert "上级目录" in window.capture_label.text()
    assert not image.parent.exists()
    if remove_session:
        assert not session.exists()


def test_missing_directory_and_session_parents_do_not_open_unrelated_root(compact, opened, tmp_path):
    window, runtime = compact
    missing = tmp_path / "removed-sessions" / "session" / "diagnostic_frames"
    runtime.directory = missing

    click_folder(window)

    assert opened == []
    assert "已不存在" in window.capture_label.text()
    assert not missing.parent.parent.exists()


def test_existing_cached_frames_win_over_missing_runtime_directory(compact, opened, tmp_path):
    window, runtime = compact
    result = saved_frame(tmp_path)
    runtime.diagnostic_frame_status.emit(result)
    runtime.directory = tmp_path / "not-saved" / "diagnostic_frames"

    click_folder(window)

    assert opened == [Path(result["image_path"]).parent]


def test_runtime_directory_error_uses_saved_path(compact, opened, tmp_path):
    window, runtime = compact
    result = saved_frame(tmp_path)
    runtime.diagnostic_frame_status.emit(result)

    def unavailable():
        raise OSError("session disappeared")

    runtime.diagnostic_frame_directory = unavailable
    click_folder(window)
    assert opened == [Path(result["image_path"]).parent]


@pytest.mark.parametrize("raises", [False, True])
def test_explorer_failure_is_inline_without_navigation(compact, tmp_path, monkeypatch, raises):
    window, runtime = compact
    result = saved_frame(tmp_path)
    runtime.diagnostic_frame_status.emit(result)

    def unavailable(_url):
        if raises:
            raise OSError("no explorer")
        return False

    monkeypatch.setattr(module.QDesktopServices, "openUrl", unavailable)
    click_folder(window)
    assert "无法打开截图目录" in window.capture_label.text()
    assert window.capture_button.isEnabled()


@pytest.mark.parametrize("source,label", [
    ("live_listener_frame", "实时监听截图"),
    ("manual_window_capture", "手动窗口截图"),
])
def test_saving_and_success_labels_distinguish_capture_source(compact, tmp_path, source, label):
    window, runtime = compact
    result = {**saved_frame(tmp_path), "source": source}
    runtime.diagnostic_frame_status.emit({**result, "status": "SAVING"})
    assert label in window.capture_label.text()
    assert not window.capture_button.isEnabled()
    runtime.diagnostic_frame_status.emit(result)
    assert f"{label}已保存" in window.capture_label.text()
    assert window.capture_button.isEnabled()
    assert str(tmp_path) not in window.capture_label.text()
    assert result["image_path"] in window.capture_label.toolTip()


@pytest.mark.parametrize("phase,label", [
    ("listener_stopped", "停止前监听帧"),
    ("listener_failed", "失败前监听帧"),
    ("waiting_capture_failed", "失败前监听帧"),
    ("geometry_recovery_failed", "失败前监听帧"),
])
def test_retained_listener_frame_label_reports_capture_time_and_age(compact, tmp_path, phase, label):
    window, runtime = compact
    result = {
        **saved_frame(tmp_path),
        "source_phase": phase,
        "captured_at": "2026-09-25T14:05:06+08:00",
        "frame_age_seconds": 12.5,
    }
    runtime.diagnostic_frame_status.emit({**result, "status": "SAVING"})
    assert label in window.capture_label.text()
    runtime.diagnostic_frame_status.emit(result)

    text = window.capture_label.text()
    assert label in text
    assert "采集 14:05:06" in text
    assert "12 秒前" in text
    assert "实时监听截图" not in text
    assert "非当前画面" in window.capture_label.toolTip()


def test_retained_time_can_be_read_from_nested_store_metadata(compact):
    window, runtime = compact
    captured = datetime.now().astimezone() - timedelta(minutes=5)
    runtime.diagnostic_frame_status.emit({
        "status": "SUCCESS",
        "metadata": {
            "source": "live_listener_frame",
            "source_phase": "listener_stopped",
            "captured_at": captured.isoformat(),
        },
    })

    assert "停止前监听帧" in window.capture_label.text()
    assert f"采集 {captured:%H:%M:%S}" in window.capture_label.text()
    assert "5 分钟前" in window.capture_label.text()


def test_retained_frame_with_missing_time_does_not_invent_freshness(compact):
    window, runtime = compact
    runtime.diagnostic_frame_status.emit({
        "status": "SUCCESS", "source": "live_listener_frame",
        "source_phase": "listener_stopped", "captured_at": "bad timestamp",
        "frame_age_seconds": float("nan"),
    })

    assert "停止前监听帧" in window.capture_label.text()
    assert "采集时间未知" in window.capture_label.text()
    assert "秒前" not in window.capture_label.text()


def test_manual_capture_from_main_window_does_not_open_full_assistant(compact, tmp_path):
    window, runtime = compact
    result = {**saved_frame(tmp_path), "source": "manual_window_capture"}
    runtime.save_latest_live_frame_to_session = lambda: result
    harness = SimpleNamespace(live_runtime=runtime, recommendation_window=window)
    navigations = []
    window.open_full_assistant_requested.connect(lambda: navigations.append("full"))
    window.open_diagnostic_requested.connect(lambda: navigations.append("diagnostics"))

    DaguandanBridgeWindow.capture_diagnostic_from_compact(harness)

    assert "手动窗口截图已保存" in window.capture_label.text()
    assert navigations == []
    assert window.capture_button.isEnabled()


def test_production_failed_listener_frame_phase_has_capture_time_and_age(compact):
    window, _runtime = compact
    window.apply_diagnostic_frame_status({
        "status": "SUCCESS", "source": "live_listener_frame",
        "source_phase": "failed_listener_frame", "captured_at": "2026-09-25T23:02:40+08:00",
        "frame_age_ms": 2300,
    })
    assert "失败前监听帧" in window.capture_label.text()
    assert "23:02:40" in window.capture_label.text()
    assert "2 秒前" in window.capture_label.text()
