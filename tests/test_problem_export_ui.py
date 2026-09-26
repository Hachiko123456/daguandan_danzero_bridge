from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from zipfile import ZipFile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from qfluentwidgets import ToolButton

from daguandan_bridge.domain.live_runtime import LiveUpdate
from daguandan_bridge.gui import main_window as main_module
from daguandan_bridge.gui import recommendation_window as module
from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
from daguandan_bridge.gui.window_debug_page import WindowDebugPage

pytestmark = pytest.mark.integration


class ExportRuntime(QObject):
    problem_export_status = Signal(object)
    update_ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.requests = []
        self.accept = True
        self.error = None
        self.synchronous_result = None
        self.stop_calls = 0

    def request_problem_export(self, *, include_images=True, case_directory=None):
        self.requests.append((include_images, case_directory))
        if self.error:
            raise self.error
        if self.accept:
            self.problem_export_status.emit({"status": "RUNNING"})
        if self.synchronous_result is not None:
            self.problem_export_status.emit(self.synchronous_result)
        return self.accept

    def stop_listening(self):
        self.stop_calls += 1


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def surfaces(app):
    runtime = ExportRuntime()
    window = RecommendationFloatWindow(runtime)
    page = WindowDebugPage(report_service=SimpleNamespace(), problem_export_ui=window.problem_export_ui)
    window.resize(420, 235)
    window.show()
    app.processEvents()
    yield window, page, runtime
    ui = window.problem_export_ui
    for dialog in (ui.confirmation_dialog, ui.result_dialog):
        if dialog is not None:
            dialog.close()
    page.hide()
    window.hide()
    page.deleteLater()
    window.deleteLater()
    app.processEvents()


def click(button):
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def start(window, include_images=True):
    click(window.problem_export_button)
    dialog = window.problem_export_ui.confirmation_dialog
    assert dialog is not None
    click(dialog.images_button if include_images else dialog.logs_button)
    return window.problem_export_ui


def archive_at(root):
    path = root / "问题包 #1 中文.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
    return path


def test_confirmation_privacy_default_and_cancel_without_leaving_compact(surfaces, app):
    window, page, runtime = surfaces
    before = window.size()
    click(window.problem_export_button)
    dialog = window.problem_export_ui.confirmation_dialog
    assert "昵称" in dialog.privacy_label.text()
    assert "头像" in dialog.privacy_label.text()
    assert "不会自动上传" in dialog.privacy_label.text()
    assert dialog.images_button.isDefault()
    assert "推荐" in dialog.images_button.text()
    assert not window.problem_export_button.isEnabled()
    assert not page.problem_export_button.isEnabled()
    click(dialog.cancel_button)
    app.processEvents()
    assert runtime.requests == []
    assert runtime.stop_calls == 0
    assert window.problem_export_button.isEnabled()
    assert page.problem_export_button.isEnabled()
    assert window.problem_export_ui.result_dialog is None
    assert window.isVisible()
    assert window.size() == before


@pytest.mark.parametrize("include_images", [True, False])
def test_shared_guard_and_nonblocking_updates_keep_420px_compact(surfaces, include_images, app):
    window, page, runtime = surfaces
    size = window.size()
    suggestion = window.suggestion_label.text()
    navigations = []
    for name in ("open_full_assistant_requested", "open_diagnostic_requested", "stop_listening_requested"):
        getattr(window, name).connect(lambda name=name: navigations.append(name))
    click(window.problem_export_button)
    confirmation = window.problem_export_ui.confirmation_dialog
    click(window.problem_export_button)
    page.request_problem_export()
    assert window.problem_export_ui.confirmation_dialog is confirmation
    click(confirmation.images_button if include_images else confirmation.logs_button)
    click(window.problem_export_button)
    page.request_problem_export()
    assert runtime.requests == [(include_images, None)]
    assert window.suggestion_label.text() == suggestion
    assert "后台" in window.problem_export_button.toolTip()
    assert not window.problem_export_button.isEnabled()
    assert not page.problem_export_button.isEnabled()
    assert window.capture_button.isEnabled()
    assert window.stop_button.isEnabled()
    runtime.update_ready.emit(LiveUpdate(
        status="running", snapshot=SimpleNamespace(current_player="right", finished_seats=frozenset()),
        advice=None,
    ))
    app.processEvents()
    assert window.suggestion_label.text() == "等待自己回合"
    assert window.isVisible()
    assert window.size() == size
    assert window.minimumWidth() == 420
    assert len(window.findChildren(ToolButton)) == 7
    assert navigations == []
    assert runtime.stop_calls == 0


@pytest.mark.parametrize("status", ["SUCCESS", "PARTIAL"])
@pytest.mark.parametrize("include_images", [True, False])
def test_signal_result_shows_one_zip_and_preserves_recommendation(surfaces, tmp_path, app, status, include_images):
    window, page, runtime = surfaces
    before_size = window.size()
    before_text = (window.suggestion_label.text(), window.detail_label.text())
    ui = start(window, include_images)
    archive = archive_at(tmp_path)
    runtime.problem_export_status.emit({
        "status": status, "archive_path": archive, "message": "已保存本机",
        "missing": ["部分历史日志"] if status == "PARTIAL" else [],
        "omitted": ["截图"] if not include_images else [], "include_images": include_images,
    })
    app.processEvents()
    dialog = ui.result_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.path_edit.text() == str(archive)
    assert dialog.open_folder_button.isEnabled()
    assert dialog.copy_path_button.isEnabled()
    assert "问题包已生成" == dialog.windowTitle()
    assert not dialog.details_view.isVisible()
    click(dialog.details_toggle)
    assert dialog.details_view.isVisible()
    if status == "PARTIAL":
        assert "部分历史日志" in dialog.details_view.toPlainText()
    if not include_images:
        assert "仅日志" in dialog.details_view.toPlainText()
        assert "不包含游戏截图" in dialog.details_view.toPlainText()
    assert window.problem_export_button.isEnabled()
    assert page.problem_export_button.isEnabled()
    assert (window.suggestion_label.text(), window.detail_label.text()) == before_text
    assert window.size() == before_size
    assert window.isVisible()
    assert dialog.height() < 360
    runtime.problem_export_status.emit({"status": status, "archive_path": archive})
    assert ui.result_dialog is dialog  # Repeated terminal signals cannot open another dialog.


def test_object_enum_result_and_background_thread_signal(surfaces, tmp_path, app):
    window, _page, runtime = surfaces
    ui = start(window)
    archive = archive_at(tmp_path)

    class Status(Enum):
        PARTIAL = "PARTIAL"

    result = SimpleNamespace(status=Status.PARTIAL, archive_path=archive, message="缺少部分截图",
                             missing=["frame"], omitted=[], include_images=True)
    thread = Thread(target=lambda: runtime.problem_export_status.emit(result))
    thread.start()
    thread.join()
    for _ in range(20):
        app.processEvents()
        if ui.result_dialog is not None:
            break
        QTest.qWait(10)
    assert ui.result_dialog is not None
    assert "问题包已生成" == ui.result_dialog.windowTitle()
    assert ui.result_dialog.thread() == app.thread()


@pytest.mark.parametrize("bad_archive", ["missing", "directory", "not_zip", "relative", "https", "file_url"])
@pytest.mark.parametrize("status", ["SUCCESS", "PARTIAL"])
def test_missing_or_nonlocal_archive_never_claims_success(surfaces, tmp_path, status, bad_archive):
    window, page, runtime = surfaces
    ui = start(window)
    nonzip = tmp_path / "readme.txt"
    nonzip.write_text("not a problem package", encoding="utf-8")
    value = {
        "missing": tmp_path / "missing.zip", "directory": tmp_path, "not_zip": nonzip,
        "relative": "bundle.zip", "https": "https://example.com/bundle.zip",
        "file_url": "file:///C:/bundle.zip",
    }[bad_archive]
    runtime.problem_export_status.emit({"status": status, "archive_path": value})
    dialog = ui.result_dialog
    assert "失败" in dialog.windowTitle()
    assert not dialog.open_folder_button.isEnabled()
    assert not dialog.copy_path_button.isEnabled()
    assert not dialog.path_edit.isVisible()
    assert window.problem_export_button.isEnabled()
    assert page.problem_export_button.isEnabled()
    assert "失败" in window.problem_export_button.toolTip()


def test_failure_is_bounded_and_does_not_expose_stale_zip(surfaces, tmp_path, app):
    window, _page, runtime = surfaces
    size = window.size()
    ui = start(window)
    runtime.problem_export_status.emit({"status": "FAILURE", "archive_path": archive_at(tmp_path),
                                       "message": "磁盘空间不足；" * 500})
    app.processEvents()
    assert "失败" in ui.result_dialog.windowTitle()
    assert "磁盘空间不足" in ui.result_dialog.details_view.toPlainText()
    assert ui.result_dialog.archive_path is None
    assert not ui.result_dialog.open_folder_button.isEnabled()
    assert not ui.result_dialog.copy_path_button.isEnabled()
    assert ui.result_dialog.height() < 360
    assert window.size() == size
    assert window.isVisible()


@pytest.mark.parametrize("failure", ["rejected", "exception"])
def test_request_refusal_or_exception_allows_retry(surfaces, failure):
    window, page, runtime = surfaces
    runtime.accept = failure != "rejected"
    runtime.error = RuntimeError("无法写入") if failure == "exception" else None
    ui = start(window)
    assert "失败" in ui.result_dialog.windowTitle()
    assert window.problem_export_button.isEnabled()
    assert page.problem_export_button.isEnabled()
    runtime.accept, runtime.error = True, None
    start(window)
    assert len(runtime.requests) == 2
    assert not window.problem_export_button.isEnabled()


def test_every_export_requires_consent_and_cancellation_preserves_prior_result(surfaces, tmp_path):
    window, _page, runtime = surfaces
    ui = start(window, False)
    runtime.problem_export_status.emit({"status": "SUCCESS", "archive_path": archive_at(tmp_path)})
    old_result = ui.result_dialog
    click(window.problem_export_button)
    assert ui.confirmation_dialog is not None
    assert len(runtime.requests) == 1
    click(ui.confirmation_dialog.cancel_button)
    assert len(runtime.requests) == 1
    assert ui.result_dialog is old_result


def test_synchronous_completion_is_not_replaced_by_request_return(surfaces, tmp_path):
    window, _page, runtime = surfaces
    runtime.synchronous_result = {"status": "SUCCESS", "archive_path": archive_at(tmp_path)}
    ui = start(window)
    assert "问题包已生成" == ui.result_dialog.windowTitle()
    assert window.problem_export_button.isEnabled()


def test_local_url_and_clipboard_handle_chinese_paths_without_navigation(surfaces, tmp_path, monkeypatch):
    window, _page, runtime = surfaces
    ui = start(window)
    archive = archive_at(tmp_path)
    runtime.problem_export_status.emit({"status": "SUCCESS", "archive_path": archive})
    urls = []
    monkeypatch.setattr(module.QDesktopServices, "openUrl", lambda url: urls.append(url) or True)
    click(ui.result_dialog.open_folder_button)
    assert len(urls) == 1 and urls[0].isLocalFile()
    assert Path(urls[0].toLocalFile()) == archive.parent
    click(ui.result_dialog.copy_path_button)
    assert QApplication.clipboard().text() == str(archive)
    assert window.isVisible()
    assert runtime.stop_calls == 0


@pytest.mark.parametrize("action", ["open_folder_button", "copy_path_button"])
def test_deleted_archive_actions_recheck_existence(surfaces, tmp_path, monkeypatch, action):
    window, _page, runtime = surfaces
    ui = start(window)
    archive = archive_at(tmp_path)
    runtime.problem_export_status.emit({"status": "SUCCESS", "archive_path": archive})
    archive.unlink()
    urls = []
    monkeypatch.setattr(module.QDesktopServices, "openUrl", lambda url: urls.append(url) or True)
    QApplication.clipboard().setText("unchanged")
    click(getattr(ui.result_dialog, action))
    assert urls == []
    assert QApplication.clipboard().text() == "unchanged"
    assert "不存在" in ui.result_dialog.message_label.text()
    assert not ui.result_dialog.open_folder_button.isEnabled()
    assert not ui.result_dialog.copy_path_button.isEnabled()


def test_open_failure_keeps_copy_path_available(surfaces, tmp_path, monkeypatch):
    window, _page, runtime = surfaces
    ui = start(window)
    runtime.problem_export_status.emit({"status": "SUCCESS", "archive_path": archive_at(tmp_path)})
    monkeypatch.setattr(module.QDesktopServices, "openUrl", lambda _url: False)
    click(ui.result_dialog.open_folder_button)
    assert "无法打开" in ui.result_dialog.message_label.text()
    assert ui.result_dialog.copy_path_button.isEnabled()


@pytest.mark.parametrize("runtime", [SimpleNamespace(), SimpleNamespace(request_problem_export=lambda **kwargs: True)])
def test_older_runtime_gracefully_reports_missing_interface(app, runtime):
    window = RecommendationFloatWindow(runtime)
    click(window.problem_export_button)
    ui = window.problem_export_ui
    assert ui.confirmation_dialog is None
    assert "不支持" in ui.result_dialog.details_view.toPlainText()
    assert "失败" in ui.result_dialog.windowTitle()
    assert window.problem_export_button.isEnabled()
    ui.result_dialog.close()
    window.hide()
    window.deleteLater()
    app.processEvents()


def test_external_export_disables_both_entries_without_unsolicited_result(surfaces):
    window, page, runtime = surfaces
    runtime.problem_export_status.emit({"status": "RUNNING"})
    click(window.problem_export_button)
    page.request_problem_export()
    assert runtime.requests == []
    assert not window.problem_export_button.isEnabled()
    assert not page.problem_export_button.isEnabled()
    runtime.problem_export_status.emit({"status": "FAILURE"})
    assert window.problem_export_button.isEnabled()
    assert page.problem_export_button.isEnabled()
    assert window.problem_export_ui.result_dialog is None


@pytest.mark.parametrize("selected", ["case", "frames", "legacy", "none"])
def test_diagnostic_page_binds_selection_not_current_runtime_case(surfaces, tmp_path, selected):
    window, page, runtime = surfaces
    historical = tmp_path / "中文历史 case"
    historical.mkdir()
    (historical / "case.json").write_text("{}", encoding="utf-8")
    (historical / "frames").mkdir()
    legacy = tmp_path / "old_session"
    (legacy / "diagnostic_frames").mkdir(parents=True)
    target = {"case": historical, "frames": historical / "frames", "legacy": legacy / "diagnostic_frames"}.get(selected)
    if target is not None:
        page.set_session_directory(target)
    expected = legacy if selected == "legacy" else historical if target is not None else None
    page.request_problem_export()
    dialog = window.problem_export_ui.confirmation_dialog
    assert dialog.parent() is page
    assert dialog.target_label.toolTip() == str(expected or "")
    # Selection changes while consent is open must not switch the export case.
    page.set_session_directory(tmp_path / "another_case")
    click(dialog.logs_button)
    assert runtime.requests == [(False, expected)]


def test_empty_selected_case_stays_bound_and_unsupported_standalone_page_is_safe(app, tmp_path):
    page = WindowDebugPage(report_service=SimpleNamespace())
    case = tmp_path / "empty_case"
    case.mkdir()
    (case / "case.json").write_text("{}", encoding="utf-8")
    assert page.set_session_directory(case) == case
    assert page._session_directory == case
    click(page.problem_export_button)
    assert "不支持" in page.policy_label.text()
    page.close()
    page.deleteLater()


def test_main_window_injects_the_same_runtime_presenter_into_both_surfaces(app, monkeypatch, tmp_path):
    class StubPage(QWidget):
        compact_mode_requested = Signal()

        def __init__(self, dependency):
            super().__init__()
            self.setObjectName("stub_" + str(id(self)))

        def shutdown(self):
            pass

    for name in ("AnnotationPage", "LiveAssistantPage", "ReplayPage"):
        monkeypatch.setattr(main_module, name, StubPage)
    runtime = ExportRuntime()
    deps = SimpleNamespace(live_runtime=runtime, annotation_service=object(), sessions_root=tmp_path)
    main = main_module.DaguandanBridgeWindow(dependencies=deps, window_debug_service=SimpleNamespace())
    assert main.window_debug_page.problem_export_ui is main.recommendation_window.problem_export_ui
    assert main.window_debug_page.problem_export_ui.runtime is runtime
    main.window_debug_page.request_problem_export()
    click(main.recommendation_window.problem_export_ui.confirmation_dialog.logs_button)
    assert runtime.requests == [(False, None)]
    assert not main.recommendation_window.problem_export_button.isEnabled()
    assert not main.window_debug_page.problem_export_button.isEnabled()
    main.close()
    main.recommendation_window.deleteLater()
    main.deleteLater()
    app.processEvents()


def test_external_running_while_confirming_dismisses_stale_consent_and_keeps_guard(surfaces):
    window, page, runtime = surfaces
    click(window.problem_export_button)
    runtime.problem_export_status.emit({"status": "RUNNING"})
    assert window.problem_export_ui.confirmation_dialog is None
    assert runtime.requests == []
    assert not window.problem_export_button.isEnabled()
    assert not page.problem_export_button.isEnabled()
    runtime.problem_export_status.emit({"status": "FAILURE"})
    assert window.problem_export_button.isEnabled()
    assert window.problem_export_ui.result_dialog is None


def test_unrelated_late_completion_does_not_unlock_consent(surfaces):
    window, page, runtime = surfaces
    click(window.problem_export_button)
    dialog = window.problem_export_ui.confirmation_dialog
    runtime.problem_export_status.emit({"status": "FAILURE"})
    assert window.problem_export_ui.confirmation_dialog is dialog
    assert not window.problem_export_button.isEnabled()
    assert not page.problem_export_button.isEnabled()
    click(dialog.cancel_button)
    assert window.problem_export_button.isEnabled()


def test_native_dialog_theme_uses_dark_background_and_collapsed_machine_details(surfaces, tmp_path, app):
    from qfluentwidgets import setTheme, Theme
    window, page, runtime = surfaces
    try:
        setTheme(Theme.DARK)
        click(window.problem_export_button)
        app.processEvents()
        confirm = window.problem_export_ui.confirmation_dialog
        assert "#202020" in confirm.styleSheet()
        click(confirm.images_button)
        runtime.problem_export_status.emit({"status":"PARTIAL", "archive_path":archive_at(tmp_path),
            "missing":[{"path":"session", "reason":"not_bound"}]})
        app.processEvents()
        result = window.problem_export_ui.result_dialog
        assert "#202020" in result.styleSheet()
        assert result.windowTitle() == "问题包已生成"
        assert not result.details_view.isVisible()
        assert "not_bound" not in result.message_label.text()
        setTheme(Theme.LIGHT)
        app.processEvents()
        assert "#f7f7f7" in result.styleSheet()
    finally:
        setTheme(Theme.LIGHT)
