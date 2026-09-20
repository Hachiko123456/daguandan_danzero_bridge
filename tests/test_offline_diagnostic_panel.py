from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Slot
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from shiboken6 import isValid

from daguandan_bridge.gui.offline_diagnostic_panel import (
    OfflineDiagnosticPanel,
    OfflineDiagnosticReplayThread,
    _ACTIVE_THREADS,
)
from daguandan_bridge.gui.replay_page import ReplayPage


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


def wait_until(app, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    assert predicate()


def result(report: Path, status="complete", modes=None):
    data = {
        "status": status, "report_path": str(report),
        "output_directory": str(report.parent),
        "advice_statuses": {"ready": 999},  # Must not become an aggregated PASS.
        "modes": modes if modes is not None else {
            "latest": {"status": "complete", "frame_count": 100},
            "synchronous": {"status": "complete", "frame_count": 200},
        },
    }
    return SimpleNamespace(to_dict=lambda: data)


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "诊断结果" / "report.md"
    path.parent.mkdir()
    path.write_text("# 离线复测报告\n\n保留已完成部分。\n", encoding="utf-8")
    return path


class HeldService:
    """Wait for an explicit test release even after observing cancellation."""

    def __init__(self, report):
        self.report = report
        self.entered = threading.Event()
        self.cancel_seen = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def run(self, input_path, *, on_progress, stop_requested, environment):
        self.calls.append((input_path, stop_requested, environment, QThread.currentThread()))
        self.entered.set()
        while not self.release.wait(0.01):
            if stop_requested():
                self.cancel_seen.set()
        cancelled = stop_requested()
        on_progress({"mode": "latest", "processed": 2, "total": 10, "report_path": str(self.report)})
        return result(self.report, "cancelled" if cancelled else "complete")


def test_callback_supports_dict_and_legacy_tuple():
    assert OfflineDiagnosticReplayThread._progress_values(2, 5, 17) == {
        "processed": 2, "total": 5, "frame_index": 17,
    }
    payload = {"mode": "latest", "phase": "replay", "processed": 3}
    assert OfflineDiagnosticReplayThread._progress_values(payload) == payload
    assert OfflineDiagnosticReplayThread._progress_values(payload) is not payload


def test_modern_signature_receives_all_options_and_ui_slots_are_queued(app, tmp_path, report):
    calls = []
    received_threads = []
    host_thread = []

    class Panel(OfflineDiagnosticPanel):
        @staticmethod
        def _host_environment():
            host_thread.append(QThread.currentThread())
            return OfflineDiagnosticPanel._host_environment()

        @Slot(object)
        def _on_progress(self, data):
            received_threads.append(QThread.currentThread())
            super()._on_progress(data)

    class Service:
        def run(self, input_path, *, on_progress, stop_requested, environment):
            calls.append((input_path, stop_requested, environment, QThread.currentThread()))
            on_progress({"mode": "latest", "processed": 9, "total": 10, "frame_index": 8})
            on_progress({"mode": "synchronous", "processed": 1, "total": 10, "frame_index": 0})
            return result(report)

    constructors = []

    def factory(**kwargs):
        constructors.append(kwargs)
        return Service()

    panel = Panel(profiles_root=tmp_path / "profiles", profile_name="test", service_factory=factory)
    try:
        assert panel.start(tmp_path / "诊断.zip")
        wait_until(app, lambda: not panel.is_running)
        path, stopped, environment, worker_thread = calls[0]
        assert len(calls) == 1
        assert constructors == [{"profiles_root": tmp_path / "profiles", "profile_name": "test"}]
        assert path == tmp_path / "诊断.zip"
        assert isinstance(stopped.__self__, threading.Event)
        assert not stopped()
        assert host_thread == [app.thread()]
        assert received_threads == [app.thread(), app.thread()]
        assert worker_thread != app.thread()
        assert environment["source"] == "host_environment"
        for screen in environment["screens"]:
            assert {"geometry", "availableGeometry", "dpr", "primary", "name"} <= screen.keys()
            assert {"x", "y", "width", "height"} == screen["geometry"].keys()
        assert "验证截图" in environment["note"]
        assert panel.progress_bar.value() == 10  # New mode resets progress.
        assert panel.summary_label.text().splitlines() == [
            "原速复测（1×）：已完成 · 100 帧", "逐帧对照：已完成 · 200 帧",
        ]
        assert "999" not in panel.summary_label.text()
        assert "PASS" not in panel.status_label.text()
        assert "production" not in panel.status_label.text()
    finally:
        panel.shutdown()
        panel.close()


def test_real_cancel_event_prevents_concurrent_start_and_preserves_report(app, tmp_path, report):
    service = HeldService(report)
    panel = OfflineDiagnosticPanel(service_factory=lambda **kw: service)
    busy = []
    panel.busy_changed.connect(busy.append)
    try:
        assert panel.start(tmp_path / "input.zip")
        assert service.entered.wait(1)
        worker = panel._thread
        assert panel.cancel_button.isEnabled()
        assert not panel.start(tmp_path / "other.zip")
        panel.cancel_button.click()
        assert service.cancel_seen.wait(1)
        assert worker.stop_event.is_set()
        assert not panel.cancel_button.isEnabled()
        assert not panel.import_button.isEnabled()
        panel._on_progress({"mode": "latest", "processed": 2, "total": 10})
        assert "正在停止" in panel.status_label.text()
        assert busy == [True]
        service.release.set()
        wait_until(app, lambda: not panel.is_running)
        assert len(service.calls) == 1
        assert busy == [True, False]
        assert "已取消" in panel.status_label.text()
        assert panel._report_target() == report
        assert not panel.report_button.isHidden()
        assert panel.import_button.isEnabled()
    finally:
        service.release.set()
        wait_until(app, lambda: not panel.is_running)
        panel.close()


@pytest.mark.parametrize("status", ["failed", "incomplete", "complete"])
def test_failed_or_incomplete_mode_is_not_a_global_pass_and_report_is_user_opened(
    app, tmp_path, report, monkeypatch, status,
):
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)

    class Service:
        def run(self, path, *, on_progress, stop_requested, environment):
            return result(report, status, {
                "latest": {"status": "failed", "advice_statuses": {"ready": 10}},
                "synchronous": {"status": "complete", "advice_statuses": {"ready": 20}},
            })

    panel = OfflineDiagnosticPanel(service_factory=lambda **kw: Service())
    try:
        assert panel.start(tmp_path / "input.zip")
        wait_until(app, lambda: not panel.is_running)
        assert "有问题" in panel.status_label.text()
        assert "未通过" in panel.status_label.text()
        assert panel.summary_label.text().splitlines() == [
            "原速复测（1×）：有问题", "逐帧对照：已完成",
        ]
        assert opened == []
        assert panel._report_path == report
        panel.report_button.click()
        panel.directory_open_button.click()
        assert [Path(p) for p in opened] == [report, report.parent]
    finally:
        panel.close()


def test_exception_preserves_progress_report_and_does_not_retry_typeerror(app, tmp_path, report):
    calls = []

    class Service:
        def run(self, path, **kwargs):
            assert set(kwargs) == {"on_progress", "stop_requested", "environment"}
            calls.append(path)
            kwargs["on_progress"]({"mode": "latest", "report_path": str(report)})
            raise TypeError("internal processing failure")

    panel = OfflineDiagnosticPanel(service_factory=lambda **kw: Service())
    try:
        assert panel.start(tmp_path / "input.zip")
        wait_until(app, lambda: not panel.is_running)
        assert len(calls) == 1
        assert "有问题" in panel.status_label.text()
        assert panel._report_target() == report
        assert not panel.report_button.isHidden()
    finally:
        panel.close()


def test_old_stub_is_supported_but_does_not_offer_fake_cancel(app, tmp_path, report):
    class Service:
        def run(self, path, on_progress):
            on_progress(1, 2, 3)
            return result(report)

    panel = OfflineDiagnosticPanel(service_factory=lambda **kw: Service())
    try:
        assert panel.start(tmp_path / "input.zip")
        assert not panel.cancel_button.isEnabled()
        assert not panel._thread.supports_cancel
        wait_until(app, lambda: not panel.is_running)
        assert panel._report_target() == report
    finally:
        panel.close()


def test_shutdown_is_bounded_and_window_close_defers_destruction_until_finished(app, tmp_path, report):
    service = HeldService(report)
    host = QWidget()
    layout = QVBoxLayout(host)
    panel = OfflineDiagnosticPanel(service_factory=lambda **kw: service, parent=host)
    layout.addWidget(panel)
    host.show()
    try:
        assert panel.start(tmp_path / "input.zip")
        assert service.entered.wait(1)
        worker = panel._thread
        started = time.monotonic()
        assert not panel.shutdown(timeout_ms=60_000)
        assert time.monotonic() - started < 0.75
        assert service.cancel_seen.wait(1)
        assert worker in _ACTIVE_THREADS
        assert worker.parent() is None
        assert not host.close()
        assert host.isVisible()
        assert panel.is_running
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        wait_until(app, lambda: bool(ticks))
        assert worker.isRunning()
        service.release.set()
        wait_until(app, lambda: not panel.is_running)
        wait_until(app, lambda: not host.isVisible())
        assert worker not in _ACTIVE_THREADS
    finally:
        service.release.set()
        wait_until(app, lambda: not panel.is_running)
        host.close()


def test_parent_deletion_does_not_destroy_running_qthread(app, tmp_path, report):
    service = HeldService(report)
    panel = OfflineDiagnosticPanel(service_factory=lambda **kw: service)
    assert panel.start(tmp_path / "input.zip")
    assert service.entered.wait(1)
    worker = panel._thread
    try:
        panel.request_cancel()
        panel.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not isValid(panel)
        assert isValid(worker) and worker.isRunning()
        assert worker in _ACTIVE_THREADS
    finally:
        service.release.set()
        assert worker.wait(2000)
        wait_until(app, lambda: worker not in _ACTIVE_THREADS)


@pytest.mark.parametrize("thread_name", ["_visual_thread", "_pure_scan_thread", "_trusted_thread", "_unverified_batch_thread"])
def test_page_rejects_conflicting_scan_and_injected_live_guard(app, tmp_path, thread_name):
    live_idle = [False]
    page = ReplayPage(tmp_path, can_start_offline_diagnostic=lambda: live_idle[0])
    try:
        assert not page.offline_diagnostic_panel.start(tmp_path / "input.zip")
        live_idle[0] = True
        assert page._can_start_offline_diagnostic()
        setattr(page, thread_name, SimpleNamespace(isRunning=lambda: True))
        assert not page._can_start_offline_diagnostic()
        assert not page.offline_diagnostic_panel.start(tmp_path / "input.zip")
    finally:
        setattr(page, thread_name, None)
        page.shutdown()
        page.close()


def test_page_restores_eligibility_and_direct_scan_routes_remain_blocked(app, tmp_path, report):
    page = ReplayPage(tmp_path)
    service = HeldService(report)
    panel = page.offline_diagnostic_panel
    panel._service_factory = lambda **kw: service
    page.visual_replay_button.setEnabled(True)
    page.scan_unverified_button.setEnabled(True)
    page.scan_initial_state_button.setEnabled(False)
    try:
        assert panel.start(tmp_path / "input.zip")
        assert service.entered.wait(1)
        assert not page.content_host.isEnabled()
        assert not page.selector_card.isEnabled()
        assert not page.visual_replay_button.isEnabled()
        assert not page.scan_unverified_button.isEnabled()
        page._scan_unverified_sessions()
        page.analyze_video_to_truth_log()
        page.replay_truth()
        page.replay_trusted_advisor(None)
        assert all(t is None for t in (page._pure_scan_thread, page._trusted_thread, page._unverified_batch_thread))
        service.release.set()
        wait_until(app, lambda: not panel.is_running)
        assert page.visual_replay_button.isEnabled()
        assert page.scan_unverified_button.isEnabled()
        assert not page.scan_initial_state_button.isEnabled()
    finally:
        service.release.set()
        wait_until(app, lambda: not panel.is_running)
        page.shutdown()
        page.close()


def test_host_environment_handles_no_screens(app, monkeypatch):
    monkeypatch.setattr(QGuiApplication, "screens", lambda: [])
    monkeypatch.setattr(QGuiApplication, "primaryScreen", lambda: None)
    assert OfflineDiagnosticPanel._host_environment()["screens"] == []
