from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.live_controller import LiveAssistantController


def _app():
    return QApplication.instance() or QApplication([])


class _CaptureServiceStub:
    def __init__(self, root):
        self.profiles_root = root


class _SlowCaptureWorker:
    is_running = True

    def stop(self):
        pass

    def wait(self, timeout_ms):
        if timeout_ms:
            time.sleep(timeout_ms / 1_000)
        return False


class _SlowAnalysisWorker:
    def stop(self, *, timeout=None):
        if timeout:
            time.sleep(timeout)
        return False


class _PauseOrchestrator:
    status = "running"

    def pause(self):
        self.status = "paused"
        return SimpleNamespace(status="paused")

    def resume(self, *, monotonic_ms):
        del monotonic_ms
        self.status = "running"
        return SimpleNamespace(status="running")


def test_pause_never_waits_for_blocked_capture_or_recognition(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    orchestrator = _PauseOrchestrator()
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    controller._capture_worker = _SlowCaptureWorker()  # type: ignore[assignment]
    controller._analysis_worker = _SlowAnalysisWorker()  # type: ignore[assignment]

    started = time.perf_counter()
    controller.pause()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2
    assert orchestrator.status == "paused"


def test_resume_restarts_capture_after_prior_blocked_capture_exits(
    tmp_path,
    monkeypatch,
):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    orchestrator = _PauseOrchestrator()
    old_worker = _SlowCaptureWorker()
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    controller._capture_worker = old_worker  # type: ignore[assignment]
    monkeypatch.setattr(controller, "_start_analysis_worker", lambda: None)
    restarted = []
    monkeypatch.setattr(
        controller,
        "_start_capture_worker",
        lambda: restarted.append(True)
        if controller._capture_worker is None
        else None,
    )

    controller.pause()
    controller.resume()
    assert restarted == []

    controller._capture_finished(old_worker)  # type: ignore[arg-type]

    assert restarted == [True]


class _FinishOrchestrator:
    status = "running"

    def __init__(self, release: threading.Event):
        self.release = release

    def begin_finalizing(self):
        self.status = "finalizing"

    def finish(self):
        assert self.release.wait(2)
        self.status = "sealed"
        return SimpleNamespace(status="sealed")


def test_finish_runs_sealing_work_outside_gui_thread(tmp_path):
    app = _app()
    release = threading.Event()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    orchestrator = _FinishOrchestrator(release)
    controller.orchestrator = orchestrator  # type: ignore[assignment]

    started = time.perf_counter()
    controller.finish()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2
    assert controller._finish_thread is not None
    release.set()
    controller._finish_thread.wait(2_000)
    app.processEvents()

    assert controller.orchestrator is None
