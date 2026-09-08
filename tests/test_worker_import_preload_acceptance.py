from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.application.ports import LiveSessionConstruction
from daguandan_bridge.gui import live_controller as controller_module
from daguandan_bridge.gui.live_controller import LiveAssistantController
from daguandan_bridge.live.orchestrator import LiveUpdate


def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in tuple(self.callbacks):
            callback(*args)


class _RecordingOneShotThread:
    def __init__(self, operation, parent=None, *, events):
        del parent
        self.operation = operation
        self.result = _Signal()
        self.error = _Signal()
        self.finished = _Signal()
        self._events = events

    def start(self):
        self._events.append("warmup_worker_start")

    def isRunning(self):
        return False

    def wait(self, timeout_ms=0):
        del timeout_ms
        return True


class _RecordingLatestOnlyWorker:
    def __init__(self, *args, events, **kwargs):
        del kwargs
        self._events = events
        operation = args[0] if args else None
        self._start_event = (
            "waiting_analysis_worker_start"
            if getattr(operation, "__name__", "") == "_recognize_waiting_frame"
            else "analysis_worker_start"
        )

    def start(self):
        self._events.append(self._start_event)

    def stop(self, *, discard_pending=True):
        del discard_pending

    def wait(self, timeout=1.0):
        del timeout
        return True


class _RecordingWorkerHandle:
    def __init__(self, operation, interval_sec, *, events):
        del operation
        self.frame_ready = _Signal()
        self.error = _Signal()
        self.finished = _Signal()
        self._events = events
        self.worker = SimpleNamespace(interval_sec=interval_sec)
        self._start_event = (
            "waiting_capture_worker_start"
            if float(interval_sec) == 1.0
            else "capture_worker_start"
        )

    @property
    def is_running(self):
        return False

    def start(self):
        self._events.append(self._start_event)

    def stop(self):
        pass

    def wait(self, timeout_ms=0):
        del timeout_ms
        return True


class FableDanAdvisor:
    def initialize(self):
        return None


class _CaptureService:
    def __init__(self, root, events):
        self.profiles_root = root
        self.events = events

    def lock_target_client_size(self, profile_name):
        del profile_name
        self.events.append("lock_capture_source")

    def open_live_source(self, profile_name):
        del profile_name
        self.events.append("open_capture_source")
        return _Source()


class _Source:
    def close(self):
        pass


class _Orchestrator:
    def __init__(self, session_id="preload-session"):
        self.snapshot = SimpleNamespace(session_id=session_id)
        self.store = SimpleNamespace()
        self.recorder = SimpleNamespace()
        self.needs_first_action_frames = False
        self.finished = False

    def bind_capture_generation(self, generation):
        return LiveUpdate(
            status="running",
            snapshot=self.snapshot,
            capture_generation=generation,
            update_sequence=1,
        )

    def begin_finalizing(self):
        return None

    def finish(self):
        self.finished = True


class _SessionFactory:
    def __init__(self, events):
        self.events = events
        self.started = False

    def start_session(self, **kwargs):
        del kwargs
        self.started = True
        self.events.append("session_start")
        return LiveSessionConstruction(_Orchestrator(), _Source(), None)


@pytest.fixture
def controller_harness(tmp_path, monkeypatch):
    _app()
    events: list[str] = []
    monkeypatch.setattr(
        controller_module,
        "OneShotThread",
        lambda operation, parent=None: _RecordingOneShotThread(
            operation, parent, events=events,
        ),
    )
    monkeypatch.setattr(
        controller_module,
        "LatestOnlyWorker",
        lambda *args, **kwargs: _RecordingLatestOnlyWorker(
            *args, events=events, **kwargs,
        ),
    )
    monkeypatch.setattr(
        controller_module,
        "WorkerHandle",
        lambda operation, interval_sec: _RecordingWorkerHandle(
            operation, interval_sec, events=events,
        ),
    )
    factory = _SessionFactory(events)
    controller = LiveAssistantController(
        _CaptureService(tmp_path, events),
        recognition_service=SimpleNamespace(),
        advisor=FableDanAdvisor(),
        session_factory=factory,
    )
    errors: list[str] = []
    controller.error.connect(errors.append)
    return controller, factory, events, errors


def test_live_start_preloads_win32_capture_and_rlcard_before_any_worker(
    controller_harness,
    monkeypatch,
):
    controller, _factory, events, _errors = controller_harness

    def preload_live_worker_dependencies(*args, **kwargs):
        del args, kwargs
        events.append("preload_win32_capture")
        events.append("preload_rlcard_rules")

    monkeypatch.setattr(
        controller_module,
        "preload_live_worker_dependencies",
        preload_live_worker_dependencies,
        raising=False,
    )

    assert controller.start_session(
        round_level="2",
        hand=("3C",),
        lead_player="self",
    )
    worker_starts = [
        index for index, event in enumerate(events) if event.endswith("_worker_start")
    ]
    assert worker_starts, events
    assert events.index("preload_win32_capture") < min(worker_starts)
    assert events.index("preload_rlcard_rules") < min(worker_starts)
    assert events.index("preload_win32_capture") < events.index("session_start")
    assert events.index("preload_rlcard_rules") < events.index("session_start")


def test_live_start_preload_failure_reports_before_creating_workers_or_session(
    controller_harness,
    monkeypatch,
):
    controller, factory, events, errors = controller_harness

    def preload_live_worker_dependencies(*args, **kwargs):
        del args, kwargs
        events.append("preload_win32_capture")
        raise RuntimeError("rlcard rules preload failed")

    monkeypatch.setattr(
        controller_module,
        "preload_live_worker_dependencies",
        preload_live_worker_dependencies,
        raising=False,
    )

    assert not controller.start_session(
        round_level="2",
        hand=("3C",),
        lead_player="self",
    )
    assert events == ["preload_win32_capture"]
    assert not factory.started
    assert controller.orchestrator is None
    assert any("rlcard rules preload failed" in message for message in errors)


def test_listening_preloads_before_warmup_waiting_workers_and_capture_source(
    controller_harness,
    monkeypatch,
):
    controller, _factory, events, _errors = controller_harness

    def preload_live_worker_dependencies(*args, **kwargs):
        del args, kwargs
        events.append("preload_win32_capture")
        events.append("preload_rlcard_rules")

    monkeypatch.setattr(
        controller_module,
        "preload_live_worker_dependencies",
        preload_live_worker_dependencies,
        raising=False,
    )

    assert controller.start_listening()
    assert events[:2] == ["preload_win32_capture", "preload_rlcard_rules"]
    for event in (
        "lock_capture_source",
        "warmup_worker_start",
        "open_capture_source",
        "waiting_analysis_worker_start",
        "waiting_capture_worker_start",
    ):
        assert event in events
        assert events.index("preload_win32_capture") < events.index(event)
        assert events.index("preload_rlcard_rules") < events.index(event)


def test_listening_preload_failure_reports_before_enabling_or_creating_workers(
    controller_harness,
    monkeypatch,
):
    controller, _factory, events, errors = controller_harness

    def preload_live_worker_dependencies(*args, **kwargs):
        del args, kwargs
        events.append("preload_win32_capture")
        raise RuntimeError("win32 capture preload failed")

    monkeypatch.setattr(
        controller_module,
        "preload_live_worker_dependencies",
        preload_live_worker_dependencies,
        raising=False,
    )

    assert not controller.start_listening()
    assert events == ["preload_win32_capture"]
    assert not controller._listening_enabled
    assert controller._waiting_analysis_worker is None
    assert controller._waiting_capture_worker is None
    assert controller._waiting_source is None
    assert any("win32 capture preload failed" in message for message in errors)
