from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.live_controller import LiveAssistantController
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.orchestrator import LiveUpdate


def _app():
    return QApplication.instance() or QApplication([])


class _CaptureServiceStub:
    def __init__(self, root):
        self.profiles_root = root


class _WarmAdvisor:
    def __init__(self):
        self.initialize_calls = 0

    def initialize(self):
        self.initialize_calls += 1


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


def test_controller_warms_one_reusable_default_advisor_in_background(tmp_path):
    app = _app()
    advisor = _WarmAdvisor()
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        advisor=advisor,
    )
    statuses = []
    controller.danzero_warmup_status.connect(statuses.append)

    controller._start_danzero_warmup()

    assert controller.danzero_advisor is advisor
    assert controller._danzero_warmup_thread is not None
    controller._danzero_warmup_thread.wait(2_000)
    app.processEvents()
    controller._start_danzero_warmup()

    assert advisor.initialize_calls == 1
    assert statuses[0] == "FableDan 模型预热中"
    assert statuses[-1].startswith("FableDan 模型已就绪")


def _initial_recognition(hand, *, round_level="2"):
    return SimpleNamespace(my_hand=hand, round_level=round_level)


def test_listener_starts_session_after_two_identical_complete_hands(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    hand = tuple(f"{rank}{suit}" for rank in ("3", "4", "5", "6", "7", "8", "9") for suit in "SHCD")[:27]
    started = []
    controller._listening_enabled = True
    monkeypatch.setattr(
        controller,
        "_start_detected_session",
        lambda result: started.append((result.round_level, result.my_hand)),
    )

    controller._consume_waiting_recognition(_initial_recognition(hand), None)
    controller._consume_waiting_recognition(_initial_recognition(hand), None)

    assert started == [("2", hand)]


def test_listener_does_not_start_session_when_complete_hand_changes(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    first = tuple(f"{rank}{suit}" for rank in ("3", "4", "5", "6", "7", "8", "9") for suit in "SHCD")[:27]
    second = first[:-1] + ("10S",)
    started = []
    controller._listening_enabled = True
    monkeypatch.setattr(controller, "_start_detected_session", lambda result: started.append(result))

    controller._consume_waiting_recognition(_initial_recognition(first), None)
    controller._consume_waiting_recognition(_initial_recognition(second), None)

    assert started == []


def test_listener_treats_different_recognition_order_as_the_same_hand(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    first = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    started = []
    controller._listening_enabled = True
    monkeypatch.setattr(controller, "_start_detected_session", lambda result: started.append(result))

    controller._consume_waiting_recognition(_initial_recognition(first), None)
    controller._consume_waiting_recognition(_initial_recognition(tuple(reversed(first))), None)

    assert len(started) == 1


def test_controller_auto_finishes_once_when_game_end_is_detected(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    calls = []
    monkeypatch.setattr(controller, "finish", lambda: calls.append("finish"))
    event = LiveEvent(
        event_id="AUX-000001",
        event_type="game_end_detected",
        session_id="session",
        seq=1,
        monotonic_ms=0,
        wall_time="2026-08-09T00:00:00+08:00",
        trick_id=1,
        turn_id=1,
        actor=None,
        payload={"control": "continue_game"},
        confidence=1.0,
        source="test",
        state_revision_before=1,
        state_revision_after=1,
    )
    update = LiveUpdate(status="running", snapshot=SimpleNamespace(), event=event)

    controller._auto_finish_on_game_end(update)
    controller._auto_finish_on_game_end(update)

    assert calls == ["finish"]


def test_controller_auto_finishes_when_terminal_event_is_in_update_events(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    calls = []
    monkeypatch.setattr(controller, "finish", lambda: calls.append("finish"))
    event = LiveEvent(
        event_id="AUX-000001",
        event_type="game_end_detected",
        session_id="session",
        seq=1,
        monotonic_ms=0,
        wall_time="2026-08-09T00:00:00+08:00",
        trick_id=1,
        turn_id=1,
        actor=None,
        payload={"control": "change_table"},
        confidence=1.0,
        source="test",
        state_revision_before=1,
        state_revision_after=1,
    )
    update = LiveUpdate(
        status="running",
        snapshot=SimpleNamespace(),
        events=(event,),
    )

    controller._auto_finish_on_game_end(update)

    assert calls == ["finish"]


def test_controller_resumes_waiting_listener_after_sealing_when_enabled(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    resumed = []
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: resumed.append(True))

    controller._finish_thread_finished()

    assert resumed == [True]
