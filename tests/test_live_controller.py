from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.live_controller import (
    LiveAssistantController,
    _AnalysisFrameTask,
    _LiveRunToken,
)
from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.image_io import StandardizationResult
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.orchestrator import AdviceRequestKey, LiveAdvice, LiveUpdate
from daguandan_bridge.models import Box, ClientRect
from daguandan_bridge.window_capture import CapturedStandardizedFrame


def _app():
    return QApplication.instance() or QApplication([])


class _CaptureServiceStub:
    def __init__(self, root):
        self.profiles_root = root


class _LockingCaptureServiceStub(_CaptureServiceStub):
    def __init__(self, root, *, error: Exception | None = None):
        super().__init__(root)
        self.error = error
        self.locked_profiles = []

    def lock_target_client_size(self, profile_name):
        self.locked_profiles.append(profile_name)
        if self.error is not None:
            raise self.error


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


class _UnknownSuitAwareRecognitionStub:
    def __init__(self) -> None:
        self.allow_unknown_suit = None

    def recognize(self, image, *, allow_unknown_suit=False):
        del image
        self.allow_unknown_suit = allow_unknown_suit
        return _initial_recognition(())


def test_waiting_scan_keeps_unknown_suits_in_the_initial_hand(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    recognition = _UnknownSuitAwareRecognitionStub()
    controller.recognition_service = recognition
    snapshot = SimpleNamespace(image=np.zeros((1, 1, 3), dtype=np.uint8))

    result, returned_snapshot = controller._recognize_waiting_frame(snapshot)

    assert recognition.allow_unknown_suit is True
    assert result.my_hand == ()
    assert returned_snapshot is snapshot


def test_listener_locks_target_client_before_starting_waiting_capture(
    tmp_path,
    monkeypatch,
):
    _app()
    capture = _LockingCaptureServiceStub(tmp_path)
    controller = LiveAssistantController(capture)
    started = []
    monkeypatch.setattr(controller, "_start_danzero_warmup", lambda: None)
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: started.append(True))

    assert controller.start_listening() is True
    assert capture.locked_profiles == ["tencent_daguandan"]
    assert started == [True]


def test_listener_does_not_start_when_target_client_lock_fails(tmp_path, monkeypatch):
    _app()
    capture = _LockingCaptureServiceStub(tmp_path, error=RuntimeError("window denied resize"))
    controller = LiveAssistantController(capture)
    errors = []
    started = []
    controller.error.connect(errors.append)
    monkeypatch.setattr(controller, "_start_danzero_warmup", lambda: None)
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: started.append(True))

    assert controller.start_listening() is False
    assert capture.locked_profiles == ["tencent_daguandan"]
    assert controller._listening_enabled is False
    assert started == []
    assert errors == ["无法锁定牌桌客户区尺寸：window denied resize"]


def test_listener_starts_session_after_two_identical_complete_hands(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    hand = tuple(f"{rank}{suit}" for rank in ("3", "4", "5", "6", "7", "8", "9") for suit in "SHCD")[:27]
    started = []
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    monkeypatch.setattr(
        controller,
        "_start_detected_session",
        lambda result: started.append((result.round_level, result.my_hand)),
    )

    controller._consume_waiting_recognition(_initial_recognition(hand), None)
    controller._consume_waiting_recognition(_initial_recognition(hand), None)

    assert started == [("2", hand)]


def test_listener_keeps_rejected_initial_state_in_memory(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    hand = tuple(f"{rank}{suit}" for rank in ("3", "4", "5", "6", "7", "8", "9") for suit in "SHCD")[:27]

    controller._consume_waiting_recognition(
        _initial_recognition(hand, round_level=""),
        SimpleNamespace(captured_at=None),
    )

    assert controller._waiting_candidate is None


def test_listener_does_not_create_a_recording_from_settlement_controls(
    tmp_path,
):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True

    controller._consume_waiting_recognition(
        SimpleNamespace(
            my_hand=(),
            round_level=None,
            buttons=("change_table", "continue_game"),
        ),
        SimpleNamespace(),
    )

    assert controller._waiting_candidate is None
    assert controller._table_anchor_observed is False


def test_stop_listener_discards_an_unconfirmed_initial_candidate(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._waiting_candidate = object()  # type: ignore[assignment]

    controller.stop_listening()

    assert controller._waiting_candidate is None


def test_listener_does_not_start_session_when_complete_hand_changes(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    first = tuple(f"{rank}{suit}" for rank in ("3", "4", "5", "6", "7", "8", "9") for suit in "SHCD")[:27]
    second = first[:-1] + ("10S",)
    started = []
    controller._listening_enabled = True
    controller._table_anchor_observed = True
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
    controller._table_anchor_observed = True
    monkeypatch.setattr(controller, "_start_detected_session", lambda result: started.append(result))

    controller._consume_waiting_recognition(_initial_recognition(first), None)
    controller._consume_waiting_recognition(_initial_recognition(tuple(reversed(first))), None)

    assert len(started) == 1


def test_listener_starts_with_multiple_unknown_suits_of_the_same_rank(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    hand = (
        "5?", "5?", "5?",
        "3S", "3H", "3C", "3D",
        "4S", "4H", "4C", "4D",
        "6S", "6H", "6C", "6D",
        "7S", "7H", "7C", "7D",
        "8S", "8H", "8C", "8D",
        "9S", "9H", "9C", "9D",
    )
    started = []
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    monkeypatch.setattr(controller, "_start_detected_session", lambda result: started.append(result))

    controller._consume_waiting_recognition(_initial_recognition(hand), None)
    controller._consume_waiting_recognition(_initial_recognition(hand), None)

    assert len(started) == 1


def test_listener_does_not_start_before_the_table_anchor(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    started = []
    controller._listening_enabled = True
    monkeypatch.setattr(controller, "_table_anchor_score", lambda _snapshot: 0.8499)
    monkeypatch.setattr(controller, "_start_detected_session", lambda result: started.append(result))

    controller._consume_waiting_recognition(_initial_recognition(hand), SimpleNamespace())
    controller._consume_waiting_recognition(_initial_recognition(hand), SimpleNamespace())

    assert started == []
    assert controller._waiting_candidate is None


def test_listener_starts_after_a_single_085_table_anchor_frame(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    started = []
    controller._listening_enabled = True
    monkeypatch.setattr(controller, "_table_anchor_score", lambda _snapshot: 0.85)
    monkeypatch.setattr(controller, "_start_detected_session", lambda result: started.append(result))

    controller._consume_waiting_recognition(_initial_recognition(hand), SimpleNamespace())
    controller._consume_waiting_recognition(_initial_recognition(hand), SimpleNamespace())

    assert len(started) == 1


def test_listener_keeps_a_late_unanchored_game_out_of_the_state_machine(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    controller._listening_enabled = True
    controller._table_anchor_observed = True

    controller._consume_waiting_recognition(
        SimpleNamespace(
            my_hand=hand,
            round_level="2",
            lead_player="right",
            current_player="self",
            events=(),
        ),
        SimpleNamespace(captured_at=None),
    )

    assert controller._waiting_candidate is None


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


def _preselection_frame():
    standardization = StandardizationResult(
        image=np.zeros((100, 200, 3), dtype=np.uint8),
        source_size=(200, 100),
        source_viewport=Box(0, 0, 200, 100),
        content_box=Box(0, 0, 200, 100),
        scale=1.0,
        padding=(0, 0, 0, 0),
        aspect_error=0.0,
        aspect_compatible=True,
    )
    return FrameSnapshot(
        CapturedStandardizedFrame(
            standardization=standardization,
            rect=ClientRect(0, 0, 200, 100),
            backend="printwindow",
            dpi=96,
            window_title="game",
        )
    )


class _TokenAnalysisOrchestrator:
    def __init__(self, session_id):
        self.snapshot = SimpleNamespace(session_id=session_id)
        self.calls = 0
        self.trace_contexts = []

    def analyze_frame(self, _image, *, monotonic_ms, trace_context):
        self.calls += 1
        self.trace_contexts.append((monotonic_ms, trace_context))
        return LiveUpdate(
            status="running",
            snapshot=self.snapshot,
        )


def test_stale_analysis_task_cannot_call_replacement_orchestrator(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    old = _TokenAnalysisOrchestrator("old")
    old_token = _LiveRunToken(old, "old", 1, 1)
    controller.orchestrator = old  # type: ignore[assignment]
    controller._active_live_token = old_token
    controller._capture_generation = 1
    replacement = _TokenAnalysisOrchestrator("replacement")
    replacement_token = _LiveRunToken(replacement, "replacement", 2, 2)
    controller.orchestrator = replacement  # type: ignore[assignment]
    controller._active_live_token = replacement_token
    controller._capture_generation = 2

    result = controller._analyze_live_frame(
        old_token,
        _AnalysisFrameTask(old_token, _preselection_frame(), 7, 123),
    )

    assert result is None
    assert old.calls == 0
    assert replacement.calls == 0


def test_current_analysis_task_uses_its_immutable_token_and_capture_metadata(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    orchestrator = _TokenAnalysisOrchestrator("session")
    token = _LiveRunToken(orchestrator, "session", 4, 9)
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    controller._active_live_token = token
    controller._capture_generation = 9

    result = controller._analyze_live_frame(
        token,
        _AnalysisFrameTask(token, _preselection_frame(), 17, 456),
    )

    assert isinstance(result, LiveUpdate)
    assert orchestrator.calls == 1
    monotonic_ms, trace_context = orchestrator.trace_contexts[-1]
    assert monotonic_ms == 456
    assert trace_context == {
        "worker_token": {"session_id": "session", "nonce": 4, "generation": 9},
        "capture_seq": 17,
        "captured_ms": 456,
    }


def test_controller_schedules_visible_non_pass_advice_once_per_request(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._latest_live_frame = _preselection_frame()
    controller._latest_live_frame_generation = controller._capture_generation
    snapshot = SimpleNamespace(
        session_id="session",
        turn_id=7,
        revision=8,
        current_player="self",
        my_hand=("3S",),
    )
    key = AdviceRequestKey("session", 7, 8)
    advice = LocalAdvice(
        strategy="fabledan",
        cards=("3S",),
        play_type="Single",
        is_pass=False,
        state_revision=8,
        elapsed_ms=1.0,
        request_id=key.request_id,
        engine_input={},
        timings={},
    )
    update = LiveUpdate(
        status="running",
        snapshot=snapshot,
        advice=LiveAdvice(key=key, status="ready", visible=True, advice=advice),
    )
    scheduled = []
    monkeypatch.setattr(
        controller,
        "_start_hand_preselection_recognition",
        lambda task: scheduled.append(task),
    )

    controller._schedule_hand_preselection(update)
    controller._schedule_hand_preselection(update)

    assert len(scheduled) == 1
    assert scheduled[0].advice_cards == ("3S",)


def test_controller_never_schedules_pass_or_hidden_advice(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._latest_live_frame = _preselection_frame()
    snapshot = SimpleNamespace(
        session_id="session",
        turn_id=7,
        revision=8,
        current_player="self",
        my_hand=("3S",),
    )
    key = AdviceRequestKey("session", 7, 8)
    scheduled = []
    monkeypatch.setattr(
        controller,
        "_start_hand_preselection_recognition",
        lambda task: scheduled.append(task),
    )
    for visible, is_pass in ((False, False), (True, True)):
        advice = LocalAdvice(
            strategy="fabledan",
            cards=(),
            play_type="PASS",
            is_pass=is_pass,
            state_revision=8,
            elapsed_ms=1.0,
            request_id=key.request_id,
            engine_input={},
            timings={},
        )
        controller._schedule_hand_preselection(
            LiveUpdate(
                status="running",
                snapshot=snapshot,
                advice=LiveAdvice(
                    key=key,
                    status="ready",
                    visible=visible,
                    advice=advice,
                ),
            )
        )

    controller._schedule_hand_preselection(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=LiveAdvice(
                key=key,
                status="withheld",
                visible=True,
                advice=LocalAdvice(
                    strategy="fabledan",
                    cards=("3S",),
                    play_type="Single",
                    is_pass=False,
                    state_revision=8,
                    elapsed_ms=1.0,
                    request_id=key.request_id,
                    engine_input={},
                    timings={},
                ),
            ),
        )
    )

    assert scheduled == []
