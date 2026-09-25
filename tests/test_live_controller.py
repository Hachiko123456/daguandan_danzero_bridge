from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.application.opening_readiness import (
    OpeningReadinessCode,
    report_for_error,
    report_for_phase,
)
from daguandan_bridge.gui.live_controller import (
    LiveAssistantController,
    _AnalysisDelivery,
    _AnalysisFrameTask,
    _LiveRunToken,
    _WaitingAnalysisTask,
    _WaitingRecognitionEnvelope,
)
from daguandan_bridge.gui.recording_dispatcher import RecordingFrame
from daguandan_bridge.opening_gate import ListeningPageSignal, serialized_result
from daguandan_bridge.capture_service import FrameSnapshot, LiveCaptureInterrupted
from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.image_io import StandardizationResult
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.live.orchestrator import AdviceRequestKey, LiveAdvice, LiveUpdate
from daguandan_bridge.models import Box, ClientRect
from daguandan_bridge.window_capture import CapturedStandardizedFrame


def _app():
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, *, timeout=3.0):
    app = _app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


class _CaptureServiceStub:
    def __init__(self, root):
        self.profiles_root = root


def _isolated_controller(root):
    return LiveAssistantController(
        _CaptureServiceStub(root),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )


class _LockingCaptureServiceStub(_CaptureServiceStub):
    def __init__(self, root, *, error: Exception | None = None):
        super().__init__(root)
        self.error = error
        self.locked_profiles = []

    def lock_target_client_size(self, profile_name):
        self.locked_profiles.append(profile_name)
        if self.error is not None:
            raise self.error


class _SourceStub:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _ReopeningCaptureServiceStub(_CaptureServiceStub):
    def __init__(self, root, *, open_error=None):
        super().__init__(root)
        self.open_error = open_error
        self.opened_profiles = []
        self.sources = []

    def open_live_source(self, profile_name):
        self.opened_profiles.append(profile_name)
        if self.open_error is not None:
            raise self.open_error
        source = _SourceStub()
        self.sources.append(source)
        return source


class _WarmAdvisor:
    strategy_id = "fabledan"

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
    capture = _ReopeningCaptureServiceStub(tmp_path)
    controller = LiveAssistantController(capture)
    orchestrator = _PauseOrchestrator()
    old_worker = _SlowCaptureWorker()
    old_source = _SourceStub()
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    controller._capture_worker = old_worker  # type: ignore[assignment]
    controller._live_source = old_source
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
    assert old_source.closed is True
    assert capture.opened_profiles == ["tencent_daguandan"]
    assert controller._live_source is capture.sources[0]
    assert orchestrator.status == "running"


def test_resume_open_failure_does_not_resume_state_or_start_workers(
    tmp_path,
    monkeypatch,
):
    _app()
    capture = _ReopeningCaptureServiceStub(
        tmp_path,
        open_error=RuntimeError("window is still minimized"),
    )
    controller = LiveAssistantController(capture)
    orchestrator = _PauseOrchestrator()
    orchestrator.status = "paused"
    old_source = _SourceStub()
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    controller._live_source = old_source
    errors = []
    started = []
    controller.error.connect(errors.append)
    monkeypatch.setattr(controller, "_start_analysis_worker", lambda: started.append("analysis"))
    monkeypatch.setattr(controller, "_start_capture_worker", lambda: started.append("capture"))

    controller.resume()

    assert old_source.closed is True
    assert orchestrator.status == "paused"
    assert controller._live_source is None
    assert started == []
    assert errors == ["重新打开采集源失败：window is still minimized"]


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



def _opening_recognition(hand, *, round_level="2", cards=("2C",)):
    return serialized_result(
        round_level=round_level,
        hand=hand,
        lead_player="left",
        current_player="self",
        events=(
            {
                "player": "left",
                "cards": tuple(cards),
                "is_pass": False,
                "confidence": 0.95,
                "source": "test-opening",
            },
        ),
    )

@pytest.mark.parametrize("phase", ["opening_seed_invalid", "confirming_opening"])
def test_opening_title_is_compact_without_losing_structured_state(tmp_path, phase):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._waiting_generation = 3
    statuses = []
    controller.listening_status.connect(statuses.append)
    controller._publish_opening_status(phase, _initial_recognition(("2C",) * 27))
    assert len(statuses) == 1
    status = statuses[0]
    assert status["state"] == "opening"
    assert status["phase"] == phase
    assert status["hand_count"] == 27
    assert status["generation"] == 3
    assert status["status"] == "WAIT"
    assert status["primary_reason"] == "OPENING_UNRESOLVED"
    assert status["report"].primary_reason.value == "OPENING_UNRESOLVED"
    assert status["readiness"]["compact_allowed"] is False
    assert status["suggested_action"]


def test_staged_opening_publishes_actual_progress_and_starts_once_with_action(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    statuses, started = [], []
    controller.listening_status.connect(statuses.append)
    monkeypatch.setattr(controller, "start_session", lambda **kw: started.append(kw) or True)
    hand = tuple(f"{r}{s}" for r in ("3", "4", "5", "6", "7", "8", "9") for s in "SHCD")[:27]
    from daguandan_bridge.opening_gate import serialized_result
    for index, confidence in enumerate((.92, .920001, .94)):
        result = serialized_result(
            round_level="5", hand=hand, lead_player="left", current_player="self",
            events=({"player": "left", "cards": ("2C",), "is_pass": False,
                     "confidence": confidence, "source": f"template:{index}"},),
        )
        task = _WaitingAnalysisTask(SimpleNamespace(captured_monotonic_ms=100 + index * 500), 0)
        controller._consume_waiting_recognition(result, task)
    assert len(started) == 1
    assert started[0]["lead_player"] == "left"
    assert started[0]["opening_action"] is not None
    assert str(started[0]["opening_action"].actor) == "left"
    assert str(started[0]["opening_action"].next_player) == "self"
    assert statuses[0]["phase"] == "confirming_hand"
    assert statuses[-1]["phase"] == "ready"
    assert all("重新连接" not in status["message"] for status in statuses)


def test_off_table_probe_skips_full_recognition_and_media_for_long_idle(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    controller.recording_mode = "all"
    recording = _ListenerRecordingStub()
    factory = _ListenerRecordingFactory(recording)
    controller.session_factory = factory

    class Recognizer:
        full_calls = 0
        def recognize_listening_page(self, image):
            return ListeningPageSignal("settlement", .2, ("continue_game", "change_table"))
        def recognize(self, image, **kwargs):
            self.full_calls += 1
            raise AssertionError("settlement must not run hand recognition")

    recognizer = Recognizer()
    controller.recognition_service = recognizer
    snapshot = SimpleNamespace(image=np.zeros((1, 1, 3), dtype=np.uint8), captured_monotonic_ms=100)
    value, envelope = controller._recognize_waiting_frame(_WaitingAnalysisTask(snapshot, 0))
    for _ in range(1800):
        controller._consume_waiting_recognition(value, envelope)
    assert recognizer.full_calls == 0
    assert factory.started_with == []
    assert recording.frames == []
    assert controller.orchestrator is None
    assert controller._listening_enabled
    controller.opening_evidence.close(.2)


def test_finish_discards_bounded_recording_tail_before_sealing_without_orphan(
    tmp_path, monkeypatch
):
    app = _app()
    write_entered = threading.Event()
    release_write = threading.Event()
    pending_discarded = threading.Event()
    finish_called = threading.Event()

    class Recorder:
        frame_count = 1
        dropped_frames = 0
        session_directory = tmp_path

        def close(self):
            return RecordingResult(
                tmp_path / "game.avi", tmp_path / "frames.jsonl", 1, 0,
                integrity={"status": "PASS", "issues": []},
            )

    class Store:
        def __init__(self):
            self.metadata = []

        def update_session_metadata(self, value):
            self.metadata.append(value)

    class Orchestrator:
        status = "running"
        snapshot = SimpleNamespace(session_id="recording-finish")

        def __init__(self):
            self.recorder = Recorder()
            self.store = Store()
            self.recording_result = None

        def begin_finalizing(self):
            self.status = "finalizing"

        def record_frame(self, image, *, monotonic_ms, wall_time):
            write_entered.set()
            assert release_write.wait(2)

        def finish(self):
            self.recording_result = self.recorder.close()
            finish_called.set()
            self.status = "sealed"
            return SimpleNamespace(status="sealed")

    controller = _isolated_controller(tmp_path)
    orchestrator = Orchestrator()
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    token = controller._activate_live_token(orchestrator)  # type: ignore[arg-type]
    original_drop = controller._recording_dropped

    def watched_drop(frame, reason):
        original_drop(frame, reason)
        if reason == "close" and (
            token.pipeline_timing.snapshot()["counters"].get(
                "recording_dispatcher_drop_close"
            ) == 2
        ):
            pending_discarded.set()

    monkeypatch.setattr(controller, "_recording_dropped", watched_drop)
    controller._start_recording_dispatcher(token)
    for sequence in (1, 2, 3):
        assert controller._recording_dispatcher.submit(
            RecordingFrame(object(), sequence * 100, f"frame-{sequence}", sequence, token)
        )
    assert write_entered.wait(1)

    controller.finish()
    assert pending_discarded.wait(1)
    assert not finish_called.is_set()
    release_write.set()
    assert controller._finish_thread.wait(2_000)
    app.processEvents()

    counters = token.pipeline_timing.snapshot()["counters"]
    assert finish_called.is_set()
    assert counters["recording_dispatcher_close_discarded"] == 2
    assert counters["recording_dispatcher_closed"] == 1
    assert orchestrator.recording_result.dropped_frames == 2
    audit = orchestrator.recording_result.integrity["recording_dispatcher"]
    assert [item["capture_sequence"] for item in audit["drops"]] == [2, 3]
    dispatcher_metadata = next(
        item["recording_dispatcher"]
        for item in orchestrator.store.metadata
        if "recording_dispatcher" in item
    )
    assert dispatcher_metadata == audit
    cadence_metadata = next(
        item["recording_cadence"]
        for item in orchestrator.store.metadata
        if "recording_cadence" in item
    )
    assert cadence_metadata["target_fps"] == 10
    assert controller._recording_dispatcher is None
    assert not any(
        thread.name == "live-recording-dispatcher" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_complete_table_recording_stops_once_on_settlement_and_rearms_next_table(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller.recording_mode = "all"
    recording = _ListenerRecordingStub()
    factory = _ListenerRecordingFactory(recording)
    controller.session_factory = factory
    snapshot = SimpleNamespace(image=np.zeros((1, 1, 3), dtype=np.uint8), captured_at=datetime.now().astimezone())
    assert controller._start_listener_recording()
    assert factory.started_with == []
    controller._apply_listening_page(ListeningPageSignal("table", .95), snapshot)
    assert len(recording.frames) == 1
    controller._apply_listening_page(ListeningPageSignal("settlement", .1), snapshot)
    for _ in range(10):
        controller._record_listener_frame(snapshot)
        controller._apply_listening_page(ListeningPageSignal("settlement", .1), snapshot)
    assert recording.closed_with == ["page_settlement"]
    assert len(recording.frames) == 1
    controller._apply_listening_page(ListeningPageSignal("table", .95), snapshot)
    assert len(factory.started_with) == 2
    controller.stop_listening()


def test_stale_page_envelope_cannot_close_new_generation_recording(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    controller._waiting_generation = 2
    recording = _ListenerRecordingStub()
    controller._listener_recording = recording
    controller._listening_page = ListeningPageSignal("table", .95)
    envelope = _WaitingRecognitionEnvelope(SimpleNamespace(), 1, None, None, ListeningPageSignal("settlement", .1))
    controller._consume_waiting_recognition(_initial_recognition(()), envelope)
    assert controller._listener_recording is recording
    assert recording.closed_with == []
    controller.stop_listening()


def test_capacity_notice_does_not_stop_recognition_or_emit_capture_error(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    errors, warnings = [], []
    controller.error.connect(errors.append)
    controller.recording_status.connect(warnings.append)
    controller._publish_recording_warning(SimpleNamespace(reason="recording_capacity_reached"))
    assert errors == []
    assert controller._listening_enabled
    assert warnings[0]["reason"] == "recording_capacity_reached"


class _UnknownSuitAwareRecognitionStub:
    def __init__(self) -> None:
        self.allow_unknown_suit = None

    def recognize(self, image, *, allow_unknown_suit=False):
        del image
        self.allow_unknown_suit = allow_unknown_suit
        return _initial_recognition(())


class _ListenerRecordingStub:
    def __init__(self) -> None:
        self.frames = []
        self.recognitions = []
        self.closed_with = []

    def record_frame(self, image, *, monotonic_ms, wall_time):
        self.frames.append((image, monotonic_ms, wall_time))

    def record_recognition(self, result):
        self.recognitions.append(result)

    def close(self, *, reason):
        self.closed_with.append(reason)


class _ListenerRecordingFactory:
    def __init__(self, recording):
        self.recording = recording
        self.started_with = []

    def start_listener_recording(self, *, recognition_strategy):
        self.started_with.append(recognition_strategy)
        return self.recording


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


def test_e002_waiting_worker_error_keeps_typed_exception_and_input_snapshot(tmp_path):
    _app()

    class TypedRecognitionError(RuntimeError):
        code = "RESOURCE-MISMATCH"

    class FailingRecognizer:
        calls = 0

        def recognize(self, _image, *, allow_unknown_suit=False):
            assert allow_unknown_suit is True
            self.calls += 1
            raise TypedRecognitionError("template unavailable")

    class EvidenceSpy:
        def __init__(self):
            self.failures = []

        def observe_failure(self, error, **kwargs):
            self.failures.append((error, kwargs))

    spy = EvidenceSpy()
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        opening_evidence_monitor=spy,
    )
    recognizer = FailingRecognizer()
    controller.recognition_service = recognizer
    snapshot = SimpleNamespace(image=np.zeros((2, 2, 3), dtype=np.uint8))
    errors = []
    controller.error.connect(errors.append)

    with pytest.raises(RuntimeError) as captured:
        controller._recognize_waiting_frame(snapshot)
    controller._accept_waiting_recognition_error(captured.value)
    assert controller.opening_evidence.flush(2)

    assert recognizer.calls == 1
    assert errors == ["template unavailable"]
    assert len(spy.failures) == 1
    original, context = spy.failures[0]
    assert isinstance(original, TypedRecognitionError)
    assert original.code == "RESOURCE-MISMATCH"
    assert context == {"stage": "recognition", "snapshot": snapshot}
    controller.shutdown()


def test_full_recording_keeps_listener_frames_when_initial_hand_is_empty(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    recording = _ListenerRecordingStub()
    factory = _ListenerRecordingFactory(recording)
    controller.session_factory = factory  # type: ignore[assignment]
    controller.recording_mode = "all"
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    controller._listening_page = ListeningPageSignal("table", 1.0)

    assert controller._start_listener_recording() is True
    snapshot = SimpleNamespace(
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        captured_at=datetime.now().astimezone(),
    )
    controller._record_listener_frame(snapshot)
    controller._consume_waiting_recognition(_initial_recognition(()), snapshot)
    controller.stop_listening()

    assert factory.started_with == ["two_valid_streak"]
    assert len(recording.frames) == 1
    assert recording.recognitions == [_initial_recognition(())]
    assert recording.closed_with == ["listener_stopped"]


def test_full_recording_seals_listener_clip_before_starting_a_confirmed_game(
    tmp_path,
    monkeypatch,
):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    recording = _ListenerRecordingStub()
    controller._listener_recording = recording
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    started = []
    monkeypatch.setattr(
        controller,
        "_start_pending_auto_session",
        lambda: started.append(True),
    )

    controller._consume_waiting_recognition(_opening_recognition(hand), None)
    controller._consume_waiting_recognition(_opening_recognition(hand), None)

    assert recording.closed_with == ["initial_state_confirmed"]
    assert started == [True]


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

    controller._consume_waiting_recognition(_opening_recognition(hand), None)
    controller._consume_waiting_recognition(_opening_recognition(hand), None)

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

    controller._consume_waiting_recognition(_opening_recognition(first), None)
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

    controller._consume_waiting_recognition(_opening_recognition(first), None)
    controller._consume_waiting_recognition(_opening_recognition(tuple(reversed(first))), None)

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

    controller._consume_waiting_recognition(_opening_recognition(hand), SimpleNamespace())
    controller._consume_waiting_recognition(_opening_recognition(hand), SimpleNamespace())

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

    controller._consume_waiting_recognition(_opening_recognition(hand), SimpleNamespace())
    controller._consume_waiting_recognition(_opening_recognition(hand), SimpleNamespace())

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


class _GeometrySource:
    def __init__(self, snapshots=(), *, error=None, state=None):
        self.snapshots = tuple(snapshots)
        self.error = error
        self.state = dict(state or {})
        self.capture_calls = 0
        self.closed = False

    def capture(self):
        self.capture_calls += 1
        if self.error is not None:
            raise self.error
        if not self.snapshots:
            raise RuntimeError("no scripted recovery frame")
        return self.snapshots[min(self.capture_calls - 1, len(self.snapshots) - 1)]

    def diagnostic_state(self):
        return dict(self.state)

    def close(self):
        self.closed = True


class _GeometryRecoveryCaptureService(_CaptureServiceStub):
    def __init__(self, root, sources):
        super().__init__(root)
        self.sources = list(sources)
        self.open_calls = 0
        self.lock_calls = 0

    def lock_target_client_size(self, _profile_name):
        self.lock_calls += 1
        return ClientRect(30, 40, 1280, 764)

    def open_live_source(self, _profile_name):
        self.open_calls += 1
        if not self.sources:
            raise RuntimeError("no scripted source")
        return self.sources.pop(0)


class _BlockingAnalysisWorker:
    def __init__(self):
        self.release = threading.Event()
        self.stop_calls = []

    def stop(self, *, timeout=None):
        self.stop_calls.append(timeout)
        if timeout is None:
            self.release.wait()
            return True
        return self.release.wait(float(timeout))


@pytest.mark.parametrize(
    ("change_type", "old_rect", "new_rect", "expected_lock_calls"),
    (
        ("move", [10, 20, 1280, 764], [30, 40, 1280, 764], 0),
        ("resize", [10, 20, 1280, 764], [10, 20, 1100, 700], 1),
    ),
)
def test_geometry_move_or_resize_recovers_then_allows_opening_gate(
    tmp_path,
    monkeypatch,
    change_type,
    old_rect,
    new_rect,
    expected_lock_calls,
):
    app = _app()
    frame = _preselection_frame()
    recovered_source = _GeometrySource(
        (frame, frame),
        state={
            "rect": [30, 40, 1280, 764],
            "dpi": 96,
            "backend": "printwindow",
        },
    )
    capture = _GeometryRecoveryCaptureService(tmp_path, (recovered_source,))
    controller = LiveAssistantController(capture)
    controller._GEOMETRY_STABLE_SAMPLE_DELAY_SEC = 0
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    recording = _ListenerRecordingStub()
    controller._listener_recording = recording
    old_source = _GeometrySource()
    old_worker = object()
    controller._waiting_source = old_source
    controller._waiting_capture_worker = old_worker  # type: ignore[assignment]
    restarted = []
    monkeypatch.setattr(
        controller,
        "_start_waiting_workers",
        lambda: restarted.append(controller._recovered_waiting_source),
    )
    started = []
    monkeypatch.setattr(
        controller,
        "_start_detected_session",
        lambda result: started.append(result),
    )
    monkeypatch.setattr(controller, "_table_anchor_score", lambda _snapshot: 1.0)
    incident = LiveCaptureInterrupted(
        "geometry changed",
        code="GEOMETRY-CHANGED",
        details={
            "old_rect": old_rect,
            "new_rect": new_rect,
            "change_types": [change_type],
            "capture_backend": "printwindow",
            "old_dpi": 96,
            "new_dpi": 96,
        },
    )

    controller._accept_waiting_error(incident, controller._waiting_generation)
    recovery_generation = controller._waiting_generation
    controller._waiting_capture_finished(old_worker)  # type: ignore[arg-type]

    assert _wait_until(lambda: not controller._geometry_recovery_active)
    assert restarted == [recovered_source]
    assert controller._listening_enabled is True
    assert old_source.closed is True
    assert recovered_source.closed is False
    assert recording.closed_with == []
    # Recovery samples are not yet classified as a table; keep them in memory.
    assert len(recording.frames) == 0
    assert capture.lock_calls == expected_lock_calls
    assert recovery_generation > 0

    hand = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    task = _WaitingAnalysisTask(frame, recovery_generation)
    opening = SimpleNamespace(
        my_hand=hand,
        round_level="2",
        lead_player="left",
        current_player="self",
        events=(SimpleNamespace(
            player="left", cards=("2C",), is_pass=False,
            confidence=0.95, source="geometry-recovery",
        ),),
        buttons=(),
    )
    controller._consume_waiting_recognition(opening, task)
    next_frame = replace(frame, captured_monotonic_ms=frame.captured_monotonic_ms + 200)
    controller._consume_waiting_recognition(
        opening, _WaitingAnalysisTask(next_frame, recovery_generation)
    )
    app.processEvents()

    assert len(started) == 1
    controller.stop_listening()


@pytest.mark.parametrize(
    ("code", "change_types", "backend"),
    (
        ("WINDOW-MINIMIZED", ["minimized"], "printwindow"),
        ("GEOMETRY-CHANGED", ["dpi"], "printwindow"),
        ("GEOMETRY-CHANGED", ["move"], "screen"),
    ),
)
def test_minimized_or_dpi_geometry_path_recovers_with_stable_samples(
    tmp_path,
    monkeypatch,
    code,
    change_types,
    backend,
):
    frame = _preselection_frame()
    recovered_source = _GeometrySource((frame, frame))
    capture = _GeometryRecoveryCaptureService(tmp_path, (recovered_source,))
    controller = LiveAssistantController(capture)
    controller._GEOMETRY_STABLE_SAMPLE_DELAY_SEC = 0
    controller._listening_enabled = True
    old_source = _GeometrySource()
    old_worker = object()
    controller._waiting_source = old_source
    controller._waiting_capture_worker = old_worker  # type: ignore[assignment]
    restarted = []
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: restarted.append(True))

    controller._accept_waiting_error(
        LiveCaptureInterrupted(
            "window changed",
            code=code,
            details={
                "old_rect": [0, 0, 1280, 764],
                "new_rect": None,
                "change_types": change_types,
                "capture_backend": backend,
                "old_dpi": 96,
                "new_dpi": 144 if change_types == ["dpi"] else None,
            },
        ),
        controller._waiting_generation,
    )
    controller._waiting_capture_finished(old_worker)  # type: ignore[arg-type]

    assert _wait_until(lambda: not controller._geometry_recovery_active)
    assert restarted == [True]
    assert recovered_source.capture_calls == 2
    assert capture.lock_calls == 1
    controller.stop_listening()


def test_geometry_recovery_waits_for_old_recognition_worker_before_restarting(
    tmp_path,
    monkeypatch,
):
    _app()
    frame = _preselection_frame()
    recovered_source = _GeometrySource((frame, frame))
    capture = _GeometryRecoveryCaptureService(tmp_path, (recovered_source,))
    controller = LiveAssistantController(capture)
    controller._GEOMETRY_STABLE_SAMPLE_DELAY_SEC = 0
    controller._GEOMETRY_ANALYSIS_DRAIN_TIMEOUT_SEC = 2.0
    controller._listening_enabled = True
    old_analysis = _BlockingAnalysisWorker()
    controller._waiting_analysis_worker = old_analysis  # type: ignore[assignment]
    old_worker = object()
    controller._waiting_source = _GeometrySource()
    controller._waiting_capture_worker = old_worker  # type: ignore[assignment]
    restarted_after_old_exit = []
    monkeypatch.setattr(
        controller,
        "_start_waiting_workers",
        lambda: restarted_after_old_exit.append(old_analysis.release.is_set()),
    )

    controller._accept_waiting_error(
        LiveCaptureInterrupted(
            "geometry changed",
            code="GEOMETRY-CHANGED",
            details={
                "change_types": ["move"],
                "capture_backend": "printwindow",
            },
        ),
        controller._waiting_generation,
    )
    controller._waiting_capture_finished(old_worker)  # type: ignore[arg-type]

    assert _wait_until(lambda: len(old_analysis.stop_calls) >= 2)
    assert restarted_after_old_exit == []
    assert capture.open_calls == 0

    old_analysis.release.set()
    assert _wait_until(lambda: restarted_after_old_exit == [True])
    assert capture.open_calls == 1
    controller.stop_listening()


def test_geometry_recovery_fails_when_old_recognition_worker_does_not_exit(
    tmp_path,
    monkeypatch,
):
    _app()
    capture = _GeometryRecoveryCaptureService(
        tmp_path,
        (_GeometrySource((_preselection_frame(), _preselection_frame())),),
    )
    controller = LiveAssistantController(capture)
    controller._GEOMETRY_ANALYSIS_DRAIN_TIMEOUT_SEC = 0.02
    controller._GEOMETRY_RECOVERY_DELAYS_MS = (0,)
    controller._listening_enabled = True
    old_analysis = _BlockingAnalysisWorker()
    controller._waiting_analysis_worker = old_analysis  # type: ignore[assignment]
    old_worker = object()
    controller._waiting_source = _GeometrySource()
    controller._waiting_capture_worker = old_worker  # type: ignore[assignment]
    restarted = []
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: restarted.append(True))

    controller._accept_waiting_error(
        LiveCaptureInterrupted("geometry changed", code="GEOMETRY-CHANGED"),
        controller._waiting_generation,
    )
    controller._waiting_capture_finished(old_worker)  # type: ignore[arg-type]

    assert _wait_until(lambda: not controller._geometry_recovery_active)
    assert controller._listening_enabled is False
    assert capture.open_calls == 0
    assert restarted == []
    assert controller._draining_waiting_analysis_worker is old_analysis
    old_analysis.release.set()
    controller.stop_listening()


def test_slow_waiting_recognition_from_old_generation_is_discarded(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    controller._table_anchor_observed = True
    old_generation = controller._waiting_generation
    frame = _preselection_frame()
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    started = []
    monkeypatch.setattr(controller, "_start_detected_session", started.append)

    controller._begin_geometry_recovery(
        LiveCaptureInterrupted(
            "moved while recognition was running",
            code="GEOMETRY-CHANGED",
        )
    )
    stale_task = _WaitingAnalysisTask(frame, old_generation)
    controller._consume_waiting_recognition(_initial_recognition(hand), stale_task)
    controller._consume_waiting_recognition(_initial_recognition(hand), stale_task)

    assert controller._waiting_candidate is None
    assert started == []
    assert controller._waiting_generation == old_generation + 1
    controller.stop_listening()


def test_continuous_geometry_changes_fail_after_finite_attempts_without_looping(
    tmp_path,
):
    _app()
    failures = [
        _GeometrySource(
            error=LiveCaptureInterrupted(
                f"still changing {index}",
                code="GEOMETRY-CHANGED",
                details={"change_types": ["resize"]},
            )
        )
        for index in range(3)
    ]
    capture = _GeometryRecoveryCaptureService(tmp_path, failures)
    controller = LiveAssistantController(capture)
    controller._GEOMETRY_RECOVERY_DELAYS_MS = (0, 0, 0)
    controller._GEOMETRY_STABLE_SAMPLE_DELAY_SEC = 0
    controller._listening_enabled = True
    recording = _ListenerRecordingStub()
    controller._listener_recording = recording
    old_worker = object()
    controller._waiting_source = _GeometrySource()
    controller._waiting_capture_worker = old_worker  # type: ignore[assignment]
    statuses = []
    controller.listening_status.connect(statuses.append)

    controller._accept_waiting_error(
        LiveCaptureInterrupted("geometry changed", code="GEOMETRY-CHANGED"),
        controller._waiting_generation,
    )
    controller._waiting_capture_finished(old_worker)  # type: ignore[arg-type]

    assert _wait_until(lambda: not controller._geometry_recovery_active)
    assert controller._listening_enabled is False
    assert controller._geometry_recovery_attempt_count == 3
    assert capture.open_calls == 3
    assert all(source.closed for source in failures)
    assert recording.closed_with == ["geometry_recovery_failed"]
    assert [status["state"] for status in statuses].count("failed") == 1
    assert controller._geometry_recovery_timer.isActive() is False
    assert controller._geometry_recovery_thread is None


def test_shutdown_cancels_geometry_stability_wait_and_closes_recovery_source(tmp_path):
    _app()
    frame = _preselection_frame()
    recovery_source = _GeometrySource((frame, frame))
    capture = _GeometryRecoveryCaptureService(tmp_path, (recovery_source,))
    controller = LiveAssistantController(capture)
    controller._GEOMETRY_STABLE_SAMPLE_DELAY_SEC = 10
    controller._listening_enabled = True
    old_worker = object()
    controller._waiting_source = _GeometrySource()
    controller._waiting_capture_worker = old_worker  # type: ignore[assignment]

    controller._accept_waiting_error(
        LiveCaptureInterrupted("geometry changed", code="GEOMETRY-CHANGED"),
        controller._waiting_generation,
    )
    controller._waiting_capture_finished(old_worker)  # type: ignore[arg-type]
    assert _wait_until(lambda: recovery_source.capture_calls >= 1)

    started = time.perf_counter()
    controller.shutdown()
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert recovery_source.closed is True
    assert controller._geometry_recovery_thread is None
    assert controller._geometry_recovery_timer.isActive() is False


def test_repeated_short_lived_geometry_recoveries_hit_cycle_circuit_breaker(
    tmp_path,
    monkeypatch,
):
    _app()
    frame = _preselection_frame()
    stable_sources = [_GeometrySource((frame, frame)) for _ in range(3)]
    capture = _GeometryRecoveryCaptureService(tmp_path, stable_sources)
    controller = LiveAssistantController(capture)
    controller._GEOMETRY_STABLE_SAMPLE_DELAY_SEC = 0
    controller._listening_enabled = True
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: None)

    for _cycle in range(3):
        old_worker = object()
        controller._waiting_source = _GeometrySource()
        controller._waiting_capture_worker = old_worker  # type: ignore[assignment]
        controller._accept_waiting_error(
            LiveCaptureInterrupted("geometry changed", code="GEOMETRY-CHANGED"),
            controller._waiting_generation,
        )
        controller._waiting_capture_finished(old_worker)  # type: ignore[arg-type]
        assert _wait_until(lambda: not controller._geometry_recovery_active)
        recovered, controller._recovered_waiting_source = (
            controller._recovered_waiting_source,
            None,
        )
        assert recovered is not None
        recovered.close()

    controller._accept_waiting_error(
        LiveCaptureInterrupted("geometry changed again", code="GEOMETRY-CHANGED"),
        controller._waiting_generation,
    )

    assert controller._listening_enabled is False
    assert controller._geometry_recovery_cycle_count == 4
    assert controller._geometry_recovery_thread is None
    assert controller._geometry_recovery_timer.isActive() is False


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
        _AnalysisFrameTask(token, replace(_preselection_frame(), captured_monotonic_ms=456), 17, 456),
    )

    assert isinstance(result, LiveUpdate)
    assert orchestrator.calls == 1
    monotonic_ms, trace_context = orchestrator.trace_contexts[-1]
    assert monotonic_ms == 456
    assert trace_context["worker_token"] == {
        "session_id": "session", "nonce": 4, "generation": 9,
    }
    assert trace_context["capture_seq"] == 17
    assert trace_context["captured_ms"] == 456
    assert trace_context["capture_generation"] == 9
    assert trace_context["frame_source"] == "canonical_envelope"
    assert trace_context["roi_version"] == "live-v2"


def test_pipeline_gui_delivery_is_queued_and_rejects_changed_generation(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    orchestrator = _TokenAnalysisOrchestrator("session")
    controller.orchestrator = orchestrator
    token = controller._activate_live_token(orchestrator)
    delivered = []
    controller.update_ready.connect(lambda update: delivered.append(threading.get_ident()))
    update = LiveUpdate(status="running", snapshot=orchestrator.snapshot)
    controller._accept_analysis_update(token, _AnalysisDelivery(token, update, time.monotonic_ns()))
    assert delivered == []
    assert _wait_until(lambda: bool(delivered))
    assert delivered == [threading.get_ident()]
    controller._accept_analysis_update(token, _AnalysisDelivery(token, update, time.monotonic_ns()))
    controller._invalidate_live_token()
    _app().processEvents()
    assert len(delivered) == 1
    assert token.pipeline_timing.snapshot()["counters"]["gui_stale_delivery"] == 1


@pytest.mark.parametrize("late_kind", ["older_sequence", "equal_sequence", "no_result", "older_revision", "unversioned"])
def test_controller_mailbox_keeps_newest_ready_before_window_drains(tmp_path, late_kind):
    from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    core = _TokenAnalysisOrchestrator("mailbox")
    controller.orchestrator = core
    token = controller._activate_live_token(core)
    window = RecommendationFloatWindow(controller)
    ready = _fault_test_update(core, generation=token.generation, sequence=10)
    late = replace(ready, advice=replace(ready.advice, status="withheld", visible=False,
                                        withhold_reason="turn_recovery_pending"))
    if late_kind == "no_result":
        late = None
    elif late_kind == "older_sequence":
        late = replace(late, update_sequence=9)
    elif late_kind == "older_revision":
        late = replace(late, update_sequence=11, snapshot=SimpleNamespace(
            **{**vars(ready.snapshot), "revision": ready.snapshot.revision - 1}))
    elif late_kind == "unversioned":
        late = replace(late, update_sequence=0)
    delivered = []
    controller.update_ready.connect(delivered.append)
    controller._accept_analysis_update(token, ready)
    controller._accept_analysis_update(token, late)
    assert delivered == []
    assert controller._pending_gui_delivery.update is ready
    _app().processEvents()
    assert delivered == [ready]
    assert window._card_badges
    window.hide()
    controller._invalidate_live_token()


def test_controller_mailbox_keeps_newer_block_and_none_cannot_erase_it(tmp_path):
    from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    core = _TokenAnalysisOrchestrator("mailbox")
    controller.orchestrator = core
    token = controller._activate_live_token(core)
    window = RecommendationFloatWindow(controller)
    ready = _fault_test_update(core, generation=token.generation, sequence=10)
    blocked = replace(ready, update_sequence=11, advice=replace(
        ready.advice, status="withheld", visible=False, withhold_reason="turn_recovery_pending"))
    controller._accept_analysis_update(token, ready)
    controller._accept_analysis_update(token, blocked)
    for _ in range(20):
        controller._accept_analysis_update(token, None)
        controller._accept_analysis_update(token, ready)
    assert controller._pending_gui_delivery.update is blocked
    _app().processEvents()
    assert not window._card_badges
    assert window.suggestion_label.text() == "确认中…"
    assert controller._pending_gui_delivery is None
    assert not controller._gui_delivery_scheduled
    window.hide()
    controller._invalidate_live_token()


def test_unowned_and_positive_old_callbacks_are_never_retagged(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    core = _TokenAnalysisOrchestrator("session")
    controller.orchestrator = core
    controller._capture_generation = 3
    token = controller._activate_live_token(core)
    unbound = _fault_test_update(core, generation=0)
    owned_old = replace(unbound, capture_generation=token.generation - 1)
    received = []
    controller.update_ready.connect(received.append)
    controller._queue_orchestrator_update(owned_old)
    controller._queue_orchestrator_update(unbound)
    _app().processEvents()
    assert received == []
    assert owned_old.capture_generation == token.generation - 1
    assert unbound.capture_generation == 0
    controller._invalidate_live_token()


def test_analysis_task_has_no_recording_admission_gate(tmp_path):
    _app()
    controller = _isolated_controller(tmp_path)
    orchestrator = _TokenAnalysisOrchestrator("session")
    controller.orchestrator = orchestrator
    token = controller._activate_live_token(orchestrator)
    task = _AnalysisFrameTask(
        token, _preselection_frame(), 1, 100,
        time.monotonic_ns(), time.monotonic_ns(),
    )

    result = controller._analyze_live_frame(token, task)

    assert isinstance(result, LiveUpdate)
    assert orchestrator.calls == 1
    assert "recording_admission_wait" not in token.pipeline_timing.snapshot()["stages"]


def test_recording_failure_notice_does_not_block_same_frame_analysis(tmp_path):
    _app()
    controller = _isolated_controller(tmp_path)
    orchestrator = _TokenAnalysisOrchestrator("session")
    controller.orchestrator = orchestrator
    controller._listening_enabled = True
    token = controller._activate_live_token(orchestrator)
    notices = []
    controller.recording_status.connect(notices.append)
    frame = RecordingFrame(object(), 100, "captured-at", 1, token)
    controller._recording_error(frame, OSError("codec failed"))
    task = _AnalysisFrameTask(
        token, _preselection_frame(), 1, 100,
        time.monotonic_ns(), time.monotonic_ns(),
    )

    result = controller._analyze_live_frame(token, task)

    assert isinstance(result, LiveUpdate)
    assert orchestrator.calls == 1
    assert notices[0]["reason"] == "recording_failed"
    assert notices[0]["captured_ms"] == 100
    assert controller._listening_enabled
    assert token.pipeline_timing.snapshot()["counters"]["recording_failed"] == 1


def _fault_test_update(orchestrator, *, generation=0, sequence=1):
    snapshot = SimpleNamespace(
        session_id=orchestrator.snapshot.session_id, current_player="self",
        turn_id=7, revision=8, finished_seats=frozenset(), trick_plays=(),
    )
    orchestrator.snapshot = snapshot
    advice = LiveAdvice(
        key=AdviceRequestKey(snapshot.session_id, 7, 8), status="ready", visible=True,
        advice=LocalAdvice(strategy="test", cards=("3S",), play_type="Single", is_pass=False,
                           state_revision=8, elapsed_ms=1, request_id="ADV-0007-0008", engine_input={}),
    )
    return LiveUpdate(status="running", snapshot=snapshot, advice=advice,
                      capture_generation=generation, update_sequence=sequence)


@pytest.mark.parametrize("kind", ["analysis", "capture", "occluded"])
def test_real_controller_fatal_incident_failure_clears_ui_until_new_generation(tmp_path, kind):
    from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    orchestrator = _TokenAnalysisOrchestrator("fault-game")
    orchestrator.status = "running"
    def fail_incident(*args, **kwargs):
        raise OSError("diagnostics disk unavailable")
    orchestrator.analysis_failed = fail_incident
    orchestrator.capture_interrupted = fail_incident
    controller.orchestrator = orchestrator
    token = controller._activate_live_token(orchestrator)
    window = RecommendationFloatWindow(controller)
    faults, errors = [], []
    controller.live_fault.connect(faults.append)
    controller.error.connect(errors.append)
    initial = _fault_test_update(orchestrator, generation=token.generation)
    controller._accept_analysis_update(token, initial)
    assert _wait_until(lambda: bool(window._card_badges))
    assert window._update_gate.generation == token.generation
    if kind == "analysis":
        controller._accept_analysis_error(token, RuntimeError("card matcher failed"))
    else:
        error = LiveCaptureInterrupted("capture failed", code="CAPTURE-OCCLUDED") if kind == "occluded" else RuntimeError("capture failed")
        controller._accept_live_error(token, error)
    assert _wait_until(lambda: bool(faults))
    assert faults == [{"session_id": "fault-game", "capture_generation": token.generation, "kind": kind}]
    assert not window._card_badges
    assert window.suggestion_label.text() == "识别已暂停"
    assert errors and "diagnostics disk unavailable" in errors[-1]
    assert controller._active_live_token is None
    controller.update_ready.emit(replace(initial, capture_generation=token.generation, update_sequence=2))
    _app().processEvents()
    assert not window._card_badges
    replacement = controller._activate_live_token(orchestrator)
    assert replacement.generation > token.generation
    controller._accept_analysis_update(replacement, replace(initial, update_sequence=3, capture_generation=replacement.generation))
    assert _wait_until(lambda: bool(window._card_badges))
    window.hide()
    controller._invalidate_live_token()


def test_old_or_queued_old_worker_fault_does_not_stop_replacement_capture(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    old = _TokenAnalysisOrchestrator("session")
    old.analysis_failed = lambda *args, **kwargs: pytest.fail("old core must not be called")
    controller.orchestrator = old
    old_token = controller._activate_live_token(old)
    faults, errors = [], []
    controller.live_fault.connect(faults.append)
    controller.error.connect(errors.append)
    controller._queue_fatal_worker_fault(old_token, kind="analysis", message="old queued error")
    replacement = _TokenAnalysisOrchestrator("session")
    controller.orchestrator = replacement
    replacement_token = controller._activate_live_token(replacement)
    controller._accept_analysis_error(old_token, RuntimeError("late analysis error"))
    controller._accept_live_error(old_token, RuntimeError("late capture error"))
    _app().processEvents()
    assert faults == errors == []
    assert controller._active_live_token is replacement_token
    controller._invalidate_live_token()


def test_non_concrete_runtime_reconnect_binds_and_resumes_new_generation(tmp_path, monkeypatch):
    _app()
    controller = LiveAssistantController(_ReopeningCaptureServiceStub(tmp_path))
    orchestrator = _TokenAnalysisOrchestrator("session")
    orchestrator.status = "running"
    update = _fault_test_update(orchestrator)
    calls = []
    bound_generation = [0]
    def pause():
        calls.append("pause")
        orchestrator.status = "paused"
    def resume(**kwargs):
        assert orchestrator.status == "paused"
        calls.append("resume")
        orchestrator.status = "running"
        return replace(update, capture_generation=bound_generation[0])
    def bind(generation):
        calls.append("bind")
        bound_generation[0] = generation
        return replace(update, capture_generation=generation)
    orchestrator.pause, orchestrator.resume = pause, resume
    orchestrator.bind_capture_generation = bind
    controller.orchestrator = orchestrator
    token = controller._activate_live_token(orchestrator)
    faults = []
    controller.live_fault.connect(faults.append)
    controller._queue_fatal_worker_fault(token, kind="analysis", message="failed incident")
    assert _wait_until(lambda: bool(faults))
    monkeypatch.setattr(controller, "_start_analysis_worker", lambda: None)
    monkeypatch.setattr(controller, "_start_capture_worker", lambda: None)
    updates = []
    controller.update_ready.connect(updates.append)
    controller.resume()
    assert calls == ["pause", "bind", "resume"]
    assert controller._active_live_token.generation > token.generation
    assert updates[-1].capture_generation == controller._active_live_token.generation
    assert controller._fatal_capture_session_id is None
    controller._invalidate_live_token()


@pytest.fixture
def real_controller_lifecycle(tmp_path, monkeypatch):
    from daguandan_bridge.domain.recognition import FastSignalResult, OpeningSignal, PlayRegionResult
    from daguandan_bridge.live.orchestrator import LiveOrchestrator
    from daguandan_bridge.live.recorder import InMemorySessionRecorder
    from daguandan_bridge.live.reducer import LiveReducer
    from daguandan_bridge.live.session_store import InMemoryLiveSessionStore
    _app()
    created, sources, worker_starts, calls, updates = [], [], [], [], []
    hand = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")
    class Recognition:
        def recognize_opening_signal(self, image):
            return OpeningSignal(False, None, None, False)
        def recognize_fast_signals(self, image, expected_player, *, allow_pass=True):
            return FastSignalResult(expected_player, None, False, False, False)
        def recognize_play_region(self, image, player, *, wild_rank, allow_pass=True):
            return PlayRegionResult(player, (), False, 0., (), (), source="lifecycle-test")
    recognition = Recognition()
    def open_source(profile_name):
        source = _SourceStub()
        sources.append(source)
        return source
    capture = SimpleNamespace(profiles_root=tmp_path, open_live_source=open_source)
    class Factory:
        fail_bind = False
        def start_session(self, *, round_level, hand, lead_player, recognition_strategy, on_update):
            profile = f"lifecycle{len(created)}"
            (tmp_path / profile).mkdir()
            store = InMemoryLiveSessionStore(tmp_path, profile)
            core = LiveOrchestrator(
                reducer=LiveReducer(store.session_id), store=store,
                recorder=InMemorySessionRecorder(store.directory), recognition_service=recognition,
                advisor=None, minimum_free_bytes=0, settle_ms=0, processing_clock_ms=lambda: 1100,
                on_update=on_update,
            )
            initial = core.start(round_level=round_level, hand=hand, lead_player=lead_player, monotonic_ms=900)
            created.append(core)
            binder, resume = core.bind_capture_generation, core.resume
            def watched_bind(generation):
                assert controller._active_live_token.generation == generation
                calls.append(("bind", generation))
                if self.fail_bind:
                    raise RuntimeError("binding failed")
                return binder(generation)
            def watched_resume(**kwargs):
                calls.append(("resume", controller._active_live_token.generation))
                return resume(**kwargs)
            core.bind_capture_generation, core.resume = watched_bind, watched_resume
            return SimpleNamespace(orchestrator=core, source=open_source("test"), initial_update=initial)
    factory = Factory()
    controller = LiveAssistantController(capture, recognition_service=recognition,
                                         advisor=object(), session_factory=factory)
    monkeypatch.setattr(controller, "_start_danzero_warmup", lambda: None)
    monkeypatch.setattr(controller, "_start_analysis_worker", lambda: worker_starts.append("analysis"))
    monkeypatch.setattr(controller, "_start_capture_worker", lambda: worker_starts.append("capture"))
    controller.update_ready.connect(updates.append)
    yield SimpleNamespace(controller=controller, factory=factory, cores=created, sources=sources,
                          workers=worker_starts, calls=calls, updates=updates, hand=hand)
    controller._invalidate_live_token()
    controller.orchestrator = None
    for core in created:
        core.finish()
    for source in sources:
        source.close()
    _app().processEvents()
    assert not list(tmp_path.rglob("*.jsonl"))


@pytest.mark.parametrize("lead_player", [None, "right"])
def test_real_core_start_binds_before_capture_and_waiting_lead_notices(real_controller_lifecycle, lead_player):
    rig = real_controller_lifecycle
    assert rig.controller.start_session(round_level="2", hand=rig.hand, lead_player=lead_player)
    token, core = rig.controller._active_live_token, rig.cores[-1]
    assert rig.calls == [("bind", token.generation)]
    assert rig.workers == ["analysis", "capture"]
    assert rig.updates[-1].capture_generation == token.generation > 0
    assert rig.updates[-1].status == ("waiting_lead" if lead_player is None else "running")
    core._notify_update_listener()
    _app().processEvents()
    assert rig.updates[-1].capture_generation == token.generation
    if lead_player is None:
        frame = _preselection_frame()
        update = rig.controller._analyze_live_frame(token, _AnalysisFrameTask(token, frame, 1, 1100))
        assert update.status == "waiting_lead"
        assert update.capture_generation == token.generation


def test_start_session_preloads_live_dependencies_before_construction_warmup_and_workers(real_controller_lifecycle, monkeypatch):
    from daguandan_bridge.gui import live_controller as module

    rig = real_controller_lifecycle
    order = []
    monkeypatch.setattr(module, "preload_live_worker_dependencies", lambda: order.append("preload"))
    original_start = rig.factory.start_session
    rig.factory.start_session = lambda **kwargs: order.append("construct") or original_start(**kwargs)
    monkeypatch.setattr(rig.controller, "_start_danzero_warmup", lambda: order.append("warmup"))
    monkeypatch.setattr(rig.controller, "_start_analysis_worker", lambda: order.append("analysis"))
    monkeypatch.setattr(rig.controller, "_start_capture_worker", lambda: order.append("capture_worker"))

    assert rig.controller.start_session(round_level="2", hand=rig.hand, lead_player="right")

    assert order == ["preload", "warmup", "construct", "analysis", "capture_worker"]


def test_start_session_preload_failure_does_not_start_session_or_workers(real_controller_lifecycle, monkeypatch):
    from daguandan_bridge.gui import live_controller as module

    rig = real_controller_lifecycle
    errors = []
    rig.controller.error.connect(errors.append)
    monkeypatch.setattr(
        module,
        "preload_live_worker_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeError("import race guard failed")),
    )

    assert not rig.controller.start_session(round_level="2", hand=rig.hand, lead_player="right")

    assert rig.cores == []
    assert rig.sources == []
    assert rig.workers == []
    assert errors == ["实时依赖预加载失败：import race guard failed"]


def test_start_listening_preloads_before_warmup_and_waiting_workers(tmp_path, monkeypatch):
    from daguandan_bridge.gui import live_controller as module

    _app()
    order = []
    controller = LiveAssistantController(_LockingCaptureServiceStub(tmp_path))
    monkeypatch.setattr(module, "preload_live_worker_dependencies", lambda: order.append("preload"))
    monkeypatch.setattr(controller, "_start_danzero_warmup", lambda: order.append("warmup"))
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: order.append("waiting_workers"))

    assert controller.start_listening()

    assert order == ["preload", "warmup", "waiting_workers"]


def test_start_listening_preload_failure_does_not_start_waiting_workers(tmp_path, monkeypatch):
    from daguandan_bridge.gui import live_controller as module

    _app()
    controller = LiveAssistantController(_LockingCaptureServiceStub(tmp_path))
    errors = []
    started = []
    controller.error.connect(errors.append)
    monkeypatch.setattr(
        module,
        "preload_live_worker_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeError("native import failed")),
    )
    monkeypatch.setattr(controller, "_start_danzero_warmup", lambda: started.append("warmup"))
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: started.append("waiting"))

    assert not controller.start_listening()

    assert controller._listening_enabled is False
    assert controller.orchestrator is None
    assert controller._waiting_capture_worker is None
    assert started == []
    assert errors == ["实时依赖预加载失败：native import failed"]


@pytest.mark.parametrize("lead_player", [None, "right"])
def test_real_core_resume_activates_and_binds_before_resume_notices(real_controller_lifecycle, lead_player):
    rig = real_controller_lifecycle
    assert rig.controller.start_session(round_level="2", hand=rig.hand, lead_player=lead_player)
    first = rig.controller._active_live_token
    rig.controller.pause()
    rig.calls.clear()
    rig.workers.clear()
    rig.controller.resume()
    second = rig.controller._active_live_token
    assert second.generation > first.generation
    assert rig.calls == [("bind", second.generation), ("resume", second.generation)]
    assert rig.updates[-1].capture_generation == second.generation
    assert rig.updates[-1].status == ("waiting_lead" if lead_player is None else "running")
    assert rig.workers == ["analysis", "capture"]
    assert rig.sources[0].closed and not rig.sources[-1].closed
    rig.cores[-1]._notify_update_listener()
    _app().processEvents()
    assert rig.updates[-1].capture_generation == second.generation


def test_real_core_initial_bind_failure_invalidates_and_closes_without_workers(real_controller_lifecycle):
    rig = real_controller_lifecycle
    rig.factory.fail_bind = True
    assert not rig.controller.start_session(round_level="2", hand=rig.hand, lead_player=None)
    assert rig.controller.orchestrator is None
    assert rig.controller._active_live_token is None
    assert rig.controller._live_source is None
    assert rig.workers == [] and rig.updates == []
    assert rig.sources[-1].closed
    assert rig.cores[-1].status == "sealed"


@pytest.mark.parametrize("failure", ["bind", "resume_after_status_change", "wrong_bound_identity"])
def test_real_core_failed_resume_cancels_token_and_leaves_source_closed(real_controller_lifecycle, failure):
    rig = real_controller_lifecycle
    assert rig.controller.start_session(round_level="2", hand=rig.hand, lead_player="right")
    rig.controller.pause()
    core = rig.cores[-1]
    rig.workers.clear()
    rig.updates.clear()
    binder, resume = core.bind_capture_generation, core.resume
    if failure == "bind":
        rig.factory.fail_bind = True
    elif failure == "wrong_bound_identity":
        core.bind_capture_generation = lambda generation: replace(binder(generation), capture_generation=generation - 1)
    else:
        def failed_resume(**kwargs):
            resume(**kwargs)
            raise RuntimeError("resume publication failed")
        core.resume = failed_resume
    rig.controller.resume()
    assert rig.controller._active_live_token is None
    assert rig.controller._live_source is None
    assert rig.sources[-1].closed
    assert rig.workers == [] and rig.updates == []
    assert core.status == "paused"
    assert rig.controller._fatal_capture_session_id == core.snapshot.session_id


def test_slow_recording_never_delays_analysis_admission_or_execution(
    tmp_path, monkeypatch
):
    from daguandan_bridge.gui import live_controller as module
    from daguandan_bridge.live.recorder import SessionRecorder
    _app()
    recording_entered, release_recording, analyzed = threading.Event(), threading.Event(), threading.Event()
    frame = _preselection_frame()
    recorder = SessionRecorder(tmp_path / "tail", size=(frame.image.shape[1], frame.image.shape[0]), fps=10)
    class RaceOrchestrator(_TokenAnalysisOrchestrator):
        needs_first_action_frames = False
        def __init__(self):
            super().__init__("tail")
            self.recorder = recorder
            self.recorded = []
        def record_frame(self, image, *, monotonic_ms, wall_time):
            self.recorded.append((image, monotonic_ms, wall_time))
            recording_entered.set()
            assert release_recording.wait(2)
            return self.recorder.write_frame(image, monotonic_ms, wall_time)
        def analyze_frame(self, image, *, monotonic_ms, trace_context):
            analyzed.set()
            return super().analyze_frame(image, monotonic_ms=monotonic_ms, trace_context=trace_context)
    class FakeSignal:
        def connect(self, callback):
            pass
    class ManualCaptureWorker:
        def __init__(self, operation, interval):
            self.operation = operation
            self.frame_ready, self.error, self.finished = FakeSignal(), FakeSignal(), FakeSignal()
            self.is_running = False
        def start(self):
            self.is_running = True
    monkeypatch.setattr(module, "WorkerHandle", ManualCaptureWorker)
    controller = _isolated_controller(tmp_path)
    orchestrator = RaceOrchestrator()
    controller.orchestrator = orchestrator
    token = controller._activate_live_token(orchestrator)
    controller._live_source = SimpleNamespace(capture=lambda: frame)
    controller._start_recording_dispatcher(token)
    controller._start_analysis_worker()
    controller._start_capture_worker()
    capture = threading.Thread(target=controller._capture_worker.operation)
    recorder_closed = False
    try:
        capture.start()
        assert recording_entered.wait(1)
        assert controller._analysis_worker.stats["submitted"] == 1
        assert analyzed.wait(1)
        capture.join(1)
        assert not capture.is_alive()
        release_recording.set()
        assert controller._recording_dispatcher.wait_idle(2)
        controller._close_recording_dispatcher(token)
        result = orchestrator.recorder.close()
        recorder_closed = True
        assert result.frame_count == 1
        assert result.integrity["decodable_frame_count"] == 1
        assert result.integrity["status"] == "PASS"
        assert len(orchestrator.recorded) == 1
        recorded_image, recorded_ms, recorded_wall = orchestrator.recorded[0]
        assert recorded_image is not frame.image
        assert np.array_equal(recorded_image, frame.image)
        assert recorded_image.flags.writeable is False
        assert recorded_ms == frame.captured_monotonic_ms
        assert recorded_wall == frame.captured_at.isoformat()
        timing = token.pipeline_timing.snapshot()
        assert "recording_admission_wait" not in timing["stages"]
        assert timing["counters"]["analysis_submitted"] == 1
        assert timing["counters"]["recording_written"] == 1
    finally:
        release_recording.set()
        capture.join(2)
        controller._invalidate_live_token()
        controller._close_recording_dispatcher(token)
        controller._analysis_worker.stop(timeout=2)
        controller._analysis_worker = None
        controller._capture_worker = None
        if not recorder_closed:
            recorder.close()


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


def test_compact_request_guard_keeps_wait_separate_from_diagnostic_compact(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    calls = []
    controller.capture_service.target_client_rect = (
        lambda profile_name: calls.append(profile_name) or "rect"
    )

    # WAIT remains unsafe for recommendation compact mode, but is eligible for
    # a separate future diagnostic/waiting compact surface.
    assert controller.compact_request_allowed() is False
    assert controller.can_show_compact_recommendation() is False
    assert controller.diagnostic_compact_request_allowed() is True
    assert controller.can_show_waiting_compact() is True
    assert controller.target_client_rect() is None
    assert calls == []

    controller._emit_listening_status(
        "opening",
        report_for_error(SimpleNamespace(code="ROI_FATAL"), stage="recognition"),
    )

    assert controller.compact_request_allowed() is False
    assert controller.diagnostic_compact_request_allowed() is False
    assert controller.can_show_waiting_compact() is False
    assert controller.compact_request_report().primary_reason is OpeningReadinessCode.ROI_FATAL
    assert controller.target_client_rect() is None
    assert calls == []


def test_compact_request_guard_allows_only_ready_recommendation_compact(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    calls = []
    controller.capture_service.target_client_rect = (
        lambda profile_name: calls.append(profile_name) or "rect"
    )
    controller._emit_listening_status(
        "opening",
        report_for_phase("ready"),
    )

    assert controller.compact_request_allowed() is True
    assert controller.can_show_compact_recommendation() is True
    assert controller.target_client_rect() == "rect"
    assert calls == ["tencent_daguandan"]


def test_minimized_recovery_emits_structured_waiting_report_immediately(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    controller._listening_enabled = True
    statuses = []
    controller.listening_status.connect(statuses.append)

    controller._begin_geometry_recovery(
        LiveCaptureInterrupted("牌桌窗口已最小化", code="WINDOW-MINIMIZED")
    )

    assert statuses
    status = statuses[-1]
    assert status["state"] == "recovering"
    assert status["status"] == "WAIT"
    assert status["primary_reason"] == "WINDOW_MINIMIZED"
    assert status["report"].primary_reason is OpeningReadinessCode.WINDOW_MINIMIZED
    assert status["compact_allowed"] is False
    assert status["diagnostic_compact_allowed"] is True
    controller._cancel_geometry_recovery()


def test_publish_ready_waiting_first_action_keeps_legacy_message_field(tmp_path):
    _app()
    controller = LiveAssistantController(_CaptureServiceStub(tmp_path))
    statuses = []
    controller.listening_status.connect(statuses.append)

    controller._publish_opening_status(
        "ready_waiting_first_action",
        SimpleNamespace(my_hand=("3S",) * 27),
    )

    assert statuses[-1]["phase"] == "ready_waiting_first_action"
    assert statuses[-1]["message"] == "已进入牌桌，等待自己首出"
    assert statuses[-1]["primary_reason"] == "READY_WAITING_FIRST_ACTION"
