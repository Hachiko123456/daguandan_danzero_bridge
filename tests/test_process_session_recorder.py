from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from daguandan_bridge.application.recording_process_protocol import (
    RecordingCommand, RecordingOperation, RecordingReady, RecordingResponse,
)
from daguandan_bridge.infrastructure.process_session_recorder import (
    ProcessSessionRecorder, RecordingProcessError,
)
from daguandan_bridge.gui.live_controller import LiveAssistantController
from daguandan_bridge.gui.recording_dispatcher import RecordingFrame
from types import SimpleNamespace


def blocking_recording_worker(config, connection, entered_path, release_path) -> None:
    video = Path(config.session_directory) / "video" / "game.avi"
    index = Path(config.session_directory) / "video" / "frame_index.jsonl"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"before")
    index.write_text("", encoding="utf-8")
    connection.send(RecordingReady(os.getpid()))
    command = connection.recv()
    assert isinstance(command, RecordingCommand)
    assert command.operation is RecordingOperation.WRITE
    index.write_text('{"captured_monotonic_ms":100}\n', encoding="utf-8")
    Path(entered_path).write_text("entered", encoding="ascii")
    while not Path(release_path).exists():
        time.sleep(.01)
    with video.open("ab") as handle:
        handle.write(b"after")
    with index.open("a", encoding="utf-8") as handle:
        handle.write('{"captured_monotonic_ms":999}\n')
    connection.send(RecordingResponse(command.request_id, "ok"))


def crashing_recording_worker(config, connection) -> None:
    connection.send(RecordingReady(os.getpid()))
    connection.recv()
    os._exit(23)


def close_blocking_recording_worker(config, connection, entered_path, release_path) -> None:
    video = Path(config.session_directory) / "video" / "game.avi"
    index = Path(config.session_directory) / "video" / "frame_index.jsonl"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"closed-before")
    index.write_text("", encoding="utf-8")
    connection.send(RecordingReady(os.getpid()))
    write = connection.recv()
    index.write_text('{"monotonic_ms":321}\n', encoding="utf-8")
    connection.send(RecordingResponse(write.request_id, "ok"))
    close = connection.recv()
    assert close.operation is RecordingOperation.CLOSE
    Path(entered_path).write_text("entered", encoding="ascii")
    while not Path(release_path).exists():
        time.sleep(.01)
    with video.open("ab") as handle:
        handle.write(b"closed-after")
    connection.send(RecordingResponse(close.request_id, "ok"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wait_for(path: Path, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(.01)
    return path.exists()


def test_timeout_kills_child_before_aborted_files_are_published(tmp_path: Path) -> None:
    session = tmp_path / "blocked"
    session.mkdir()
    entered, release = tmp_path / "entered", tmp_path / "release"
    manifest = session / "manifest.json"
    recorder = ProcessSessionRecorder(
        session, size=(64, 32), fps=10, write_timeout=.1,
        worker_module=__name__, worker_name="blocking_recording_worker",
        worker_args=(str(entered), str(release)),
    )
    with pytest.raises(RecordingProcessError) as captured:
        recorder.write_frame(np.zeros((32, 64, 3), np.uint8), 100, "wall-100")
    assert captured.value.code == "recording_write_timeout"
    assert _wait_for(entered) and not recorder.worker_alive

    result = recorder.close()
    manifest.write_text(json.dumps({
        "status": "aborted", "dropped_frames": result.dropped_frames,
        "recording_integrity": result.integrity,
    }, sort_keys=True), encoding="utf-8")
    paths = (result.video_path, result.index_path, manifest)
    before = [(path.stat().st_size, path.stat().st_mtime_ns, _hash(path)) for path in paths]
    release.write_text("released", encoding="ascii")
    after = [(path.stat().st_size, path.stat().st_mtime_ns, _hash(path)) for path in paths]

    assert before == after
    assert result.integrity["status"] == "FAIL"
    assert result.integrity["recording_state"] == "aborted"
    assert result.integrity["worker_confirmed_dead"] is True
    assert result.integrity["final_files"]["video"]["sha256"] == _hash(result.video_path)


def test_real_session_recorder_child_preserves_frame_and_timestamp(tmp_path: Path) -> None:
    session = tmp_path / "normal"
    session.mkdir()
    recorder = ProcessSessionRecorder(session, size=(64, 32), fps=10)
    frame = np.zeros((32, 64, 3), np.uint8)
    frame[:, :, 1] = 180

    assert recorder.write_frame(frame, 1234, "wall-1234") is None
    result = recorder.close()

    assert not recorder.worker_alive
    assert result.frame_count == 1 and result.dropped_frames == 0
    assert result.integrity["status"] == "PASS"
    record = json.loads(result.index_path.read_text("utf-8").strip())
    assert record["monotonic_ms"] == 1234
    assert record["wall_time"] == "wall-1234"
    capture = cv2.VideoCapture(str(result.video_path))
    ok, decoded = capture.read(); capture.release()
    assert ok and decoded is not None
    assert float(decoded[:, :, 1].mean()) > 160


def test_normal_close_has_an_independent_bounded_audit_budget() -> None:
    default = inspect.signature(ProcessSessionRecorder).parameters["close_timeout"].default
    assert 30 <= default <= 60


def test_crashed_child_is_classified_and_confirmed_dead(tmp_path: Path) -> None:
    session = tmp_path / "crash"
    session.mkdir()
    recorder = ProcessSessionRecorder(
        session, size=(64, 32), fps=10, write_timeout=1,
        worker_module=__name__, worker_name="crashing_recording_worker",
    )
    with pytest.raises(RecordingProcessError) as captured:
        recorder.write_frame(np.zeros((32, 64, 3), np.uint8), 1, "wall")
    result = recorder.close()

    assert captured.value.code in {"recording_worker_crashed", "recording_worker_unavailable"}
    assert not recorder.worker_alive
    assert result.integrity["status"] == "FAIL"
    assert result.integrity["worker_confirmed_dead"] is True


def test_close_timeout_is_classified_killed_and_forensically_stable(tmp_path: Path) -> None:
    session = tmp_path / "close-timeout"
    session.mkdir()
    entered, release = tmp_path / "close-entered", tmp_path / "close-release"
    recorder = ProcessSessionRecorder(
        session, size=(64, 32), fps=10, close_timeout=.1,
        worker_module=__name__, worker_name="close_blocking_recording_worker",
        worker_args=(str(entered), str(release)),
    )
    assert recorder.write_frame(np.zeros((32, 64, 3), np.uint8), 321, "wall") is None

    result = recorder.close()

    assert _wait_for(entered)
    assert not recorder.worker_alive
    assert result.integrity["status"] == "FAIL"
    assert result.integrity["issues"] == ["recording_close_timeout"]
    assert result.integrity["worker_failure"]["code"] == "recording_close_timeout"
    paths = (result.video_path, result.index_path)
    before = [(path.stat().st_size, path.stat().st_mtime_ns, _hash(path)) for path in paths]
    release.write_text("released", encoding="ascii")
    after = [(path.stat().st_size, path.stat().st_mtime_ns, _hash(path)) for path in paths]
    assert before == after


def test_controller_failed_recording_finish_is_stable_and_resumes_listener(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    session = tmp_path / "controller"
    session.mkdir()
    entered, release = tmp_path / "controller-entered", tmp_path / "controller-release"
    process_recorder = ProcessSessionRecorder(
        session, size=(64, 32), fps=10, write_timeout=.1,
        worker_module=__name__, worker_name="blocking_recording_worker",
        worker_args=(str(entered), str(release)),
    )

    class Store:
        def __init__(self): self.metadata = []
        def update_session_metadata(self, value): self.metadata.append(value)

    class Orchestrator:
        status = "running"
        snapshot = SimpleNamespace(session_id="controller-recording")
        automatic_log_delivery_result = None
        recognition_service = SimpleNamespace()
        def __init__(self):
            self.recorder, self.store, self.result = process_recorder, Store(), None
        def record_frame(self, image, *, monotonic_ms, wall_time):
            return self.recorder.write_frame(image, monotonic_ms, wall_time)
        def begin_finalizing(self): self.status = "finalizing"
        def finish(self):
            self.result = self.recorder.close()
            manifest = session / "manifest.json"
            manifest.write_text(json.dumps({
                "status": "sealed", "health": "FAIL",
                "dropped_frames": self.result.dropped_frames,
                "recording_integrity": self.result.integrity,
            }, sort_keys=True), encoding="utf-8")
            self.status = "sealed"
            return SimpleNamespace(status="sealed")

    capture = SimpleNamespace(profiles_root=tmp_path)
    controller = LiveAssistantController(
        capture, recognition_service=SimpleNamespace(), advisor=object(),
        session_factory=SimpleNamespace(),
    )
    orchestrator = Orchestrator()
    controller.orchestrator = orchestrator
    controller._listening_enabled = True
    resumed = []
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: resumed.append(True))
    token = controller._activate_live_token(orchestrator)
    controller._start_recording_dispatcher(token)
    controller._recording_dispatcher.submit(
        RecordingFrame(
            np.zeros((32, 64, 3), np.uint8), 100, "wall", 1, token,
        )
    )
    assert _wait_for(entered)

    controller.finish()
    assert controller._finish_thread.wait(10_000)
    app.processEvents()
    assert orchestrator.result.integrity["status"] == "FAIL"
    assert not process_recorder.worker_alive
    assert resumed == [True]
    manifest = session / "manifest.json"
    paths = (process_recorder.video_path, process_recorder.index_path, manifest)
    before = [(path.stat().st_size, path.stat().st_mtime_ns, _hash(path)) for path in paths]
    release.write_text("released", encoding="ascii")
    after = [(path.stat().st_size, path.stat().st_mtime_ns, _hash(path)) for path in paths]
    assert before == after
