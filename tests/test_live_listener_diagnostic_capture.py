from __future__ import annotations

import hashlib
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.gui.live_controller import (
    LiveAssistantController,
    _LiveFrameDelivery,
    _LiveRunToken,
    _WaitingFrameDelivery,
)
from daguandan_bridge.image_io import StandardizationResult
from daguandan_bridge.models import Box, ClientRect
from daguandan_bridge.window_capture import CapturedStandardizedFrame


class _ManualSource:
    def __init__(self, snapshot=None, error: Exception | None = None):
        self.snapshot = snapshot
        self.error = error
        self.capture_calls = 0
        self.closed = False

    def capture(self):
        self.capture_calls += 1
        if self.error is not None:
            raise self.error
        return self.snapshot

    def close(self):
        self.closed = True


class _CaptureServiceStub:
    def __init__(self, root: Path, *, manual_snapshot=None, manual_error: Exception | None = None):
        self.profiles_root = root
        self.manual_snapshot = manual_snapshot
        self.manual_error = manual_error
        self.open_calls: list[str] = []
        self.sources: list[_ManualSource] = []

    def open_live_source(self, profile_name: str):
        self.open_calls.append(profile_name)
        source = _ManualSource(self.manual_snapshot, self.manual_error)
        self.sources.append(source)
        return source


class _DiagnosticStore:
    instances: list["_DiagnosticStore"] = []
    counts: dict[Path, int] = {}

    def __init__(self):
        self.calls: list[dict[str, object]] = []
        type(self).instances.append(self)

    def save_snapshot(
        self,
        session_directory: Path,
        snapshot: FrameSnapshot,
        *,
        session_id: str,
        capture_generation: int,
        capture_seq: int,
        source: str = "live_listener_frame",
        source_phase: str | None = None,
    ) -> dict[str, object]:
        self.counts[session_directory] = self.counts.get(session_directory, 0) + 1
        count = self.counts[session_directory]
        self.calls.append({
            "session_directory": session_directory,
            "snapshot": snapshot,
            "session_id": session_id,
            "capture_generation": capture_generation,
            "capture_seq": capture_seq,
            "source": source,
            "source_phase": source_phase,
        })
        return {
            "sequence": count,
            "image_path": session_directory / f"diagnostic_frames/{count:06d}.png",
            "metadata_path": session_directory / f"diagnostic_frames/{count:06d}.json",
            "count": count,
            "message": "saved",
            "source": source,
            "source_phase": source_phase,
        }


def _install_store(monkeypatch: pytest.MonkeyPatch) -> None:
    _DiagnosticStore.instances.clear()
    _DiagnosticStore.counts.clear()
    module = types.ModuleType("daguandan_bridge.application.session_diagnostic_frames")
    module.SessionDiagnosticFrameStore = _DiagnosticStore
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _snapshot(value: int = 7) -> FrameSnapshot:
    image = np.array(
        [[[value, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8,
    )
    standardization = StandardizationResult(
        image=image,
        source_size=(2, 2),
        source_viewport=Box(0, 0, 2, 2),
        content_box=Box(0, 0, 2, 2),
        scale=1.0,
        padding=(0, 0, 0, 0),
        aspect_error=0.0,
        aspect_compatible=True,
    )
    frame = CapturedStandardizedFrame(
        standardization=standardization,
        rect=ClientRect(10, 20, 2, 2),
        backend="screen",
        dpi=96,
        window_title="牌桌",
        raw_image=image.copy(),
    )
    return FrameSnapshot(
        frame=frame,
        captured_at=datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc),
        captured_monotonic_ms=1234,
        evidence_frame_id=f"evidence-{value}",
    )


def _controller_with_active_session(tmp_path: Path) -> LiveAssistantController:
    _app()
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    session_directory = tmp_path / "sessions" / "session-1"
    store = SimpleNamespace(directory=session_directory, session_id="session-1")
    orchestrator = SimpleNamespace(
        store=store,
        snapshot=SimpleNamespace(session_id="session-1"),
    )
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    controller._capture_generation = 4
    controller._active_live_token = _LiveRunToken(
        orchestrator=orchestrator,  # type: ignore[arg-type]
        session_id="session-1",
        nonce=1,
        generation=4,
    )
    return controller


def _prime_frame(
    controller: LiveAssistantController,
    snapshot: FrameSnapshot,
    *,
    generation: int = 4,
    capture_seq: int = 12,
) -> None:
    controller._latest_live_frame = snapshot
    controller._latest_live_frame_generation = generation
    controller._latest_live_frame_capture_seq = capture_seq


def test_save_latest_live_frame_delegates_without_capture_and_copies_pixels(
    tmp_path, monkeypatch
):
    controller = _controller_with_active_session(tmp_path)
    _install_store(monkeypatch)
    snapshot = _snapshot()
    _prime_frame(controller, snapshot, capture_seq=12)
    controller.capture_service.capture_frame = lambda *_args, **_kwargs: pytest.fail(
        "save_latest_live_frame_to_session must not recapture"
    )

    result = controller.save_latest_live_frame_to_session()

    assert result["session_id"] == "session-1"
    assert result["sequence"] == 1
    assert result["capture_seq"] == 12
    assert result["capture_generation"] == 4
    assert result["evidence_frame_id"] == snapshot.evidence_frame_id
    assert result["raw_sha256"] == hashlib.sha256(snapshot.image.tobytes(order="C")).hexdigest()
    assert result["source"] == "live_listener_frame"
    assert result["source_phase"] == "live_session"
    assert result["count"] == 1
    call = _DiagnosticStore.instances[0].calls[0]
    assert call["session_directory"] == tmp_path / "sessions" / "session-1"
    saved = call["snapshot"]
    assert saved is not snapshot
    assert saved.image is not snapshot.image
    assert np.array_equal(saved.image, snapshot.image)
    assert saved.frame.raw_image is not snapshot.frame.raw_image
    assert np.array_equal(saved.frame.raw_image, snapshot.frame.raw_image)


def test_save_latest_live_frame_requires_active_session_latest_frame_and_generation(
    tmp_path, monkeypatch
):
    _install_store(monkeypatch)
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    with pytest.raises(RuntimeError):
        controller.save_latest_live_frame_to_session()

    active = _controller_with_active_session(tmp_path)
    with pytest.raises(RuntimeError):
        active.save_latest_live_frame_to_session()

    _prime_frame(active, _snapshot(), generation=3)
    with pytest.raises(RuntimeError):
        active.save_latest_live_frame_to_session()


def test_save_latest_live_frame_rejects_missing_store_and_generation_identity(tmp_path):
    _app()
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    controller._capture_generation = 1
    orchestrator = SimpleNamespace(snapshot=SimpleNamespace(session_id="session-1"))
    controller.orchestrator = orchestrator  # type: ignore[assignment]
    controller._active_live_token = _LiveRunToken(
        orchestrator=orchestrator,  # type: ignore[arg-type]
        session_id="session-1",
        nonce=1,
        generation=1,
    )
    with pytest.raises(RuntimeError):
        controller.save_latest_live_frame_to_session()


def test_save_latest_live_frame_sequence_increments_and_binds_session_metadata(
    tmp_path, monkeypatch
):
    controller = _controller_with_active_session(tmp_path)
    _install_store(monkeypatch)
    _prime_frame(controller, _snapshot(7), capture_seq=20)
    first = controller.save_latest_live_frame_to_session()
    _prime_frame(controller, _snapshot(8), capture_seq=21)
    second = controller.save_latest_live_frame_to_session()

    assert first["sequence"] == 1
    assert second["sequence"] == 2
    assert second["count"] == 2
    calls = [
        call
        for instance in _DiagnosticStore.instances
        for call in instance.calls
    ]
    assert [call["session_id"] for call in calls] == ["session-1", "session-1"]
    assert [call["capture_generation"] for call in calls] == [4, 4]
    assert [call["capture_seq"] for call in calls] == [20, 21]
    assert [call["snapshot"].evidence_frame_id for call in calls] == [
        "evidence-7", "evidence-8"
    ]


def test_accept_live_frame_preserves_capture_seq_without_changing_public_signal(tmp_path):
    _app()
    controller = _controller_with_active_session(tmp_path)
    token = controller._active_live_token
    assert token is not None
    emitted = []
    controller.frame_ready.connect(emitted.append)
    snapshot = _snapshot()
    controller._accept_live_frame(token, _LiveFrameDelivery(snapshot, 33))

    assert emitted == [snapshot]
    assert controller._latest_live_frame is snapshot
    assert controller._latest_live_frame_capture_seq == 33
    assert controller._latest_live_frame_generation == 4


def test_background_save_freezes_exact_frame_before_png_work(tmp_path, monkeypatch):
    app = _app()
    controller = _controller_with_active_session(tmp_path)
    _install_store(monkeypatch)
    snapshot = _snapshot(21)
    _prime_frame(controller, snapshot, capture_seq=44)
    statuses: list[dict[str, object]] = []
    controller.diagnostic_frame_status.connect(statuses.append)

    assert controller.save_latest_live_frame_to_session_background() is True
    # The controller must have copied the listener frame before handing PNG
    # encoding and filesystem work to the one-shot thread.
    snapshot.image[:, :, :] = 200
    thread = controller._diagnostic_frame_thread
    assert thread is not None
    assert thread.wait(5_000)
    app.processEvents()

    assert [item.get("status") for item in statuses] == ["SAVING", "SUCCESS"]
    saved = _DiagnosticStore.instances[0].calls[0]["snapshot"]
    assert np.array_equal(saved.image, _snapshot(21).image)
    assert statuses[-1]["capture_seq"] == 44



def _prime_waiting_frame(
    controller: LiveAssistantController,
    snapshot: FrameSnapshot,
    *,
    generation: int,
    capture_seq: int,
) -> None:
    controller._listening_enabled = True
    controller._waiting_generation = generation
    controller._begin_preopening_diagnostic_round()
    controller._accept_waiting_frame(
        _WaitingFrameDelivery(snapshot, generation, capture_seq),
    )


def test_listening_without_orchestrator_saves_waiting_frame_to_stable_round_directory(
    tmp_path, monkeypatch
):
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    _install_store(monkeypatch)
    snapshot = _snapshot(31)
    _prime_waiting_frame(controller, snapshot, generation=5, capture_seq=1)
    controller.capture_service.capture_frame = lambda *_args, **_kwargs: pytest.fail(
        "waiting diagnostic save must not recapture"
    )

    result = controller.save_latest_live_frame_to_session()

    # Before the listener recording exists, a waiting-frame save is an
    # explicit manual fallback.  It still uses the exact cached listener
    # frame, and all saves in this round must reuse this one fallback path.
    assert result["source"] == "live_listener_frame"
    assert result["source_phase"] == "preopening_listener"
    assert result["session_id"].startswith("diagnostic_")
    session_directory = Path(result["session_directory"])
    assert session_directory.parent.name == "manual_diagnostic"
    assert session_directory.name == result["session_id"]
    assert result["capture_generation"] == 5
    assert result["capture_seq"] == 1
    call = _DiagnosticStore.instances[0].calls[0]
    assert call["snapshot"] is not snapshot
    assert np.array_equal(call["snapshot"].image, snapshot.image)


def test_waiting_frames_in_one_round_share_directory_and_keep_ordered_sequence(
    tmp_path, monkeypatch
):
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    _install_store(monkeypatch)
    controller._listening_enabled = True
    controller._waiting_generation = 7
    controller._begin_preopening_diagnostic_round()
    controller._accept_waiting_frame(_WaitingFrameDelivery(_snapshot(41), 7, 3))
    first = controller.save_latest_live_frame_to_session()
    controller._accept_waiting_frame(_WaitingFrameDelivery(_snapshot(42), 7, 4))
    second = controller.save_latest_live_frame_to_session()

    assert first["session_directory"] == second["session_directory"]
    assert first["sequence"] == 1
    assert second["sequence"] == 2
    calls = [call for instance in _DiagnosticStore.instances for call in instance.calls]
    assert [call["capture_seq"] for call in calls] == [3, 4]


def test_new_listening_round_allocates_fallback_only_on_save_and_uses_new_path_next_round(
    tmp_path, monkeypatch
):
    _app()
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    _install_store(monkeypatch)
    monkeypatch.setattr("daguandan_bridge.gui.live_controller.preload_live_worker_dependencies", lambda: None)
    monkeypatch.setattr(controller, "_start_danzero_warmup", lambda: None)
    monkeypatch.setattr(controller, "_start_waiting_workers", lambda: None)

    assert controller.start_listening() is True
    # No recording and no screenshot: no episode or diagnostic directory is
    # allocated merely by starting a listening round.
    assert controller._preopening_diagnostic_directory is None
    assert controller._preopening_diagnostic_session_id == ""
    sessions_root = tmp_path / "tencent_daguandan" / "sessions"
    assert not (sessions_root / ".preopening").exists()

    controller._waiting_generation = 1
    controller._accept_waiting_frame(_WaitingFrameDelivery(_snapshot(71), 1, 1))
    first = controller.save_latest_live_frame_to_session()
    first_directory = Path(first["session_directory"])
    assert first_directory.parent.name == "manual_diagnostic"
    assert first["source_phase"] == "preopening_listener"
    assert controller._preopening_diagnostic_directory is None

    controller.stop_listening()
    assert controller.start_listening() is True
    assert controller._preopening_diagnostic_directory is None
    assert controller._preopening_diagnostic_session_id == ""

    controller._waiting_generation = 2
    controller._accept_waiting_frame(_WaitingFrameDelivery(_snapshot(72), 2, 1))
    second = controller.save_latest_live_frame_to_session()
    second_directory = Path(second["session_directory"])
    assert second_directory.parent.name == "manual_diagnostic"
    assert second_directory != first_directory
    assert second["session_id"] != first["session_id"]


def test_waiting_frame_stale_generation_is_rejected(tmp_path, monkeypatch):
    controller = LiveAssistantController(
        _CaptureServiceStub(tmp_path),
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    _install_store(monkeypatch)
    controller._listening_enabled = True
    controller._waiting_generation = 9
    controller._begin_preopening_diagnostic_round()
    snapshot = _snapshot(51)
    controller._accept_waiting_frame(_WaitingFrameDelivery(snapshot, 8, 1))

    with pytest.raises(RuntimeError):
        controller.save_latest_live_frame_to_session()


def test_formal_live_frame_remains_preferred_over_waiting_frame(tmp_path, monkeypatch):
    controller = _controller_with_active_session(tmp_path)
    _install_store(monkeypatch)
    controller._listening_enabled = True
    controller._waiting_generation = 2
    controller._begin_preopening_diagnostic_round()
    controller._accept_waiting_frame(_WaitingFrameDelivery(_snapshot(61), 2, 1))
    formal = _snapshot(62)
    _prime_frame(controller, formal, capture_seq=22)

    result = controller.save_latest_live_frame_to_session()

    assert result["source_phase"] == "live_session"
    assert result["session_id"] == "session-1"
    saved = _DiagnosticStore.instances[0].calls[0]["snapshot"]
    assert saved.evidence_frame_id == formal.evidence_frame_id


class _CaptureFailure(RuntimeError):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code
        self.details = {"test": True}


def test_manual_capture_uses_open_live_source_capture_and_records_exact_identity(
    tmp_path, monkeypatch
):
    _install_store(monkeypatch)
    snapshot = _snapshot(71)
    capture = _CaptureServiceStub(tmp_path, manual_snapshot=snapshot)
    controller = LiveAssistantController(
        capture,
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    capture.capture_frame = lambda *_args, **_kwargs: pytest.fail(
        "manual capture must not use capture_frame"
    )

    result = controller.save_latest_live_frame_to_session()

    assert capture.open_calls == ["tencent_daguandan"]
    assert capture.sources[0].capture_calls == 1
    assert capture.sources[0].closed is True
    assert result["source"] == "manual_window_capture"
    assert result["source_phase"] == "manual_window_capture"
    assert result["capture_seq"] == 1
    assert result["evidence_frame_id"] == snapshot.evidence_frame_id
    assert result["raw_sha256"] == hashlib.sha256(snapshot.image.tobytes(order="C")).hexdigest()
    assert str(result["session_directory"]).replace("\\", "/").endswith(
        "/manual_diagnostic/" + str(result["session_id"])
    )


def test_manual_capture_background_reuses_directory_and_increments_capture_seq(
    tmp_path, monkeypatch
):
    app = _app()
    _install_store(monkeypatch)
    capture = _CaptureServiceStub(tmp_path, manual_snapshot=_snapshot(72))
    controller = LiveAssistantController(
        capture,
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    statuses: list[dict[str, object]] = []
    controller.diagnostic_frame_status.connect(statuses.append)

    assert controller.save_latest_live_frame_to_session_background() is True
    first_thread = controller._diagnostic_frame_thread
    assert first_thread is not None
    assert first_thread.wait(5_000)
    app.processEvents()
    assert controller.save_latest_live_frame_to_session_background() is True
    second_thread = controller._diagnostic_frame_thread
    assert second_thread is not None
    assert second_thread.wait(5_000)
    app.processEvents()

    successes = [item for item in statuses if item.get("status") == "SUCCESS"]
    assert [item["capture_seq"] for item in successes] == [1, 2]
    assert successes[0]["session_directory"] == successes[1]["session_directory"]
    assert [source.capture_calls for source in capture.sources] == [1, 1]
    assert [item["source"] for item in successes] == [
        "manual_window_capture",
        "manual_window_capture",
    ]


def test_manual_capture_failure_is_structured_and_writes_no_frame(tmp_path, monkeypatch):
    _install_store(monkeypatch)
    capture = _CaptureServiceStub(
        tmp_path,
        manual_error=_CaptureFailure("没有找到目标窗口", "WINDOW-NOT-FOUND"),
    )
    controller = LiveAssistantController(
        capture,
        recognition_service=SimpleNamespace(),
        advisor=object(),
        session_factory=SimpleNamespace(),
    )
    statuses: list[dict[str, object]] = []
    controller.diagnostic_frame_status.connect(statuses.append)

    assert controller.save_latest_live_frame_to_session_background() is True
    thread = controller._diagnostic_frame_thread
    assert thread is not None
    assert thread.wait(5_000)
    _app().processEvents()

    failure = statuses[-1]
    assert failure["status"] == "FAILURE"
    assert failure["source"] == "manual_window_capture"
    assert failure["source_phase"] == "manual_window_capture"
    assert failure["error_code"] == "WINDOW-NOT-FOUND"
    assert "没有找到" in str(failure["message"])
    assert _DiagnosticStore.instances == []
    assert controller._manual_diagnostic_directory is None
