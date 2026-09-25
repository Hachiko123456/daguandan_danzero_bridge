from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.capture_service import LiveCaptureInterrupted
from daguandan_bridge.gui.live_controller import (
    LiveAssistantController,
    _WaitingRecognitionEnvelope,
)
from daguandan_bridge.opening_gate import ListeningPageSignal


class _Capture:
    def __init__(self, root):
        self.profiles_root = root


class _Store:
    def __init__(self):
        self.updates = []

    def update_session_metadata(self, value):
        self.updates.append(dict(value))

    def append_recognition_trace(self, value):
        self.updates.append({"recognition_trace": dict(value)})


class _Recording:
    def __init__(self):
        self.store = _Store()
        self.closed_with = []
        self.frames = []

    def record_frame(self, image, *, monotonic_ms, wall_time):
        self.frames.append((image, monotonic_ms, wall_time))

    def close(self, *, reason):
        self.closed_with.append(reason)


def _app():
    return QApplication.instance() or QApplication([])


def _snapshot():
    return SimpleNamespace(
        image=object(),
        captured_at=datetime.now(timezone.utc),
        captured_monotonic_ms=123,
    )


def _controller(tmp_path):
    controller = LiveAssistantController(_Capture(tmp_path))
    controller._listening_enabled = True
    controller._listener_recording = _Recording()
    controller._test_recording = controller._listener_recording
    controller._listening_page = ListeningPageSignal("table", 0.95)
    controller._last_stable_page_stage = "table"
    controller._table_anchor_observed = True
    return controller


def test_one_unknown_frame_then_table_keeps_episode_context_and_recovers(tmp_path):
    _app()
    controller = _controller(tmp_path)
    candidate = object()
    controller._waiting_candidate = candidate
    snapshot = _snapshot()
    statuses = []
    controller.listening_status.connect(statuses.append)

    controller._apply_listening_page(ListeningPageSignal("unknown", 0.0), snapshot)

    assert controller._listening_enabled is True
    assert controller._table_anchor_observed is True
    assert controller._waiting_candidate is candidate
    assert controller._page_unknown_retry_count == 1
    assert controller._test_recording.closed_with == []
    page_update = next(
        item for item in controller._test_recording.store.updates
        if "page_recovery" in item
    )
    assert page_update["page_recovery"]["reason"] == "transient_page_unknown"

    controller._apply_listening_page(ListeningPageSignal("table", 0.95), snapshot)

    assert controller._listening_enabled is True
    assert controller._page_unknown_retry_count == 0
    assert controller._last_stable_page_stage == "table"
    assert controller._test_recording.closed_with == []
    assert any(item["state"] == "recovered" for item in statuses)


def test_unknown_delivery_publishes_recovery_stage_reason_and_retry_count(tmp_path):
    _app()
    controller = _controller(tmp_path)
    statuses = []
    controller.listening_status.connect(statuses.append)
    snapshot = _snapshot()
    envelope = _WaitingRecognitionEnvelope(
        snapshot=snapshot,
        generation=controller._waiting_generation,
        trace=None,
        opening_seed_valid=None,
        page=ListeningPageSignal("unknown", 0.0),
    )

    controller._consume_waiting_recognition(
        SimpleNamespace(my_hand=(), round_level=None),
        envelope,
    )

    recovery = statuses[-1]
    assert recovery["state"] == "recovering"
    assert recovery["stage"] == "page"
    assert recovery["reason"] == "transient_page_unknown"
    assert recovery["retry_count"] == 1
    assert recovery["page_stage"] == "unknown"
    assert controller._test_recording.store.updates[-1]["recognition_trace"]["phase"] == (
        "page_recovery"
    )


def test_unknown_frames_stop_only_after_bounded_recovery_budget(tmp_path):
    _app()
    controller = _controller(tmp_path)
    controller._PAGE_UNKNOWN_MAX_RETRIES = 2
    statuses = []
    faults = []
    controller.listening_status.connect(statuses.append)
    controller.live_fault.connect(faults.append)
    snapshot = _snapshot()

    controller._apply_listening_page(ListeningPageSignal("unknown", 0.0), snapshot)
    controller._apply_listening_page(ListeningPageSignal("unknown", 0.0), snapshot)
    assert controller._listening_enabled is True
    assert controller._test_recording.closed_with == []

    controller._apply_listening_page(ListeningPageSignal("unknown", 0.0), snapshot)

    assert controller._listening_enabled is False
    assert controller._test_recording.closed_with == [
        "transient_page_unknown_budget_exhausted"
    ]
    assert faults[-1]["kind"] == "page_recovery"
    assert faults[-1]["stage"] == "page"
    assert faults[-1]["retry_count"] == 3
    assert faults[-1]["reason"] == "transient_page_unknown_budget_exhausted"
    assert statuses[-1]["state"] == "failed"
    assert statuses[-1]["failure_class"] == "transient_page_recovery_exhausted"
    assert statuses[-1]["retry_count"] == 3


def test_lobby_and_settlement_are_explicit_boundaries_not_transient_unknown(
    tmp_path,
):
    _app()
    for stage in ("lobby", "settlement"):
        controller = _controller(tmp_path)
        controller._waiting_candidate = object()
        controller._page_unknown_retry_count = 2
        controller._apply_listening_page(ListeningPageSignal(stage, 0.1), _snapshot())

        assert controller._listening_enabled is True
        assert controller._table_anchor_observed is False
        assert controller._waiting_candidate is None
        assert controller._page_unknown_retry_count == 0
        assert controller._last_stable_page_stage == stage
        assert controller._test_recording.closed_with == [f"page_{stage}"]


def test_geometry_capture_failure_still_enters_existing_geometry_recovery(
    tmp_path,
    monkeypatch,
):
    _app()
    controller = _controller(tmp_path)
    started = []
    monkeypatch.setattr(
        controller,
        "_begin_geometry_recovery",
        lambda error: started.append(error),
    )

    controller._accept_waiting_error(
        LiveCaptureInterrupted("geometry changed", code="GEOMETRY-CHANGED"),
        controller._waiting_generation,
    )

    assert len(started) == 1
    assert controller._listening_enabled is True
    assert controller._test_recording.closed_with == []
