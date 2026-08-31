from __future__ import annotations

from concurrent.futures import Future
import json
import hashlib
import time
from types import SimpleNamespace

import numpy as np

from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.image_io import standardize_to_base
from daguandan_bridge.models import ClientRect
from daguandan_bridge.opening_evidence import (
    OPENING_CAPTURE_BLACK_FRAME,
    OPENING_HAND_COUNT_MISMATCH,
    OPENING_LEVEL_MISSING,
    OPENING_WINDOW_MINIMIZED,
    NonBlockingOpeningEvidenceSink,
    OpeningEvidenceMonitor,
    classify_opening_failure,
)
from daguandan_bridge.window_capture import CapturedStandardizedFrame


def _snapshot(value: int = 127) -> FrameSnapshot:
    raw = np.full((72, 128, 3), value, dtype=np.uint8)
    standardized = standardize_to_base(raw, (128, 72), detect_black_bars=False)
    return FrameSnapshot(
        CapturedStandardizedFrame(
            standardization=standardized,
            rect=ClientRect(10, 20, 128, 72),
            backend="printwindow",
            dpi=120,
            window_title="Test",
            raw_image=raw,
        )
    )


def _result(*, level=None, hand=()):
    return SimpleNamespace(
        round_level=level,
        wild_rank=level,
        my_hand=tuple(hand),
        lead_player=None,
        current_player=None,
        field_confidences={},
        sources={},
        unresolved_fields=("round_level", "my_hand"),
        diagnostics=("not ready",),
        elapsed_ms=1.25,
    )


def test_opening_ring_is_bounded_and_reports_drop_metrics(tmp_path):
    now = [0]
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        max_age_seconds=1,
        max_bytes=128 * 72 * 3 * 2 + 1,
        clock_ms=lambda: now[0],
    )
    monitor.begin(monotonic_ms=0)

    for value in (10, 20, 30):
        now[0] += 100
        monitor.observe_frame(_snapshot(value), monotonic_ms=now[0])

    metrics = monitor.metrics()
    assert metrics.retained_frames == 1
    assert metrics.dropped_budget == 2
    assert metrics.latest_seq == 3


def test_opening_timeout_is_deduplicated_and_exports_complete_evidence(tmp_path):
    now = [0]
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        max_age_seconds=10,
        max_bytes=8 * 1024 * 1024,
        field_timeout_seconds=1,
        clock_ms=lambda: now[0],
    )
    monitor.begin(monotonic_ms=0)
    snapshot = _snapshot()
    now[0] = 1_100
    monitor.observe_frame(snapshot, monotonic_ms=now[0])
    partial = _result(hand=("2S",))
    monitor.observe_recognition(snapshot, partial, {"schema": "guandan.recognition-trace/1"})
    now[0] = 2_200
    monitor.observe_recognition(snapshot, partial, {"schema": "guandan.recognition-trace/1"})
    assert monitor.flush(5)

    incident_files = sorted((tmp_path / "opening" / "incidents").glob("*/incident.json"))
    codes = [json.loads(path.read_text(encoding="utf-8"))["code"] for path in incident_files]
    assert codes.count(OPENING_LEVEL_MISSING) == 1
    assert codes.count(OPENING_HAND_COUNT_MISMATCH) == 1
    evidence_path = incident_files[0].with_name("opening_evidence.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema"] == "guandan.opening-evidence/1"
    assert evidence["frames"][0]["capture"]["backend"] == "printwindow"
    assert evidence["frames"][0]["capture"]["dpi"] == 120
    assert evidence["privacy"]["support_export_requires_explicit_image_opt_in"] is True
    assert list(incident_files[0].parent.glob("frames/standardized_*.png"))
    monitor.close()


def test_empty_listener_emits_generic_and_anchor_timeouts_after_begin(tmp_path):
    now = [0]
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        field_timeout_seconds=1,
        clock_ms=lambda: now[0],
    )
    monitor.begin(monotonic_ms=0)
    snapshot = _snapshot()
    monitor.observe_frame(snapshot, monotonic_ms=0)
    now[0] = 5_000
    monitor.observe_anchor(snapshot, 0.1, required_score=0.85)
    assert monitor.flush(1)
    incident_files = (tmp_path / "opening" / "incidents").glob("*/incident.json")
    codes = {
        json.loads(path.read_text(encoding="utf-8"))["code"]
        for path in incident_files
    }
    assert {"OPENING-TIMEOUT", "OPENING-ANCHOR-TIMEOUT"}.issubset(codes)


def test_nonblocking_proxy_contains_slow_and_throwing_sinks():
    class Sink:
        def observe_frame(self, _snapshot, **_kwargs):
            time.sleep(0.2)
            raise OSError("disk full")

    proxy = NonBlockingOpeningEvidenceSink(Sink(), queue_size=1)
    started = time.perf_counter()
    proxy.observe_frame(object())
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert proxy.flush(2)
    assert proxy.failed_calls == 1
    proxy.close()


def test_black_frame_and_window_failures_have_stable_codes(tmp_path):
    monitor = OpeningEvidenceMonitor(diagnostics_root=tmp_path)
    monitor.begin(monotonic_ms=0)
    monitor.observe_frame(_snapshot(0), monotonic_ms=1)
    assert monitor.flush(5)

    incident = next((tmp_path / "opening" / "incidents").glob("*/incident.json"))
    assert json.loads(incident.read_text(encoding="utf-8"))["code"] == OPENING_CAPTURE_BLACK_FRAME
    assert classify_opening_failure("目标窗口处于最小化状态") == OPENING_WINDOW_MINIMIZED
    monitor.close()


def test_diagnostic_writer_failure_never_raises_or_blocks_observer(tmp_path, monkeypatch):
    monitor = OpeningEvidenceMonitor(diagnostics_root=tmp_path)
    monitor.begin(monotonic_ms=0)
    monitor.observe_frame(_snapshot(), monotonic_ms=1)
    monkeypatch.setattr(monitor, "_write_incident", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))

    assert monitor.emit_incident("OPENING-TEST-FAILURE", field="test", reason="boom")
    assert monitor.flush(5) is False
    assert monitor.metrics().writer_failures == 1
    monitor.close()


def test_e003_persists_only_explicit_analysis_delivery_and_drop_timing(tmp_path):
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        delivery_settle_seconds=0,
    )
    monitor.begin()
    delivered = _snapshot(80)
    dropped = _snapshot(90)
    for snapshot in (delivered, dropped):
        monitor.observe_frame(snapshot)
        monitor.observe_analysis_submitted(snapshot)
    monitor.observe_analysis_started(delivered)
    trace_hash = hashlib.sha256(memoryview(np.ascontiguousarray(delivered.image))).hexdigest()
    monitor.observe_recognition(
        delivered,
        _result(level=None, hand=()),
        {"input_sha256": trace_hash, "candidates": []},
    )
    monitor.observe_delivery(delivered, gate_eligible=True)
    monitor.observe_analysis_dropped(dropped, reason="latest_replaced")
    assert monitor.emit_incident("OPENING-DELIVERY-AUDIT", field="test", reason="audit")
    assert monitor.flush(5)

    evidence_path = next(
        (tmp_path / "opening" / "incidents").glob("*/opening_evidence.json")
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    frames = {item["frame_id"]: item for item in evidence["frames"]}
    delivered_analysis = frames[delivered.evidence_frame_id]["analysis"]
    dropped_analysis = frames[dropped.evidence_frame_id]["analysis"]
    assert delivered_analysis["gate_delivered"] is True
    assert delivered_analysis["submitted_ms"] <= delivered_analysis["started_ms"]
    assert delivered_analysis["completed_ms"] <= delivered_analysis["delivered_ms"]
    assert dropped_analysis["status"] == "dropped"
    assert dropped_analysis["drop_reason"] == "latest_replaced"
    assert dropped_analysis["gate_delivered"] is False
    monitor.close()


def test_e006_slow_successful_unresolved_recognition_keeps_exact_trace(tmp_path):
    now = [0]
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        max_bytes=128 * 72 * 3 * 2 + 1,
        field_timeout_seconds=1,
        clock_ms=lambda: now[0],
    )
    monitor.begin(monotonic_ms=0)
    slow = _snapshot(101)
    monitor.observe_frame(slow, monotonic_ms=1)
    monitor.observe_analysis_submitted(slow)
    monitor.observe_analysis_started(slow)
    monitor.observe_frame(_snapshot(102), monotonic_ms=2)
    now[0] = 2_000
    trace_hash = hashlib.sha256(memoryview(np.ascontiguousarray(slow.image))).hexdigest()
    monitor.observe_recognition(
        slow,
        _result(level=None, hand=()),
        {
            "schema": "guandan.recognition-trace/1",
            "input_sha256": trace_hash,
            "candidates": [],
        },
    )
    monitor.observe_delivery(slow, gate_eligible=True)
    assert monitor.flush(5)

    evidence_files = list(
        (tmp_path / "opening" / "incidents").glob("*/opening_evidence.json")
    )
    assert evidence_files
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in evidence_files]
    correlated = [
        frame
        for document in documents
        for frame in document["frames"]
        if frame["frame_id"] == slow.evidence_frame_id
        and frame.get("recognition") is not None
    ]
    assert correlated
    assert all(frame["analysis"]["gate_delivered"] is True for frame in correlated)
    assert all(
        frame["analysis"]["recovered_exact_analysis_frame"] is True
        for frame in correlated
    )
    trace_files = [path.with_name("recognition_trace.jsonl") for path in evidence_files]
    assert any(trace_hash in path.read_text(encoding="utf-8") for path in trace_files)
    monitor.close()


def test_e006_atomic_slow_frame_restore_is_trace_bound_before_any_writer(
    tmp_path,
    monkeypatch,
):
    class SynchronousExecutor:
        def submit(self, operation, *args):
            future = Future()
            try:
                operation(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)
            return future

        def shutdown(self, **_kwargs):
            return None

    now = [0]
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        max_bytes=128 * 72 * 3 * 2 + 1,
        field_timeout_seconds=1,
        delivery_settle_seconds=0,
        clock_ms=lambda: now[0],
    )
    monitor.begin(monotonic_ms=0)
    slow = _snapshot(103)
    monitor.observe_frame(slow, monotonic_ms=1)
    monitor.observe_analysis_submitted(slow)
    monitor.observe_analysis_started(slow)
    monitor.observe_frame(_snapshot(104), monotonic_ms=2)
    assert all(item.snapshot is not slow for item in monitor._ring)
    monitor._executor = SynchronousExecutor()
    original_append = monitor._append_frame_locked

    def append_and_publish(*args, **kwargs):
        restored = original_append(*args, **kwargs)
        assert restored.recognition is not None
        assert restored.recognition_trace is not None
        assert restored.analysis["status"] == "completed"
        assert restored.analysis["submitted_ms"] is not None
        assert restored.analysis["started_ms"] is not None
        assert restored.analysis["delivered_ms"] is None
        assert restored.analysis["gate_delivered"] is False
        monitor.emit_incident(
            "OPENING-ATOMIC-RESTORE-RACE",
            field="round_level",
            reason="force a synchronous writer before restore returns",
        )
        return restored

    monkeypatch.setattr(monitor, "_append_frame_locked", append_and_publish)
    now[0] = 2_000
    trace_hash = hashlib.sha256(
        memoryview(np.ascontiguousarray(slow.image))
    ).hexdigest()

    monitor.observe_recognition(
        slow,
        _result(level=None, hand=()),
        {
            "schema": "guandan.recognition-trace/1",
            "input_sha256": trace_hash,
            "candidates": [],
        },
    )
    assert monitor.flush(5)

    evidence_files = list(
        (tmp_path / "opening" / "incidents").glob("*/opening_evidence.json")
    )
    assert len(evidence_files) >= 2
    for evidence_path in evidence_files:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        correlated = [
            frame
            for frame in evidence["frames"]
            if frame["frame_id"] == slow.evidence_frame_id
        ]
        assert correlated
        assert all(frame["recognition"] is not None for frame in correlated)
        assert all(frame["analysis"]["completed_ms"] == now[0] for frame in correlated)
        assert all(frame["analysis"]["submitted_ms"] is not None for frame in correlated)
        assert all(frame["analysis"]["started_ms"] is not None for frame in correlated)
        assert all(frame["analysis"]["gate_delivered"] is False for frame in correlated)
        trace_path = evidence_path.with_name("recognition_trace.jsonl")
        assert trace_hash in trace_path.read_text(encoding="utf-8")
    monitor.close()


def test_e007_png_actual_bytes_never_exceed_budget_or_leave_partial_files(tmp_path):
    monitor = OpeningEvidenceMonitor(
        diagnostics_root=tmp_path,
        max_persisted_image_bytes=8,
        delivery_settle_seconds=0,
    )
    monitor.begin()
    monitor.observe_frame(_snapshot(77))
    assert monitor.emit_incident("OPENING-PNG-BUDGET", field="test", reason="budget")
    assert monitor.flush(5)
    incident = next((tmp_path / "opening" / "incidents").iterdir())
    evidence = json.loads((incident / "opening_evidence.json").read_text(encoding="utf-8"))
    assert evidence["privacy"]["contains_sensitive_images"] is False
    assert all(not frame["artifacts"] for frame in evidence["frames"])
    assert not list(incident.rglob("*.png"))
    assert not any(path.name.endswith(".tmp") for path in incident.rglob("*"))
    monitor.close()
