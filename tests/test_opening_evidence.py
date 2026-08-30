from __future__ import annotations

import json
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
