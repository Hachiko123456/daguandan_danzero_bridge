from __future__ import annotations

from types import SimpleNamespace

from daguandan_bridge.domain.frame import FrameEnvelope
from daguandan_bridge.live.frame_pipeline import analyze_frame_envelope


def test_frame_envelope_carries_capture_identity_and_trace_context():
    envelope = FrameEnvelope(
        image=object(),
        captured_monotonic_ms=1234,
        wall_time="2026-09-12T12:00:00+08:00",
        frame_index=7,
        capture_seq=8,
        capture_generation=2,
        evidence_frame_id="evidence-7",
    )

    assert envelope.trace_context({"historical_scan": True}) == {
        "frame_source": "canonical_envelope",
        "source_wall_time": "2026-09-12T12:00:00+08:00",
        "captured_ms": 1234,
        "capture_generation": 2,
        "roi_version": "capture-v1",
        "source_id": "capture-v1",
        "frame_index": 7,
        "capture_seq": 8,
        "evidence_frame_id": "evidence-7",
        "historical_scan": True,
    }


def test_frame_pipeline_uses_one_runtime_entry_point():
    calls = []

    class Runtime:
        def analyze_frame(self, image, *, monotonic_ms, trace_context):
            calls.append((image, monotonic_ms, trace_context))
            return SimpleNamespace(status="ok")

    image = object()
    envelope = FrameEnvelope(image, 44, "t0", frame_index=3, capture_seq=4)
    result = analyze_frame_envelope(Runtime(), envelope, trace_context={"test": 1})

    assert result.status == "ok"
    assert calls[0][0] is image
    assert calls[0][1] == 44
    assert calls[0][2]["frame_index"] == 3
    assert calls[0][2]["capture_seq"] == 4
    assert calls[0][2]["test"] == 1