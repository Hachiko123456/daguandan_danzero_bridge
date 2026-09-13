from __future__ import annotations

from daguandan_bridge.application import live_v2_recorded_replay as replay_module
from daguandan_bridge.domain.frame import FrameEnvelope


def test_recorded_replay_records_before_using_canonical_frame_entry(monkeypatch):
    calls = []

    class Runtime:
        def record_frame(self, image, *, monotonic_ms, wall_time):
            calls.append(("record", image, monotonic_ms, wall_time))

        def analyze_frame(self, *args, **kwargs):  # pragma: no cover - must not be used
            raise AssertionError("recorded replay must use analyze_frame_envelope")

    runtime = Runtime()
    envelope = FrameEnvelope(
        image=object(),
        captured_monotonic_ms=123,
        wall_time="2026-09-13T12:00:00+08:00",
        frame_index=7,
        capture_seq=8,
        capture_generation=1,
        evidence_frame_id="frame-7",
    )

    def fake_analyze_frame_envelope(runtime_arg, envelope_arg, *, trace_context):
        calls.append(("analyze", runtime_arg, envelope_arg, trace_context))
        return "update"

    monkeypatch.setattr(
        replay_module,
        "analyze_frame_envelope",
        fake_analyze_frame_envelope,
    )

    result = replay_module._record_and_analyze_frame(
        runtime,
        envelope,
        trace_context={"replay_mode": "opening_confirmation_frame"},
    )

    assert result == "update"
    assert calls[0] == (
        "record",
        envelope.image,
        envelope.captured_monotonic_ms,
        envelope.wall_time,
    )
    assert calls[1] == (
        "analyze",
        runtime,
        envelope,
        {"replay_mode": "opening_confirmation_frame"},
    )
