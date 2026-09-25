from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from daguandan_bridge.application import live_v2_recorded_replay as replay_module
from daguandan_bridge.domain.frame import FrameEnvelope
from daguandan_bridge.opening_gate import serialized_result


def test_recorded_replay_pacing_uses_capture_clock_and_can_cancel(monkeypatch):
    now = iter((0.0, 0.5))
    sleeps = []
    monkeypatch.setattr(replay_module.time, "monotonic", lambda: next(now))
    monkeypatch.setattr(replay_module.time, "sleep", sleeps.append)

    assert replay_module._wait_for_recorded_deadline(
        captured_monotonic_ms=500,
        source_started_ms=0,
        playback_started_at=0.0,
        stop_requested=None,
    )
    assert sleeps == [0.05]

    monkeypatch.setattr(replay_module.time, "monotonic", lambda: 0.0)
    assert not replay_module._wait_for_recorded_deadline(
        captured_monotonic_ms=500,
        source_started_ms=0,
        playback_started_at=0.0,
        stop_requested=lambda: True,
    )


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


def test_recorded_replay_hands_confirmed_opening_seed_to_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("3", "4", "5", "6", "7", "8", "9")
        for suit in "SHCD"
    )[:27]
    envelopes = tuple(
        FrameEnvelope(
            image=object(),
            captured_monotonic_ms=100 + index * 100,
            wall_time=f"2026-09-19T03:00:0{index}+08:00",
            frame_index=index,
            capture_seq=index + 1,
            capture_generation=1,
            evidence_frame_id=f"opening-{index}",
        )
        for index in range(2)
    )

    class Source:
        indexed_frame_count = 2
        warnings = ()

        def __init__(self, *_args, **_kwargs):
            pass

        def envelopes(self, *, capture_generation):
            assert capture_generation == 1
            return iter(envelopes)

    class Recognition:
        def recognize_listening_page(self, _image):
            return SimpleNamespace(stage="table", anchor_score=0.99, buttons=())

        def recognize(self, _image, *, allow_unknown_suit):
            assert allow_unknown_suit is True
            return serialized_result(
                round_level="5",
                hand=hand,
                lead_player="left",
                current_player="self",
                events=({
                    "player": "left",
                    "cards": ("7?",),
                    "suit_options": (("7C", "7D"),),
                    "is_pass": False,
                    "confidence": 0.93,
                    "source": "recorded-opening",
                },),
            )

    class Runtime:
        def __init__(self):
            self.status = "new"
            self.calls = []
            self.snapshot = SimpleNamespace(current_player=None, revision=2)

        def start(self, **kwargs):
            self.calls.append(("start", kwargs))
            self.status = "running"
            return SimpleNamespace(snapshot=SimpleNamespace(current_player="left", revision=1))

        def bind_capture_generation(self, generation):
            self.calls.append(("bind", generation))
            return SimpleNamespace(snapshot=SimpleNamespace(current_player="left", revision=1))

        def record_frame(self, image, *, monotonic_ms, wall_time):
            self.calls.append(("record", image, monotonic_ms, wall_time))

        def bootstrap_opening_action(self, **kwargs):
            self.calls.append(("bootstrap", kwargs))
            return SimpleNamespace(
                snapshot=SimpleNamespace(current_player="self", revision=2),
                event=None,
                events=(),
            )

        def analyze_frame(self, *_args, **_kwargs):
            raise AssertionError("confirming opening frame must not be analyzed twice")

        def wait_for_advice_idle(self, *, timeout):
            assert timeout == 60.0
            return True

        def finish(self):
            self.calls.append(("finish",))
            self.status = "sealed"

    runtime = Runtime()
    profile = tmp_path / "profile"
    profile.mkdir()
    session = tmp_path / "session"
    session.mkdir()
    output = tmp_path / "report"
    monkeypatch.setattr(replay_module, "VideoReplaySource", Source)
    monkeypatch.setattr(replay_module, "build_session_manifest", lambda *_args: {})

    result = replay_module.replay_video_through_production_live_v2(
        session,
        Recognition(),
        profile_root=profile,
        output_root=output,
        runtime_builder=lambda **_kwargs: runtime,
        advisor_backend="fabledan",
    )

    assert [call[0] for call in runtime.calls[:4]] == [
        "start", "bind", "record", "bootstrap",
    ]
    start = runtime.calls[0][1]
    assert start["lead_player"] == "left"
    bootstrap = runtime.calls[3][1]
    assert bootstrap["actor"] == "left"
    assert bootstrap["cards"] == ("7?",)
    assert bootstrap["expected_next_player"] == "self"
    assert bootstrap["suit_options"] == (("7C", "7D"),)
    assert result.opening["confirmed"]["lead_player"] == "left"
    assert result.runtime_identity["legacy_orchestrator_used"] is False
