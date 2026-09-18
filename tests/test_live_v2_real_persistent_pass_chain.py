from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.application.live_v2_frame_pipeline import LiveV2FramePipeline
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live_v2.identity import Seat
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService

from test_live_v2_low_confidence_chain import HAND, TRUTH


SESSION = PROFILES_ROOT / "tencent_daguandan/sessions/game_20260814_004447_aab3dc"
EXPECTED = TRUTH[6:11]
FRAME_WINDOWS = ((254, 262), (262, 266), (266, 280), (280, 292), (290, 300))


class _Vision:
    def __init__(self, pipeline): self.pipeline = pipeline
    def start(self, **kwargs): pass
    def close(self, **kwargs): pass
    def process_frame(self, *args, **kwargs): return self.pipeline.process_frame(*args, **kwargs)


class _Advice:
    def start(self, **kwargs): pass
    def close(self, **kwargs): pass
    def submit(self, *args, **kwargs): return ()
    def drain_results(self): return ()


def test_real_frames_254_300_commit_persistent_pass_chain_without_drift(tmp_path: Path) -> None:
    video = SESSION / "video/game.avi"
    index_path = SESSION / "video/frame_index.jsonl"
    if not video.is_file() or not index_path.is_file():
        pytest.skip("historical AVI/index is unavailable")
    rows = [json.loads(line) for line in index_path.read_text("utf-8").splitlines()]
    recognizer = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT), TemplateService(PROFILES_ROOT),
        diagnostic_tracing=False,
    )
    pipeline = LiveV2FramePipeline(recognizer)
    store = LiveSessionStore(tmp_path, "profile", session_id="real-pass-chain")
    store.start({"target_fps": 10})
    runtime = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store), store=store,
        recorder=InMemorySessionRecorder(store.directory),
        recognition_service=recognizer,
        vision_factory=lambda version: _Vision(pipeline),
        advice_runtime_factory=lambda version: _Advice(),
        local_hint_window_ms=0,
    )
    runtime.start(
        round_level="6", hand=HAND, lead_player="left",
        monotonic_ms=rows[220]["monotonic_ms"],
    )
    runtime.bind_capture_generation(1)
    for seat, cards, is_pass, confidence in TRUTH[:6]:
        runtime.commit_trusted_action(
            actor=seat.value, cards=cards, is_pass=is_pass,
            monotonic_ms=rows[220]["monotonic_ms"] + len(runtime.snapshot.play_history) + 1,
            confidence=confidence,
        )
    assert len(runtime.snapshot.play_history) == 6
    assert runtime.snapshot.current_player == "right"

    fast_evidence: list[tuple[int, str, str | None, tuple[str, ...]]] = []
    capture = cv2.VideoCapture(str(video))
    try:
        for source_frame in range(254, 301):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = capture.read()
            assert ok and image is not None
            expected_before = runtime.snapshot.current_player
            update = runtime.analyze_frame(
                image, monotonic_ms=int(rows[source_frame]["monotonic_ms"]),
                trace_context={
                    "capture_generation": 1, "capture_seq": source_frame + 1,
                    "captured_ms": int(rows[source_frame]["monotonic_ms"]),
                },
            )
            if update.fast_signals is not None:
                fast_evidence.append((
                    source_frame, str(expected_before),
                    update.fast_signals.active_player,
                    tuple(update.fast_signals.pass_marker_players),
                ))
    finally:
        capture.release()

    actions = runtime.rule_session.confirmed_actions
    actual = actions[6:11]
    assert [
        (item.seat, tuple(sorted(item.cards)), item.kind.value == "pass")
        for item in actual
    ] == [
        (seat, tuple(sorted(cards)), is_pass)
        for seat, cards, is_pass, _confidence in EXPECTED
    ]
    for action, (lower, upper) in zip(actual, FRAME_WINDOWS, strict=True):
        assert lower <= action.first_frame.frame_sequence - 1 <= upper
        assert lower <= action.last_frame.frame_sequence - 1 <= upper
    for previous, current in zip(actual, actual[1:], strict=False):
        assert current.first_frame.frame_sequence > previous.last_frame.frame_sequence
        assert current.first_frame.captured_ms > previous.last_frame.captured_ms
    for seat, (_lower, _upper) in zip(
        (Seat.OPPOSITE, Seat.LEFT, Seat.SELF), FRAME_WINDOWS[1:4], strict=True
    ):
        assert any(expected == seat.value and seat.value in markers
                   for _frame, expected, _active, markers in fast_evidence)
    assert any(expected == "right" and active == "opposite"
               for _frame, expected, active, _markers in fast_evidence)
    runtime.finish()
