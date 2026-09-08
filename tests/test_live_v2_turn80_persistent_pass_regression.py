"""Historical regression for the turn-80 persistent opposite PASS stall."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.application.live_v2_frame_pipeline import (
    FramePipelineConfig,
    LiveV2FramePipeline,
    _persistent_pass_turnovers,
)
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.application.live_v2_vision_protocol import VisionWorkerConfig
from daguandan_bridge.application.live_v2_vision_runtime import LiveV2VisionRuntime
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.live_v2.identity import Seat
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService

from test_live_v2_async_visual_production_regression import (
    AuditedVisionRuntime,
    DelayedPipelineHost,
    MemoryRecorder,
    MemoryStore,
    NoAdviceRuntime,
)


ROOT = Path(__file__).parents[1]
SESSION = (
    ROOT
    / "data/profiles/tencent_daguandan/sessions/game_20260814_004447_aab3dc"
)
FIRST_VISUAL_TURN = 77
LAST_VISUAL_TURN = 84
FIRST_SOURCE_FRAME = 1429
LAST_SOURCE_FRAME = 1510


def _fixture():
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (SESSION / "video/frame_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    turns = tuple(
        {
            "turn_id": int(turn["turn_id"]),
            "actor": str(turn["actor"]),
            "is_pass": bool(turn["is_pass"]),
            "cards": tuple(str(card) for card in turn["cards"]),
            "source_frame": int(turn["evidence"]["frame_indices"][0]),
        }
        for turn in truth["turns"]
    )
    return truth["initial_state"], turns, rows


def _signature(action) -> tuple[str, bool, tuple[str, ...]]:
    return action.seat.value, action.kind.value == "pass", tuple(action.cards)


class HistoricalFrameRecognition:
    """Real recognizer with the bind hook required by the delayed test host."""

    def __init__(self) -> None:
        self.delegate = ScreenshotRecognitionService(
            AnnotationService(PROFILES_ROOT),
            TemplateService(PROFILES_ROOT),
            diagnostic_tracing=False,
        )
        self.source_frame = -1
        self.full_frame_shapes: list[tuple[int, ...]] = []

    def bind(self, frame, image) -> None:
        self.source_frame = frame.frame_sequence - 1
        self.full_frame_shapes.append(tuple(image.shape))

    def play_roi(self, image, seat):
        return self.delegate.play_roi(image, seat)

    def recognize_fast_signals(self, image, expected_player, *, allow_pass=True):
        return self.delegate.recognize_fast_signals(
            image,
            expected_player,
            allow_pass=allow_pass,
        )

    def recognize_play_region(self, image, seat, **kwargs):
        return self.delegate.recognize_play_region(image, seat, **kwargs)


def test_turnover_rearms_only_marked_expected_seat_across_finished_player():
    fast = FastSignalResult(
        expected_player="self",
        active_player="opposite",
        pass_visible=True,
        self_action_buttons_visible=False,
        effect_visible=False,
        pass_marker_player="self",
        pass_marker_players=("self",),
    )

    assert _persistent_pass_turnovers(Seat.SELF, fast) == (Seat.SELF,)
    assert Seat.RIGHT not in _persistent_pass_turnovers(Seat.SELF, fast)


def test_real_async_frames_rearm_turn80_pass_after_formal_turn79_boundary():
    video = SESSION / "video/game.avi"
    index = SESSION / "video/frame_index.jsonl"
    if not video.is_file() or not index.is_file():
        pytest.skip("historical AVI/index is unavailable")
    initial, turns, rows = _fixture()
    visual_truth = tuple(
        turn
        for turn in turns
        if FIRST_VISUAL_TURN <= int(turn["turn_id"]) <= LAST_VISUAL_TURN
    )
    assert [turn["source_frame"] for turn in visual_truth] == [
        1431, 1463, 1473, 1475, 1479, 1492, 1496, 1499,
    ]

    recognition = HistoricalFrameRecognition()
    pipeline = LiveV2FramePipeline(
        recognition,
        config=FramePipelineConfig(max_deep_reads_per_frame=4),
    )
    host = DelayedPipelineHost(pipeline, recognition)
    vision: LiveV2VisionRuntime = AuditedVisionRuntime(
        VisionWorkerConfig(str(ROOT / "data/profiles")),
        host=host,
        session_id=MemoryStore.session_id,
        capture_generation=1,
    )
    store = MemoryStore()
    store.start({"schema": "test.turn80-persistent-pass/1"})
    runtime = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store),
        store=store,
        recorder=MemoryRecorder(),
        recognition_service=recognition,
        vision_factory=lambda version: vision,
        advice_runtime_factory=lambda version: NoAdviceRuntime(),
        processing_clock_ms=lambda: host.now_ms,
        roi_version="historical-turn80-v1",
        source_id=f"avi:{SESSION.name}",
        local_hint_window_ms=0,
    )
    capture = cv2.VideoCapture(str(video))
    try:
        runtime.start(
            round_level=str(initial["round_level"]),
            hand=tuple(str(card) for card in initial["my_hand"]),
            lead_player=str(initial["lead_player"]),
            monotonic_ms=int(rows[0]["monotonic_ms"]),
        )
        runtime.bind_capture_generation(1)
        for turn in turns[: FIRST_VISUAL_TURN - 1]:
            runtime.commit_trusted_action(
                actor=str(turn["actor"]),
                cards=tuple(turn["cards"]),
                is_pass=bool(turn["is_pass"]),
                monotonic_ms=int(rows[int(turn["source_frame"])]["monotonic_ms"]),
                confidence=1.0,
            )
        assert len(runtime.rule_session.confirmed_actions) == 76
        assert runtime.snapshot.current_player == "opposite"

        for source_frame in range(FIRST_SOURCE_FRAME, LAST_SOURCE_FRAME + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, image = capture.read()
            assert ok and image is not None
            runtime.analyze_frame(
                image,
                monotonic_ms=int(rows[source_frame]["monotonic_ms"]),
                trace_context={
                    "capture_generation": 1,
                    "capture_seq": source_frame + 1,
                    "captured_ms": int(rows[source_frame]["monotonic_ms"]),
                    "roi_version": "historical-turn80-v1",
                    "source_id": f"avi:{SESSION.name}",
                },
            )

        actual_actions = runtime.rule_session.confirmed_actions[76:84]
        actual = tuple(_signature(action) for action in actual_actions)
        expected = tuple(
            (
                str(turn["actor"]),
                bool(turn["is_pass"]),
                tuple(turn["cards"]),
            )
            for turn in visual_truth
        )
        assert actual == expected, {
            "actual": actual,
            "candidate_frames": {
                frame: values
                for frame, values in host.candidate_signatures.items()
                if frame >= 1468 and values
            },
            "submission_revisions": {
                frame: revision
                for frame, revision in host.submission_revisions.items()
                if frame >= 1468
            },
            "deliveries": [item for item in host.deliveries if item[0] >= 1468],
        }
        assert len(actual_actions) == 8
        turn79 = actual_actions[2]
        turn80 = actual_actions[3]
        assert turn79.seat.value == "self" and turn79.kind.value == "pass"
        assert turn80.seat.value == "opposite" and turn80.kind.value == "pass"
        assert turn80.first_frame.frame_sequence - 1 > 1473
        assert turn80.first_frame.frame_sequence > turn79.last_frame.frame_sequence
        assert turn80.first_frame.captured_ms > turn79.last_frame.captured_ms
        for previous, current in zip(actual_actions, actual_actions[1:]):
            assert current.first_frame.frame_sequence > previous.last_frame.frame_sequence
            assert current.first_frame.captured_ms > previous.last_frame.captured_ms
        for action in actual_actions:
            assert len(action.evidence_ids) >= 2
            assert action.first_frame.frame_sequence < action.last_frame.frame_sequence
            assert action.first_frame.captured_ms < action.last_frame.captured_ms
    finally:
        capture.release()
        if runtime.status != "sealed":
            runtime.finish()
