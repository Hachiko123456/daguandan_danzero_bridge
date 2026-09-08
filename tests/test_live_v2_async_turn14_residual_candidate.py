from __future__ import annotations

import cv2

from daguandan_bridge.application.live_v2_frame_pipeline import (
    FramePipelineConfig,
    LiveV2FramePipeline,
)
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.application.live_v2_vision_protocol import VisionWorkerConfig
from daguandan_bridge.domain.recognition import PlayRegionResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.identity import Seat

from test_live_v2_async_visual_production_regression import (
    ROOT,
    SESSION,
    AuditedVisionRuntime,
    DelayedPipelineHost,
    MemoryRecorder,
    MemoryStore,
    NoAdviceRuntime,
    TruthFrameRecognition,
    _fixture,
    _signature,
)


FIRST_SOURCE_FRAME = 350
LAST_SOURCE_FRAME = 430
RESIDUAL_FRAMES = range(382, 385)
RESIDUAL_CARDS = ("8S", "9S", "10S", "JS", "QS")
RESIDUAL_OPTIONS = tuple((card, f"{card[:-1]}D") for card in RESIDUAL_CARDS)


class Turn14ResidualRecognition(TruthFrameRecognition):
    """Truth adapter retaining one unrelated, suit-uncertain right surface."""

    def play_roi(self, image, seat):
        roi = super().play_roi(image, seat)
        if Seat(seat) is Seat.RIGHT and self.source_frame in RESIDUAL_FRAMES:
            roi[8:40, 12:84] = 242
            roi[18:30, 24:72] = 18
        return roi

    def recognize_play_region(self, image, seat, **kwargs):
        if Seat(seat) is Seat.RIGHT and self.source_frame in RESIDUAL_FRAMES:
            return PlayRegionResult(
                Seat.RIGHT.value,
                RESIDUAL_CARDS,
                False,
                0.99,
                ("residual_uncertain_right",),
                (),
                suit_options=RESIDUAL_OPTIONS,
            )
        return super().recognize_play_region(image, seat, **kwargs)


class FinalDrainDelayedHost(DelayedPipelineHost):
    """Let final-frame work finish after capture stops, without another frame."""

    def __init__(self, pipeline, recognition) -> None:
        super().__init__(pipeline, recognition)
        self.raw_candidates = []
        self.scheduled_delays: list[int] = []
        self._final_drain_advanced = False

    def submit(self, request):
        values = super().submit(request)
        success = self.pending[-1][1].payload
        self.raw_candidates.extend(success.pipeline_result.candidates)
        self.scheduled_delays.append(int(success.elapsed_ms))
        return values

    def drain_results(self):
        if (
            self.analyzed_source_frames
            and self.analyzed_source_frames[-1] == LAST_SOURCE_FRAME
            and not self._final_drain_advanced
        ):
            self.now_ms += 720
            self._final_drain_advanced = True
        return super().drain_results()


def _strictly_after(current, previous) -> bool:
    if current.captured_ms <= previous.captured_ms:
        return False
    current_stream = (
        current.session_id,
        current.capture_generation,
        current.roi_version,
        current.source_id,
    )
    previous_stream = (
        previous.session_id,
        previous.capture_generation,
        previous.roi_version,
        previous.source_id,
    )
    return (
        current_stream != previous_stream
        or current.frame_sequence > previous.frame_sequence
    )


def test_turn14_self_play_is_not_blocked_by_unrelated_right_residual_candidate() -> None:
    initial, all_turns, frame_rows = _fixture()
    visual_turns = all_turns[13:18]
    recognition = Turn14ResidualRecognition(visual_turns)
    pipeline = LiveV2FramePipeline(
        recognition, config=FramePipelineConfig(max_deep_reads_per_frame=4)
    )
    host = FinalDrainDelayedHost(pipeline, recognition)
    vision = AuditedVisionRuntime(
        VisionWorkerConfig(str(ROOT / "data/profiles")),
        host=host,
        session_id=MemoryStore.session_id,
        capture_generation=1,
    )
    store = MemoryStore()
    store.start({"schema": "test.async-turn14-residual/1"})
    runtime = LiveV2SessionRuntime(
        rule_session=ProductionRuleSession(store),
        store=store,
        recorder=MemoryRecorder(),
        recognition_service=recognition,
        vision_factory=lambda version: vision,
        advice_runtime_factory=lambda version: NoAdviceRuntime(),
        processing_clock_ms=lambda: host.now_ms,
        roi_version="historical-avi-v1",
        source_id=f"avi:{SESSION.name}",
        local_hint_window_ms=0,
    )
    capture = cv2.VideoCapture(str(SESSION / "video/game.avi"))
    try:
        runtime.start(
            round_level=str(initial["round_level"]),
            hand=tuple(str(card) for card in initial["my_hand"]),
            lead_player=str(initial["lead_player"]),
            monotonic_ms=0,
        )
        runtime.bind_capture_generation(1)
        for index, turn in enumerate(all_turns[:13], 1):
            runtime.commit_trusted_action(
                actor=str(turn["actor"]),
                cards=tuple(str(card) for card in turn["cards"]),
                is_pass=bool(turn["is_pass"]),
                monotonic_ms=index,
                confidence=1.0,
            )
        assert runtime.snapshot.current_player == Seat.SELF.value

        capture.set(cv2.CAP_PROP_POS_FRAMES, FIRST_SOURCE_FRAME)
        for source_frame in range(FIRST_SOURCE_FRAME, LAST_SOURCE_FRAME + 1):
            ok, image = capture.read()
            assert ok, f"historical AVI failed to decode source frame {source_frame}"
            row = frame_rows[source_frame]
            runtime.analyze_frame(
                image,
                monotonic_ms=int(row["monotonic_ms"]),
                trace_context={
                    "capture_generation": 1,
                    "capture_seq": source_frame + 1,
                    "captured_ms": int(row["monotonic_ms"]),
                    "roi_version": "historical-avi-v1",
                    "source_id": f"avi:{SESSION.name}",
                },
            )

        actions = runtime.rule_session.confirmed_actions
        residuals = [
            item for item in host.raw_candidates
            if item.seat is Seat.RIGHT and item.cards == RESIDUAL_CARDS
        ]
        actual_visual = tuple(_signature(item) for item in actions[13:17])
        expected_visual = tuple(
            (turn["actor"], turn["is_pass"], turn["cards"])
            for turn in all_turns[13:17]
        )
        assert len(actions) == 17, (
            "turn14 self JC JC JD JH was blocked by the unrelated right "
            f"residual; committed={actual_visual!r}"
        )
        assert actual_visual == expected_visual
        assert residuals
        assert any(all(len(options) > 1 for options in item.suit_options) for item in residuals)

        turn18 = all_turns[17]
        runtime.commit_trusted_action(
            actor=str(turn18["actor"]),
            cards=tuple(str(card) for card in turn18["cards"]),
            is_pass=bool(turn18["is_pass"]),
            monotonic_ms=int(frame_rows[int(turn18["source_frame"])]["monotonic_ms"]),
            confidence=1.0,
        )
        actions = runtime.rule_session.confirmed_actions
        assert len(actions) == 18
        assert _signature(actions[17]) == (
            turn18["actor"], turn18["is_pass"], turn18["cards"],
        )

        assert host.analyzed_source_frames == list(
            range(FIRST_SOURCE_FRAME, LAST_SOURCE_FRAME + 1)
        )
        assert len(host.scheduled_delays) == (
            LAST_SOURCE_FRAME - FIRST_SOURCE_FRAME + 1
        )
        assert all(480 <= delay <= 720 for delay in host.scheduled_delays)
        for previous, current in zip(actions[12:18], actions[13:18]):
            assert _strictly_after(current.first_frame, previous.last_frame)
    finally:
        capture.release()
        if runtime.status != "sealed":
            runtime.finish()
