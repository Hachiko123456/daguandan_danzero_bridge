from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from daguandan_bridge.application.live_v2_advice_protocol import (
    AdviceRequestIdentity,
    AdviceRuntimeResult,
    AdviceRuntimeStatus,
)
from daguandan_bridge.application.live_v2_frame_pipeline import (
    FramePipelineConfig,
    LiveV2FramePipeline,
)
from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.application.live_v2_vision_protocol import (
    VisionRuntimeResult,
    VisionRuntimeStatus,
    VisionWorkerConfig,
    VisionWorkerSuccess,
)
from daguandan_bridge.application.live_v2_vision_runtime import LiveV2VisionRuntime
from daguandan_bridge.application.live_v2_worker_protocol import (
    WorkerReady,
    WorkerRequest,
    WorkerResult,
    WorkerResultStatus,
)
from daguandan_bridge.domain.recognition import FastSignalResult, PlayRegionResult
from daguandan_bridge.domain.recording import RecordingResult
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat


ROOT = Path(__file__).parents[1]
SESSION = (
    ROOT
    / "data/profiles/tencent_daguandan/sessions/game_20260814_004447_aab3dc"
)
LAST_SOURCE_FRAME = 750
ACTION_LIMIT = 30


class MemoryStore:
    session_id = "async-visual-production-regression"
    directory = Path("memory-async-visual-production-regression")
    persistence_enabled = True
    automatic_log_delivery_enabled = False

    def __init__(self) -> None:
        self.batches: list[tuple[object, ...]] = []

    def start(self, manifest): self.manifest = manifest
    def append_event(self, event): pass
    def append_event_batch(self, events): self.batches.append(tuple(events))
    def append_advice(self, record): pass
    def append_observation(self, record): pass
    def append_recognition_trace(self, record): pass
    def update_runtime_identity(self, identity): pass
    def upsert_decision(self, record): pass
    def create_incident(self, **kwargs): return self.directory
    def append_incident_occurrence(self, *args, **kwargs): pass
    def seal(self, **kwargs): pass
    def append_post_seal_health_audit(self, report, **kwargs): pass
    def record_automatic_log_delivery(self, result): pass


class MemoryRecorder:
    frame_count = 0

    def write_frame(self, frame, monotonic_ms, wall_time): return None

    def close(self):
        return RecordingResult(Path("game.avi"), Path("frame_index.jsonl"), 0, 0)


class NoAdviceRuntime:
    def start(self, *, timeout=10.0): pass
    def close(self, *, timeout=5.0): pass
    def drain_results(self): return ()

    def submit(self, snapshot, opportunity, *, request_sequence, timeout_ms=3000):
        identity = AdviceRequestIdentity(
            opportunity.version, request_sequence, opportunity.opportunity_id
        )
        return (AdviceRuntimeResult(
            identity, AdviceRuntimeStatus.BLOCKED, 1, 91_001,
            failure_code="acceptance_test_no_adviser",
        ),)


class TruthFrameRecognition:
    """Cheap recorded-result adapter; source pixels still traverse the real pipeline."""

    def __init__(self, turns: tuple[dict[str, object], ...]) -> None:
        self.turns = turns
        self.source_frame = -1
        self.full_frame_shapes: list[tuple[int, ...]] = []

    def bind(self, frame: FrameIdentity, image: np.ndarray) -> None:
        self.source_frame = frame.frame_sequence - 1
        self.full_frame_shapes.append(tuple(image.shape))

    def _visible(self, turn: dict[str, object]) -> bool:
        target = int(turn["source_frame"])
        return target - 2 <= self.source_frame <= target

    def _turns_for(self, seat: Seat) -> tuple[dict[str, object], ...]:
        return tuple(
            turn for turn in self.turns
            if turn["actor"] == seat.value and self._visible(turn)
        )

    def play_roi(self, image, seat):
        del image
        seat = Seat(seat)
        roi = np.zeros((48, 96, 3), dtype=np.uint8)
        roi[:, :] = (36, 96, 36)
        if any(not bool(turn["is_pass"]) for turn in self._turns_for(seat)):
            roi[8:40, 12:84] = 242
            roi[18:30, 24:72] = 18
        return roi

    def recognize_fast_signals(self, image, expected_player, *, allow_pass=True):
        del image
        assert allow_pass is True
        expected = Seat(expected_player)
        passes = tuple(
            Seat(str(turn["actor"]))
            for turn in self.turns
            if bool(turn["is_pass"]) and self._visible(turn)
        )
        visible = next((turn for turn in self.turns if self._visible(turn)), None)
        if visible is not None and bool(visible["is_pass"]):
            # The worker may still be bound to the preceding formal turn.  A
            # real screenshot can retain both the old and new seat markers;
            # keep the expected marker so crossed-pass proof remains possible.
            passes = tuple(dict.fromkeys((expected,) + passes))
        active = None
        self_buttons = False
        if visible is not None:
            actor = Seat(str(visible["actor"]))
            self_buttons = actor is Seat.SELF and not bool(visible["is_pass"])
            if bool(visible["is_pass"]):
                index = self.turns.index(visible)
                if index + 1 < len(self.turns):
                    active = str(self.turns[index + 1]["actor"])
            else:
                active = actor.value
        return FastSignalResult(
            expected.value,
            active,
            expected in passes,
            self_buttons,
            False,
            expected.value if expected in passes else None,
            tuple(seat.value for seat in passes),
        )

    def recognize_play_region(self, image, seat, **kwargs):
        del image, kwargs
        seat = Seat(seat)
        visible = self._turns_for(seat)
        if not visible:
            return PlayRegionResult(seat.value, (), False, 0.0, (), ())
        turn = visible[-1]
        if bool(turn["is_pass"]):
            return PlayRegionResult(seat.value, (), True, 0.99, (), ())
        cards = tuple(str(card) for card in turn["cards"])
        return PlayRegionResult(
            seat.value, cards, False, 0.99,
            (f"source_frame={self.source_frame}",), (),
            suit_options=tuple((card,) for card in cards),
        )


class DelayedPipelineHost:
    """Process-host contract with a stable PID and deterministic completion clock."""

    state = "new"
    worker_pid = 91_337
    worker_generation = 1

    def __init__(
        self, pipeline: LiveV2FramePipeline, recognition: TruthFrameRecognition
    ) -> None:
        self.pipeline = pipeline
        self.recognition = recognition
        self.now_ms = 0
        self.pending: list[tuple[int, WorkerResult]] = []
        self.analyzed_source_frames: list[int] = []
        self.candidate_signatures: dict[int, tuple[tuple[object, ...], ...]] = {}
        self.bindings: list[tuple[int, int]] = []
        self.submission_revisions: dict[int, int] = {}
        self.deliveries: list[tuple[int, int]] = []
        self.restart_calls = 0

    def start(self, *, timeout=10.0):
        self.state = "ready"
        return WorkerReady(self.worker_generation, self.worker_pid, self.now_ms)

    def bind_version(self, *, session_id, capture_generation, state_revision):
        del session_id, capture_generation
        self.bindings.append((state_revision, self.worker_pid))

    def submit(self, request: WorkerRequest):
        payload = request.payload
        frame = payload.identity.frame
        self.now_ms = frame.captured_ms
        source_frame = frame.frame_sequence - 1
        self.analyzed_source_frames.append(source_frame)
        self.submission_revisions[source_frame] = payload.identity.version.state_revision
        self.recognition.bind(frame, payload.image)
        result = self.pipeline.process_frame(
            payload.image,
            frame=frame,
            version=payload.identity.version,
            wild_rank=payload.wild_rank,
            expected_seat=payload.expected_seat,
            now_ms=frame.captured_ms,
            formal_action_boundary=payload.formal_action_boundary,
        )
        self.candidate_signatures[source_frame] = tuple(
            (item.seat.value, item.kind.value, item.cards) for item in result.candidates
        )
        delay_ms = 480 + source_frame % 4 * 80
        success = VisionWorkerSuccess(
            payload.identity, result, self.worker_pid, float(delay_ms), False
        )
        completed = WorkerResult.terminal(
            request,
            status=WorkerResultStatus.SUCCESS,
            worker_generation=self.worker_generation,
            worker_pid=self.worker_pid,
            started_processing_ms=request.submitted_processing_ms,
            finished_processing_ms=request.submitted_processing_ms + delay_ms,
            payload=success,
        )
        self.pending.append((frame.captured_ms + delay_ms, completed))
        return ()

    def drain_results(self):
        ready = tuple(item for due, item in self.pending if due <= self.now_ms)
        self.pending = [(due, item) for due, item in self.pending if due > self.now_ms]
        current_source_frame = self.analyzed_source_frames[-1]
        self.deliveries.extend(
            (item.payload.identity.frame.frame_sequence - 1, current_source_frame)
            for item in ready
        )
        return ready

    def get_result(self, *, timeout=None):
        del timeout
        values = self.drain_results()
        if not values:
            raise TimeoutError("no virtual completion is due")
        return values[0]

    def restart(self, **kwargs):
        del kwargs
        self.restart_calls += 1
        self.worker_generation += 1
        self.state = "ready"
        return WorkerReady(self.worker_generation, self.worker_pid, self.now_ms)

    def close(self, *, timeout=5.0):
        del timeout
        self.state = "closed"


class AuditedVisionRuntime(LiveV2VisionRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.completions: list[VisionRuntimeResult] = []

    def submit(self, *args, **kwargs):
        values = super().submit(*args, **kwargs)
        self.completions.extend(values)
        return values

    def drain_results(self):
        values = super().drain_results()
        self.completions.extend(values)
        return values

def _fixture() -> tuple[dict[str, object], tuple[dict[str, object], ...], list[dict]]:
    truth = json.loads((SESSION / "truth_log.json").read_text(encoding="utf-8"))
    turns = tuple({
        "turn_id": int(turn["turn_id"]),
        "actor": str(turn["actor"]),
        "is_pass": bool(turn["is_pass"]),
        "cards": tuple(str(card) for card in turn["cards"]),
        "source_frame": int(turn["evidence"]["frame_indices"][0]),
    } for turn in truth["turns"][:ACTION_LIMIT])
    rows = [
        json.loads(line)
        for line in (SESSION / "video/frame_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return truth["initial_state"], turns, rows


def _signature(value) -> tuple[object, ...]:
    return value.seat.value, value.kind.value == "pass", value.cards


def test_async_real_frame_pipeline_keeps_old_state_completions_and_one_worker_pid() -> None:
    initial, turns, frame_rows = _fixture()
    assert turns[-1]["source_frame"] < LAST_SOURCE_FRAME
    assert len(frame_rows) > LAST_SOURCE_FRAME
    assert all(
        int(left["source_frame"]) < int(right["source_frame"])
        for left, right in zip(turns, turns[1:])
    )

    recognition = TruthFrameRecognition(turns)
    pipeline = LiveV2FramePipeline(
        recognition, config=FramePipelineConfig(max_deep_reads_per_frame=4)
    )
    host = DelayedPipelineHost(pipeline, recognition)
    vision = AuditedVisionRuntime(
        VisionWorkerConfig(str(ROOT / "data/profiles")),
        host=host,
        session_id=MemoryStore.session_id,
        capture_generation=1,
    )
    store = MemoryStore()
    store.start({"schema": "test.async-visual-production/1"})
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
            monotonic_ms=int(frame_rows[0]["monotonic_ms"]),
        )
        runtime.bind_capture_generation(1)
        for source_frame, row in enumerate(frame_rows[: LAST_SOURCE_FRAME + 1]):
            ok, image = capture.read()
            assert ok, f"historical AVI failed to decode source frame {source_frame}"
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

        expected = tuple(
            (turn["actor"], turn["is_pass"], turn["cards"]) for turn in turns
        )
        actual_actions = runtime.rule_session.confirmed_actions[:ACTION_LIMIT]
        actual = tuple(_signature(item) for item in actual_actions)

        assert host.analyzed_source_frames == list(range(LAST_SOURCE_FRAME + 1))
        assert len(set(host.analyzed_source_frames)) == LAST_SOURCE_FRAME + 1
        assert all(shape[:2] == recognition.full_frame_shapes[0][:2]
                   for shape in recognition.full_frame_shapes)
        assert host.restart_calls == 0
        assert {pid for _revision, pid in host.bindings} == {host.worker_pid}
        assert len({revision for revision, _pid in host.bindings}) > 1
        assert any(
            host.candidate_signatures.get(submitted_frame)
            and host.submission_revisions[delivered_frame]
            > host.submission_revisions[submitted_frame]
            for submitted_frame, delivered_frame in host.deliveries
        ), "fixture must deliver a candidate after the formal revision advanced"
        for previous, current in zip(actual_actions, actual_actions[1:]):
            assert current.first_frame.frame_sequence > previous.last_frame.frame_sequence
            assert current.first_frame.captured_ms > previous.last_frame.captured_ms

        if actual != expected:
            mismatch = next(
                index for index in range(ACTION_LIMIT)
                if index >= len(actual) or actual[index] != expected[index]
            )
            stale_candidate_frames = [
                result.identity.frame.frame_sequence - 1
                for result in vision.completions
                if result.status is VisionRuntimeStatus.REJECTED
                and result.failure_code == "stale_state_revision"
                and host.candidate_signatures.get(
                    result.identity.frame.frame_sequence - 1, ()
                )
            ]
            first_stale = min(stale_candidate_frames) if stale_candidate_frames else None
            turn = turns[mismatch]
            if first_stale is not None:
                candidate_signatures = host.candidate_signatures[first_stale]
                stale_turn = next(
                    item for item in turns
                    if abs(int(item["source_frame"]) - first_stale) <= 2
                    and (
                        item["actor"],
                        "pass" if item["is_pass"] else "play",
                        item["cards"],
                    ) in candidate_signatures
                )
                raise AssertionError(
                    "first old-state visual completion discarded: "
                    f"truth_action={stale_turn['turn_id']}, "
                    f"source_frame={first_stale}, "
                    f"submitted_revision={host.submission_revisions[first_stale]}, "
                    f"candidate={candidate_signatures!r}"
                )
            raise AssertionError(
                "visual action signature diverged before truth action "
                f"{mismatch + 1}: truth_source_frame={turn['source_frame']}, "
                f"expected={expected[mismatch]!r}, "
                f"actual={actual[mismatch] if mismatch < len(actual) else '<missing>'!r}"
            )
    finally:
        capture.release()
        if runtime.status != "sealed":
            runtime.finish()
