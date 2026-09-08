from __future__ import annotations

import os
from pathlib import Path
import pickle
import threading
from time import perf_counter

import cv2
import numpy as np
import pytest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.application.live_v2_frame_pipeline import LiveV2FramePipeline
from daguandan_bridge.application.live_v2_frame_types import (
    FramePipelineConfig,
    FramePipelineResult,
)
from daguandan_bridge.application.live_v2_vision_protocol import (
    VisionRequestIdentity,
    VisionRuntimeStatus,
    VisionWorkerConfig,
    VisionWorkerPayload,
    VisionWorkerSuccess,
)
from daguandan_bridge.application.live_v2_vision_runtime import LiveV2VisionRuntime
from daguandan_bridge.application.live_v2_worker_protocol import (
    WorkerReady,
    WorkerReference,
    WorkerRequest,
)
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.infrastructure.live_v2_vision_service_factory import (
    build_live_v2_vision_runtime,
)
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, VersionIdentity
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService


_STUB_STREAMS: set[tuple[object, ...]] = set()


class _ApplicationHostStub:
    state = "ready"
    worker_pid = 101
    worker_generation = 1

    def __init__(self) -> None:
        self.requests: list[WorkerRequest] = []

    def start(self, *, timeout=10.0):
        return WorkerReady(1, 101, 0)

    def bind_version(self, **kwargs):
        return None

    def submit(self, request):
        self.requests.append(request)
        return ()

    def get_result(self, *, timeout=None):
        raise TimeoutError

    def drain_results(self):
        return ()

    def restart(self, **kwargs):
        self.worker_generation += 1
        return WorkerReady(self.worker_generation, 101, 0)

    def close(self, *, timeout=5.0):
        self.state = "closed"


def _stub_vision_worker(request: WorkerRequest) -> VisionWorkerSuccess:
    payload = request.payload
    assert isinstance(payload, VisionWorkerPayload)
    marker = int(payload.image.flat[0])
    if marker == 1:
        threading.Event().wait(0.3)
    if marker == 2:
        threading.Event().wait(1.0)
    if marker == 99:
        os._exit(39)
    identity = payload.identity
    stream = (
        identity.frame.session_id,
        identity.frame.capture_generation,
        identity.frame.roi_version,
        identity.frame.source_id,
    )
    hit = stream in _STUB_STREAMS
    _STUB_STREAMS.add(stream)
    effective = Seat.SELF if payload.visual_self_opportunity else payload.expected_seat
    fast = FastSignalResult((effective or Seat.SELF).value, None, False, False, False)
    result = FramePipelineResult(identity.frame, fast, (), (), (), (), (), 0, ())
    return VisionWorkerSuccess(identity, result, os.getpid(), 0.25, hit)


def _stub_reference() -> WorkerReference:
    return WorkerReference.from_callable(_stub_vision_worker)


def _frame(sequence: int, generation: int = 1, session: str = "session-a") -> FrameIdentity:
    return FrameIdentity(
        session,
        generation,
        sequence,
        sequence * 100,
        "roi-v1",
        "window-1",
    )


def _version(
    update: int,
    generation: int = 1,
    session: str = "session-a",
    revision: int = 0,
) -> VersionIdentity:
    return VersionIdentity(session, generation, revision, update, update // 4)


def _runtime(*, worker: WorkerReference | None = None) -> LiveV2VisionRuntime:
    kwargs = {} if worker is None else {"worker": worker}
    return build_live_v2_vision_runtime(
        VisionWorkerConfig(str(PROFILES_ROOT)),
        session_id="session-a",
        capture_generation=1,
        **kwargs,
    )


def _submit(
    runtime: LiveV2VisionRuntime,
    sequence: int,
    *,
    marker: int = 0,
    generation: int = 1,
    session: str = "session-a",
    timeout_ms: int = 2_000,
) -> tuple:
    image = np.full((8, 16, 3), marker, dtype=np.uint8)
    return runtime.submit(
        image,
        frame=_frame(sequence, generation, session),
        version=_version(sequence, generation, session),
        expected_seat=Seat.LEFT,
        visual_self_opportunity=False,
        wild_rank="10",
        request_sequence=sequence,
        timeout_ms=timeout_ms,
    )


def test_latest_only_drops_queued_frame_but_accepts_completed_same_state_frames() -> None:
    runtime = _runtime(worker=_stub_reference())
    try:
        runtime.start()
        _submit(runtime, 1, marker=1)
        _submit(runtime, 2)
        immediate = _submit(runtime, 3)
        assert [(item.identity.request_sequence, item.status) for item in immediate] == [
            (2, VisionRuntimeStatus.SUPERSEDED)
        ]
        completed = [runtime.get_result(timeout=10), runtime.get_result(timeout=10)]
        by_sequence = {item.identity.request_sequence: item for item in completed}
        assert by_sequence[1].status is VisionRuntimeStatus.FRAME
        assert by_sequence[1].pipeline_result is not None
        assert by_sequence[3].status is VisionRuntimeStatus.FRAME
        assert by_sequence[3].pipeline_result is not None
        assert {item.worker_pid for item in completed} == {runtime.worker_pid}
    finally:
        runtime.close()


def test_application_runtime_accepts_only_the_injected_host_port() -> None:
    host = _ApplicationHostStub()
    runtime = LiveV2VisionRuntime(
        VisionWorkerConfig(str(PROFILES_ROOT)),
        host=host, session_id="session-a", capture_generation=1,
    )
    assert _submit(runtime, 2) == ()
    rejected = _submit(runtime, 1)
    assert rejected[0].status is VisionRuntimeStatus.REJECTED
    assert len(host.requests) == 1
    runtime.close()


def test_stale_submission_does_not_replace_the_active_identity() -> None:
    runtime = _runtime(worker=_stub_reference())
    try:
        _submit(runtime, 5)
        current = runtime.get_result(timeout=10)
        rejected = _submit(runtime, 4)
        assert current.status is VisionRuntimeStatus.FRAME
        assert rejected[0].status is VisionRuntimeStatus.REJECTED
        assert rejected[0].failure_code == "stale_update_sequence"
        assert rejected[0].pipeline_result is None

        _submit(runtime, 1, session="session-b")
        assert runtime.get_result(timeout=10).status is VisionRuntimeStatus.FRAME
        retired = _submit(runtime, 6, session="session-a")
        assert retired[0].status is VisionRuntimeStatus.REJECTED
        assert retired[0].failure_code == "stale_session"
    finally:
        runtime.close()


def test_cross_generation_restarts_process_and_clears_worker_cache() -> None:
    runtime = _runtime(worker=_stub_reference())
    try:
        _submit(runtime, 1)
        first = runtime.get_result(timeout=10)
        _submit(runtime, 2)
        second = runtime.get_result(timeout=10)
        old_pid, old_generation = second.worker_pid, second.worker_generation
        assert first.cache_hit is False and second.cache_hit is True

        _submit(runtime, 1, generation=2)
        restarted = runtime.get_result(timeout=10)
        assert restarted.status is VisionRuntimeStatus.FRAME
        assert restarted.cache_hit is False
        assert restarted.worker_pid != old_pid
        assert restarted.worker_generation > old_generation
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("marker", "timeout_ms", "expected"),
    ((2, 50, VisionRuntimeStatus.WORKER_TIMEOUT), (99, 2_000, VisionRuntimeStatus.WORKER_CRASHED)),
)
def test_timeout_and_crash_are_structured_without_empty_observations(
    marker: int,
    timeout_ms: int,
    expected: VisionRuntimeStatus,
) -> None:
    runtime = _runtime(worker=_stub_reference())
    try:
        _submit(runtime, 1, marker=marker, timeout_ms=timeout_ms)
        failed = runtime.get_result(timeout=10)
        assert failed.status is expected
        assert failed.pipeline_result is None
        previous_pid = failed.worker_pid
        previous_generation = failed.worker_generation
        _submit(runtime, 2)
        recovered = runtime.get_result(timeout=10)
        assert recovered.status is VisionRuntimeStatus.FRAME
        assert recovered.worker_pid != previous_pid
        assert recovered.worker_generation == previous_generation + 1
    finally:
        runtime.close()


def test_close_is_idempotent_and_closed_submit_has_no_observation() -> None:
    runtime = _runtime(worker=_stub_reference())
    runtime.start()
    runtime.close()
    runtime.close()
    closed = _submit(runtime, 1)
    assert closed[0].status is VisionRuntimeStatus.SERVICE_CLOSED
    assert closed[0].pipeline_result is None


def test_payload_contains_one_owned_full_frame_and_serialization_is_measured(
    record_property,
) -> None:
    source = np.zeros((72, 128, 3), dtype=np.uint8)
    identity = VisionRequestIdentity(_frame(1), _version(1), 1)
    payload = VisionWorkerPayload(
        VisionWorkerConfig(str(PROFILES_ROOT)),
        identity,
        Seat.RIGHT,
        True,
        "10",
        source,
    )
    arrays = [value for value in vars_for_slots(payload) if isinstance(value, np.ndarray)]
    started = perf_counter()
    serialized = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed_ms = (perf_counter() - started) * 1000
    record_property("serialized_bytes", len(serialized))
    record_property("serialization_elapsed_ms", elapsed_ms)
    assert arrays == [source]
    assert len(serialized) >= source.nbytes
    assert elapsed_ms >= 0


def test_production_worker_matches_synchronous_pipeline_on_historical_frames() -> None:
    video = (
        PROFILES_ROOT
        / "tencent_daguandan/sessions/game_20260814_004447_aab3dc/video/game.avi"
    )
    if not video.is_file():
        pytest.skip(f"missing historical smoke video: {video}")
    capture = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    try:
        for _ in range(2):
            ok, image = capture.read()
            if ok:
                frames.append(image)
    finally:
        capture.release()
    if len(frames) < 2:
        pytest.skip("historical smoke video did not decode two frames")

    config = VisionWorkerConfig(
        str(PROFILES_ROOT),
        pipeline=FramePipelineConfig(max_deep_reads_per_frame=1),
    )
    recognition = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
        diagnostic_tracing=False,
    )
    synchronous = LiveV2FramePipeline(recognition, config=config.pipeline, surface_config=config.surface)
    runtime = build_live_v2_vision_runtime(
        config,
        session_id="historical-smoke",
        capture_generation=1,
    )
    try:
        for sequence, image in enumerate(frames, 1):
            frame = FrameIdentity("historical-smoke", 1, sequence, sequence * 100, "roi-v1", "avi")
            version = VersionIdentity("historical-smoke", 1, 0, sequence, 0)
            expected = synchronous.process_frame(
                image, frame=frame, version=version, wild_rank="10", expected_seat=Seat.LEFT
            )
            runtime.submit(
                image,
                frame=frame,
                version=version,
                expected_seat=Seat.LEFT,
                visual_self_opportunity=False,
                wild_rank="10",
                request_sequence=sequence,
                timeout_ms=20_000,
            )
            actual = runtime.get_result(timeout=25)
            assert actual.status is VisionRuntimeStatus.FRAME
            assert actual.worker_pid != os.getpid()
            assert actual.pipeline_result is not None
            assert _semantic(actual.pipeline_result) == _semantic(expected)
            assert actual.cache_hit is (sequence > 1)
            assert actual.end_to_end_ms >= 0
            assert actual.elapsed_ms >= 0
    finally:
        runtime.close()


def vars_for_slots(value: object) -> tuple[object, ...]:
    return tuple(getattr(value, name) for name in value.__slots__)


def _semantic(result: FramePipelineResult) -> tuple[object, ...]:
    observations = tuple(
        (item.seat, item.kind, item.cards, item.suit_options, item.reason, item.confidence)
        for item in result.observations
    )
    candidates = tuple(
        (item.seat, item.kind, item.cards, item.suit_options, item.reason, item.confidence)
        for item in result.candidates
    )
    metrics = tuple(
        (
            item.seat,
            item.fingerprint,
            item.content_changed,
            item.visible_surface,
            item.animating,
            item.pass_visible,
        )
        for item in result.surface_metrics
    )
    return result.fast_signals, metrics, observations, candidates, result.pending_seats


def test_vision_modules_remain_small() -> None:
    root = Path(__file__).parents[1]
    paths = (
        root / "src/daguandan_bridge/application/live_v2_vision_protocol.py",
        root / "src/daguandan_bridge/application/live_v2_vision_runtime.py",
        root / "src/daguandan_bridge/infrastructure/live_v2_vision_worker.py",
        root / "src/daguandan_bridge/infrastructure/live_v2_vision_service_factory.py",
        Path(__file__),
    )
    assert all(len(path.read_text(encoding="utf-8").splitlines()) < 400 for path in paths)
