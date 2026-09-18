from __future__ import annotations

import numpy as np
import pytest

from daguandan_bridge.application.live_v2_frame_types import FramePipelineResult
from daguandan_bridge.application.live_v2_runtime_updates import consume_vision
from daguandan_bridge.application.live_v2_vision_protocol import (
    VisionRequestIdentity, VisionRuntimeResult, VisionRuntimeStatus,
    VisionWorkerConfig, VisionWorkerPayload, VisionWorkerSuccess,
)
from daguandan_bridge.application.live_v2_vision_runtime import LiveV2VisionRuntime
from daguandan_bridge.application.live_v2_worker_protocol import (
    WorkerReady, WorkerRequest, WorkerResult, WorkerResultStatus,
)
from daguandan_bridge.domain.recognition import FastSignalResult
from daguandan_bridge.live_v2.candidates import (
    ActionCandidate, ActionKind, CandidateReason,
)
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, VersionIdentity
from daguandan_bridge.live_v2.observations import (
    ObservationKind, ObservationReason, SeatObservation,
)


class Host:
    state, worker_pid, worker_generation = "ready", 101, 1

    def __init__(self) -> None:
        self.requests: list[WorkerRequest] = []
        self.results: list[WorkerResult] = []

    def start(self, *, timeout=10.0): return WorkerReady(1, 101, 0)
    def bind_version(self, **kwargs): pass
    def submit(self, request): self.requests.append(request); return ()
    def get_result(self, *, timeout=None): return self.results.pop(0)
    def drain_results(self): values, self.results = tuple(self.results), []; return values
    def restart(self, **kwargs):
        self.worker_generation += 1
        return WorkerReady(self.worker_generation, 101, 0)
    def close(self, *, timeout=5.0): self.state = "closed"


def frame(sequence: int, generation: int) -> FrameIdentity:
    return FrameIdentity("session-a", generation, sequence, sequence * 100, "roi", "window")


def submit(runtime: LiveV2VisionRuntime, version: VersionIdentity, sequence: int) -> None:
    runtime.submit(
        np.zeros((4, 8, 3), dtype=np.uint8),
        frame=frame(sequence, version.capture_generation), version=version,
        expected_seat=Seat.RIGHT, visual_self_opportunity=False,
        wild_rank="10", request_sequence=sequence,
    )


def success(request: WorkerRequest) -> WorkerResult:
    identity = request.payload.identity
    result = FramePipelineResult(
        identity.frame, FastSignalResult("right", None, False, False, False),
        (), (), (), (), (), 0, (),
    )
    return WorkerResult.terminal(
        request, status=WorkerResultStatus.SUCCESS, worker_generation=1,
        worker_pid=101, finished_processing_ms=request.submitted_processing_ms + 1,
        payload=VisionWorkerSuccess(identity, result, 101, 0.1, False),
    )


@pytest.mark.parametrize(
    "latest",
    (
        VersionIdentity("session-a", 1, 2, 2, 1),
        VersionIdentity("session-a", 1, 1, 2, 1),
    ),
)
def test_success_from_old_formal_state_in_same_stream_remains_a_frame(latest) -> None:
    host = Host()
    runtime = LiveV2VisionRuntime(
        VisionWorkerConfig("profiles"), host=host,
        session_id="session-a", capture_generation=1,
    )
    submit(runtime, VersionIdentity("session-a", 1, 1, 1, 0), 1)
    submit(runtime, latest, 2)
    host.results.append(success(host.requests[0]))
    result = runtime.drain_results()[0]
    assert result.status is VisionRuntimeStatus.FRAME
    assert result.identity.version == VersionIdentity("session-a", 1, 1, 1, 0)
    assert result.pipeline_result.frame == frame(1, 1)


def test_process_frame_sync_waits_for_exact_frame_identity():
    host = Host()
    runtime = LiveV2VisionRuntime(
        VisionWorkerConfig("profiles"), host=host,
        session_id="session-a", capture_generation=1,
    )
    # Leave an older result in the host queue. The synchronous API must not
    # return it merely because it is a valid FRAME result.
    submit(runtime, VersionIdentity("session-a", 1, 1, 1, 0), 1)
    host.results.append(success(host.requests[0]))
    requested_version = VersionIdentity("session-a", 1, 1, 2, 0)
    requested_frame = frame(2, 1)
    host.results.append(
        success(WorkerRequest(
            session_id="session-a", capture_generation=1, request_sequence=2,
            state_revision=1, submitted_processing_ms=0,
            payload=VisionWorkerPayload(
                VisionWorkerConfig("profiles"),
                VisionRequestIdentity(requested_frame, requested_version, 2),
                Seat.RIGHT, False, "10", np.zeros((4, 8, 3), dtype=np.uint8),
            ),
        ))
    )
    result = runtime.process_frame_sync(
        np.zeros((4, 8, 3), dtype=np.uint8), frame=requested_frame,
        version=requested_version, expected_seat=Seat.RIGHT,
        wild_rank="10", request_sequence=2,
    )
    assert result.frame == requested_frame
    assert result.frame.frame_sequence == 2


def test_success_from_old_capture_generation_is_rejected() -> None:
    host = Host()
    runtime = LiveV2VisionRuntime(
        VisionWorkerConfig("profiles"), host=host,
        session_id="session-a", capture_generation=1,
    )
    submit(runtime, VersionIdentity("session-a", 1, 1, 1, 0), 1)
    submit(runtime, VersionIdentity("session-a", 2, 1, 2, 0), 2)
    host.results.append(success(host.requests[0]))
    result = runtime.drain_results()[0]
    assert result.status is VisionRuntimeStatus.REJECTED
    assert result.failure_code == "stale_capture_generation"
    assert result.pipeline_result is None


class CompletedVision:
    def __init__(self, result: VisionRuntimeResult) -> None:
        self.result = result

    def submit(self, *_args, **_kwargs): return (self.result,)
    def drain_results(self): return ()


def _candidate(
    candidate_id: str, version: VersionIdentity, first: int, last: int,
) -> ActionCandidate:
    return ActionCandidate(
        candidate_id, version, Seat.OPPOSITE, ActionKind.PLAY, ("5D",),
        (("5D",),), (candidate_id + "-1", candidate_id + "-2"), 1,
        frame(first, 1), frame(last, 1), last * 100 + 1, 0.95,
        CandidateReason.STABLE_PLAY,
    )


def test_consume_salvages_only_post_boundary_candidates_from_ancestor_state() -> None:
    old = VersionIdentity("session-a", 1, 1, 1, 0)
    current = VersionIdentity("session-a", 1, 2, 2, 1)
    observation = SeatObservation(
        "old-observation", frame(4, 1), Seat.OPPOSITE, ObservationKind.EMPTY,
        (), 0.9, ObservationReason.STABLE_EMPTY, 401,
    )
    pipeline = FramePipelineResult(
        frame(4, 1), FastSignalResult("opposite", None, False, False, False),
        (), (observation,),
        (
            _candidate("overlap", old, 1, 3),
            _candidate("boundary-witness", old, 2, 3),
            _candidate("next", old, 3, 4),
        ),
        (), (), 0, (),
    )
    runtime_result = VisionRuntimeResult(
        VisionRequestIdentity(frame(4, 1), old, 4),
        VisionRuntimeStatus.FRAME, 1, 101, pipeline_result=pipeline,
    )
    results, failures = consume_vision(
        CompletedVision(runtime_result), np.zeros((4, 8, 3), dtype=np.uint8),
        frame=frame(5, 1), version=current, wild_rank="10",
        expected_seat=Seat.OPPOSITE, processing_ms=501,
        formal_action_boundary=frame(2, 1),
    )
    assert failures == () and len(results) == 1
    assert results[0].observations == ()
    assert [item.candidate_id for item in results[0].candidates] == [
        "boundary-witness", "next"
    ]
    assert results[0].candidates[0].version == current
