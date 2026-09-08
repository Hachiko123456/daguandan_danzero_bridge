"""Production composition root for the live-v2 vision runtime."""

from __future__ import annotations

from ..application.live_v2_vision_protocol import VisionWorkerConfig
from ..application.live_v2_vision_runtime import LiveV2VisionRuntime
from ..application.live_v2_worker_protocol import WorkerReference
from .live_v2_worker_host import LiveV2WorkerHost, RebindPolicy


DEFAULT_VISION_WORKER = WorkerReference(
    "daguandan_bridge.infrastructure.live_v2_vision_worker",
    "run_live_v2_vision_worker",
)


def build_live_v2_vision_runtime(
    config: VisionWorkerConfig,
    *,
    session_id: str,
    capture_generation: int,
    state_revision: int = 0,
    worker: WorkerReference = DEFAULT_VISION_WORKER,
) -> LiveV2VisionRuntime:
    """Wire the application runtime to the concrete process supervisor."""

    host = LiveV2WorkerHost(
        worker,
        session_id=session_id,
        capture_generation=capture_generation,
        state_revision=state_revision,
        rebind_policy=RebindPolicy.PRESERVE_IN_FLIGHT_WITHIN_STREAM,
        fifo_capacity=0,
    )
    return LiveV2VisionRuntime(
        config,
        host=host,
        session_id=session_id,
        capture_generation=capture_generation,
    )


__all__ = ["DEFAULT_VISION_WORKER", "build_live_v2_vision_runtime"]
