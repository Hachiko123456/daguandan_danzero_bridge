"""Production composition root for the live-v2 FableDan advice service."""

from __future__ import annotations

from ..application.live_v2_advice_protocol import (
    AdviceWorkerConfig,
    WorkerHostPort,
)
from ..application.live_v2_advice_runtime import LiveV2AdviceRuntime
from ..application.live_v2_worker_protocol import WorkerReference
from ..live_v2.identity import VersionIdentity
from .live_v2_worker_host import LiveV2WorkerHost, RebindPolicy


ADVICE_WORKER_REFERENCE = WorkerReference(
    "daguandan_bridge.infrastructure.live_v2_advice_worker",
    "run_live_v2_advice_worker",
)


def create_advice_worker_host(
    *, session_id: str, capture_generation: int, state_revision: int
) -> WorkerHostPort:
    return LiveV2WorkerHost(
        ADVICE_WORKER_REFERENCE,
        session_id=session_id,
        capture_generation=capture_generation,
        state_revision=state_revision,
        rebind_policy=RebindPolicy.PRESERVE_IN_FLIGHT_WITHIN_STREAM,
    )


def create_live_v2_advice_runtime(
    config: AdviceWorkerConfig,
    version: VersionIdentity,
) -> LiveV2AdviceRuntime:
    return LiveV2AdviceRuntime(
        config,
        session_id=version.session_id,
        capture_generation=version.capture_generation,
        state_revision=version.state_revision,
        host_factory=create_advice_worker_host,
    )


__all__ = [
    "ADVICE_WORKER_REFERENCE",
    "create_advice_worker_host",
    "create_live_v2_advice_runtime",
]
