from __future__ import annotations

from dataclasses import replace
import os
from time import perf_counter

import pytest

from daguandan_bridge.application.live_v2_advice_protocol import AdviceWorkerConfig
from daguandan_bridge.infrastructure.live_v2_advice_service_factory import (
    create_live_v2_advice_runtime,
)
from daguandan_bridge.live_v2.game_state import SeatCardCount, TrustedGameSnapshot
from daguandan_bridge.live_v2.identity import Seat, VersionIdentity
from daguandan_bridge.live_v2.results import (
    AdviceOpportunity, OpportunityReason, OpportunityStatus,
)
from daguandan_bridge.config import PROFILES_ROOT


def _snapshot() -> TrustedGameSnapshot:
    version = VersionIdentity("backend-smoke", 1, 0, 0, 0)
    hand = tuple(f"{rank}{suit}" for rank in "3456789" for suit in "HDSC")[:27]
    return TrustedGameSnapshot(
        version, "6", "6", 1, Seat.SELF, Seat.SELF, hand, (), (),
        tuple(SeatCardCount(seat, 27) for seat in Seat), (), True, False, 100,
    )


def _opportunity(
    snapshot: TrustedGameSnapshot, backend: str, sequence: int = 1
) -> AdviceOpportunity:
    return AdviceOpportunity(
        f"backend-{backend}-{sequence}", snapshot.version, Seat.SELF,
        OpportunityStatus.READY, OpportunityReason.TRUSTED_STATE, 100, 110,
    )


def test_advice_backend_is_required_and_unknown_values_fail_closed() -> None:
    with pytest.raises(TypeError):
        AdviceWorkerConfig(str(PROFILES_ROOT))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="advisor_backend"):
        AdviceWorkerConfig(str(PROFILES_ROOT), "unknown")


@pytest.mark.parametrize(
    ("backend", "expected_strategy"),
    (("fabledan", "fabledan-numpy"), ("danzero", "danzero")),
)
def test_real_spawn_uses_selected_backend_and_complete_snapshot(
    backend: str, expected_strategy: str,
) -> None:
    snapshot = _snapshot()
    config = AdviceWorkerConfig(
        str(PROFILES_ROOT), backend, fabledan_runtime_policy="model_required"
    )
    runtime = create_live_v2_advice_runtime(config, snapshot.version)
    try:
        ready = runtime.start(timeout=20)
        assert ready.advisor_backend == backend
        assert ready.runtime_backend == ("numpy" if backend == "fabledan" else "danzero")
        assert ready.model_identity
        assert ready.worker_pid != os.getpid()
        assert ready.worker_pid == runtime.worker_pid
        assert ready.worker_generation == runtime.worker_generation
        assert ready.elapsed_ms >= 0
        started = perf_counter()
        runtime.submit(
            snapshot, _opportunity(snapshot, backend), request_sequence=1,
            timeout_ms=30_000,
        )
        result = runtime.get_result(timeout=35)
        assert perf_counter() - started < 3.0
        assert result.status.value == "advice"
        assert result.advice and result.advice.strategy == expected_strategy
        assert result.advice.state_revision == snapshot.version.state_revision
        assert result.identity.version == snapshot.version
        assert result.worker_pid != os.getpid()
        assert result.advisor_cache_hit is True
        assert result.advisor_ready is not None
        assert result.advisor_ready.model_identity == ready.model_identity
        assert result.advisor_ready.worker_pid == ready.worker_pid
        rebound = replace(
            snapshot, version=replace(snapshot.version, state_revision=1)
        )
        runtime.submit(
            rebound, _opportunity(rebound, backend, 2), request_sequence=2,
            timeout_ms=30_000,
        )
        second = runtime.get_result(timeout=35)
        assert second.status.value == "advice"
        assert second.advice and second.advice.strategy == expected_strategy
        assert second.worker_pid == result.worker_pid
        assert second.worker_generation == result.worker_generation
        assert second.advisor_cache_hit is True
    finally:
        runtime.close()
