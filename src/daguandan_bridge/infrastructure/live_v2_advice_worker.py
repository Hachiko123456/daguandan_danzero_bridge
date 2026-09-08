"""Spawn-importable live-v2 advisor worker and snapshot replay adapter."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol

from ..application.live_v2_advice_protocol import (
    AdvicePrewarmPayload,
    AdviceFailureKind,
    AdviceWorkerConfig,
    AdviceWorkerFailure,
    AdviceWorkerPayload,
    AdviceWorkerSuccess,
    AdvisorReady,
)
from ..application.live_v2_worker_protocol import WorkerRequest
from ..domain.advice import AdviceResult
from ..live_v2.events import ActionKind
from ..live_v2.game_state import TrustedGameSnapshot
from ..live_v2.identity import Seat
from ..live_v2.results import OpportunityStatus
from ..live_v2.reducer_snapshot import physical_action_entities
from .live_v2_legacy_gateway import replay_trusted_snapshot_projection


class _Advisor(Protocol):
    def recommend(
        self, state: GuanDanState, *, request_id: str = ""
    ) -> AdviceResult: ...


_ADVISORS: dict[tuple[str, str, str, str], _Advisor] = {}
_ADVISOR_READY: dict[tuple[str, str, str, str], AdvisorReady] = {}


def _advisor_key(config: AdviceWorkerConfig) -> tuple[str, str, str, str]:
    return (
        str(Path(config.profile_root).resolve()), config.profile_name,
        config.advisor_backend, config.fabledan_runtime_policy,
    )


def _advisor_for(config: AdviceWorkerConfig) -> _Advisor:
    key = _advisor_key(config)
    cached = _ADVISORS.get(key)
    if cached is not None:
        return cached
    if config.advisor_backend == "fabledan":
        from ..fabledan.advisor import FableDanAdvisor
        advisor = FableDanAdvisor(
            key[0], key[1], runtime_policy=config.fabledan_runtime_policy,
            diagnostics="off", write_decision_log=False,
        )
    elif config.advisor_backend == "danzero":
        from ..danzero import DanzeroAdvisor
        advisor = DanzeroAdvisor(key[0], key[1])
    else:
        raise ValueError(f"unsupported advisor backend: {config.advisor_backend}")
    advisor.initialize()
    _ADVISORS[key] = advisor
    return advisor


def _advisor_ready(
    config: AdviceWorkerConfig,
    *,
    worker_generation: int,
    elapsed_ms: float,
    cache_hit: bool,
) -> AdvisorReady:
    advisor = _advisor_for(config)
    audit_provider = getattr(advisor, "audit_info", None)
    audit = audit_provider() if callable(audit_provider) else {}
    runtime_backend = str(audit.get("backend") or config.advisor_backend)
    model_path = str(audit.get("model_path") or audit.get("path") or "")
    model_digest = str(audit.get("model_hash") or audit.get("digest") or "")
    model_schema = str(audit.get("schema") or "")
    model_status = str(audit.get("status") or "initialized")
    model_identity = model_digest or model_path or (
        f"{runtime_backend}:{config.profile_name}:{config.fabledan_runtime_policy}"
    )
    return AdvisorReady(
        advisor_backend=config.advisor_backend,
        runtime_backend=runtime_backend,
        model_identity=model_identity,
        worker_generation=worker_generation,
        worker_pid=os.getpid(),
        elapsed_ms=elapsed_ms,
        cache_hit=cache_hit,
        model_path=model_path,
        model_digest=model_digest,
        model_schema=model_schema,
        model_status=model_status,
    )


def run_live_v2_advice_worker(
    request: WorkerRequest,
) -> AdvisorReady | AdviceWorkerSuccess | AdviceWorkerFailure:
    """Replay fresh state while reusing the child-initialized adviser."""

    started = perf_counter()
    payload = request.payload
    if isinstance(payload, AdvicePrewarmPayload):
        key = _advisor_key(payload.config)
        cache_hit = key in _ADVISORS
        _advisor_for(payload.config)
        ready = _advisor_ready(
            payload.config,
            worker_generation=payload.worker_generation,
            elapsed_ms=(perf_counter() - started) * 1000,
            cache_hit=cache_hit,
        )
        _ADVISOR_READY[key] = ready
        return ready
    if not isinstance(payload, AdviceWorkerPayload):
        return _failure(
            request, payload, AdviceFailureKind.INVALID_REQUEST,
            "invalid_payload", TypeError("expected AdviceWorkerPayload"), started,
        )
    try:
        _validate_request(request, payload)
        state = replay_trusted_snapshot(payload.snapshot)
    except Exception as exc:
        return _failure(
            request, payload, AdviceFailureKind.SNAPSHOT_MISMATCH,
            "snapshot_replay_failed", exc, started,
        )
    try:
        key = _advisor_key(payload.config)
        cache_hit = key in _ADVISORS
        advice = _advisor_for(payload.config).recommend(
            state, request_id=payload.request_id
        )
        advice = replace(
            advice,
            state_revision=payload.snapshot.version.state_revision,
            request_id=payload.request_id,
        )
    except Exception as exc:
        return _failure(
            request, payload, AdviceFailureKind.MODEL_ERROR,
            "model_execution_failed", exc, started,
        )
    return AdviceWorkerSuccess(
        opportunity_id=payload.opportunity.opportunity_id,
        version=payload.snapshot.version,
        advice=advice,
        replayed_actions=len(payload.snapshot.play_history),
        elapsed_ms=(perf_counter() - started) * 1000,
        advisor_cache_hit=cache_hit,
        advisor_ready=(
            replace(_ADVISOR_READY[key], cache_hit=cache_hit)
            if key in _ADVISOR_READY
            else None
        ),
    )


def replay_trusted_snapshot(snapshot: TrustedGameSnapshot):
    """Rebuild a fresh reducer and reject any semantic disagreement."""

    state, reduced = replay_trusted_snapshot_projection(snapshot)
    expected_remaining = {item.seat.value: item.count for item in snapshot.remaining}
    checks = {
        "trick_index": (reduced.trick_id, snapshot.trick_index),
        "current_seat": (reduced.current_player, _seat(snapshot.current_seat)),
        "lead_seat": (reduced.lead_player, _seat(snapshot.lead_seat)),
        "my_hand": (Counter(reduced.my_hand), Counter(snapshot.my_hand)),
        "remaining": (reduced.remaining_cards, expected_remaining),
        "finished": (set(reduced.finished_seats), {seat.value for seat in snapshot.finished}),
        "history": (_actions(reduced.play_history), _actions(snapshot.play_history)),
        "current_trick": (_actions(reduced.trick_plays), _actions(snapshot.current_trick)),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError("snapshot disagrees with reducer: " + ", ".join(mismatches))
    return state


def _validate_request(request: WorkerRequest, payload: AdviceWorkerPayload) -> None:
    snapshot, opportunity = payload.snapshot, payload.opportunity
    if opportunity.status is not OpportunityStatus.READY:
        raise ValueError("worker accepts READY opportunities only")
    if snapshot.current_seat is not opportunity.seat:
        raise ValueError("opportunity seat and snapshot current seat differ")
    if opportunity.version != snapshot.version:
        raise ValueError("opportunity and snapshot versions differ")
    version = snapshot.version
    if (request.session_id, request.capture_generation, request.state_revision) != (
        version.session_id, version.capture_generation, version.state_revision,
    ):
        raise ValueError("worker request and snapshot versions differ")


def _seat(seat: Seat | None) -> str | None:
    return None if seat is None else seat.value


def _actions(values: object) -> tuple[tuple[object, ...], ...]:
    result: list[tuple[object, ...]] = []
    for item in values:  # type: ignore[union-attr]
        is_pass = getattr(item, "is_pass", None)
        kind = getattr(item, "kind", None)
        cards = tuple(getattr(item, "cards", ()))
        supplied = tuple(getattr(item, "suit_options", ()))
        entities = physical_action_entities(cards, supplied)
        result.append((
            getattr(getattr(item, "seat", None), "value", None)
            or getattr(item, "player", None),
            bool(is_pass) if is_pass is not None else kind is ActionKind.PASS,
            entities,
        ))
    return tuple(result)


def _failure(
    request: WorkerRequest,
    payload: object,
    kind: AdviceFailureKind,
    code: str,
    exc: Exception,
    started: float,
) -> AdviceWorkerFailure:
    opportunity = getattr(payload, "opportunity", None)
    snapshot = getattr(payload, "snapshot", None)
    version = getattr(snapshot, "version", None)
    if version is None:
        from ..live_v2.identity import VersionIdentity
        version = VersionIdentity(
            request.session_id, request.capture_generation,
            request.state_revision, 0, 0,
        )
    return AdviceWorkerFailure(
        opportunity_id=getattr(opportunity, "opportunity_id", "invalid-request"),
        version=version,
        kind=kind,
        code=code,
        error_type=type(exc).__name__,
        message=str(exc),
        elapsed_ms=(perf_counter() - started) * 1000,
    )


run_fabledan_advice_worker = run_live_v2_advice_worker

__all__ = [
    "replay_trusted_snapshot", "run_fabledan_advice_worker",
    "run_live_v2_advice_worker",
]
