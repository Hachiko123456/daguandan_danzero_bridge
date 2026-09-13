"""Spawn-importable live-v2 advisor worker and snapshot replay adapter."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
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
from ..danzero.state import GuanDanState, PlayEvent
from ..domain.advice import AdviceResult
from ..live_v2.events import ActionKind
from ..live_v2.game_state import GameAction, TrustedGameSnapshot
from ..live_v2.identity import Seat
from ..live_v2.results import OpportunityStatus


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
    """Project fresh trusted state while reusing the child-initialized adviser."""

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
            "snapshot_projection_failed", exc, started,
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


def replay_trusted_snapshot(snapshot: TrustedGameSnapshot) -> GuanDanState:
    """Project the authoritative LiveV2 snapshot directly for FableDan.

    ``TrustedGameSnapshot`` has already been cross-checked against the rule
    backend before it reaches the advice pump.  Replaying it through a second
    ``LiveReducer`` used to normalize unknown-suit options differently and
    produced false ``history/current_trick`` mismatches.  This projection has
    one source of truth: the immutable snapshot itself.
    """

    if not isinstance(snapshot, TrustedGameSnapshot):
        raise TypeError("snapshot must be a TrustedGameSnapshot")
    if not snapshot.trusted or snapshot.terminal:
        raise ValueError("advice requires a trusted active snapshot")
    if snapshot.current_seat is None or snapshot.lead_seat is None:
        raise ValueError("advice snapshot requires current and lead seats")

    events_by_action_id: dict[str, PlayEvent] = {}
    history: list[PlayEvent] = []
    for action in snapshot.play_history:
        event = _project_action(action)
        history.append(event)
        events_by_action_id[action.action_id] = event
    try:
        current_trick = [
            events_by_action_id[action.action_id]
            for action in snapshot.current_trick
        ]
    except KeyError as exc:
        raise ValueError("current_trick is not part of play_history") from exc

    return GuanDanState(
        round_level=snapshot.round_level,
        wild_rank=snapshot.wild_rank,
        phase="playing",
        current_player=snapshot.current_seat.value,
        lead_player=snapshot.lead_seat.value,
        my_hand=tuple(snapshot.my_hand),
        trick_plays=current_trick,
        play_history=history,
        remaining_cards={
            item.seat.value: int(item.count) for item in snapshot.remaining
        },
        revision=snapshot.version.state_revision,
    )


def _project_action(action: GameAction) -> PlayEvent:
    semantics = None if action.semantics is None else action.semantics.to_metadata()
    audit = {
        "action_id": action.action_id,
        "action_epoch": action.action_epoch,
        "captured_ms": action.captured_ms,
        "evidence_ids": list(action.evidence_ids),
    }
    metadata = audit if semantics is None else {**audit, **semantics}
    return PlayEvent(
        player=action.seat.value,
        cards=tuple(action.cards),
        is_pass=action.kind is ActionKind.PASS,
        observed_at=datetime.fromtimestamp(
            action.captured_ms / 1000, tz=timezone.utc
        ),
        source="live_v2_snapshot_projection",
        suit_options=tuple(tuple(options) for options in action.suit_options),
        action_metadata=metadata,
    )


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
