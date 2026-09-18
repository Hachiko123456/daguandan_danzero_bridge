"""Presentation mapping for the live-v2 production session runtime."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from ..domain.live import LiveEvent, LiveSnapshot
from ..domain.live_runtime import AdviceRequestKey, LiveAdvice, LiveStatus, LiveUpdate
from ..live_v2.candidates import (
    ActionCandidate, ActionKind, CandidateReason, EvidenceOrigin,
)
from ..live_v2.action_semantics import ActionSemantics
from ..live_v2.corrections import ConfirmedCorrection
from ..live_v2.engine import EngineResult
from ..live_v2.game_state import GameAction, TrustedGameSnapshot
from ..live_v2.identity import FrameIdentity, Seat, VersionIdentity
from ..live_v2.results import AdviceOpportunity, GapPhase, OpportunityStatus
from .live_v2_advice_protocol import AdviceRuntimeResult, AdviceRuntimeStatus
from .live_v2_frame_types import FramePipelineResult
from .live_v2_vision_protocol import VisionRuntimeStatus
from .live_v2_vision_rebind import salvage_vision_result


class VisionRuntimeLike(Protocol):
    """Lifecycle shared by synchronous adapters and process-backed vision."""

    def start(self, *, timeout: float = 10.0) -> None: ...
    def close(self, *, timeout: float = 5.0) -> None: ...


def _observed_at(action: GameAction) -> datetime:
    # Capture timestamps are monotonic, not wall-clock timestamps. Preserve
    # ordering without pretending they are civil time.
    return datetime.fromtimestamp(action.captured_ms / 1000, tz=timezone.utc)


def action_to_play_event(action: GameAction):
    from ..danzero.state import PlayEvent

    audit_metadata = {
        "action_id": action.action_id,
        "action_epoch": action.action_epoch,
        "captured_ms": action.captured_ms,
        "evidence_ids": list(action.evidence_ids),
    }
    semantic_metadata = (
        None if action.semantics is None else action.semantics.to_metadata()
    )
    expose_audit = action.semantics is None or action.semantics.selection_source in {
        "rules_unique",
        "rules_strongest_wildcard",
    }
    return PlayEvent(
        player=action.seat.value,
        cards=action.cards,
        is_pass=action.kind.value == "pass",
        observed_at=_observed_at(action),
        source="live_v2_confirmed",
        suit_options=action.suit_options,
        action_metadata=(
            audit_metadata
            if semantic_metadata is None
            else {**audit_metadata, **semantic_metadata}
            if expose_audit
            else semantic_metadata
        ),
    )


def trusted_to_live_snapshot(snapshot: TrustedGameSnapshot) -> LiveSnapshot:
    history = tuple(action_to_play_event(action) for action in snapshot.play_history)
    trick_count = len(snapshot.current_trick)
    trick = history[-trick_count:] if trick_count else ()
    return LiveSnapshot(
        session_id=snapshot.version.session_id,
        round_level=snapshot.round_level,
        wild_rank=snapshot.wild_rank,
        current_player=(snapshot.current_seat.value if snapshot.current_seat else None),
        lead_player=(snapshot.lead_seat.value if snapshot.lead_seat else None),
        my_hand=snapshot.my_hand,
        trick_plays=trick,
        play_history=history,
        remaining_cards={item.seat.value: item.count for item in snapshot.remaining},
        finished_seats=frozenset(seat.value for seat in snapshot.finished),
        trick_id=snapshot.trick_index,
        turn_id=snapshot.version.turn_index + 1,
        revision=snapshot.version.state_revision,
        initialized=True,
    )


def request_key(snapshot: TrustedGameSnapshot) -> AdviceRequestKey:
    return AdviceRequestKey(
        snapshot.version.session_id,
        snapshot.version.turn_index + 1,
        snapshot.version.state_revision,
    )


def requested_advice(snapshot: TrustedGameSnapshot) -> LiveAdvice:
    return LiveAdvice(key=request_key(snapshot), status="requested")


def opportunity_advice(
    snapshot: TrustedGameSnapshot, opportunity: AdviceOpportunity
) -> LiveAdvice | None:
    if opportunity.status is OpportunityStatus.READY:
        return requested_advice(snapshot)
    if opportunity.status is OpportunityStatus.BLOCKED:
        return LiveAdvice(
            key=request_key(snapshot), status="withheld", visible=False,
            withhold_reason=opportunity.reason.value,
        )
    if opportunity.status is OpportunityStatus.CLOSED:
        return LiveAdvice(key=request_key(snapshot), status="stale", visible=False)
    return None


def runtime_result_to_advice(
    result: AdviceRuntimeResult, snapshot: TrustedGameSnapshot
) -> LiveAdvice:
    key = request_key(snapshot)
    if result.status is AdviceRuntimeStatus.ADVICE:
        return LiveAdvice(
            key=key, status="ready", advice=result.advice, visible=True,
        )
    if result.status in {
        AdviceRuntimeStatus.SUPERSEDED,
        AdviceRuntimeStatus.CLOSED,
        AdviceRuntimeStatus.SERVICE_CLOSED,
    }:
        return LiveAdvice(key=key, status="stale", visible=False)
    if result.status is AdviceRuntimeStatus.BLOCKED:
        return LiveAdvice(
            key=key, status="withheld", visible=False,
            withhold_reason=result.failure_code or result.message,
        )
    return LiveAdvice(
        key=key, status="failed", visible=False,
        error=result.message or result.failure_code or result.status.value,
    )


def trusted_candidate(
    *, version: VersionIdentity, seat: Seat, cards: tuple[str, ...],
    is_pass: bool, captured_ms: int, processing_ms: int, confidence: float,
    origin: EvidenceOrigin, reason: CandidateReason,
    evidence_refs: tuple[str, ...], sequence: int,
    suit_options: tuple[tuple[str, ...], ...] = (),
    requested_semantics: ActionSemantics | None = None,
) -> ActionCandidate:
    audit = evidence_refs[0] if evidence_refs else f"{origin.value}-{sequence}-{captured_ms}"
    frame = FrameIdentity(
        version.session_id, version.capture_generation, sequence, captured_ms,
        "trusted", f"{origin.value}:{audit}",
    )
    return ActionCandidate(
        f"candidate:{audit}", version, seat,
        ActionKind.PASS if is_pass else ActionKind.PLAY,
        () if is_pass else cards,
        () if is_pass else (suit_options or tuple((card,) for card in cards)),
        (audit,), version.turn_index, frame, frame, processing_ms,
        confidence, reason, origin, requested_semantics=requested_semantics,
    )


def manual_confirmation_candidate(
    candidate: ActionCandidate,
    *,
    version: VersionIdentity,
    captured_ms: int,
    processing_ms: int,
    sequence: int,
) -> ActionCandidate:
    """Turn one reviewed visual candidate into one explicit audit fact."""

    audit = f"manual:confirm:{candidate.candidate_id}"
    frame = FrameIdentity(
        version.session_id,
        version.capture_generation,
        sequence,
        captured_ms,
        candidate.last_frame.roi_version,
        audit,
    )
    return ActionCandidate(
        audit,
        version,
        candidate.seat,
        candidate.kind,
        candidate.cards,
        candidate.suit_options,
        candidate.evidence_ids,
        candidate.action_epoch,
        frame,
        frame,
        processing_ms,
        candidate.confidence,
        CandidateReason.LOCAL_ACTION_CONFIRMED,
        EvidenceOrigin.MANUAL,
        (audit,),
        requested_semantics=candidate.requested_semantics,
    )


def correction_event(
    correction: ConfirmedCorrection,
    *,
    snapshot: TrustedGameSnapshot,
) -> LiveEvent:
    """Project the authoritative correction receipt for LiveUpdate consumers."""

    return LiveEvent(
        event_id=correction.correction_id,
        event_type="event_correction",
        session_id=snapshot.version.session_id,
        seq=snapshot.version.state_revision,
        monotonic_ms=correction.corrected_ms,
        wall_time="",
        trick_id=snapshot.trick_index,
        turn_id=snapshot.version.turn_index + 1,
        actor=correction.seat.value,
        payload={
            "target_action_id": correction.target_action_id,
            "cards": list(correction.corrected_cards),
            "is_pass": correction.corrected_kind is ActionKind.PASS,
            "correction_id": correction.correction_id,
            "correction_reason": correction.reason.value,
            "evidence_origin": correction.evidence_origin.value,
            **(
                {}
                if correction.corrected_semantics is None
                else {
                    "move_semantics": correction.corrected_semantics.to_metadata()
                }
            ),
        },
        confidence=correction.confidence,
        source="live_v2_rule_session",
        state_revision_before=correction.version_before.state_revision,
        state_revision_after=correction.version_after.state_revision,
        evidence_refs=(correction.evidence_id,),
    )


def consume_vision(
    vision: VisionRuntimeLike, image: Any, *, frame: FrameIdentity,
    version: VersionIdentity, wild_rank: str, expected_seat: Seat | None,
    processing_ms: int, formal_action_boundary: FrameIdentity | None,
    repair_seats: tuple[Seat, ...] = (),
    synchronous: bool = False,
) -> tuple[tuple[FramePipelineResult, ...], tuple[str, ...]]:
    if synchronous:
        process_sync = getattr(vision, "process_frame_sync", None)
        if not callable(process_sync):
            raise TypeError("synchronous vision runtime must provide process_frame_sync")
        kwargs: dict[str, object] = {
            "image": image, "frame": frame, "version": version,
            "wild_rank": wild_rank, "expected_seat": expected_seat,
            "request_sequence": frame.frame_sequence,
            "formal_action_boundary": formal_action_boundary,
        }
        if repair_seats:
            kwargs["repair_seats"] = repair_seats
        try:
            result = process_sync(**kwargs)
        except TypeError as exc:
            # Keep compatibility with lightweight test/legacy adapters that
            # predate the optional correction-read parameter.
            if repair_seats and "repair_seats" in str(exc):
                kwargs.pop("repair_seats", None)
                result = process_sync(**kwargs)
            else:
                raise
        return (result,), ()

    process = getattr(vision, "process_frame", None)
    if callable(process):
        kwargs = {
            "image": image, "frame": frame, "version": version,
            "wild_rank": wild_rank, "expected_seat": expected_seat,
            "now_ms": processing_ms,
            "formal_action_boundary": formal_action_boundary,
        }
        if repair_seats:
            kwargs["repair_seats"] = repair_seats
        try:
            result = process(**kwargs)
        except TypeError as exc:
            if repair_seats and "repair_seats" in str(exc):
                kwargs.pop("repair_seats", None)
                result = process(**kwargs)
            else:
                raise
        return (result,), ()

    submit = getattr(vision, "submit", None)
    if not callable(submit):
        raise TypeError("vision runtime must provide process_frame or submit")
    submit_kwargs: dict[str, object] = {
        "image": image, "frame": frame, "version": version,
        "expected_seat": expected_seat,
        "visual_self_opportunity": expected_seat is Seat.SELF,
        "wild_rank": wild_rank, "request_sequence": frame.frame_sequence,
        "formal_action_boundary": formal_action_boundary,
    }
    if repair_seats:
        submit_kwargs["repair_seats"] = repair_seats
    try:
        values = list(submit(**submit_kwargs))
    except TypeError as exc:
        if repair_seats and "repair_seats" in str(exc):
            submit_kwargs.pop("repair_seats", None)
            values = list(submit(**submit_kwargs))
        else:
            raise
    drain = getattr(vision, "drain_results", None)
    if callable(drain):
        values.extend(drain())
    completed: list[FramePipelineResult] = []
    failures: list[str] = []
    for item in values:
        if item.status is not VisionRuntimeStatus.FRAME:
            if item.status not in {VisionRuntimeStatus.SUPERSEDED, VisionRuntimeStatus.REJECTED}:
                failures.append(item.message or item.failure_code or item.status.value)
            continue
        result = salvage_vision_result(
            item.pipeline_result, source_version=item.identity.version,
            current_version=version, formal_action_boundary=formal_action_boundary,
        )
        if result is not None:
            completed.append(result)
    return tuple(completed), tuple(failures)


def project_engine_result(
    *, result: EngineResult, snapshot: TrustedGameSnapshot,
    action_events: tuple[LiveEvent, ...], status: LiveStatus,
    latest_advice: LiveAdvice | None, sequence: int,
    fast_signals: object | None = None,
) -> tuple[LiveUpdate, LiveStatus, LiveAdvice | None, int]:
    update = result.update
    if update is None:
        sequence += 1
        projected = live_update(
            status=status, snapshot=snapshot, sequence=sequence,
            advice=latest_advice, block_reason="engine_input_rejected",
        )
        return projected, status, latest_advice, sequence
    sequence = max(sequence, update.version.update_sequence)
    opportunity = update.advice_opportunity
    if opportunity and opportunity.status is not OpportunityStatus.READY:
        latest_advice = opportunity_advice(snapshot, opportunity)
    elif opportunity and (
        latest_advice is None
        or latest_advice.key.state_revision != update.version.state_revision
    ):
        latest_advice = requested_advice(snapshot)
    gap = update.gap
    blocked = gap.reason.value if gap and gap.phase is not GapPhase.CLEAR else ""
    if gap and gap.phase in {GapPhase.BLOCKING, GapPhase.EXPIRED}:
        status = "review_required"
    elif status == "review_required" and (not gap or gap.phase is GapPhase.CLEAR):
        status = "running"
    missing = gap.expected_seats[0].value if gap and gap.expected_seats else None
    projected = live_update(
        status=status, snapshot=snapshot, sequence=sequence,
        advice=latest_advice,
        events=action_events,
        fast_signals=fast_signals, block_reason=blocked,
        missing_player=missing, missing_action_kind="action" if missing else "",
    )
    return projected, status, latest_advice, sequence


def live_update(
    *,
    status: LiveStatus,
    snapshot: TrustedGameSnapshot,
    sequence: int,
    advice: LiveAdvice | None = None,
    events: tuple[LiveEvent, ...] = (),
    fast_signals: object | None = None,
    local_rule_hint: object | None = None,
    block_reason: str = "",
    missing_player: str | None = None,
    missing_action_kind: str = "",
) -> LiveUpdate:
    return LiveUpdate(
        status=status,
        snapshot=trusted_to_live_snapshot(snapshot),
        event=events[-1] if events else None,
        events=events,
        advice=advice,
        fast_signals=fast_signals,
        local_rule_hint=local_rule_hint,
        capture_generation=snapshot.version.capture_generation,
        update_sequence=sequence,
        block_reason=block_reason,
        missing_player=missing_player,
        missing_action_kind=missing_action_kind,
    )


__all__ = [
    "VisionRuntimeLike", "consume_vision", "correction_event", "live_update",
    "manual_confirmation_candidate", "opportunity_advice",
    "project_engine_result", "requested_advice", "runtime_result_to_advice",
    "trusted_candidate", "trusted_to_live_snapshot",
]
