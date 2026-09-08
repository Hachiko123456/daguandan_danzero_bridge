"""Pure DTO projection helpers for the infrastructure legacy gateway."""

from __future__ import annotations

from ..domain.live import LiveEvent
from ..live_v2.corrections import ConfirmedCorrection
from ..live_v2.game_state import SeatCardCount
from ..live_v2.reducer_snapshot import (
    ReducedActionState,
    ReducedGameState,
    build_trusted_snapshot,
)
from ..live_v2.types import ActionKind, ConfirmedAction, Seat, VersionIdentity


def extract_reduced_state(reducer: object) -> ReducedGameState:
    snapshot = reducer.snapshot()

    def action_state(event: object) -> ReducedActionState:
        is_pass = bool(getattr(event, "is_pass"))
        return ReducedActionState(
            seat=Seat(str(getattr(event, "player"))),
            kind=ActionKind.PASS if is_pass else ActionKind.PLAY,
            cards=tuple(str(card) for card in getattr(event, "cards")),
            suit_options=tuple(
                tuple(str(choice) for choice in options)
                for options in getattr(event, "suit_options")
            ),
        )

    return ReducedGameState(
        session_id=snapshot.session_id,
        revision=snapshot.revision,
        round_level=snapshot.round_level,
        wild_rank=snapshot.wild_rank,
        trick_index=int(snapshot.trick_id),
        current_seat=(
            None if snapshot.current_player is None else Seat(snapshot.current_player)
        ),
        lead_seat=None if snapshot.lead_player is None else Seat(snapshot.lead_player),
        my_hand=tuple(snapshot.my_hand),
        play_history=tuple(action_state(item) for item in snapshot.play_history),
        current_trick=tuple(action_state(item) for item in snapshot.trick_plays),
        remaining=tuple(
            SeatCardCount(seat, int(snapshot.remaining_cards[seat.value]))
            for seat in Seat
        ),
        finished=tuple(
            seat for seat in Seat if seat.value in snapshot.finished_seats
        ),
        initialized=bool(snapshot.initialized),
    )


def event_matches_action(event: LiveEvent, action: ConfirmedAction) -> bool:
    expected_type = (
        "player_passed" if action.kind is ActionKind.PASS else "player_played"
    )
    semantics = event.payload.get("action_semantics", event.payload.get("move_semantics"))
    return (
        event.event_type == expected_type
        and event.session_id == action.version_before.session_id
        and event.actor == action.seat.value
        and event.state_revision_before == action.version_before.state_revision
        and event.state_revision_after == action.version_after.state_revision
        and event.monotonic_ms == action.captured_ms
        and event.evidence_refs == action.event_evidence_ids
        and (
            action.kind is ActionKind.PASS
            or action.semantics is None
            and semantics is None
            or action.semantics is not None
            and isinstance(semantics, dict)
            and semantics == action.semantics.to_metadata()
        )
    )


def validate_reducer_seed(
    reducer: object,
    version: VersionIdentity,
    actions: tuple[ConfirmedAction, ...],
    corrections: tuple[ConfirmedCorrection, ...],
) -> None:
    watermark = max((item.captured_ms for item in actions), default=0)
    watermark = max(
        watermark,
        max((item.corrected_ms for item in corrections), default=0),
    )
    build_trusted_snapshot(
        state=extract_reduced_state(reducer),
        actions=actions,
        corrections=corrections,
        version=version,
        captured_ms=watermark,
    )
    action_events = tuple(
        event
        for event in reducer.events
        if event.event_type in {
            "player_played", "player_passed", "manual_confirmed_event"
        }
    )
    correction_events = tuple(
        event for event in reducer.events if event.event_type == "event_correction"
    )
    if len(action_events) != len(actions) or not all(
        event_matches_action(event, action)
        for event, action in zip(action_events, actions, strict=True)
    ):
        raise ValueError("seed action count or base revision does not match reducer")
    if len(correction_events) != len(corrections) or not all(
        _event_matches_correction(event, correction)
        for event, correction in zip(correction_events, corrections, strict=True)
    ):
        raise ValueError("seed correction ledger differs from reducer events")


def _event_matches_correction(
    event: LiveEvent, correction: ConfirmedCorrection
) -> bool:
    return (
        event.event_type == "event_correction"
        and event.session_id == correction.version_before.session_id
        and event.actor == correction.seat.value
        and event.state_revision_before == correction.version_before.state_revision
        and event.state_revision_after == correction.version_after.state_revision
        and event.monotonic_ms == correction.corrected_ms
        and event.evidence_refs == (correction.evidence_id,)
        and event.payload.get("target_action_id") == correction.target_action_id
        and event.payload.get("correction_id") == correction.correction_id
        and event.payload.get("correction_reason") == correction.reason.value
        and event.payload.get("evidence_origin") == correction.evidence_origin.value
    )


__all__ = ["event_matches_action", "extract_reduced_state", "validate_reducer_seed"]
