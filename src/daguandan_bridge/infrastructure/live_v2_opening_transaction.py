"""Atomic lead-plus-first-action staging behind the legacy gateway."""

from dataclasses import dataclass, replace

from ..domain.live import LiveEvent
from ..live_v2.candidates import (
    ActionCandidate, ActionKind, CandidateReason, ConfirmedAction, EvidenceOrigin,
)
from ..live_v2.corrections import ConfirmedCorrection
from ..live_v2.results import CommitReason, ProjectionReason
from ..live_v2.rules_adapter import LiveReducerRuleAdapter
from ..live_v2.identity import VersionIdentity
from .live_v2_legacy_gateway import confirm_lead_reducer, legacy_identity
from .live_v2_rule_backend import create_reducer_backend


@dataclass(frozen=True, slots=True)
class StagedOpeningAction:
    reducer: object
    version: VersionIdentity
    action: ConfirmedAction
    events: tuple[LiveEvent, LiveEvent]


def stage_opening_action(
    reducer: object,
    *,
    current: VersionIdentity,
    seed_actions: tuple[ConfirmedAction, ...],
    seed_corrections: tuple[ConfirmedCorrection, ...],
    candidate: ActionCandidate,
    processing_ms: int,
) -> StagedOpeningAction:
    if seed_actions:
        raise ValueError("opening action requires empty confirmed history")
    if candidate.kind is not ActionKind.PLAY:
        raise ValueError("opening action cannot be PASS")
    if (
        candidate.evidence_origin is not EvidenceOrigin.VISUAL
        or candidate.reason is not CandidateReason.STABLE_PLAY
    ):
        raise ValueError("opening action requires stable visual evidence")
    if (
        candidate.version.session_id != current.session_id
        or candidate.version.capture_generation != current.capture_generation
        or candidate.version.state_version != current.state_version
    ):
        raise ValueError("opening candidate belongs to another state or generation")

    staged, lead_event = confirm_lead_reducer(
        reducer,
        candidate.seat,
        monotonic_ms=candidate.last_captured_ms,
        evidence_id=f"opening:visual:{candidate.candidate_id}",
    )
    session_id, revision, turn_index = legacy_identity(staged)
    lead_version = VersionIdentity(
        session_id, current.capture_generation, revision,
        current.update_sequence + 1, turn_index,
    )
    backend = create_reducer_backend(
        staged, version=lead_version, seed_actions=seed_actions,
        seed_corrections=seed_corrections, in_memory=True,
    )
    adapter = LiveReducerRuleAdapter(backend, lead_version)
    visual = replace(candidate, version=lead_version)
    projection = adapter.project(
        base_version=lead_version, candidates=(visual,), processing_ms=processing_ms
    )
    if projection.reason is not ProjectionReason.ACCEPTED:
        raise ValueError(f"opening action rejected: {projection.reason.value}")
    committed = adapter.commit(
        expected_version=lead_version, actions=projection.actions
    )
    if committed.reason is not CommitReason.COMMITTED or len(
        committed.committed_actions
    ) != 1:
        raise ValueError(f"opening action commit failed: {committed.reason.value}")
    action = committed.committed_actions[0]
    action_event = backend.events_for_actions((action,))[0]
    return StagedOpeningAction(
        staged, adapter.version, action, (lead_event, action_event)
    )


__all__ = ["StagedOpeningAction", "stage_opening_action"]
