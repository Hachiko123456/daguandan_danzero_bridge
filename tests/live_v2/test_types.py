from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from enum import Enum
from typing import get_type_hints

import pytest

from daguandan_bridge.live_v2.protocols import (
    AdviceConsumer,
    EventJournal,
    ObservationInput,
    ProcessingClock,
    RuleProjector,
    TransactionalActionCommitter,
)
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionKind,
    AdviceOpportunity,
    CandidateReason,
    CommitReason,
    CommitResult,
    ConfirmationReason,
    ConfirmedAction,
    EngineUpdate,
    EngineUpdateReason,
    EvidenceDropReason,
    EvidenceOrigin,
    FrameIdentity,
    GapPhase,
    GapReason,
    GapState,
    ObservationKind,
    ObservationReason,
    OpportunityReason,
    OpportunityStatus,
    ProjectionReason,
    ProjectionResult,
    ScheduledItemKind,
    SchedulingDropReason,
    Seat,
    SeatObservation,
    StateVersion,
    VersionIdentity,
)


def frame(**changes: object) -> FrameIdentity:
    values = {
        "session_id": "session-1",
        "capture_generation": 2,
        "frame_sequence": 7,
        "captured_ms": 1_000,
        "roi_version": "roi-v3",
        "source_id": "window-42",
    }
    values.update(changes)
    return FrameIdentity(**values)  # type: ignore[arg-type]


def version(**changes: object) -> VersionIdentity:
    values = {
        "session_id": "session-1",
        "capture_generation": 2,
        "state_revision": 4,
        "update_sequence": 9,
        "turn_index": 3,
    }
    values.update(changes)
    return VersionIdentity(**values)  # type: ignore[arg-type]


def observation(**changes: object) -> SeatObservation:
    values = {
        "observation_id": "obs-1",
        "frame": frame(),
        "seat": Seat.RIGHT,
        "kind": ObservationKind.PLAY,
        "cards": ("5D", "5D"),
        "confidence": 0.91,
        "reason": ObservationReason.CARDS_RECOGNIZED,
        "processing_ms": 1_030,
        "suit_options": (("5D",), ("5D",)),
        "diagnostics": ("template_match",),
    }
    values.update(changes)
    if "cards" in changes and "suit_options" not in changes:
        cards = changes["cards"]
        if isinstance(cards, tuple):
            values["suit_options"] = tuple((card,) for card in cards)
    return SeatObservation(**values)  # type: ignore[arg-type]


def candidate(**changes: object) -> ActionCandidate:
    values = {
        "candidate_id": "candidate-1",
        "version": version(),
        "seat": Seat.RIGHT,
        "kind": ActionKind.PLAY,
        "cards": ("5D", "5D"),
        "suit_options": (("5D",), ("5D",)),
        "evidence_ids": ("obs-1", "obs-2"),
        "action_epoch": 6,
        "first_frame": frame(),
        "last_frame": frame(frame_sequence=8, captured_ms=1_100),
        "processing_ms": 1_130,
        "confidence": 0.9,
        "reason": CandidateReason.STABLE_PLAY,
    }
    values.update(changes)
    if "cards" in changes and "suit_options" not in changes:
        cards = changes["cards"]
        if isinstance(cards, tuple):
            values["suit_options"] = tuple((card,) for card in cards)
    return ActionCandidate(**values)  # type: ignore[arg-type]


def action(**changes: object) -> ConfirmedAction:
    before = version().state_version
    after = replace(before, state_revision=5, turn_index=4)
    source = candidate()
    values = {
        "action_id": "action-1",
        "source_candidate": source,
        "version_before": before,
        "version_after": after,
        "cards": ("5D", "5D"),
        "suit_options": (("5D",), ("5D",)),
        "action_epoch": 6,
        "processing_ms": 1_140,
        "reason": ConfirmationReason.RULE_VALIDATED,
    }
    values.update(changes)
    return ConfirmedAction(**values)  # type: ignore[arg-type]


def clear_gap(current_version: VersionIdentity | None = None) -> GapState:
    return GapState(
        version=current_version or version(),
        phase=GapPhase.CLEAR,
        reason=GapReason.NONE,
        expected_seats=(),
        evidence_ids=(),
        opened_captured_ms=1_000,
        processing_ms=1_010,
    )


def ready_opportunity(current_version: VersionIdentity | None = None) -> AdviceOpportunity:
    return AdviceOpportunity(
        opportunity_id="opportunity-1",
        version=current_version or version(),
        seat=Seat.SELF,
        status=OpportunityStatus.READY,
        reason=OpportunityReason.TRUSTED_STATE,
        captured_ms=1_100,
        processing_ms=1_120,
    )


def test_frame_and_version_identities_are_immutable_and_generation_bound() -> None:
    identity = frame()
    assert version().belongs_to(identity)
    assert not replace(version(), capture_generation=3).belongs_to(identity)
    with pytest.raises(FrozenInstanceError):
        identity.frame_sequence = 8  # type: ignore[misc]
    with pytest.raises(ValueError, match="session_id"):
        frame(session_id=" ")
    with pytest.raises(ValueError, match="capture_generation"):
        frame(capture_generation=-1)
    with pytest.raises(ValueError, match="frame_sequence"):
        frame(frame_sequence=True)
    with pytest.raises(ValueError, match="roi_version"):
        frame(roi_version="")
    assert FrameIdentity("s", 1, 1, 10, "roi-v1").source_id == "roi-v1"


def test_state_version_is_independent_from_capture_generation() -> None:
    current = version()
    state = current.state_version
    assert state == StateVersion("session-1", 4, 3)
    rebound = VersionIdentity.from_state(
        state,
        capture_generation=9,
        update_sequence=20,
    )
    assert rebound.state_version is not state
    assert rebound.state_version == state
    assert rebound.capture_generation == 9
    assert current.with_state(state, update_sequence=10).capture_generation == 2
    with pytest.raises(ValueError, match="another session"):
        current.with_state(StateVersion("other", 4, 3))


@pytest.mark.parametrize(
    ("kind", "cards", "reason"),
    [
        (ObservationKind.EMPTY, (), ObservationReason.STABLE_EMPTY),
        (ObservationKind.PASS, (), ObservationReason.PASS_MARKER),
        (ObservationKind.PLAY, ("3H",), ObservationReason.CARDS_RECOGNIZED),
        (ObservationKind.ANIMATING, (), ObservationReason.ANIMATION_DETECTED),
        (ObservationKind.UNKNOWN, (), ObservationReason.UNREADABLE),
        (ObservationKind.UNKNOWN, (), ObservationReason.CONFLICTING_SIGNALS),
        (ObservationKind.UNKNOWN, (), ObservationReason.DETECTOR_FAILURE),
    ],
)
def test_all_observation_kinds_have_explicit_semantics(
    kind: ObservationKind,
    cards: tuple[str, ...],
    reason: ObservationReason,
) -> None:
    result = observation(
        kind=kind,
        cards=cards,
        reason=reason,
        suit_options=tuple((card,) for card in cards),
    )
    assert result.kind is kind
    assert result.cards == cards


def test_observation_rejects_ambiguous_payloads_and_mixed_time_domains() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        observation(cards=())
    with pytest.raises(ValueError, match="cannot contain cards"):
        observation(
            kind=ObservationKind.PASS,
            reason=ObservationReason.PASS_MARKER,
            cards=("3H",),
            suit_options=(),
        )
    with pytest.raises(ValueError, match="requires reason"):
        observation(kind=ObservationKind.EMPTY, cards=(), suit_options=())
    with pytest.raises(ValueError, match="precede captured_ms"):
        observation(processing_ms=999)
    with pytest.raises(TypeError, match="ObservationKind"):
        observation(kind="play")


def test_physical_cards_are_tuples_and_duplicate_copies_are_preserved() -> None:
    duplicate_deck_cards = ("5D", "5D", "BJ", "BJ")
    result = observation(
        cards=duplicate_deck_cards,
        suit_options=(("5D",), ("5D",), ("BJ",), ("BJ",)),
    )
    assert result.cards == duplicate_deck_cards
    assert len(result.cards) == 4
    with pytest.raises(TypeError, match="cards must be a tuple"):
        observation(cards=["5D", "5D"])  # type: ignore[arg-type]


def test_play_suit_options_align_with_every_physical_card_entity() -> None:
    result = observation(
        cards=("5D", "5D"),
        suit_options=(("5D", "5H"), ("5D",)),
        diagnostics=("weak_suit", "weak_suit", "stable_rank"),
    )
    assert result.suit_options == (("5D", "5H"), ("5D",))
    assert result.diagnostics == ("weak_suit", "stable_rank")
    with pytest.raises(ValueError, match="one-to-one"):
        observation(suit_options=(("5D",),))
    with pytest.raises(ValueError, match="must occur"):
        observation(suit_options=(("5H",), ("5D",)))
    with pytest.raises(ValueError, match="cannot contain suit_options"):
        observation(
            kind=ObservationKind.PASS,
            cards=(),
            reason=ObservationReason.PASS_MARKER,
            suit_options=(("5D",),),
        )


def test_candidate_is_distinct_from_formal_action_and_validates_evidence() -> None:
    proposal = candidate()
    committed = action()
    assert type(proposal) is ActionCandidate
    assert type(committed) is ConfirmedAction
    assert proposal.cards == ("5D", "5D")
    assert proposal.suit_options == (("5D",), ("5D",))
    assert proposal.action_epoch == 6
    with pytest.raises(ValueError, match="at least two"):
        candidate(evidence_ids=())
    with pytest.raises(ValueError, match="must not contain duplicates"):
        candidate(evidence_ids=("obs-1", "obs-1"))
    with pytest.raises(ValueError, match="strictly increase"):
        candidate(
            first_frame=frame(frame_sequence=9, captured_ms=1_101),
            last_frame=frame(frame_sequence=8, captured_ms=1_100),
        )
    with pytest.raises(ValueError, match="captured evidence"):
        candidate(processing_ms=1_099)


def test_candidate_retains_complete_ordered_frame_identity() -> None:
    result = candidate()
    assert result.first_frame.roi_version == "roi-v3"
    assert result.last_frame.frame_sequence == 8
    assert result.first_captured_ms == 1_000
    assert result.last_captured_ms == 1_100
    with pytest.raises(ValueError, match="one stream"):
        candidate(last_frame=frame(frame_sequence=8, captured_ms=1_100, source_id="other"))
    with pytest.raises(ValueError, match="strictly increase"):
        candidate(last_frame=frame(frame_sequence=7, captured_ms=1_100))
    with pytest.raises(TypeError, match="first_frame"):
        candidate(first_frame=1_000)
    with pytest.raises(ValueError, match="action_epoch"):
        candidate(action_epoch=-1)
    with pytest.raises(ValueError, match="one-to-one"):
        candidate(suit_options=())


@pytest.mark.parametrize(
    ("reason", "origin", "source_id"),
    [
        (
            CandidateReason.LOCAL_ACTION_CONFIRMED,
            EvidenceOrigin.MANUAL,
            "manual:operator-confirmation-17",
        ),
        (
            CandidateReason.OPENING_ACTION_CONFIRMED,
            EvidenceOrigin.OPENING,
            "opening:first-seat-marker-3",
        ),
        (
            CandidateReason.LOCAL_ACTION_CONFIRMED,
            EvidenceOrigin.TRUSTED,
            "trusted:imported-action-42",
        ),
    ],
)
def test_trusted_non_visual_candidate_allows_one_auditable_evidence(
    reason: CandidateReason,
    origin: EvidenceOrigin,
    source_id: str,
) -> None:
    single = frame(source_id=source_id)
    result = candidate(
        reason=reason,
        evidence_origin=origin,
        evidence_ids=("trusted-evidence-1",),
        first_frame=single,
        last_frame=single,
    )
    assert result.first_frame is result.last_frame
    assert result.evidence_origin is origin
    confirmed = ConfirmedAction.from_candidate(
        action_id="trusted-action-1",
        candidate=result,
        version_before=result.version.state_version,
        version_after=replace(
            result.version.state_version,
            state_revision=result.version.state_revision + 1,
            turn_index=result.version.turn_index + 1,
        ),
        processing_ms=result.processing_ms + 1,
        reason=ConfirmationReason.LOCAL_ACTION_COMMITTED,
    )
    assert confirmed.evidence_origin is origin
    assert confirmed.source_candidate is result


def test_reviewed_candidate_keeps_visual_evidence_and_one_formal_audit() -> None:
    audit = "manual:confirm:visual-42"
    single = frame(source_id=audit)
    reviewed = candidate(
        candidate_id=audit,
        reason=CandidateReason.LOCAL_ACTION_CONFIRMED,
        evidence_origin=EvidenceOrigin.MANUAL,
        evidence_ids=("visual-42-a", "visual-42-b"),
        audit_evidence_ids=(audit,),
        action_epoch=42,
        first_frame=single,
        last_frame=single,
    )
    assert reviewed.evidence_ids == ("visual-42-a", "visual-42-b")
    assert reviewed.event_evidence_ids == (audit,)
    assert reviewed.all_evidence_ids == ("visual-42-a", "visual-42-b", audit)


def test_visual_candidate_cannot_be_downgraded_to_one_frame() -> None:
    single = frame()
    with pytest.raises(ValueError, match="at least two observations"):
        candidate(
            evidence_ids=("visual-only-once",),
            first_frame=single,
            last_frame=single,
        )


def test_non_visual_candidate_rejects_spoofed_or_mismatched_source() -> None:
    ordinary_window = frame(source_id="window-42")
    with pytest.raises(ValueError, match="origin and audit ID"):
        candidate(
            reason=CandidateReason.LOCAL_ACTION_CONFIRMED,
            evidence_origin=EvidenceOrigin.MANUAL,
            evidence_ids=("claimed-manual",),
            first_frame=ordinary_window,
            last_frame=ordinary_window,
        )
    opening = frame(source_id="opening:marker-1")
    with pytest.raises(ValueError, match="invalid evidence origin"):
        candidate(
            reason=CandidateReason.LOCAL_ACTION_CONFIRMED,
            evidence_origin=EvidenceOrigin.OPENING,
            evidence_ids=("wrong-reason",),
            first_frame=opening,
            last_frame=opening,
        )
    unauditable = frame(source_id="manual:")
    with pytest.raises(ValueError, match="origin and audit ID"):
        candidate(
            reason=CandidateReason.LOCAL_ACTION_CONFIRMED,
            evidence_origin=EvidenceOrigin.MANUAL,
            evidence_ids=("missing-audit-id",),
            first_frame=unauditable,
            last_frame=unauditable,
        )


def test_pass_candidate_and_action_cannot_smuggle_card_payloads() -> None:
    proposal = candidate(
        kind=ActionKind.PASS,
        cards=(),
        reason=CandidateReason.FRESH_PASS_EDGE,
    )
    assert proposal.cards == ()
    with pytest.raises(ValueError, match="pass candidate"):
        candidate(kind=ActionKind.PASS, cards=("3H",))
    with pytest.raises(ValueError, match="pass candidate"):
        candidate(
            kind=ActionKind.PASS,
            cards=(),
            suit_options=(("3H",),),
            reason=CandidateReason.FRESH_PASS_EDGE,
        )
    with pytest.raises(ValueError, match="confirmed pass"):
        action(
            source_candidate=candidate(
                kind=ActionKind.PASS,
                cards=(),
                reason=CandidateReason.FRESH_PASS_EDGE,
            ),
            cards=("3H",),
            suit_options=(),
        )


def test_confirmed_action_requires_one_atomic_version_transition() -> None:
    before = version().state_version
    with pytest.raises(ValueError, match="exactly once"):
        action(version_after=replace(before, state_revision=6))
    with pytest.raises(ValueError, match="cross session"):
        action(
            version_after=replace(
                before,
                session_id="other-session",
                state_revision=5,
            )
        )
    committed = action()
    assert committed.evidence_ids == ("obs-1", "obs-2")
    assert committed.action_epoch == 6
    assert committed.suit_options == (("5D",), ("5D",))
    assert committed.captured_ms == committed.last_frame.captured_ms
    with pytest.raises(ValueError, match="turn_index exactly once"):
        action(
            version_after=replace(
                before,
                state_revision=5,
                turn_index=5,
            )
        )
    with pytest.raises(ValueError, match="action_epoch must match"):
        action(action_epoch=7)


def test_confirmed_action_is_a_complete_trace_of_its_source_candidate() -> None:
    source = candidate(
        cards=("7D",),
        suit_options=(("7D", "7S"),),
        action_epoch=9,
    )
    before = source.version.state_version
    after = replace(before, state_revision=5, turn_index=4)
    confirmed = ConfirmedAction.from_candidate(
        action_id="action-uncertain-7",
        candidate=source,
        version_before=before,
        version_after=after,
        cards=("7?",),
        processing_ms=1_140,
        reason=ConfirmationReason.RULE_VALIDATED,
    )
    assert confirmed.source_candidate is source
    assert confirmed.candidate_ids == (source.candidate_id,)
    assert confirmed.evidence_ids == source.evidence_ids
    assert confirmed.action_epoch == source.action_epoch
    assert confirmed.first_frame is source.first_frame
    assert confirmed.last_frame is source.last_frame
    assert confirmed.suit_options == source.suit_options
    with pytest.raises(ValueError, match="match source candidate"):
        replace(confirmed, suit_options=(("7D",),))


def test_gap_state_requires_a_reason_and_discards_evidence_when_clear() -> None:
    assert clear_gap().phase is GapPhase.CLEAR
    gap = GapState(
        version=version(),
        phase=GapPhase.BLOCKING,
        reason=GapReason.MISSING_EXPECTED_ACTION,
        expected_seats=(Seat.OPPOSITE,),
        evidence_ids=("obs-4",),
        opened_captured_ms=2_000,
        processing_ms=2_050,
    )
    assert gap.expected_seats == (Seat.OPPOSITE,)
    with pytest.raises(ValueError, match="requires reason NONE"):
        replace(clear_gap(), reason=GapReason.STALE_EVIDENCE)
    with pytest.raises(ValueError, match="concrete GapReason"):
        replace(gap, reason=GapReason.NONE)
    with pytest.raises(ValueError, match="cannot retain"):
        replace(clear_gap(), evidence_ids=("stale",))


def test_advice_ready_is_only_for_current_local_trusted_state() -> None:
    assert ready_opportunity().status is OpportunityStatus.READY
    with pytest.raises(ValueError, match="local seat"):
        replace(ready_opportunity(), seat=Seat.LEFT)
    with pytest.raises(ValueError, match="requires TRUSTED_STATE"):
        replace(ready_opportunity(), reason=OpportunityReason.HISTORY_GAP)
    blocked = replace(
        ready_opportunity(),
        status=OpportunityStatus.BLOCKED,
        reason=OpportunityReason.HISTORY_GAP,
    )
    assert blocked.reason is OpportunityReason.HISTORY_GAP


def test_engine_update_enforces_one_exact_stream_and_nested_version() -> None:
    current = version()
    update = EngineUpdate(
        version=current,
        reason=EngineUpdateReason.OPPORTUNITY_CHANGED,
        captured_ms=1_100,
        processing_ms=1_150,
        observations=(observation(),),
        candidates=(candidate(),),
        gap=clear_gap(current),
        advice_opportunity=ready_opportunity(current),
    )
    assert update.observations[0].frame.captured_ms < update.processing_ms
    with pytest.raises(ValueError, match="another session or generation"):
        replace(update, observations=(observation(frame=frame(capture_generation=8)),))
    with pytest.raises(ValueError, match="exact update version"):
        replace(update, gap=clear_gap(replace(current, update_sequence=10)))
    with pytest.raises(TypeError, match="observations must be a tuple"):
        replace(update, observations=[])  # type: ignore[arg-type]


def test_projection_and_commit_results_make_transaction_outcome_explicit() -> None:
    confirmed = action()
    current = version()
    projected = ProjectionResult(
        base_version=current,
        actions=(confirmed,),
        rejected_candidate_ids=(),
        reason=ProjectionReason.ACCEPTED,
    )
    committed = CommitResult(
        expected_version=current,
        resulting_version=current.with_state(
            confirmed.version_after,
            update_sequence=current.update_sequence + 1,
        ),
        committed_actions=(confirmed,),
        reason=CommitReason.COMMITTED,
    )
    assert projected.actions == committed.committed_actions
    with pytest.raises(ValueError, match="must contain an action"):
        replace(projected, actions=())
    with pytest.raises(ValueError, match="failed commit cannot advance"):
        replace(committed, committed_actions=(), reason=CommitReason.VERSION_CONFLICT)


def test_projection_and_commit_reject_non_contiguous_action_chains() -> None:
    confirmed = action()
    current = version()
    wrong_start = replace(
        confirmed,
        version_before=replace(confirmed.version_before, state_revision=3),
        version_after=replace(confirmed.version_after, state_revision=4),
    )
    with pytest.raises(ValueError, match="contiguous version chain"):
        ProjectionResult(
            base_version=current,
            actions=(wrong_start,),
            rejected_candidate_ids=(),
            reason=ProjectionReason.ACCEPTED,
        )
    with pytest.raises(ValueError, match="end of the action chain"):
        CommitResult(
            expected_version=current,
            resulting_version=replace(
                current.with_state(confirmed.version_after),
                state_revision=6,
                turn_index=5,
            ),
            committed_actions=(confirmed,),
            reason=CommitReason.COMMITTED,
        )


def test_every_reason_field_is_typed_by_an_enum() -> None:
    contract_types = (
        SeatObservation,
        ActionCandidate,
        ConfirmedAction,
        GapState,
        AdviceOpportunity,
        EngineUpdate,
        ProjectionResult,
        CommitResult,
    )
    for contract_type in contract_types:
        reason_field = next(item for item in fields(contract_type) if item.name == "reason")
        resolved = get_type_hints(contract_type)[reason_field.name]
        assert isinstance(resolved, type) and issubclass(resolved, Enum)


def test_evidence_and_scheduling_reason_values_are_enums() -> None:
    assert EvidenceDropReason.CONSUMED_BOUNDARY.value == "consumed_boundary"
    assert SchedulingDropReason.RAW_REPLACED.value == "raw_replaced"
    assert ScheduledItemKind.CANDIDATE.value == "candidate"


def test_ports_are_narrow_runtime_checkable_structural_contracts() -> None:
    class Adapter:
        def project(self, **_: object) -> object:
            return object()

        def commit(self, **_: object) -> object:
            return object()

        def processing_ms(self) -> int:
            return 1

        def receive(self) -> tuple[SeatObservation, ...]:
            return ()

        def append(self, _: EngineUpdate) -> None:
            return None

        def publish(self, _: AdviceOpportunity) -> None:
            return None

    adapter = Adapter()
    assert isinstance(adapter, RuleProjector)
    assert isinstance(adapter, TransactionalActionCommitter)
    assert isinstance(adapter, ProcessingClock)
    assert isinstance(adapter, ObservationInput)
    assert isinstance(adapter, EventJournal)
    assert isinstance(adapter, AdviceConsumer)
