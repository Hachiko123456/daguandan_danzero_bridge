"""Durability contract for reviewed wildcard action semantics.

These tests intentionally exercise the public live-v2 ledger rather than the
legacy reducer's transient ``action_metadata`` dictionaries.  A recorded
wildcard declaration is authoritative evidence: persistence, generation
rebinding, corrections, replay projection, and FableDan must not infer a
different declaration from the same physical cards.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from daguandan_bridge.application.live_v2_runtime_updates import (
    trusted_to_live_snapshot,
)
from daguandan_bridge.infrastructure.live_v2_rule_session import (
    ProductionRuleSession,
)
from daguandan_bridge.danzero.state import GuanDanState
from daguandan_bridge.fabledan import FableDanAdvisor
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines
from daguandan_bridge.live_v2.corrections import (
    ConfirmedCorrection,
    CorrectionCommand,
    CorrectionReason,
)
from daguandan_bridge.live_v2.types import (
    ActionCandidate,
    ActionSemantics,
    ActionKind,
    CandidateReason,
    ConfirmedAction,
    EvidenceOrigin,
    FrameIdentity,
    Seat,
)
from daguandan_bridge.live_v2.game_state import GameAction


PHYSICAL_CARDS = ("10C", "6C", "6H", "7C", "9C")
SUIT_OPTIONS = tuple((card,) for card in PHYSICAL_CARDS)
STRAIGHT = {
    "move_type": "Straight",
    "key": "6",
    "logical_label": "678910",
    "wildcard_assignments": [{"physical_card": "6H", "as_rank": "8"}],
}
STRAIGHT_FLUSH = {
    "move_type": "StraightFlush",
    "key": "6",
    "logical_label": "678910",
    "wildcard_assignments": [{"physical_card": "6H", "as_rank": "8"}],
}
AMBIGUOUS_FULL_HOUSE_CARDS = ("JC", "JD", "9H", "2D", "2H")
FULL_HOUSE_J = {
    "move_type": "ThreeWithTwo",
    "key": "J",
    "logical_label": "JJJ22",
    "wildcard_assignments": [{"physical_card": "9H", "as_rank": "J"}],
}
FULL_HOUSE_2 = {
    "move_type": "ThreeWithTwo",
    "key": "2",
    "logical_label": "222JJ",
    "wildcard_assignments": [{"physical_card": "9H", "as_rank": "2"}],
}


def _hand() -> tuple[str, ...]:
    ranks = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
    return tuple(f"{rank}{suit}" for suit in "SHC" for rank in ranks)[:27]


def _store(tmp_path: Path) -> LiveSessionStore:
    store = LiveSessionStore(tmp_path, "profile", session_id="semantic-session")
    store.start({})
    return store


def _semantics(*, selected: dict[str, object] | None, source: str):
    contract = ActionSemantics
    assert dataclasses.is_dataclass(contract), "ActionSemantics must be a dataclass"
    assert contract.__dataclass_params__.frozen is True, "ActionSemantics must be frozen"
    assert selected is not None, "unresolved ambiguity has no ActionSemantics value"
    value = contract.requested_from_metadata({
        "interpretation_ambiguous": False,
        "candidate_interpretations": [selected],
        "selected_interpretation": selected,
        "selection_source": source,
    })
    assert value is not None
    encoded = json.dumps(
        value.to_metadata(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert json.loads(encoded.decode("utf-8")) == value.to_metadata()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        value.selection_source = "mutated"
    return value


def _full_house_semantics(*, selected: dict[str, object], source: str):
    contract = ActionSemantics
    value = contract.requested_from_metadata({
        "interpretation_ambiguous": True,
        "candidate_interpretations": [FULL_HOUSE_J, FULL_HOUSE_2],
        "selected_interpretation": selected,
        "selection_source": source,
    })
    assert value is not None
    return value


def _assert_same_semantic_decision(actual, expected) -> None:
    assert actual.selected == expected.selected
    assert actual.selection_source == expected.selection_source
    assert set(actual.candidates) == set(expected.candidates)


def _candidate(
    session: ProductionRuleSession,
    semantics,
    *,
    origin: EvidenceOrigin = EvidenceOrigin.TRUSTED,
    cards: tuple[str, ...] = PHYSICAL_CARDS,
) -> ActionCandidate:
    fields = {field.name for field in dataclasses.fields(ActionCandidate)}
    assert "requested_semantics" in fields, (
        "ActionCandidate must carry requested ActionSemantics"
    )
    version = session.version
    trusted = origin in {EvidenceOrigin.MANUAL, EvidenceOrigin.TRUSTED}
    source_id = f"{origin.value}:semantic-evidence" if trusted else "window"
    first = FrameIdentity(
        "semantic-session", version.capture_generation, 10, 100, "roi", source_id
    )
    last = first if trusted else FrameIdentity(
        "semantic-session", version.capture_generation, 11, 110, "roi", source_id
    )
    return ActionCandidate(
        candidate_id="wildcard-action",
        version=version,
        seat=Seat.RIGHT,
        kind=ActionKind.PLAY,
        cards=cards,
        suit_options=tuple((card,) for card in cards),
        evidence_ids=(
            ("semantic-evidence",)
            if trusted
            else ("semantic-evidence-a", "semantic-evidence-b")
        ),
        action_epoch=1,
        first_frame=first,
        last_frame=last,
        processing_ms=120,
        confidence=1.0,
        reason=(
            CandidateReason.LOCAL_ACTION_CONFIRMED
            if trusted
            else CandidateReason.STABLE_PLAY
        ),
        evidence_origin=origin,
        requested_semantics=semantics,
    )


def _commit(session: ProductionRuleSession, candidate: ActionCandidate):
    binding = session.bind_generation(session.version.capture_generation)
    projected = binding.adapter.project(
        base_version=binding.version,
        candidates=(candidate,),
        processing_ms=candidate.processing_ms,
    )
    result = binding.adapter.commit(
        expected_version=binding.version,
        actions=projected.actions,
    )
    assert len(result.committed_actions) == 1
    return result.committed_actions[0]


def _semantic_payload(record: dict[str, object]) -> dict[str, object]:
    payload = record.get("payload")
    assert isinstance(payload, dict)
    value = payload.get(
        "action_semantics",
        payload.get("move_semantics", payload.get("action_metadata")),
    )
    if value is None and "selected_interpretation" in payload:
        value = {
            key: payload[key]
            for key in (
                "interpretation_ambiguous",
                "candidate_interpretations",
                "selected_interpretation",
                "selection_source",
            )
            if key in payload
        }
    assert isinstance(value, dict), "durable action event must embed action semantics"
    return value


def test_live_v2_exposes_semantics_at_every_durable_boundary():
    required = {
        "ActionCandidate": (ActionCandidate, {"requested_semantics"}),
        "ConfirmedAction": (ConfirmedAction, {"semantics"}),
        "CorrectionCommand": (CorrectionCommand, {"requested_semantics"}),
        "ConfirmedCorrection": (
            ConfirmedCorrection,
            {"previous_semantics", "corrected_semantics"},
        ),
        "GameAction": (GameAction, {"semantics"}),
    }
    missing = {
        name: sorted(fields - {field.name for field in dataclasses.fields(contract)})
        for name, (contract, fields) in required.items()
        if fields - {field.name for field in dataclasses.fields(contract)}
    }

    assert not missing, f"wildcard semantics disappear at durable boundaries: {missing}"


def test_action_semantics_survives_durable_event_round_trip_byte_for_byte(tmp_path: Path):
    semantics = _semantics(selected=STRAIGHT_FLUSH, source="exact_engine_state")
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    session.initialize(
        round_level="6",
        hand=_hand(),
        lead_player=Seat.RIGHT,
        monotonic_ms=1,
        capture_generation=1,
    )

    confirmed = _commit(session, _candidate(session, semantics))
    rows = read_json_lines(store.timeline_path)
    durable = next(row for row in rows if row["event_type"] == "player_played")
    raw_line = next(
        line
        for line in store.timeline_path.read_bytes().splitlines()
        if b'"event_type":"player_played"' in line
    )

    _assert_same_semantic_decision(confirmed.semantics, semantics)
    assert _semantic_payload(durable) == semantics.to_metadata()
    assert json.loads(raw_line.decode("utf-8"))["payload"] == durable["payload"]


def test_semantics_survives_seed_and_cross_generation_snapshot(tmp_path: Path):
    semantics = _semantics(selected=STRAIGHT_FLUSH, source="exact_engine_state")
    session = ProductionRuleSession(_store(tmp_path))
    session.initialize(
        round_level="6",
        hand=_hand(),
        lead_player=Seat.RIGHT,
        monotonic_ms=1,
        capture_generation=1,
    )
    _commit(session, _candidate(session, semantics))

    rebound = session.bind_generation(2)
    trusted = session.snapshot(captured_ms=200)
    replay_projection = trusted_to_live_snapshot(trusted)

    assert rebound.version.capture_generation == 2
    _assert_same_semantic_decision(session.confirmed_actions[0].semantics, semantics)
    _assert_same_semantic_decision(trusted.play_history[0].semantics, semantics)
    replayed = ActionSemantics.requested_from_metadata(
        replay_projection.play_history[0].action_metadata
    )
    assert replayed is not None
    _assert_same_semantic_decision(replayed, semantics)
    advisor_state = GuanDanState()
    advisor_state.set_context(
        round_level="6",
        wild_rank="6",
        current_player="self",
        lead_player="right",
    )
    advisor_state.confirm_hand(_hand())
    recorded = replay_projection.play_history[0]
    advisor_state.record_play(
        "right",
        recorded.cards,
        suit_options=recorded.suit_options,
        action_metadata=recorded.action_metadata,
    )
    advisor_state.record_pass("opposite")
    advisor_state.record_pass("left")
    advisor_state.remaining_cards = dict(replay_projection.remaining_cards)
    decision = FableDanAdvisor(
        tmp_path,
        "profile",
        runtime_policy="rule_only",
        diagnostics="full",
        write_decision_log=False,
    ).recommend_detailed(advisor_state, request_id="durable-semantics")
    resolution = decision.advice.engine_input["fabledan_trace"][
        "adapter_observation"
    ]["events"][0]["semantic_resolution"]
    assert resolution["selection_source"] == "exact_engine_state"
    assert resolution["selected_interpretation"]["type"] == "SFLUSH"


def test_correction_replaces_semantics_and_replay_never_keeps_stale_selection(
    tmp_path: Path,
):
    original = _full_house_semantics(
        selected=FULL_HOUSE_2, source="exact_engine_state"
    )
    reviewed = _full_house_semantics(selected=FULL_HOUSE_J, source="manual_review")
    store = _store(tmp_path)
    session = ProductionRuleSession(store)
    session.initialize(
        round_level="9",
        hand=_hand(),
        lead_player=Seat.RIGHT,
        monotonic_ms=1,
        capture_generation=1,
    )
    action = _commit(
        session,
        _candidate(session, original, cards=AMBIGUOUS_FULL_HOUSE_CARDS),
    )
    command_fields = {field.name for field in dataclasses.fields(CorrectionCommand)}
    assert "requested_semantics" in command_fields, (
        "CorrectionCommand must carry replacement ActionSemantics"
    )
    command = CorrectionCommand(
        correction_id="review-wildcard-1",
        expected_version=session.version,
        target_action_id=action.action_id,
        kind=ActionKind.PLAY,
        cards=AMBIGUOUS_FULL_HOUSE_CARDS,
        suit_options=tuple((card,) for card in AMBIGUOUS_FULL_HOUSE_CARDS),
        reason=CorrectionReason.MANUAL_REVIEW,
        evidence_id="operator-reviewed-wildcard",
        evidence_origin=EvidenceOrigin.MANUAL,
        confidence=1.0,
        corrected_ms=200,
        requested_semantics=reviewed,
    )

    correction = session.correct_latest(command)
    trusted = session.snapshot(captured_ms=200)
    replay_projection = trusted_to_live_snapshot(trusted)
    correction_row = next(
        row
        for row in read_json_lines(store.timeline_path)
        if row["event_type"] == "event_correction"
    )

    _assert_same_semantic_decision(correction.previous_semantics, original)
    _assert_same_semantic_decision(correction.corrected_semantics, reviewed)
    _assert_same_semantic_decision(trusted.play_history[0].semantics, reviewed)
    replayed = ActionSemantics.requested_from_metadata(
        replay_projection.play_history[0].action_metadata
    )
    durable_correction = ActionSemantics.requested_from_metadata(
        _semantic_payload(correction_row)
    )
    assert replayed is not None and durable_correction is not None
    _assert_same_semantic_decision(replayed, reviewed)
    _assert_same_semantic_decision(durable_correction, reviewed)
    assert replayed.selected != original.selected


def test_manual_ambiguity_requires_an_explicit_reviewed_selection(tmp_path: Path):
    reviewed = _full_house_semantics(selected=FULL_HOUSE_J, source="manual_review")
    session = ProductionRuleSession(_store(tmp_path))
    session.initialize(
        round_level="9",
        hand=_hand(),
        lead_player=Seat.RIGHT,
        monotonic_ms=1,
        capture_generation=1,
    )

    binding = session.bind_generation(session.version.capture_generation)
    unresolved = binding.adapter.project(
        base_version=binding.version,
        candidates=(
            _candidate(
                session,
                None,
                origin=EvidenceOrigin.MANUAL,
                cards=AMBIGUOUS_FULL_HOUSE_CARDS,
            ),
        ),
        processing_ms=120,
    )
    assert unresolved.actions == (), (
        "manual ambiguous wildcard input must be rejected until a reviewed selection exists"
    )

    confirmed = _commit(
        session,
        _candidate(
            session,
            reviewed,
            origin=EvidenceOrigin.MANUAL,
            cards=AMBIGUOUS_FULL_HOUSE_CARDS,
        ),
    )
    _assert_same_semantic_decision(confirmed.semantics, reviewed)
