"""The sole gateway allowed to import and execute legacy rule machinery."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from collections import Counter

from ..action_semantics import canonical_fabledan_type
from ..danzero.rules import (
    action_for_cards,
    actions_for_cards,
    logical_action_label,
    logical_action_ranks,
    play_beats_table,
    wildcard_substitutions,
)
from ..danzero.state import GameStateError, GuanDanState, PlayEvent
from ..domain.live import LiveEvent
from ..fabledan._vendor.fabledan.cards import RANK_NAMES, is_wildcard
from ..fabledan._vendor.fabledan.combos import Move, TYPE_NAMES, beats, gen_moves
from ..live.card_uncertainty import feasible_action_variants
from ..live.reducer import LiveReducer
from ..live.turns import WindCatchPolicy
from ..live_v2.action_semantics import ActionInterpretation, ActionSemantics
from ..live_v2.corrections import ConfirmedCorrection, CorrectionCommand
from ..live_v2.game_state import GameAction, TrustedGameSnapshot
from ..live_v2.identity import Seat
from ..live_v2.types import (
    ActionCandidate,
    ActionKind,
    CandidateReason,
    ConfirmationReason,
    ConfirmedAction,
    EvidenceOrigin,
    StateVersion,
    VersionIdentity,
    FrameIdentity,
)


@dataclass(frozen=True, slots=True)
class StagedReducerResult:
    reducer: LiveReducer | None
    actions: tuple[ConfirmedAction, ...] = ()
    events: tuple[LiveEvent, ...] = ()
    needs_more_evidence: bool = False
@dataclass(frozen=True, slots=True)
class StagedCorrectionResult:
    reducer: LiveReducer
    correction: ConfirmedCorrection
    event: LiveEvent
class _NeedMoreEvidence(Exception):
    pass


def create_legacy_reducer(session_id: str) -> object:
    return LiveReducer(
        session_id,
        wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    )


def is_legacy_reducer(value: object) -> bool:
    return isinstance(value, LiveReducer)


def legacy_events(reducer: object) -> tuple[LiveEvent, ...]:
    return _require_reducer(reducer).events


def legacy_snapshot(reducer: object):
    return _require_reducer(reducer).snapshot()


def legacy_identity(reducer: object) -> tuple[str, int, int]:
    value = _require_reducer(reducer)
    snapshot = value.snapshot()
    return value.session_id, snapshot.revision, max(0, snapshot.turn_id - 1)


def copy_reducer(reducer: object) -> object:
    return _copy_reducer(_require_reducer(reducer))


def adopt_reducer(target: object, staged: object) -> None:
    _require_reducer(target).adopt_staged(_require_reducer(staged))


def initialize_reducer(
    reducer: object,
    *,
    round_level: str,
    hand: tuple[str, ...],
    lead_player: Seat | None,
    monotonic_ms: int,
    wall_time: str | None,
    evidence_id: str,
) -> tuple[object, LiveEvent]:
    staged = _require_reducer(reducer).clone_empty()
    event = staged.confirm_initial_state(
        round_level=round_level,
        hand=hand,
        lead_player=None if lead_player is None else lead_player.value,
        confidence=1.0,
        source="live_v2_rule_session",
        evidence_refs=(evidence_id,),
        monotonic_ms=monotonic_ms,
        wall_time=wall_time,
    )
    return staged, event


def confirm_lead_reducer(
    reducer: object,
    lead_player: Seat,
    *,
    monotonic_ms: int,
    evidence_id: str,
) -> tuple[object, LiveEvent]:
    generated_on = _copy_reducer(_require_reducer(reducer))
    generated = generated_on.confirm_lead_player(lead_player.value)
    event = replace(
        generated,
        monotonic_ms=monotonic_ms,
        source="live_v2_rule_session",
        evidence_refs=(evidence_id,),
    )
    staged = _copy_reducer(_require_reducer(reducer))
    staged.apply(event)
    return staged, event


def replay_trusted_snapshot(snapshot: TrustedGameSnapshot) -> GuanDanState:
    """Project the authoritative LiveV2 snapshot directly for an advisor.

    This gateway remains the sole infrastructure boundary that knows the
    legacy advisor DTO. It deliberately does not create a second LiveReducer:
    the immutable trusted snapshot is the only source of game history.
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
        event = _project_snapshot_action(action)
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


def _project_snapshot_action(action: GameAction) -> PlayEvent:
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


def _require_reducer(value: object) -> LiveReducer:
    if not isinstance(value, LiveReducer):
        raise TypeError("value is not a legacy reducer token")
    return value


def stage_reducer(
    *,
    reducer: LiveReducer,
    ordered: tuple[ActionCandidate, ...],
    base_version: VersionIdentity,
    processing_ms: int,
) -> StagedReducerResult:
    """Validate an entire candidate chain on an isolated reducer clone."""

    event_count = len(reducer.events)
    staged = _copy_reducer(reducer)
    actions: list[ConfirmedAction] = []
    version = base_version.state_version
    try:
        for candidate in ordered:
            cards, semantics = _apply_candidate(staged, candidate)
            after = StateVersion(
                version.session_id,
                version.state_revision + 1,
                version.turn_index + 1,
            )
            actions.append(
                ConfirmedAction.from_candidate(
                    action_id=(
                        f"action:{version.session_id}:"
                        f"{version.state_revision + 1}:"
                        f"{candidate.candidate_id}"
                    ),
                    candidate=candidate,
                    version_before=version,
                    version_after=after,
                    processing_ms=max(processing_ms, candidate.processing_ms),
                    reason=_confirmation_reason(candidate),
                    cards=cards,
                    semantics=semantics,
                )
            )
            version = after
    except _NeedMoreEvidence:
        return StagedReducerResult(None, needs_more_evidence=True)
    except (GameStateError, ValueError, ImportError, ModuleNotFoundError):
        return StagedReducerResult(None)
    return StagedReducerResult(
        staged,
        tuple(actions),
        tuple(staged.events[event_count:]),
    )


def stage_action_correction(
    *,
    reducer: LiveReducer,
    target_action: ConfirmedAction,
    target_event: LiveEvent,
    command: CorrectionCommand,
) -> StagedCorrectionResult:
    """Validate corrected semantics on the pre-action state, then rebuild fully."""

    events = reducer.events
    try:
        target_index = events.index(target_event)
    except ValueError as exc:
        raise ValueError("target action event is absent from reducer") from exc
    prefix = reducer.clone_empty()
    for event in events[:target_index]:
        prefix.apply(event)
    frame = FrameIdentity(
        session_id=command.expected_version.session_id,
        capture_generation=command.expected_version.capture_generation,
        frame_sequence=command.corrected_ms,
        captured_ms=command.corrected_ms,
        roi_version="explicit-correction",
        source_id=f"{command.evidence_origin.value}:{command.evidence_id}",
    )
    candidate = ActionCandidate(
        candidate_id=f"correction:{command.correction_id}",
        version=VersionIdentity.from_state(
            target_action.version_before,
            capture_generation=command.expected_version.capture_generation,
            update_sequence=command.expected_version.update_sequence,
        ),
        seat=target_action.seat,
        kind=command.kind,
        cards=command.cards,
        suit_options=command.suit_options,
        evidence_ids=(command.evidence_id,),
        action_epoch=target_action.action_epoch,
        first_frame=frame,
        last_frame=frame,
        processing_ms=command.corrected_ms,
        confidence=command.confidence,
        reason=(
            CandidateReason.VISUAL_CORRECTION
            if command.evidence_origin is EvidenceOrigin.VISUAL
            else CandidateReason.LOCAL_ACTION_CONFIRMED
        ),
        evidence_origin=command.evidence_origin,
        requested_semantics=command.requested_semantics,
    )
    validation = stage_reducer(
        reducer=prefix,
        ordered=(candidate,),
        base_version=candidate.version,
        processing_ms=command.corrected_ms,
    )
    if validation.reducer is None:
        raise ValueError("corrected action is not legal in the target state")
    corrected_semantics = validation.actions[0].semantics

    play_history = reducer.snapshot().play_history
    target_turn_index = int(target_action.version_before.turn_index)
    if target_turn_index < 0 or target_turn_index >= len(play_history):
        raise ValueError("correction target is outside reducer play history")
    current_play = play_history[target_turn_index]
    previous_kind = ActionKind.PASS if current_play.is_pass else ActionKind.PLAY
    previous_options = _public_suit_options(
        tuple(current_play.cards),
        tuple(tuple(item) for item in current_play.suit_options),
    )
    correction = ConfirmedCorrection(
        correction_id=command.correction_id,
        target_action_id=target_action.action_id,
        version_before=command.expected_version.state_version,
        version_after=StateVersion(
            command.expected_version.session_id,
            command.expected_version.state_revision + 1,
            command.expected_version.turn_index,
        ),
        seat=target_action.seat,
        previous_kind=previous_kind,
        previous_cards=tuple(current_play.cards),
        previous_suit_options=previous_options,
        corrected_kind=command.kind,
        corrected_cards=command.cards,
        corrected_suit_options=command.suit_options,
        reason=command.reason,
        evidence_id=command.evidence_id,
        evidence_origin=command.evidence_origin,
        confidence=command.confidence,
        corrected_ms=command.corrected_ms,
        previous_semantics=target_action.semantics,
        corrected_semantics=corrected_semantics,
    )
    generator = _copy_reducer(reducer)
    generated = generator.correct_event(
        target_event.event_id,
        cards=command.cards,
        is_pass=command.kind is ActionKind.PASS,
        reason=command.reason.value,
        confidence=command.confidence,
        source=f"live_v2_correction:{command.evidence_origin.value}",
    )
    enriched = replace(
        generated,
        monotonic_ms=command.corrected_ms,
        payload={
            **generated.payload,
            "target_action_id": target_action.action_id,
            "correction_id": command.correction_id,
            "correction_reason": command.reason.value,
            "evidence_origin": command.evidence_origin.value,
            **(
                {}
                if corrected_semantics is None
                else {"move_semantics": corrected_semantics.to_metadata()}
            ),
        },
        evidence_refs=(command.evidence_id,),
    )
    staged = _copy_reducer(reducer)
    staged.apply(enriched)
    return StagedCorrectionResult(staged, correction, enriched)


# Compatibility name retained for existing callers; the implementation now
# accepts any still-uncorrected action in the committed history.
stage_latest_correction = stage_action_correction

def _public_suit_options(
    cards: tuple[str, ...],
    legacy_options: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []
    for index, card in enumerate(cards):
        supplied = legacy_options[index] if index < len(legacy_options) else ()
        choices = tuple(
            option if len(option) > 1 else f"{card[:-1]}{option}"
            for option in supplied
        )
        # The public correction contract requires the displayed card token
        # itself to remain among its physical options, including ``5?``.
        values = tuple(dict.fromkeys((card, *choices)))
        result.append(values or ((card,) if card[-1:] in "SHCD" else ()))
    return tuple(result)


def _copy_reducer(source: LiveReducer) -> LiveReducer:
    staged = source.clone_empty()
    for event in source.events:
        staged.apply(event)
    return staged


def _confirmation_reason(candidate: ActionCandidate) -> ConfirmationReason:
    if candidate.reason is CandidateReason.RECONCILIATION_EVIDENCE:
        return ConfirmationReason.ATOMIC_RECONCILIATION
    if candidate.reason is CandidateReason.LOCAL_ACTION_CONFIRMED:
        return ConfirmationReason.LOCAL_ACTION_COMMITTED
    return ConfirmationReason.RULE_VALIDATED


def _legacy_cards_and_options(
    candidate: ActionCandidate,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    cards: list[str] = []
    suit_options: list[tuple[str, ...]] = []
    for card, choices in zip(candidate.cards, candidate.suit_options, strict=True):
        if len(choices) == 1:
            cards.append(card)
            suit_options.append((card[-1],) if card[-1:] in "SHCD" else ())
            continue
        ranks = {choice[:-1] for choice in choices if choice[-1:] in "SHCD"}
        suits = tuple(
            dict.fromkeys(choice[-1] for choice in choices if choice[-1:] in "SHCD")
        )
        if len(ranks) != 1 or not suits:
            raise GameStateError("suit options must describe one physical rank")
        cards.append(f"{next(iter(ranks))}?")
        suit_options.append(suits)
    return tuple(cards), tuple(suit_options)


def _known_cards_and_options(
    staged: LiveReducer,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    snapshot = staged.snapshot()
    cards = tuple(snapshot.my_hand) + tuple(
        card for event in snapshot.play_history for card in event.cards
    )
    options = tuple(() for _card in snapshot.my_hand) + tuple(
        option for event in snapshot.play_history for option in event.suit_options
    )
    return cards, options


def _table_variants(
    staged: LiveReducer,
    known_cards: tuple[str, ...],
    known_options: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...] | None:
    snapshot = staged.snapshot()
    table = next(
        (event for event in reversed(snapshot.trick_plays) if not event.is_pass),
        None,
    )
    if table is None:
        return None
    return feasible_action_variants(
        cards=table.cards,
        suit_options=table.suit_options,
        known_cards=known_cards,
        known_suit_options=known_options,
        candidate_already_known=True,
        limit=64,
    )


def _apply_candidate(
    staged: LiveReducer, candidate: ActionCandidate
) -> tuple[tuple[str, ...], ActionSemantics | None]:
    seat = candidate.seat.value
    source = "manual_candidate_confirmation" if candidate.audit_evidence_ids else "live_v2"
    if candidate.kind is ActionKind.PASS:
        staged.record_pass(
            seat,
            confidence=candidate.confidence,
            source=source,
            evidence_refs=candidate.event_evidence_ids,
            monotonic_ms=candidate.last_captured_ms,
        )
        return (), None

    physical_cards, suit_options = _legacy_cards_and_options(candidate)
    snapshot = staged.snapshot()
    known_cards, known_options = _known_cards_and_options(staged)
    variants = feasible_action_variants(
        cards=physical_cards,
        suit_options=suit_options,
        known_cards=known_cards,
        known_suit_options=known_options,
        candidate_already_known=seat == "self",
        limit=64,
    )
    tables = _table_variants(staged, known_cards, known_options)
    validity = tuple(
        bool(actions_for_cards(cards, snapshot.wild_rank))
        and (
            tables is None
            or bool(tables)
            and all(
                play_beats_table(cards, table, snapshot.wild_rank)
                for table in tables
            )
        )
        for cards in variants
    )
    if any(validity) and not all(validity):
        raise _NeedMoreEvidence
    valid = tuple(
        cards
        for cards, is_valid in zip(variants, validity, strict=True)
        if is_valid
    )
    if not valid:
        raise GameStateError("candidate is not a legal play in staged state")
    if seat == "self" and len(set(valid)) != 1:
        raise _NeedMoreEvidence
    committed_cards = valid[0] if seat == "self" else physical_cards
    semantics = _resolve_action_semantics(
        candidate, valid, snapshot.wild_rank, snapshot.trick_plays
    )
    staged.record_play(
        seat,
        committed_cards,
        confidence=candidate.confidence,
        source=source,
        evidence_refs=candidate.event_evidence_ids,
        suit_options=() if committed_cards != physical_cards else suit_options,
        action_metadata={"move_semantics": semantics.to_metadata()},
        monotonic_ms=candidate.last_captured_ms,
    )
    return committed_cards, semantics


def _resolve_action_semantics(
    candidate: ActionCandidate,
    valid_variants: tuple[tuple[str, ...], ...],
    wild_rank: str,
    trick_plays: tuple[object, ...] | list[object],
) -> ActionSemantics:
    interpretations = tuple(
        _interpretations_for_cards(cards, wild_rank) for cards in valid_variants
    )
    if not interpretations or any(items != interpretations[0] for items in interpretations):
        raise _NeedMoreEvidence
    rule_candidates = interpretations[0]
    if not rule_candidates:
        raise GameStateError("candidate has no legal action interpretation")
    exact_moves = _exact_moves(valid_variants[0], wild_rank)
    exact_candidates = tuple(item for _move, item in exact_moves)
    requested = candidate.requested_semantics
    if requested is not None:
        if not any(
            _matches_requested(requested.selected, item)
            for item in (*rule_candidates, *exact_candidates)
        ):
            raise GameStateError("explicit action interpretation is not uniquely legal")
        return requested
    if candidate.evidence_origin is EvidenceOrigin.MANUAL and len(exact_candidates) > 1:
        raise _NeedMoreEvidence
    if candidate.evidence_origin is EvidenceOrigin.TRUSTED and exact_candidates:
        selected = _select_exact_move(exact_moves, trick_plays, wild_rank)
        return ActionSemantics(
            selected, exact_candidates, "exact_engine_state"
        )
    else:
        strongest = action_for_cards(valid_variants[0], wild_rank)
        if strongest is None:
            raise GameStateError("candidate has no strongest legal interpretation")
        strongest_interpretation = _interpretation(strongest, wild_rank)
        selected = next(
            (
                item
                for item in exact_candidates
                if _matches_requested(strongest_interpretation, item)
            ),
            strongest_interpretation,
        )
        candidates = exact_candidates or rule_candidates
        source = (
            "rules_strongest_wildcard"
            if len(candidates) > 1
            else "rules_unique"
        )
    return ActionSemantics(selected, candidates, source)


def _interpretations_for_cards(
    cards: tuple[str, ...], wild_rank: str
) -> tuple[ActionInterpretation, ...]:
    return tuple(
        dict.fromkeys(
            _interpretation(action, wild_rank)
            for action in actions_for_cards(cards, wild_rank)
        )
    )


def _interpretation(action: list[object], wild_rank: str) -> ActionInterpretation:
    return ActionInterpretation(
        move_type=str(action[0]),
        key=str(action[1]),
        logical_label=logical_action_label(action, wild_rank),
        wildcard_assignments=wildcard_substitutions(action, wild_rank),
        claim_ranks=logical_action_ranks(action, wild_rank),
    )


def _matches_requested(
    requested: ActionInterpretation, actual: ActionInterpretation
) -> bool:
    requested_type = _fabledan_type(requested)
    actual_type = _fabledan_type(actual)
    cross_schema = (requested.type_id is None) != (actual.type_id is None)
    requested_ranks = _declared_ranks(requested)
    actual_ranks = _declared_ranks(actual)
    declarations_match = (
        bool(requested_ranks)
        and bool(actual_ranks)
        and requested_ranks == actual_ranks
    )
    if cross_schema:
        key_matches = _project_key_matches_claims(
            requested if requested.type_id is None else actual,
            actual_ranks if actual.type_id is not None else requested_ranks,
        )
    elif requested.type_id is not None:
        key_matches = requested.key == actual.key
    else:
        key_matches = _canonical_rank(requested.key) == _canonical_rank(actual.key)
    requested_wildcards = _normalized_wildcards(requested)
    actual_wildcards = _normalized_wildcards(actual)
    return (
        requested_type == actual_type
        and key_matches
        and (
            not requested_ranks
            or not actual_ranks
            or declarations_match
        )
        and (
            not requested_wildcards
            or not actual_wildcards
            or requested_wildcards == actual_wildcards
        )
    )


def _project_key_matches_claims(
    project: ActionInterpretation,
    claims: tuple[str, ...],
) -> bool:
    move_type = _fabledan_type(project)
    key = _canonical_rank(project.key)
    if not claims or move_type is None:
        return False
    if move_type == "SINGLE":
        return claims == (key,)
    if move_type == "PAIR":
        return claims == (key, key)
    if move_type == "TRIPLE":
        return claims == (key, key, key)
    if move_type == "FULL":
        return len(claims) == 5 and claims[:3] == (key, key, key)
    if move_type in {"STRAIGHT", "SFLUSH", "PLATE", "TUBE"}:
        return claims[0] == key
    if move_type == "BOMB":
        return len(claims) >= 4 and set(claims) == {key}
    return False


def _declared_ranks(interpretation: ActionInterpretation) -> tuple[str, ...]:
    if interpretation.claim_ranks:
        return tuple(_canonical_rank(rank) for rank in interpretation.claim_ranks)
    return _ranks_from_label(interpretation.logical_label)


def _ranks_from_label(label: str) -> tuple[str, ...]:
    text = str(label).strip()
    result: list[str] = []
    while text:
        token = next(
            (
                item
                for item in ("small_joker", "big_joker", "小王", "大王", "10")
                if text.startswith(item)
            ),
            text[0],
        )
        result.append(_canonical_rank(token))
        text = text[len(token) :]
    return tuple(result)


def _canonical_rank(value: object) -> str:
    raw = str(value).strip()
    aliases = {
        "b": "SJ",
        "sj": "SJ",
        "小王": "SJ",
        "small_joker": "SJ",
        "r": "BJ",
        "bj": "BJ",
        "大王": "BJ",
        "big_joker": "BJ",
        "t": "10",
    }
    return aliases.get(raw.casefold(), raw.upper())


def _normalized_wildcards(
    interpretation: ActionInterpretation,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                _canonical_card(card),
                _canonical_rank(rank),
            )
            for card, rank in interpretation.wildcard_assignments
        )
    )


def _canonical_card(value: object) -> str:
    raw = str(value).strip()
    rank = _canonical_rank(raw)
    if rank in {"SJ", "BJ"}:
        return "small_joker" if rank == "SJ" else "big_joker"
    if len(raw) >= 2 and raw[-1:].upper() in "SHCD":
        return f"{_canonical_rank(raw[:-1])}{raw[-1].upper()}"
    return raw


def _fabledan_type(interpretation: ActionInterpretation) -> str | None:
    if interpretation.move_type:
        return canonical_fabledan_type(interpretation.move_type)
    if interpretation.type_id is None or not 0 <= interpretation.type_id < len(TYPE_NAMES):
        return None
    return TYPE_NAMES[interpretation.type_id]


def _exact_moves(
    cards: tuple[str, ...], wild_rank: str
) -> tuple[tuple[Move, ActionInterpretation], ...]:
    card_ids = _fabledan_card_ids(cards)
    level = RANK_NAMES.index(wild_rank)
    unique: dict[tuple[object, ...], tuple[Move, ActionInterpretation]] = {}
    for move in gen_moves(card_ids, level, None):
        if Counter(int(card) for card in move.cards) != Counter(card_ids):
            continue
        signature = (
            int(move.type),
            int(move.key),
            tuple(int(rank) for rank in move.claim_ranks),
        )
        unique.setdefault(
            signature,
            (
                move,
            ActionInterpretation(
                move_type="",
                type_id=int(move.type),
                key=int(move.key),
                claim_ranks=tuple(RANK_NAMES[int(rank)] for rank in move.claim_ranks),
                wildcard_assignments=tuple(
                    (
                        _fabledan_card_code(int(card)),
                        RANK_NAMES[int(rank)],
                    )
                    for card, rank in zip(move.cards, move.claim_ranks, strict=True)
                    if is_wildcard(int(card), level)
                    and _canonical_rank(RANK_NAMES[int(rank)])
                    != _canonical_rank(wild_rank)
                ),
                ),
            ),
        )
    return tuple(unique.values())


def _select_exact_move(
    candidates: tuple[tuple[Move, ActionInterpretation], ...],
    trick_plays: tuple[object, ...] | list[object],
    wild_rank: str,
) -> ActionInterpretation:
    table = next(
        (item for item in reversed(trick_plays) if not bool(getattr(item, "is_pass"))),
        None,
    )
    if table is None:
        return candidates[0][1]
    metadata = getattr(table, "action_metadata", None)
    table_semantics = ActionSemantics.requested_from_metadata(metadata)
    table_moves = _exact_moves(tuple(getattr(table, "cards")), wild_rank)
    selected_table = next(
        (
            move
            for move, interpretation in table_moves
            if table_semantics is not None
            and _matches_requested(table_semantics.selected, interpretation)
        ),
        table_moves[0][0] if len(table_moves) == 1 else None,
    )
    if selected_table is None:
        raise _NeedMoreEvidence
    level = RANK_NAMES.index(wild_rank)
    beating = tuple(
        interpretation
        for move, interpretation in candidates
        if beats(move, selected_table, level)
    )
    if not beating:
        raise GameStateError("selected action interpretation does not beat table")
    return beating[0]


def _fabledan_card_ids(cards: tuple[str, ...]) -> list[int]:
    suits = {"H": 0, "D": 1, "S": 2, "C": 3}
    seen: Counter[str] = Counter()
    result: list[int] = []
    for card in cards:
        if card == "small_joker":
            base = 52
        elif card == "big_joker":
            base = 53
        else:
            base = RANK_NAMES.index(card[:-1]) * 4 + suits[card[-1]]
        result.append(base + 54 * seen[card])
        seen[card] += 1
    return result


def _fabledan_card_code(card_id: int) -> str:
    base = int(card_id) % 54
    if base == 52:
        return "small_joker"
    if base == 53:
        return "big_joker"
    suits = ("H", "D", "S", "C")
    return f"{RANK_NAMES[base // 4]}{suits[base % 4]}"


__all__ = [
    "StagedCorrectionResult",
    "StagedReducerResult",
    "adopt_reducer",
    "confirm_lead_reducer",
    "copy_reducer",
    "create_legacy_reducer",
    "initialize_reducer",
    "is_legacy_reducer",
    "legacy_events",
    "legacy_identity",
    "legacy_snapshot",
    "replay_trusted_snapshot",
    "stage_latest_correction",
    "stage_reducer",
]
