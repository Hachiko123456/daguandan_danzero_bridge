"""Build an auditable draft :class:`TruthLog` from canonical scan actions.

This module is deliberately a pure data conversion boundary.  It accepts an
already-loaded baseline ``TruthLog`` and an in-memory canonical action trace;
it does not open session files, read scan reports, or mutate either input.
Actions that cannot be represented by the current TruthLog schema are kept in
``review_items`` rather than being silently discarded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..danzero.state import SEATS
from ..domain.truth import LabelProvenance, TruthEvidence, TruthOutcome
from ..live.truth_log import TruthInitialState, TruthLog, TruthTurn, card_text_to_code
from .placement_projection import derive_finish_order


# A single source string is used because the existing TruthLog provenance
# schema has one ``source`` field.  Keeping both tokens in the value makes the
# conversion auditable without changing the established schema.
DRAFT_PROVENANCE_SOURCE = "video_scan+canonical_reconciliation"

_RELIABLE_TRICK_STATUSES = frozenset(
    {
        "confirmed",
        "exact",
        "reliable",
        "verified",
        "human_review",
        "canonical_reconciliation",
        "inferred_unique",
    }
)


@dataclass(frozen=True)
class TruthLogFromScanResult:
    """Generated draft plus non-lossy review records.

    ``truth_log`` contains every canonical action that can be represented by
    the current TruthLog schema.  ``review_items`` contains one audit record
    for every action marked uncertain and every action that required omission
    or normalization.  The source action is copied into each review item so a
    caller can render a review queue without consulting another report.
    """

    truth_log: TruthLog
    review_items: tuple[dict[str, object], ...] = ()

    @property
    def review_actions(self) -> tuple[dict[str, object], ...]:
        """Compatibility/readability alias for consumers building a queue."""

        return self.review_items

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe, detached representation of the conversion."""

        return {
            "truth_log": self.truth_log.to_dict(),
            "review_items": deepcopy(list(self.review_items)),
        }


def build_truth_log_from_scan(
    baseline: TruthLog,
    canonical_actions: Iterable[object] | Mapping[str, object],
) -> TruthLogFromScanResult:
    """Convert canonical scan actions into a new draft TruthLog.

    Only the baseline's session/source metadata and initial state are copied:
    ``source_session_id``, ``initial_state``, ``source_video`` and
    ``frame_index_path``.  Existing turns, outcome, label status and
    provenance are intentionally not copied.  The generated log is always a
    draft and its provenance names both conversion stages.

    Input action order is stable.  Valid actions receive fresh consecutive
    TruthLog turn ids (1..N), independent of source ``action_id`` values.
    Explicit ``trick_id`` values are accepted only when the source marks them
    reliable.  The existing TruthLog constructor temporarily infers missing
    ids while normalizing the object; this module then rebuilds its detached
    output turns and restores ``None`` for every non-reliable id.  No value is
    guessed from an uncertain card reading.
    """

    if not isinstance(baseline, TruthLog):
        raise TypeError("baseline must be a TruthLog")

    turns: list[TruthTurn] = []
    reliable_turn_ids: set[int] = set()
    review_items: list[dict[str, object]] = []
    for source_position, source_action in enumerate(
        _action_rows(canonical_actions), start=1
    ):
        row = _as_mapping(source_action)
        if row is None:
            review_items.append(
                _review_item(
                    source_position=source_position,
                    source_action=source_action,
                    action_id=None,
                    output_turn_id=None,
                    actor=None,
                    is_pass=None,
                    cards=(),
                    statuses=(),
                    reasons=("action_not_an_object",),
                )
            )
            continue

        action_id = _first_value(row, "action_id", "source_action_id", "id")
        actor_raw = _first_value(row, "actor", "seat", "player")
        actor = str(actor_raw).strip() if actor_raw is not None else ""
        is_pass = _as_bool(
            _first_value(row, "is_pass", "recognized_pass", "pass"), default=False
        )
        statuses = _source_statuses(row)
        reasons: list[str] = []
        fatal_reasons: list[str] = []
        uncertainty = _uncertainty(row)

        raw_cards = _first_present(row, "cards", "recognized_cards", "physical_cards")
        card_values = _card_values(raw_cards)
        cards: tuple[str, ...] = ()

        if actor not in SEATS:
            fatal_reasons.append("invalid_actor")

        if is_pass:
            # TruthLog requires PASS cards to be empty.  Preserve the original
            # cards in the review item rather than silently dropping evidence.
            if card_values:
                reasons.append("pass_cards_present_normalized_to_empty")
        else:
            try:
                cards = tuple(card_text_to_code(str(card).strip()) for card in card_values)
            except (TypeError, ValueError) as exc:
                fatal_reasons.append(f"invalid_card:{exc}")
            if not cards and not fatal_reasons:
                fatal_reasons.append("missing_cards")
            if cards:
                duplicate_cards = [
                    card for card, count in Counter(cards).items() if count > 2
                ]
                if duplicate_cards:
                    fatal_reasons.append("card_multiplicity_exceeds_two")
                if any(card.endswith("?") for card in cards) and "unknown_suit" not in uncertainty:
                    uncertainty = (*uncertainty, "unknown_suit")

        if fatal_reasons:
            review_items.append(
                _review_item(
                    source_position=source_position,
                    source_action=source_action,
                    action_id=action_id,
                    output_turn_id=None,
                    actor=actor or None,
                    is_pass=is_pass,
                    cards=cards or tuple(str(card) for card in card_values),
                    statuses=statuses,
                    reasons=tuple(_unique((*fatal_reasons, *reasons))),
                    uncertainty=uncertainty,
                )
            )
            # Invalid actor/card/missing-card actions cannot be represented as
            # a valid TruthTurn.  They remain fully available in review_items.
            continue

        evidence, frame_index, monotonic_ms = _evidence(row)
        reliable_trick_id, trick_id_was_present = _reliable_trick_id(row)
        if trick_id_was_present and reliable_trick_id is None:
            reasons.append("unreliable_trick_id_ignored")

        move_semantics = _move_semantics(row)
        if move_semantics is not None and is_pass:
            reasons.append("pass_move_semantics_ignored")
            move_semantics = None

        output_turn_id = len(turns) + 1
        try:
            turn = TruthTurn(
                index=output_turn_id,
                actor=actor,  # type: ignore[arg-type]
                is_pass=is_pass,
                cards=() if is_pass else cards,
                frame_index=frame_index,
                monotonic_ms=monotonic_ms,
                trick_id=reliable_trick_id,
                evidence=evidence,
                label_status="draft",
                provenance=LabelProvenance(
                    source=DRAFT_PROVENANCE_SOURCE,
                    confidence=_confidence(row),
                ),
                uncertainty=uncertainty,
                move_semantics=move_semantics,
            )
        except ValueError as exc:
            # Keep the action as a draft even if optional semantic metadata is
            # outside the current TruthLog vocabulary; retain the source row
            # and make the loss explicit in the review queue.
            if move_semantics is None:
                raise
            reasons.append(f"invalid_move_semantics:{exc}")
            turn = TruthTurn(
                index=output_turn_id,
                actor=actor,  # type: ignore[arg-type]
                is_pass=is_pass,
                cards=() if is_pass else cards,
                frame_index=frame_index,
                monotonic_ms=monotonic_ms,
                trick_id=reliable_trick_id,
                evidence=evidence,
                label_status="draft",
                provenance=LabelProvenance(
                    source=DRAFT_PROVENANCE_SOURCE,
                    confidence=_confidence(row),
                ),
                uncertainty=uncertainty,
            )
        turns.append(turn)
        if reliable_trick_id is not None:
            reliable_turn_ids.add(output_turn_id)

        if statuses or uncertainty or reasons:
            review_items.append(
                _review_item(
                    source_position=source_position,
                    source_action=source_action,
                    action_id=action_id,
                    output_turn_id=output_turn_id,
                    actor=actor,
                    is_pass=is_pass,
                    cards=(tuple(str(card) for card in card_values) if is_pass else cards),
                    statuses=statuses,
                    reasons=tuple(_unique(reasons)),
                    uncertainty=uncertainty,
                )
            )

    # Construct a fresh initial state as well as a fresh TruthLog.  This keeps
    # all tuple/dict boundaries detached from the baseline object and makes it
    # explicit that no baseline turns or outcome are carried into the draft.
    initial_state = TruthInitialState(
        round_level=str(baseline.initial_state.round_level),
        lead_player=baseline.initial_state.lead_player,
        my_hand=tuple(str(card) for card in baseline.initial_state.my_hand),
        seat_hand_sizes=baseline.initial_state.seat_hand_sizes,
    )
    generated = TruthLog(
        source_session_id=str(baseline.source_session_id),
        initial_state=initial_state,
        turns=tuple(turns),
        source_video=str(baseline.source_video),
        frame_index_path=str(baseline.frame_index_path),
        label_status="draft",
        provenance=LabelProvenance(source=DRAFT_PROVENANCE_SOURCE),
        outcome=TruthOutcome(
            complete=False,
            finish_order=derive_finish_order(
                turns,
                initial_hand_size=len(initial_state.my_hand),
            ),
            team_result="unknown",
            reward=None,
            reward_scheme="",
        ),
    )
    # TruthLog.__post_init__ infers absent trick ids for legacy compatibility.
    # Rebuild our own turns after construction so the draft's audit contract is
    # stricter: only explicitly reliable trick ids remain populated.  This is
    # detached from the baseline and from the input action objects.
    detached_turns: list[TruthTurn] = []
    for turn in generated.turns:
        detached = TruthTurn(
            index=turn.index,
            actor=turn.actor,
            is_pass=turn.is_pass,
            cards=turn.cards,
            frame_index=turn.frame_index,
            monotonic_ms=turn.monotonic_ms,
            trick_id=turn.trick_id if turn.index in reliable_turn_ids else None,
            evidence=turn.evidence,
            label_status=turn.label_status,
            provenance=turn.provenance,
            uncertainty=turn.uncertainty,
            move_semantics=turn.move_semantics,
        )
        if turn.index not in reliable_turn_ids:
            # TruthTurn's constructor normalizes None to its legacy sentinel
            # 0; overwrite only this fresh detached instance to preserve the
            # optional value required by this conversion boundary.
            object.__setattr__(detached, "trick_id", None)
        detached_turns.append(detached)
    object.__setattr__(generated, "turns", tuple(detached_turns))
    return TruthLogFromScanResult(generated, tuple(review_items))


def truth_log_from_canonical_actions(
    baseline: TruthLog,
    canonical_actions: Iterable[object] | Mapping[str, object],
) -> TruthLogFromScanResult:
    """Alias with a name matching the source/target data types."""

    return build_truth_log_from_scan(baseline, canonical_actions)


def convert_canonical_actions_to_truth_log(
    baseline: TruthLog,
    canonical_actions: Iterable[object] | Mapping[str, object],
) -> TruthLogFromScanResult:
    """Descriptive alias for callers that prefer an imperative API name."""

    return build_truth_log_from_scan(baseline, canonical_actions)


def _action_rows(value: Iterable[object] | Mapping[str, object]) -> Iterable[object]:
    """Accept a list or a canonical trace envelope without reading files."""

    if isinstance(value, Mapping):
        candidate: object = _first_present(
            value, "actions", "canonical_actions", "action_trace", "rows"
        )
        if isinstance(candidate, Mapping):
            candidate = _first_present(candidate, "actions", "rows")
        if candidate is None:
            return ()
        return candidate if _is_action_iterable(candidate) else (candidate,)
    if _is_action_iterable(value):
        return value
    return (value,)


def _is_action_iterable(value: object) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping))


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, Mapping):
        return attrs
    return None


def _first_present(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in row:
            return row[name]
    return None


def _first_value(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "pass"}:
            return True
        if normalized in {"false", "0", "no", "n", "play"}:
            return False
    return bool(value)


def _card_values(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return tuple(value)
    return ()


def _uncertainty(row: Mapping[str, object]) -> tuple[str, ...]:
    value = _first_present(row, "uncertainty", "uncertainties")
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        return (str(value),)
    return tuple(_unique(str(item) for item in value if str(item)))
_REVIEW_STATUSES = frozenset(
    {"needs_review", "unresolved", "unknown", "ambiguous", "failed"}
)


def _source_statuses(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return only source statuses that make an action reviewable.

    ``review_status=unverified`` and ``repair_status=resolved`` are normal
    scan states, not review work.  Including them would turn every clean
    canonical action into a false-positive review item.
    """

    values: list[str] = []
    for name in ("status", "review_status", "repair_status"):
        value = row.get(name)
        normalized = str(value or "").strip().lower()
        if normalized in _REVIEW_STATUSES:
            values.append(f"{name}:{normalized}")
    return tuple(_unique(values))

def _evidence(
    row: Mapping[str, object],
) -> tuple[TruthEvidence, int | None, int | None]:
    nested = row.get("evidence")
    evidence_row = nested if isinstance(nested, Mapping) else {}

    frame_values: list[int] = []
    candidates = _first_present(row, "evidence_frames", "frame_indices")
    if candidates is None:
        candidates = _first_present(evidence_row, "frame_indices", "evidence_frames")
    for value in _card_values(candidates):
        parsed = _nonnegative_int(value)
        if parsed is not None and parsed not in frame_values:
            frame_values.append(parsed)

    frame_index = _first_present(row, "frame_index", "best_frame", "frame_start")
    if frame_index is None:
        frame_index = _first_present(evidence_row, "frame_index")
    frame = _nonnegative_int(frame_index)
    if frame is not None and frame not in frame_values:
        frame_values.insert(0, frame)

    monotonic_value = _first_present(
        row, "monotonic_ms", "timestamp_ms", "timestamp_start_ms"
    )
    if monotonic_value is None:
        monotonic_value = evidence_row.get("monotonic_ms")
    monotonic_ms = _nonnegative_int(monotonic_value)
    roi_name = str(_first_value(row, "roi_name") or evidence_row.get("roi_name") or "")
    evidence = TruthEvidence(
        frame_indices=tuple(frame_values),
        monotonic_ms=monotonic_ms,
        roi_name=roi_name,
    )
    return evidence, frame, monotonic_ms


def _reliable_trick_id(row: Mapping[str, object]) -> tuple[int | None, bool]:
    raw = _first_present(row, "trick_id")
    if raw is None or raw == "":
        return None, False
    parsed = _positive_int(raw)
    if parsed is None:
        return None, True

    explicitly_unreliable = row.get("trick_id_reliable") is False
    if explicitly_unreliable:
        return None, True
    if row.get("trick_id_reliable") is True:
        return parsed, True

    confidence = _number(
        _first_present(row, "trick_id_confidence", "trick_confidence")
    )
    if confidence is not None and confidence >= 0.9:
        return parsed, True

    status = str(
        _first_value(row, "trick_id_status", "trick_status", "trick_id_source") or ""
    ).strip().lower()
    if status in _RELIABLE_TRICK_STATUSES:
        return parsed, True
    return None, True


def _move_semantics(row: Mapping[str, object]) -> dict[str, object] | None:
    value = _first_present(row, "move_semantics", "semantics")
    if not isinstance(value, Mapping):
        return None
    return _snapshot(value)  # type: ignore[return-value]


def _confidence(row: Mapping[str, object]) -> float | None:
    value = _number(_first_present(row, "confidence", "best_confidence"))
    if value is None or not 0 <= value <= 1:
        return None
    return value


def _nonnegative_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _positive_int(value: object) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _review_item(
    *,
    source_position: int,
    source_action: object,
    action_id: object,
    output_turn_id: int | None,
    actor: str | None,
    is_pass: bool | None,
    cards: tuple[object, ...] | tuple[str, ...],
    statuses: tuple[str, ...],
    reasons: tuple[str, ...],
    uncertainty: tuple[str, ...] = (),
) -> dict[str, object]:
    all_reasons = tuple(_unique((*reasons, *uncertainty)))
    return {
        "source_position": source_position,
        "action_id": _snapshot(action_id),
        "output_turn_id": output_turn_id,
        "actor": actor,
        "is_pass": is_pass,
        "cards": [_snapshot(card) for card in cards],
        "statuses": list(statuses),
        "uncertainty": list(uncertainty),
        "reasons": list(all_reasons),
        "provenance": {"source": DRAFT_PROVENANCE_SOURCE},
        "source_action": _snapshot(source_action),
    }


def _snapshot(value: object) -> Any:
    """Make a detached JSON-like snapshot without mutating/serializing input."""

    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "DRAFT_PROVENANCE_SOURCE",
    "TruthLogFromScanResult",
    "build_truth_log_from_scan",
    "truth_log_from_canonical_actions",
    "convert_canonical_actions_to_truth_log",
]
