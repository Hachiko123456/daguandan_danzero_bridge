"""Project recorded finish events onto a TruthLog without inventing anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..live.truth_log import TruthTurn


PLACEMENT_ORDER = ("head", "second", "third", "last")
PLACEMENT_LABELS = {
    "head": "头游",
    "second": "二游",
    "third": "三游",
    "last": "末游",
}
_ACTION_TYPES = frozenset({"player_played", "player_passed", "manual_confirmed_event"})


@dataclass(frozen=True)
class PlacementProjection:
    placement: str
    actor: str
    anchor_turn_id: int | None
    source_event_id: str = ""

    @property
    def label(self) -> str:
        return PLACEMENT_LABELS[self.placement]


def project_recorded_placements(
    events: Iterable[Mapping[str, object]],
    turns: Iterable[TruthTurn],
) -> tuple[PlacementProjection, ...]:
    """Return the recorded finish order and only safe row-level anchors.

    Older recordings contain a visual ``player_finished`` event but no direct
    action reference.  Its row is safe to infer only when that player's final
    non-pass action is immediately before the event's action boundary.  This
    avoids attaching a late visual badge to an unrelated old action.
    """

    event_list = tuple(events)
    turn_by_id = {turn.index: turn for turn in turns}
    action_by_event_id = {
        str(event.get("event_id", "")): event
        for event in event_list
        if str(event.get("event_type", "")) in _ACTION_TYPES
        and str(event.get("event_id", ""))
    }
    result: list[PlacementProjection] = []
    seen: set[str] = set()
    for event in event_list:
        if str(event.get("event_type", "")) != "player_finished":
            continue
        payload = event.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        placement = str(payload.get("placement", "")).strip().lower()
        actor = str(event.get("actor", ""))
        if placement not in PLACEMENT_LABELS or not actor or placement in seen:
            continue
        seen.add(placement)
        anchor = _explicit_anchor(payload, action_by_event_id, turn_by_id, actor)
        if anchor is None:
            anchor = _legacy_immediate_anchor(event, turn_by_id, actor)
        result.append(
            PlacementProjection(
                placement=placement,
                actor=actor,
                anchor_turn_id=anchor,
                source_event_id=str(event.get("event_id", "")),
            )
        )
    return tuple(sorted(result, key=lambda item: PLACEMENT_ORDER.index(item.placement)))


def format_placement_summary(
    projections: Iterable[PlacementProjection],
    seat_labels: Mapping[str, str],
) -> str:
    """Produce the one-line, always rank-ordered log header."""

    items = tuple(projections)
    if not items:
        return "出完顺序：暂无录像名次记录"
    return "出完顺序：" + "  →  ".join(
        f"{index} {seat_labels.get(item.actor, item.actor)}·{item.label}"
        for index, item in enumerate(items, start=1)
    )


def _explicit_anchor(
    payload: Mapping[str, object],
    action_by_event_id: Mapping[str, Mapping[str, object]],
    turn_by_id: Mapping[int, TruthTurn],
    actor: str,
) -> int | None:
    target_id = str(payload.get("trigger_action_event_id", "")).strip()
    target = action_by_event_id.get(target_id)
    if target is None or str(target.get("actor", "")) != actor:
        return None
    return _matching_play_turn(target, turn_by_id, actor)


def _legacy_immediate_anchor(
    event: Mapping[str, object],
    turn_by_id: Mapping[int, TruthTurn],
    actor: str,
) -> int | None:
    try:
        boundary = int(event.get("turn_id", 0))
    except (TypeError, ValueError):
        return None
    candidate = turn_by_id.get(boundary - 1)
    if candidate is None or candidate.actor != actor or candidate.is_pass:
        return None
    return candidate.index


def _matching_play_turn(
    event: Mapping[str, object],
    turn_by_id: Mapping[int, TruthTurn],
    actor: str,
) -> int | None:
    try:
        turn_id = int(event.get("turn_id", 0))
    except (TypeError, ValueError):
        return None
    candidate = turn_by_id.get(turn_id)
    if candidate is None or candidate.actor != actor or candidate.is_pass:
        return None
    return candidate.index
