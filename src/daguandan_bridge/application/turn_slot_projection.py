"""Turn-slot ledger derived from video-only observations.

The scan's raw card/PASS surfaces are intentionally independent per seat and
can overlap for many frames.  This module creates an explicit turn ledger from
current-player transitions before any consumer interprets missing visual
surfaces as a skipped player.  A slot is never silently removed: it is either
matched to one canonical action, recovered from its own ROI evidence, or
reported as ``needs_review``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

SEATS = ("self", "right", "opposite", "left")
_SIGNAL_GAP = 4
_RECOVERY_AFTER_NEXT_SIGNAL = 8


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _cards(value: object) -> tuple[str, ...]:
    return tuple(str(card) for card in value) if isinstance(value, (list, tuple)) else ()


def _confidence(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class _SignalRun:
    actor: str
    start: int
    end: int


def _signal_runs(observations: list[Mapping[str, object]]) -> list[_SignalRun]:
    runs: list[_SignalRun] = []
    for row in observations:
        frame = _as_int(row.get("frame_index"))
        if frame is None or row.get("decode_ok", True) is False:
            continue
        opening = row.get("opening")
        current = opening.get("current_player_signal") if isinstance(opening, Mapping) else row.get("current_player")
        actor = str(current) if current in SEATS else ""
        if not actor:
            continue
        if runs and runs[-1].actor == actor:
            previous = runs[-1]
            runs[-1] = _SignalRun(actor, previous.start, frame)
        else:
            runs.append(_SignalRun(actor, frame, frame))
    return runs


def _action_bounds(action: Mapping[str, object]) -> tuple[int | None, int | None]:
    frames = [_as_int(action.get("frame_start")), _as_int(action.get("frame_end"))]
    raw = action.get("evidence_frames", ())
    if isinstance(raw, (list, tuple)):
        frames.extend(_as_int(value) for value in raw)
    usable = [frame for frame in frames if frame is not None]
    return (min(usable), max(usable)) if usable else (None, None)


def _best_recovery(
    actor: str,
    start: int,
    end: int,
    observations: list[Mapping[str, object]],
) -> dict[str, object] | None:
    readings: list[tuple[bool, tuple[str, ...], int, float, str]] = []
    for row in observations:
        frame = _as_int(row.get("frame_index"))
        if frame is None or frame < start or frame > end:
            continue
        regions = row.get("regions")
        region = regions.get(actor) if isinstance(regions, Mapping) else None
        if not isinstance(region, Mapping):
            continue
        cards = _cards(region.get("cards"))
        is_pass = bool(region.get("is_pass", False))
        if not cards and not is_pass:
            continue
        readings.append((is_pass, cards, frame, _confidence(region.get("confidence")), str(region.get("source", ""))))
    if not readings:
        return None
    grouped: Counter[tuple[bool, tuple[str, ...]]] = Counter((passed, tuple(sorted(cards))) for passed, cards, *_ in readings)

    def _recovery_key(item: tuple[tuple[bool, tuple[str, ...]], int]) -> tuple[int, int, float]:
        (passed, cards), count = item
        # 遮挡帧（我方按钮盖住左家牌尾）会产生带未知花色的读数组；同一槽位的
        # 恢复必须优先采用按钮消失后的完整读数，而不是出现次数更多的遮挡读数。
        no_unknown = 0 if any(str(card).endswith("?") for card in cards) else 1
        confidence_sum = sum(
            confidence
            for passed_r, cards_r, _frame, confidence, _source in readings
            if (passed_r, tuple(sorted(cards_r))) == (passed, cards)
        )
        return (no_unknown, count, confidence_sum)

    key = max(grouped, key=lambda item: _recovery_key((item, grouped[item])))
    matched = [item for item in readings if (item[0], tuple(sorted(item[1]))) == key]
    # Preserve the most supported visual card order while keeping PASS empty.
    passed = key[0]
    cards = () if passed else max(matched, key=lambda item: (item[3], item[2]))[1]
    return {
        "actor": actor,
        "is_pass": passed,
        "cards": list(cards),
        "evidence_frames": [item[2] for item in matched],
        "confidence": max(item[3] for item in matched),
        "source": "turn_slot_recovery",
    }


def _remaining_before(frame: int, actions: list[dict[str, object]], self_hand_size: int) -> dict[str, int]:
    remaining = {seat: 27 for seat in SEATS}
    remaining["self"] = self_hand_size or 27
    for action in actions:
        start, _end = _action_bounds(action)
        if start is None or start >= frame:
            continue
        actor = str(action.get("actor", ""))
        if actor in remaining and not bool(action.get("is_pass", False)):
            remaining[actor] -= len(_cards(action.get("cards")))
    return remaining


def _skipped_between(actor: str, following: str, remaining: Mapping[str, int]) -> tuple[str, ...]:
    if actor not in SEATS or following not in SEATS:
        return ()
    start = SEATS.index(actor)
    result = []
    for offset in range(1, len(SEATS)):
        seat = SEATS[(start + offset) % len(SEATS)]
        if seat == following:
            break
        if remaining.get(seat, 1) > 0:
            result.append(seat)
    return tuple(result)


def project_turn_slots(
    observations: Iterable[Mapping[str, object]],
    actions: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Create a complete ledger of observed current-player turn slots.

    This function intentionally does not use TruthLog/timeline data.  Slots
    use the timer/current-player signal as their identity; action surfaces only
    fill a slot and cannot replace its actor.  The leading on-screen state is
    reported as ``initial_context`` instead of fabricated as formal turns.
    """
    rows = [dict(row) for row in observations if isinstance(row, Mapping)]
    rows.sort(key=lambda row: _as_int(row.get("frame_index")) or -1)
    self_hand_size = max(
        (len(row.get("opening", {}).get("my_hand") or ()) for row in rows if isinstance(row.get("opening"), Mapping)),
        default=27,
    )
    runs = _signal_runs(rows)
    candidates = [dict(action) for action in actions if isinstance(action, Mapping)]
    claimed: set[int] = set()
    slots: list[dict[str, object]] = []
    first_signal = runs[0].start if runs else None
    initial_visible: list[dict[str, object]] = []
    if first_signal is not None:
        for row in rows:
            frame = _as_int(row.get("frame_index"))
            if frame is None or frame > first_signal:
                break
            regions = row.get("regions")
            if not isinstance(regions, Mapping):
                continue
            for actor in SEATS:
                region = regions.get(actor)
                if not isinstance(region, Mapping):
                    continue
                cards = _cards(region.get("cards"))
                if cards:
                    initial_visible.append({"actor": actor, "cards": list(cards), "frame": frame})
        if initial_visible:
            slots.append({
                "slot_id": 0,
                "kind": "initial_context",
                "status": "needs_review",
                "start_frame": 0,
                "end_frame": first_signal,
                "visible_cards": initial_visible,
                "reason": "recording_started_with_existing_table_surface",
            })
    for index, (run, following) in enumerate(zip(runs, runs[1:]), start=1):
        # A player's action surface can remain visible until that same player
        # receives another turn. Keep the slot recoverable for that complete
        # horizon instead of only a few frames after the timer changes.
        next_own_turn = next(
            (later.start for later in runs[index + 1:] if later.actor == run.actor),
            None,
        )
        recovery_end = (next_own_turn - 1) if next_own_turn is not None else following.start + _RECOVERY_AFTER_NEXT_SIGNAL
        candidate_matches: list[tuple[tuple[int, int], int]] = []
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in claimed or str(candidate.get("actor", "")) != run.actor:
                continue
            start, end = _action_bounds(candidate)
            if start is None or end is None:
                continue
            # A canonical action normally appears immediately before/after the
            # timer advances. Do not let a later same-seat action fill this
            # slot merely because that seat has not yet received another turn.
            if start <= following.start + _RECOVERY_AFTER_NEXT_SIGNAL and end >= run.start - 4:
                candidate_matches.append(((abs(start - following.start), start), candidate_index))
        match_index = min(candidate_matches)[1] if candidate_matches else None
        slot = {
            "slot_id": index,
            "kind": "turn",
            "actor": run.actor,
            "start_frame": run.start,
            "close_frame": following.start,
            "recovery_end_frame": recovery_end,
            "next_current_player": following.actor,
        }
        if match_index is not None:
            claimed.add(match_index)
            action = candidates[match_index]
            slot.update({
                "status": "resolved",
                "action_id": action.get("action_id"),
                "action": {
                    "actor": action.get("actor"),
                    "is_pass": bool(action.get("is_pass", False)),
                    "cards": list(_cards(action.get("cards"))),
                },
            })
        else:
            recovered = _best_recovery(run.actor, run.start, recovery_end, rows)
            if recovered is not None:
                slot.update({"status": "recovered", "action": recovered})
            else:
                slot.update({
                    "status": "needs_review",
                    "reason": "no_action_surface_before_actor_next_turn",
                })
        slots.append(slot)
        remaining = _remaining_before(following.start, candidates, self_hand_size)
        for skipped_actor in _skipped_between(run.actor, following.actor, remaining):
            # Timer OCR can briefly skip an actor even though that actor later
            # appears normally. Only surface a missing slot when the actor
            # remains unaccounted for through the rest of the recording.
            future_action_exists = any(
                str(action.get("actor", "")) == skipped_actor
                and (_action_bounds(action)[0] or -1) >= following.start
                for action in candidates
            )
            if future_action_exists:
                continue
            slots.append({
                "slot_id": f"{index}:missing:{skipped_actor}",
                "kind": "missing_turn",
                "actor": skipped_actor,
                "start_frame": run.end,
                "close_frame": following.start,
                "status": "needs_review",
                "reason": "current_player_skipped_active_actor",
                "remaining_cards_before": remaining.get(skipped_actor),
            })
    unresolved = [slot for slot in slots if slot.get("status") == "needs_review" and slot.get("kind") in {"turn", "missing_turn"}]
    return {
        "schema": "guandan.turn-slot-ledger/1",
        "initial_context_present": bool(initial_visible),
        "signal_runs": [
            {"actor": run.actor, "start_frame": run.start, "end_frame": run.end}
            for run in runs
        ],
        "slots": slots,
        "counts": {
            "signal_runs": len(runs),
            "turn_slots": sum(slot.get("kind") == "turn" for slot in slots),
            "resolved": sum(slot.get("status") == "resolved" for slot in slots),
            "recovered": sum(slot.get("status") == "recovered" for slot in slots),
            "needs_review": len(unresolved),
        },
    }


__all__ = ["project_turn_slots"]
