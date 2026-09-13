"""Reconcile scan-projected action windows into a canonical action trace.

The projector intentionally records what each frame looked like.  This module
is the small, pure post-processing step that turns those observations into a
stable action ledger without changing the raw frame evidence.  It is kept
independent from the video scanner so callers can use it with persisted scan
artifacts, repaired actions, or synthetic observations.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

SEATS = ("self", "right", "opposite", "left")
DEFAULT_MAX_GAP_FRAMES = 5
PASS_MAX_GAP_FRAMES = 3

# A recognizer may briefly miss the current-player indicator while the UI
# animates. Treat only small holes as one signal run; a longer absence is not
# evidence that a player still owns the turn.
_SIGNAL_RUN_GAP_FRAMES = 4
_TURN_SUPPORT_BEFORE_FRAMES = 3
_TURN_SUPPORT_AFTER_FRAMES = 4
_PASS_RECOVERY_AFTER_FRAMES = 8


def _cards(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()


def _rank(card: str) -> str:
    if card in {"small_joker", "big_joker"}:
        return card
    return card[:-1] if card.endswith(("S", "H", "C", "D", "?")) else card


def _ranks(cards: Sequence[str]) -> Counter[str]:
    return Counter(_rank(str(card)) for card in cards)


def _unknown(cards: Sequence[str]) -> bool:
    return any(str(card).endswith("?") for card in cards)


def _complete(cards: Sequence[str]) -> bool:
    return bool(cards) and not _unknown(cards)


def _frame(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _confidence(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _is_rank_subset(left: Sequence[str], right: Sequence[str]) -> bool:
    a, b = _ranks(left), _ranks(right)
    return all(count <= b[rank] for rank, count in a.items())


def _joker_only(cards: Sequence[str]) -> bool:
    return bool(cards) and all(str(card) in {"small_joker", "big_joker"} for card in cards)


def _compatible(left: Sequence[str], right: Sequence[str]) -> tuple[bool, str | None]:
    """Return whether two nearby card displays can be one play.

    Complete displays with different ordinary rank multisets are deliberately
    not compatible.  A partial rank read, unknown suit, or joker label flicker
    is the evidence needed to reconcile a pair.
    """
    left, right = tuple(left), tuple(right)
    if Counter(left) == Counter(right):
        return True, "duplicate_display"
    if _joker_only(left) and _joker_only(right) and (
        _is_rank_subset(left, right) or _is_rank_subset(right, left) or len(left) == len(right) == 1
    ):
        return True, "joker_flicker"
    if not (_is_rank_subset(left, right) or _is_rank_subset(right, left)):
        return False, None
    if _unknown(left) or _unknown(right):
        return True, "suit_or_partial_variant"
    # Equal-sized complete hands with the same ranks are suit/template
    # variants.  The caller already requires a same-seat, nearby window, so
    # this cannot collapse ordinary consecutive plays from other seats.
    if len(left) == len(right) and _ranks(left) == _ranks(right):
        return True, "suit_variant"
    if len(left) != len(right):
        return True, "progressive_variant"
    return False, None


def _as_actions(value: Iterable[Mapping[str, object]] | Mapping[str, object]) -> list[dict[str, object]]:
    if isinstance(value, Mapping):
        value = value.get("actions", ())
    return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _action_frame_bounds(action: Mapping[str, object]) -> tuple[int | None, int | None]:
    frames = [_frame(action.get("frame_start")), _frame(action.get("frame_end"))]
    frames.extend(_frame(item) for item in action.get("evidence_frames", ()) if _frame(item) is not None)
    for variant in action.get("observed_variants", ()):
        if isinstance(variant, Mapping):
            frames.append(_frame(variant.get("frame_index")))
    frames = [item for item in frames if item is not None]
    return (min(frames), max(frames)) if frames else (None, None)


@dataclass
class _FrameRead:
    frame: int
    timestamp_ms: int | None
    actor: str
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    source: str
    current_player: str | None = None
    buttons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PlayerSignalRun:
    actor: str
    frame_start: int
    frame_end: int


@dataclass
class _Action:
    original: dict[str, object]
    position: int
    original_id: object
    actor: str
    is_pass: bool
    cards: tuple[str, ...]
    start: int | None
    end: int | None
    evidence_frames: list[int] = field(default_factory=list)
    variants: list[dict[str, object]] = field(default_factory=list)
    reads: list[_FrameRead] = field(default_factory=list)
    merged_ids: list[object] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return self.start if self.start is not None else 2**63 - 1

    @property
    def last_frame(self) -> int:
        return self.end if self.end is not None else self.first_frame


def _read_frames(observations: Iterable[Mapping[str, object]] | Mapping[str, object]) -> list[_FrameRead]:
    if isinstance(observations, Mapping):
        observations = observations.get("frame_observations", observations.get("observations", ()))
    result: list[_FrameRead] = []
    for raw in observations:
        if not isinstance(raw, Mapping) or raw.get("decode_ok", True) is False:
            continue
        frame = _frame(raw.get("frame_index"))
        if frame is None:
            continue
        timestamp = _frame(raw.get("timestamp_ms"))
        current = raw.get("current_player")
        opening = raw.get("opening")
        if current in (None, "") and isinstance(opening, Mapping):
            current = opening.get("current_player_signal")
        current_player = str(current) if current not in (None, "") else None
        buttons_value = raw.get("buttons", ())
        buttons = tuple(str(item) for item in buttons_value) if isinstance(buttons_value, (list, tuple, set)) else ()
        regions = raw.get("regions")
        if not isinstance(regions, Mapping):
            continue
        for actor, region in regions.items():
            if not isinstance(region, Mapping):
                continue
            cards = _cards(region.get("cards"))
            is_pass = bool(region.get("is_pass", False))
            if not cards and not is_pass:
                continue
            result.append(_FrameRead(
                frame=frame,
                timestamp_ms=timestamp,
                actor=str(actor),
                cards=cards,
                is_pass=is_pass,
                confidence=_confidence(region.get("confidence")),
                source=str(region.get("source", "")),
                current_player=current_player,
                buttons=buttons,
            ))
    return result


def _make_actions(raw_actions: list[dict[str, object]], reads: list[_FrameRead]) -> list[_Action]:
    result: list[_Action] = []
    for position, raw in enumerate(raw_actions):
        actor = str(raw.get("actor") or raw.get("seat") or raw.get("player") or "")
        cards = _cards(raw.get("cards"))
        start, end = _action_frame_bounds(raw)
        frames = []
        for item in raw.get("evidence_frames", ()):
            value = _frame(item)
            if value is not None:
                frames.append(value)
        variants = [deepcopy(dict(item)) for item in raw.get("observed_variants", ()) if isinstance(item, Mapping)]
        action = _Action(
            original=raw,
            position=position,
            original_id=raw.get("action_id", position + 1),
            actor=actor,
            is_pass=bool(raw.get("is_pass", False)),
            cards=cards,
            start=start,
            end=end,
            evidence_frames=list(dict.fromkeys(frames)),
            variants=variants,
        )
        lo, hi = start, end
        if lo is not None and hi is not None:
            action.reads = [item for item in reads if item.actor == actor and lo <= item.frame <= hi]
        result.append(action)
    result.sort(key=lambda item: (item.first_frame, item.position))
    return result



def _split_pass_actions_at_turn_boundaries(
    items: list[_Action],
    signal_runs: Sequence[_PlayerSignalRun],
) -> list[_Action]:
    """Split a persistent PASS template into one candidate per owned turn.

    A passed badge can remain painted well after the action.  When the current
    player signal returns to the same seat later, that is a distinct turn and
    must never be absorbed into the earlier PASS display.  Without signals we
    keep the raw action unchanged for conservative compatibility.

    Runs are matched by overlap with the action window rather than by run
    start alone: the badge often renders a few frames after the seat's signal
    run began, so a purely start-anchored match would miss the owning run and
    leave two distinct turns fused into one action.

    A single long run can overlap *two* neighbouring badge surfaces (one
    ending just before the run, one starting just after).  Cloning both would
    fabricate a second PASS for the same turn, so the candidates are collapsed
    to at most one per ``(seat, run)`` — keeping the surface whose evidence
    best sits inside the run — and the losing surfaces are recorded as an
    auditable ``duplicate_pass_run_clone`` drop.
    """
    result: list[_Action] = []
    per_turn: dict[tuple[str, int], list[object]] = {}
    order: list[tuple[str, int]] = []

    def resolve(key: tuple[str, int], score: tuple[int, int], clone: _Action) -> None:
        entry = per_turn.get(key)
        if entry is None:
            per_turn[key] = [score, clone, []]
            order.append(key)
            return
        if score > entry[0]:  # type: ignore[operator]
            entry[2].append(entry[1].original_id)  # type: ignore[union-attr]
            entry[0] = score
            entry[1] = clone
        else:
            entry[2].append(clone.original_id)  # type: ignore[union-attr]

    for item in items:
        if not item.is_pass or item.start is None or item.end is None:
            result.append(item)
            continue
        runs = [
            run for run in signal_runs
            if run.actor == item.actor
            and run.frame_end >= item.start - 2
            and run.frame_start <= item.end + 2
        ]
        if len(runs) <= 1:
            result.append(item)
            continue
        for ordinal, run in enumerate(runs, start=1):
            reads = [
                read for read in item.reads
                if run.frame_start - 2 <= read.frame <= run.frame_end + 2
                and read.is_pass
            ]
            if not reads:
                continue
            clone = deepcopy(item)
            clone.original_id = f"{item.original_id}@turn-{run.frame_start}"
            clone.position = item.position * 1000 + ordinal
            clone.start = min(read.frame for read in reads)
            clone.end = max(read.frame for read in reads)
            clone.evidence_frames = [read.frame for read in reads]
            clone.reads = reads
            clone.variants = [
                {"frame_index": read.frame, "cards": [], "confidence": read.confidence, "source": read.source}
                for read in reads
            ]
            clone.events = [*item.events, {"type": "split", "reason": "current_player_turn_boundary", "source_action_id": item.original_id, "run_frame_start": run.frame_start}]
            inside = sum(1 for read in reads if run.frame_start <= read.frame <= run.frame_end)
            resolve(
                (item.actor, run.frame_start),
                (inside, -abs(clone.start - run.frame_start)),
                clone,
            )

    for key in order:
        score, clone, superseded = per_turn[key]
        if superseded:
            clone.events = [*clone.events, {
                "type": "drop",
                "reason": "duplicate_pass_run_clone",
                "action_ids": list(superseded),
                "run_frame_start": key[1],
            }]
        result.append(clone)
    return sorted(result, key=lambda action: (action.first_frame, action.position))


def _has_full_response_cycle(
    left: _Action,
    right: _Action,
    all_items: Sequence[_Action],
    signal_runs: Sequence[_PlayerSignalRun],
) -> bool:
    """Return true when a seat left and later re-entered a completed turn cycle.

    A stale ROI can reappear while other seats act.  Two intervening actors
    alone are insufficient: both candidate displays must also be followed by
    the same next-current-player signal, which is the UI signature of the
    same actor having completed two distinct turns.
    """
    if left.actor != right.actor or left.position >= right.position:
        return False
    intervening = {
        item.actor
        for item in all_items
        if left.position < item.position < right.position and item.actor != left.actor
    }
    if len(intervening) < 2:
        return False
    def next_signal(item: _Action) -> str | None:
        if item.start is None:
            return None
        return next((run.actor for run in signal_runs if run.frame_start >= item.start), None)
    return next_signal(left) is not None and next_signal(left) == next_signal(right)


def _actor_received_new_turn_between(
    left: _Action,
    right: _Action,
    signal_runs: Sequence[_PlayerSignalRun],
) -> bool:
    """Whether two same-seat surfaces belong to distinct actor turns."""
    if left.actor != right.actor or left.start is None or right.start is None:
        return True
    return any(
        run.actor == left.actor and left.start < run.frame_start <= right.start
        for run in signal_runs
    )


_GHOST_RUN_LAG_FRAMES = 4
# Animation ghosting flickers into a neighbouring ROI for a frame or two.  A
# surface that stays on screen longer than this is a real display.
_GHOST_MAX_EVIDENCE_FRAMES = 3


def _out_of_turn_card_ghost(
    item: _Action,
    signal_runs: Sequence[_PlayerSignalRun],
) -> bool:
    """Whether a card surface never overlaps its own actor's turn signal.

    Play animations can flicker into a neighbouring ROI for a frame or two
    (for example a straight flying across the centre registers inside the
    opposite seat's region).  A real play is always anchored to the actor's
    own current-player run — at worst the first reads lag the run end by the
    button/animation transition, which a small slack covers.  A surface whose
    every evidence frame falls outside all of the actor's runs while a foreign
    seat owns the timer is animation ghosting, not a play.

    Three conservative guards keep this from deleting real plays:

    * the actor must own at least one run somewhere — recordings can begin
      mid-turn or the signal can already have advanced to the next seat, and
      without any own run the "never overlaps its own run" test is vacuous;
    * a surface that was already on screen at scan start is the visible
      leader, not a ghost;
    * ghosting is brief, so a persistent surface is never treated as a ghost.
    """
    if item.is_pass or item.start is None or not signal_runs:
        return False
    if not any(run.actor == item.actor for run in signal_runs):
        return False
    if "display_present_at_scan_start" in item.original.get("uncertainty", ()):
        return False
    frames = list(item.evidence_frames) or [item.start]
    if item.end is not None:
        frames.append(item.end)
    if len(set(frames)) > _GHOST_MAX_EVIDENCE_FRAMES:
        return False
    for frame in frames:
        if any(
            run.actor == item.actor
            and run.frame_start - 2 <= frame <= run.frame_end + _GHOST_RUN_LAG_FRAMES
            for run in signal_runs
        ):
            return False
    return any(
        run.actor != item.actor
        and run.frame_start <= item.start <= run.frame_end + 2
        for run in signal_runs
    )

def _near(left: _Action, right: _Action, max_gap_frames: int) -> bool:
    if left.start is None or right.start is None:
        return False
    if left.start <= right.start:
        return right.start <= left.last_frame + max_gap_frames + 1
    return left.start <= right.last_frame + max_gap_frames + 1


def _variant_records(action: _Action) -> list[dict[str, object]]:
    records = list(action.variants)
    for read in action.reads:
        records.append({
            "frame_index": read.frame,
            "cards": list(read.cards),
            "confidence": read.confidence,
            "source": read.source,
        })
    if action.cards:
        records.append({"frame_index": action.start, "cards": list(action.cards), "confidence": _confidence(action.original.get("best_confidence", action.original.get("confidence"))), "source": "projected"})
    return records


def _preferred_observed_order(
    cluster: Sequence[_Action],
    cards: Sequence[str],
) -> tuple[str, ...] | None:
    """Choose the most consistently observed visual order for equal card sets.

    Card compatibility deliberately ignores ordering, but the trace keeps the
    order emitted by the recognizer. Prefer repeated real frame/variant order
    over a single projected summary ordering; this is deterministic and does
    not need a TruthLog.
    """
    wanted = Counter(cards)
    counts: Counter[tuple[str, ...]] = Counter()
    confidence: dict[tuple[str, ...], float] = {}
    latest_frame: dict[tuple[str, ...], int] = {}
    for action in cluster:
        visual_records = list(action.variants)
        visual_records.extend({
            "cards": list(read.cards),
            "confidence": read.confidence,
            "frame_index": read.frame,
        } for read in action.reads)
        for record in visual_records:
            observed = _cards(record.get("cards"))
            if not observed or not _complete(observed) or Counter(observed) != wanted:
                continue
            counts[observed] += 1
            confidence[observed] = confidence.get(observed, 0.0) + _confidence(record.get("confidence"))
            frame = _frame(record.get("frame_index"))
            latest_frame[observed] = max(latest_frame.get(observed, -1), frame if frame is not None else -1)
    if not counts:
        return None
    return max(
        counts,
        key=lambda observed: (counts[observed], confidence[observed], latest_frame[observed], observed),
    )


def _choose_cards(cluster: list[_Action]) -> tuple[tuple[str, ...], int | None, float, bool]:
    candidates: list[tuple[tuple[str, ...], int | None, float, int, int, bool]] = []
    reference = next((item.cards for item in cluster if item.cards), ())
    records: list[tuple[tuple[str, ...], int | None, float, int]] = []
    for action in cluster:
        records.append((action.cards, action.start, _confidence(action.original.get("best_confidence", action.original.get("confidence"))), action.position))
        for variant in _variant_records(action):
            cards = _cards(variant.get("cards"))
            if cards:
                records.append((cards, _frame(variant.get("frame_index")), _confidence(variant.get("confidence")), action.position))
    for cards, frame, confidence, position in records:
        if not cards:
            continue
        compatible = all(_compatible(cards, other.cards)[0] or Counter(cards) == Counter(other.cards) for other in cluster if other.cards)
        if not compatible:
            continue
        support = sum(1 for other_cards, _, _, _ in records if Counter(other_cards) == Counter(cards))
        current_bonus = sum(1 for action in cluster for read in action.reads if Counter(read.cards) == Counter(cards) and read.current_player == action.actor)
        candidates.append((cards, frame, confidence, support + current_bonus, position, _complete(cards)))
    if not candidates:
        return reference, None, 0.0, _unknown(reference)
    # The projector's action-level card order is the canonical visual order
    # for an already-complete window.  Do not replace it with a later OCR
    # variant merely because that variant scored a little higher.  Progressive
    # and unknown-suit windows still fall through to candidate selection.
    first = cluster[0]
    max_complete_len = max((len(item[0]) for item in candidates if item[5]), default=0)
    if _complete(first.cards) and len(first.cards) >= max_complete_len:
        if not (_joker_only(first.cards) and any(item[0] != first.cards and _joker_only(item[0]) for item in candidates)):
            confidence = _confidence(first.original.get("best_confidence", first.original.get("confidence")))
            return _preferred_observed_order(cluster, first.cards) or first.cards, first.start, confidence, False
    # Prefer complete/full displays.  Suit completeness must outrank card
    # count: an occluded frame (own buttons covering the tail of a long play)
    # can hallucinate a ghost card, and a count-first ranking would keep the
    # phantom over the later clean reading.  For joker alternatives, support
    # and confidence decide between small/big instead of blindly preferring
    # either.
    candidates.sort(key=lambda item: (
        int(item[5]),
        -sum(str(card).endswith("?") for card in item[0]),
        len(item[0]),
        item[3],
        item[2],
        item[1] if item[1] is not None else -1,
    ))
    selected = candidates[-1]
    ordered = _preferred_observed_order(cluster, selected[0]) if selected[5] else None
    return ordered or selected[0], selected[1], selected[2], not selected[5]


def _merge(left: _Action, right: _Action, reason: str) -> None:
    left.end = max(item for item in (left.end, right.end) if item is not None) if left.end is not None or right.end is not None else None
    if left.start is None:
        left.start = right.start
    left.evidence_frames = list(dict.fromkeys(left.evidence_frames + right.evidence_frames))
    left.variants.extend(deepcopy(right.variants))
    left.reads.extend(right.reads)
    left.merged_ids.extend([right.original_id, *right.merged_ids])
    left.events.append({"type": "merge", "reason": reason, "action_ids": [right.original_id, *right.merged_ids]})


def _mergeable(left: _Action, right: _Action, max_gap_frames: int) -> tuple[bool, str | None]:
    if left.actor != right.actor or left.is_pass != right.is_pass:
        return False, None
    gap = PASS_MAX_GAP_FRAMES if left.is_pass else max_gap_frames
    if not _near(left, right, gap):
        return False, None
    if left.is_pass:
        return True, "brief_missing_frame_duplicate"
    return _compatible(left.cards, right.cards)


def _reconcile_cluster(cluster: list[_Action]) -> dict[str, object]:
    base = deepcopy(cluster[0].original)
    cards, best_frame, best_confidence, unresolved = _choose_cards(cluster)
    if cluster[0].is_pass:
        cards, unresolved = (), False
    all_ids = [item.original_id for item in cluster]
    all_frames: list[int] = []
    for item in cluster:
        all_frames.extend(item.evidence_frames)
        all_frames.extend(read.frame for read in item.reads)
        all_frames.extend(_frame(v.get("frame_index")) for v in item.variants if _frame(v.get("frame_index")) is not None)
    all_frames = [item for item in all_frames if item is not None]
    if all_frames:
        base["frame_start"] = min(all_frames)
        base["frame_end"] = max(all_frames)
        base["evidence_frames"] = list(dict.fromkeys(sorted(all_frames)))
    elif cluster[0].start is not None:
        base["frame_start"] = cluster[0].start
        if cluster[-1].end is not None:
            base["frame_end"] = cluster[-1].end
    variants: list[dict[str, object]] = []
    for item in cluster:
        for variant in _variant_records(item):
            if variant not in variants:
                variants.append(variant)
    if variants:
        base["observed_variants"] = variants
    if cluster[0].is_pass:
        base["cards"] = []
    elif cards:
        # Preserve the selected visual variant order; card compatibility uses Counter.
        base["cards"] = list(cards)
    unknown_seen = _unknown(cards) or any(_unknown(_cards(v.get("cards"))) for v in variants)
    before = _cards(base.get("cards_before_repair"))
    if not before:
        before = next((_cards(v.get("cards")) for v in variants if _unknown(_cards(v.get("cards")))), ())
    if before:
        base["cards_before_repair"] = sorted(before)
    if best_frame is not None:
        base["best_frame"] = best_frame
    if best_confidence and "best_confidence" in base:
        base["best_confidence"] = best_confidence
    if unknown_seen:
        if _complete(cards):
            # A prior suit reread is authoritative provenance.  Reconciliation
            # may add its own audit metadata, but must not rewrite the repair
            # reason/frame supplied by the scanner.
            base.setdefault("repair_status", "resolved")
            if "repair_frame" not in base:
                base["repair_frame"] = best_frame
            if "repair_reason" not in base:
                base["repair_reason"] = "action_trace_reconciliation"
        else:
            base["repair_status"] = "unresolved"
            base["repair_frame"] = None
            base.setdefault("repair_reason", "reconciliation_left_unknown_suit")
    uncertainty = list(dict.fromkeys(str(item) for item in base.get("uncertainty", ()) if item))
    if _complete(cards):
        uncertainty = [item for item in uncertainty if item != "unknown_suit"]
    elif unknown_seen and "unknown_suit" not in uncertainty:
        uncertainty.append("unknown_suit")
    base["uncertainty"] = uncertainty
    base["review_status"] = "needs_review" if uncertainty else base.get("review_status", "unverified")
    merged_ids = [item for action in cluster for item in action.merged_ids]
    reconciliation = deepcopy(base.get("reconciliation", {})) if isinstance(base.get("reconciliation"), Mapping) else {}
    reconciliation.update({
        "source_action_ids": all_ids,
        "merged_action_ids": merged_ids,
        "merged": bool(merged_ids),
        "events": [event for action in cluster for event in action.events],
    })
    base["reconciliation"] = reconciliation
    return base


def _compressed_player_signal_runs(rows: Iterable[Mapping[str, object]]) -> list[_PlayerSignalRun]:
    """Return bounded runs of valid current-player readings.

    ``None``/unread frames are ignored, but only across a small gap. This
    makes the evidence resilient to a transient OCR miss without allowing an
    old turn signal to justify a much later candidate.
    """
    readings: list[tuple[int, str]] = []
    for row in rows:
        frame = _frame(row.get("frame_index")) if isinstance(row, Mapping) else None
        if frame is None:
            continue
        opening = row.get("opening") if isinstance(row, Mapping) else None
        signal = (
            opening.get("current_player_signal")
            if isinstance(opening, Mapping)
            else row.get("current_player")
        )
        if signal in SEATS:
            readings.append((frame, str(signal)))
    readings.sort()

    runs: list[_PlayerSignalRun] = []
    for frame, actor in readings:
        if (
            runs
            and runs[-1].actor == actor
            # Missing timer frames are absence of evidence, not a player
            # transition. The same actor remains one run until a different
            # valid seat is observed.
        ):
            previous = runs[-1]
            runs[-1] = _PlayerSignalRun(actor, previous.frame_start, frame)
        else:
            runs.append(_PlayerSignalRun(actor, frame, frame))
    return runs


def _has_turn_transition_support(
    item: _Action,
    signal_runs: Sequence[_PlayerSignalRun],
    terminal_frames: Sequence[int],
) -> bool:
    """Whether a short candidate is locally bounded by its actor's turn."""
    if item.start is None:
        return False
    end = item.end if item.end is not None else item.start
    for index, run in enumerate(signal_runs):
        if run.actor != item.actor:
            continue
        if not (item.start - _TURN_SUPPORT_BEFORE_FRAMES <= run.frame_end <= end + _TURN_SUPPORT_AFTER_FRAMES):
            continue
        if any(item.start - _TURN_SUPPORT_BEFORE_FRAMES <= frame <= end + _TURN_SUPPORT_AFTER_FRAMES for frame in terminal_frames):
            return True
        if index + 1 < len(signal_runs):
            following = signal_runs[index + 1]
            if following.actor != item.actor and following.frame_start <= end + _TURN_SUPPORT_AFTER_FRAMES:
                return True
    return False



def _latest_terminal_cards(
    cluster: Sequence[_Action],
    terminal_frame: int,
    expected_remaining: int | None = None,
) -> tuple[tuple[str, ...], int | None] | None:
    """Choose the last stable complete visual reading before terminal UI."""
    records: list[tuple[tuple[str, ...], int, float]] = []
    for action in cluster:
        for variant in _variant_records(action):
            cards = _cards(variant.get("cards"))
            frame = _frame(variant.get("frame_index"))
            if cards and _complete(cards) and frame is not None and frame < terminal_frame:
                records.append((cards, frame, _confidence(variant.get("confidence"))))
    if not records:
        return None
    if expected_remaining is not None and expected_remaining > 0:
        exact_count = [item for item in records if len(item[0]) == expected_remaining]
        if exact_count:
            records = exact_count
    latest = max(frame for _cards_value, frame, _confidence_value in records)
    # A final action can be partly covered by its effect. Prefer the stable
    # post-effect reading nearest the terminal control, not the first fragment.
    window = [item for item in records if item[1] >= latest - 5]
    counts: Counter[tuple[str, ...]] = Counter(cards for cards, _frame_value, _confidence_value in window)
    cards = max(
        counts,
        key=lambda value: (
            counts[value],
            sum(confidence for observed, _frame_value, confidence in window if observed == value),
            max(frame for observed, frame, _confidence_value in window if observed == value),
        ),
    )
    frame = max(frame for observed, frame, _confidence_value in window if observed == cards)
    return cards, frame


def _initial_context_required_pass_ids(
    items: Sequence[_Action],
    signal_runs: Sequence[_PlayerSignalRun],
) -> set[object]:
    """Return PASS surfaces needed to bridge the opening table to current turn.

    A recording can begin after a player has already led. The first frame then
    contains that play plus responses that happened before the visible current
    player. Only seats on the forward path from the visible leader to the first
    current player belong to this trick; other visible PASS badges are stale.
    """
    if not signal_runs:
        return set()
    first_current = signal_runs[0]
    leaders = [
        item for item in items
        if not item.is_pass and item.start is not None
        and item.start <= first_current.frame_start
        and "display_present_at_scan_start" in item.original.get("uncertainty", ())
    ]
    if not leaders:
        return set()
    leader = min(leaders, key=lambda item: (item.first_frame, item.position))
    required_actors: list[str] = []
    index = SEATS.index(leader.actor)
    for offset in range(1, len(SEATS)):
        actor = SEATS[(index + offset) % len(SEATS)]
        if actor == first_current.actor:
            break
        required_actors.append(actor)
    required: set[object] = set()
    for actor in required_actors:
        candidates = [
            item for item in items
            if item.actor == actor and item.is_pass and item.start is not None
            and item.start <= first_current.frame_start + 2
        ]
        if candidates:
            required.add(min(candidates, key=lambda item: (item.first_frame, item.position)).original_id)
    return required


def _action_frame_span(action: Mapping[str, object]) -> list[int]:
    frames: list[int] = []
    for key in ("frame_start", "frame_end", "best_frame"):
        value = _frame(action.get(key))
        if value is not None:
            frames.append(value)
    evidence = action.get("evidence_frames")
    if isinstance(evidence, (list, tuple)):
        for item in evidence:
            value = _frame(item)
            if value is not None:
                frames.append(value)
    return frames


def _anchor_actions_to_turn_spine(
    kept: list[dict[str, object]],
    signal_runs: Sequence[_PlayerSignalRun],
) -> list[dict[str, object]]:
    """Re-seat canonical actions onto the current-player turn spine.

    A dense, clean signal stream enumerates the real turn order
    (``self -> right -> opposite -> left``, skipping finished seats).  When a
    recording starts mid-play the projected surfaces can drift by a turn — a
    leader's cards stay painted while the next seats act, and a persistent
    badge can be cloned onto a neighbouring turn.  Anchoring each action to the
    run it belongs to, then emitting at most one action per run, restores a
    trace the counter-clockwise rules can explain.

    Only applied when the signal stream is dense enough to be authoritative;
    sparse recordings keep the conservative projection path untouched.
    """
    if not signal_runs or not kept:
        return kept
    spans = [_action_frame_span(action) for action in kept]
    buckets: dict[int, list[int]] = {}
    unassigned: list[int] = []
    for index, action in enumerate(kept):
        frames = spans[index]
        actor = str(action.get("actor", ""))
        if not frames:
            unassigned.append(index)
            continue
        start = min(frames)
        candidates = [i for i, run in enumerate(signal_runs) if run.actor == actor]
        containing = [
            i for i in candidates
            if signal_runs[i].frame_start - 2 <= start <= signal_runs[i].frame_end + 2
        ]
        if containing:
            chosen = max(
                containing,
                key=lambda i: (
                    sum(1 for frame in frames if signal_runs[i].frame_start <= frame <= signal_runs[i].frame_end),
                    -i,
                ),
            )
        else:
            # A surface can persist after its own turn ended (a leader's cards
            # stay on the table, a PASS badge lingers) so the nearest PRECEDING
            # run owns it; only a surface that precedes every own run belongs to
            # the following one.
            preceding = [i for i in candidates if signal_runs[i].frame_end < start]
            following = [i for i in candidates if signal_runs[i].frame_start > start]
            if preceding:
                chosen = max(preceding)
            elif following:
                chosen = min(following)
            else:
                unassigned.append(index)
                continue
        buckets.setdefault(chosen, []).append(index)

    anchored: list[dict[str, object]] = []
    for run_index in range(len(signal_runs)):
        members = buckets.get(run_index)
        if not members:
            # The turn exists (the timer moved here) but no surface survived the
            # independent per-seat reconciliation.  Recover the turn as an
            # auditable PASS rather than dropping it, which would let the seat
            # order skip and break the counter-clockwise rules.
            run = signal_runs[run_index]
            anchored.append({
                "actor": run.actor,
                "is_pass": True,
                "cards": [],
                "frame_start": run.frame_start,
                "frame_end": run.frame_end,
                "evidence_frames": [],
                "uncertainty": [],
                "review_status": "unverified",
                "reconciliation": {
                    "source_action_ids": [],
                    "merged_action_ids": [],
                    "merged": False,
                    "events": [{
                        "type": "recover",
                        "reason": "turn_spine_gap_pass",
                        "run_frame_start": run.frame_start,
                    }],
                },
            })
            continue
        if len(members) > 1:
            plays = [i for i in members if not bool(kept[i].get("is_pass", False))]
            pool = plays or members
            winner = max(pool, key=lambda i: (len(spans[i]), -i))
            losers = [i for i in members if i != winner]
            reconciliation = kept[winner].setdefault("reconciliation", {})
            if isinstance(reconciliation, dict):
                reconciliation.setdefault("events", []).append({
                    "type": "drop",
                    "reason": "duplicate_turn_anchor",
                    "action_ids": [
                        (kept[i].get("reconciliation") or {}).get("source_action_ids", [])
                        for i in losers
                    ],
                    "run_frame_start": signal_runs[run_index].frame_start,
                })
        else:
            winner = members[0]
        anchored.append(kept[winner])
    anchored.extend(kept[i] for i in unassigned)
    return anchored


def _resolve_paired_unknown_suits(kept: list[dict[str, object]]) -> list[dict[str, object]]:
    """Resolve a ``?`` sitting beside a same-rank known card in the same action.

    The canonical trace must not carry unknown suits.  With two decks the
    identical encoding may legally appear twice, so a same-rank known card is
    decisive evidence — ``QD Q?`` is ``QD QD``.  A ``?`` without that support is
    left untouched for review instead of being guessed.
    """
    for output in kept:
        cards = list(_cards(output.get("cards")))
        if not cards or not any(card.endswith("?") for card in cards):
            continue
        counts = Counter(card for card in cards if not card.endswith("?"))
        outcome: list[str] = []
        resolved_any = False
        for card in cards:
            if not card.endswith("?"):
                outcome.append(card)
                continue
            rank = _rank(card)
            choice = next(
                (
                    other for other in cards
                    if not other.endswith("?") and _rank(other) == rank and counts[other] < 2
                ),
                None,
            )
            if choice is None:
                outcome.append(card)
                continue
            counts[choice] += 1
            outcome.append(choice)
            resolved_any = True
        if not resolved_any:
            continue
        output["cards"] = outcome
        output.setdefault("cards_before_repair", sorted(cards))
        output.setdefault("repair_reason", "unknown_suit_resolved_in_trace")
        if not _unknown(outcome):
            output["repair_status"] = "resolved"
            uncertainty = [item for item in output.get("uncertainty", ()) if item != "unknown_suit"]
            output["uncertainty"] = uncertainty
            if output.get("review_status") == "needs_review" and not uncertainty:
                output["review_status"] = "resolved"
    return kept


def reconcile_action_trace(
    actions: Iterable[Mapping[str, object]] | Mapping[str, object],
    frame_observations: Iterable[Mapping[str, object]],
    *,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
) -> list[dict[str, object]]:
    """Return canonical actions from projected/repaired actions and raw frames.

    The function is pure: inputs are deep-copied and never modified.  Nearby
    same-seat windows are merged only when their card displays are identical,
    rank-compatible progressive/unknown-suit variants, or a one-card joker
    label flicker.  A one-frame unknown-suit reading is dropped only when a
    complete same-seat reading immediately supports it; otherwise it remains
    an auditable unresolved action.

    Every output action carries ``reconciliation`` metadata with source IDs,
    merge events, and any dropped source IDs.  Action IDs are assigned
    continuously from one in chronological order.
    """
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative")
    raw_actions = _as_actions(actions)
    if isinstance(frame_observations, Mapping):
        frame_iter = list(frame_observations.get("frame_observations", frame_observations.get("observations", ())))
    else:
        frame_iter = list(frame_observations)
    reads = _read_frames(frame_iter)
    items = _make_actions(raw_actions, reads)
    terminal_frames: list[int] = []
    for row in frame_iter:
        if isinstance(row, Mapping):
            frame = _frame(row.get("frame_index"))
            buttons = row.get("buttons", ())
            if frame is not None and isinstance(buttons, (list, tuple, set)) and any(button in {"change_table", "continue_game"} for button in buttons):
                terminal_frames.append(frame)
    signal_runs = _compressed_player_signal_runs(frame_iter)
    raw_card_totals = {
        seat: sum(len(item.cards) for item in items if item.actor == seat and not item.is_pass)
        for seat in SEATS
    }
    # A recording that starts while a play surface is already on screen has no
    # clean initial signal/action boundary. In that case use the dense signal
    # stream to recover each subsequent turn. A recording with a clean opening
    # is left on the conservative projector path; signal OCR may lag there.
    initial_signal_frame = signal_runs[0].frame_start if signal_runs else None
    has_initial_in_progress_display = bool(
        initial_signal_frame is not None
        and any(item.first_frame <= initial_signal_frame and not item.is_pass for item in items)
    )
    signal_boundary_mode = bool(
        has_initial_in_progress_display
        and len(signal_runs) >= max(8, int(len(items) * 0.75))
    )
    # A recording can begin with a clean opening (so the in-progress flag above
    # stays unset) and still carry a dense, clean current-player stream.  That
    # stream enumerates the real turn order and is used as the authoritative
    # turn spine when re-seating the canonical actions.
    turn_spine_mode = bool(
        signal_runs
        and len(signal_runs) >= 8
        and len(items) <= 2 * len(signal_runs)
    )
    # The PASS badge persists across trick boundaries, so one raw PASS surface
    # can span several of the seat's turns. Splitting at current-player turn
    # boundaries is always safe when signal evidence exists — restricting it to
    # signal_boundary_mode let clean-opening recordings fuse distinct turns.
    if signal_runs:
        items = _split_pass_actions_at_turn_boundaries(items, signal_runs)
    initial_required_pass_ids = _initial_context_required_pass_ids(items, signal_runs)
    dropped: list[dict[str, object]] = []
    filtered_items: list[_Action] = []
    for item in items:
        nearby_signal = next(
            (run.actor for run in signal_runs if item.start is not None and run.frame_start - 2 <= item.start <= run.frame_end + 2),
            None,
        )
        signal_actor_started = any(
            item.start is not None and other.actor == nearby_signal and other.start is not None
            and item.start < other.start <= item.start + 3
            for other in items
        )
        if (
            item.is_pass
            and item.original_id not in initial_required_pass_ids
            and nearby_signal is not None
            and item.actor != nearby_signal
        ):
            if not signal_boundary_mode and signal_actor_started:
                dropped.append({"type": "drop", "reason": "out_of_turn_pass_display", "action_ids": [item.original_id], "actor": item.actor, "frame": item.start, "current_player_signal": nearby_signal})
                continue
            if signal_boundary_mode:
                prior_index = next(
                    (
                        index
                        for index in range(len(signal_runs) - 1, -1, -1)
                        if signal_runs[index].actor == item.actor
                        and 0 <= item.start - signal_runs[index].frame_end <= _PASS_RECOVERY_AFTER_FRAMES
                    ),
                    None,
                )
                # A PASS badge can render after the timer advances once. If
                # several later player turns have already begun, it is stale
                # evidence from an earlier pass rather than this actor's turn.
                owned_prior_run = prior_index is not None and sum(
                    run.frame_start <= item.start
                    for run in signal_runs[prior_index + 1:]
                ) <= 1
                if not owned_prior_run:
                    dropped.append({"type": "drop", "reason": "out_of_turn_pass_display", "action_ids": [item.original_id], "actor": item.actor, "frame": item.start, "current_player_signal": nearby_signal})
                    continue
        # Unknown-suit readings are evidence, not automatically noise.  A short
        # candidate is removable only when a nearby complete compatible reading
        # proves it is a transient rendering of the same display.  In particular,
        # preserve a lone ``A?`` as an auditable needs-review action rather than
        # silently deleting a potentially real play.
        # Deletion needs an equal-rank complete replacement.  A larger complete
        # hand is a progressive recognition candidate and must survive to the
        # merge phase, where it can retain repair provenance.
        direct_complete_support = any(
            _complete(read.cards)
            and _ranks(item.cards) == _ranks(read.cards)
            and _compatible(item.cards, read.cards)[0]
            for read in item.reads
        )
        nearby_complete_support = any(
            other is not item and other.actor == item.actor
            and _near(item, other, max_gap_frames)
            and _complete(other.cards)
            and _ranks(item.cards) == _ranks(other.cards)
            and _compatible(item.cards, other.cards)[0]
            for other in items
        )
        nearby_pass_support = any(
            other is not item and other.actor == item.actor and other.is_pass
            and item.start is not None and other.start is not None
            and -1 <= other.start - item.start <= _PASS_RECOVERY_AFTER_FRAMES
            for other in items
        )
        unknown_frame_count = len(set(item.evidence_frames or ([item.start] if item.start is not None else [])))
        short_unknown = (
            not item.is_pass and _unknown(item.cards)
            and unknown_frame_count <= (3 if nearby_pass_support else 2)
            and (direct_complete_support or nearby_complete_support or nearby_pass_support)
        )
        if short_unknown:
            dropped.append({
                "type": "drop",
                "reason": "one_frame_unknown_suit_noise",
                "action_ids": [item.original_id],
                "actor": item.actor,
                "frame": item.start,
                "supported_by_complete_read": True,
            })
            continue
        # With no current-player signal at all, safely degrade by preserving the
        # uncertain action for review. Where signal evidence exists, a short
        # unknown candidate without an actor-to-next-seat/terminal transition is
        # locally disproved and can be removed auditably.
        if (
            not item.is_pass
            and _unknown(item.cards)
            and len(set(item.evidence_frames or ([item.start] if item.start is not None else []))) <= 2
            and signal_runs
            and not _has_turn_transition_support(item, signal_runs, terminal_frames)
        ):
            dropped.append({
                "type": "drop",
                "reason": "no_turn_transition_support",
                "action_ids": [item.original_id],
                "actor": item.actor,
                "frame": item.start,
                "signal_runs_present": True,
            })
            continue
        # A card surface that never overlaps its own actor's current-player
        # signal while a foreign seat owns the timer is animation ghosting
        # (e.g. cards flying across a neighbouring ROI), not a real play.
        if _out_of_turn_card_ghost(item, signal_runs):
            dropped.append({
                "type": "drop",
                "reason": "out_of_turn_card_ghost",
                "action_ids": [item.original_id],
                "actor": item.actor,
                "frame": item.start,
            })
            continue
        filtered_items.append(item)
    items = filtered_items
    # Reconcile each seat's display timeline independently.  A duplicate can
    # straddle another seat's action (for example, a stale card read while a
    # player passes), so global adjacency would miss exactly those windows.
    by_actor: dict[str, list[_Action]] = {}
    for item in items:
        by_actor.setdefault(item.actor, []).append(item)
    clusters: list[list[_Action]] = []
    for actor_items in by_actor.values():
        actor_items.sort(key=lambda item: (item.first_frame, item.position))
        actor_clusters: list[list[_Action]] = []
        for item in actor_items:
            if actor_clusters:
                previous = actor_clusters[-1][-1]
                ok, reason = _mergeable(previous, item, max_gap_frames)
                compatible, compatible_reason = (
                    (True, "brief_missing_frame_duplicate")
                    if previous.is_pass and item.is_pass
                    else _compatible(previous.cards, item.cards)
                )
                if (
                    not ok
                    and not previous.is_pass
                    and not item.is_pass
                    and compatible
                    and not _actor_received_new_turn_between(previous, item, signal_runs)
                ):
                    ok, reason = True, "stale_surface_reappearance_without_new_turn"
                # Two PASS surfaces separated by the seat's new turn are
                # distinct actions even when the badge gap is brief; merging
                # them would silently erase one whole turn from the trace.
                if (
                    ok
                    and previous.is_pass
                    and item.is_pass
                    and _actor_received_new_turn_between(previous, item, signal_runs)
                ):
                    ok, reason = False, None
                # Two card surfaces separated by the seat's own new turn are
                # distinct plays too.  A short hand whose ranks happen to be a
                # subset of the previous play — a ``2C`` read right after an
                # A2345 straight — must not be absorbed into that earlier play.
                if (
                    ok
                    and not previous.is_pass
                    and not item.is_pass
                    and _actor_received_new_turn_between(previous, item, signal_runs)
                ):
                    ok, reason = False, None
                if ok and signal_boundary_mode and _has_full_response_cycle(previous, item, items, signal_runs):
                    ok, reason = False, None
                if ok:
                    _merge(previous, item, reason or "compatible_display")
                    actor_clusters[-1].append(item)
                    continue
            actor_clusters.append([item])
        clusters.extend(actor_clusters)
    # Keep canonical output in stable temporal order.  ``position`` only breaks
    # ties for overlapping/identical frame spans; it must not globally reorder
    # actions whose source projection positions happen to differ.
    clusters.sort(key=lambda cluster: (
        min(item.first_frame for item in cluster),
        min(item.position for item in cluster),
    ))

    kept: list[dict[str, object]] = []
    self_starting = max(
        (
            len(row.get("opening", {}).get("my_hand") or ())
            for row in frame_iter
            if isinstance(row.get("opening"), Mapping)
        ),
        default=27,
    )
    remaining_cards = {seat: 27 for seat in SEATS}
    remaining_cards["self"] = self_starting or 27
    first_terminal_frame = min(terminal_frames) if terminal_frames else None
    terminal_actor = (
        next(
            (run.actor for run in reversed(signal_runs) if first_terminal_frame is not None and run.frame_end < first_terminal_frame),
            None,
        )
        if first_terminal_frame is not None
        else None
    )
    for cluster in clusters:
        output = _reconcile_cluster(cluster)
        cards = _cards(output.get("cards"))
        source_ids = [item.original_id for item in cluster]
        near_terminal = (
            first_terminal_frame is not None
            and cluster[0].last_frame >= first_terminal_frame - 20
        )
        if near_terminal and bool(output.get("is_pass", False)):
            recent_owned_turn = any(
                run.actor == cluster[0].actor
                and 0 <= cluster[0].first_frame - run.frame_end <= _PASS_RECOVERY_AFTER_FRAMES
                for run in signal_runs
            )
            if not recent_owned_turn:
                dropped.append({"type": "drop", "reason": "post_game_residual_pass", "action_ids": source_ids, "actor": cluster[0].actor, "frame": cluster[0].first_frame})
                continue
        if near_terminal and not bool(output.get("is_pass", False)):
            # A terminal control often appears immediately after the final
            # player commits cards. Preserve that last actor's stable card
            # display; other simultaneously revealed hands are residual UI.
            is_last_turn_action = (
                cluster[0].actor == terminal_actor
                and bool(cards)
                and not bool(output.get("is_pass", False))
                and cluster[0].first_frame < first_terminal_frame
            )
            if not is_last_turn_action:
                dropped.append({"type": "drop", "reason": "post_game_residual_hand", "action_ids": source_ids, "actor": cluster[0].actor, "frame": cluster[0].first_frame})
                continue
            terminal_cards = _latest_terminal_cards(
                cluster,
                first_terminal_frame,
                remaining_cards.get(cluster[0].actor),
            )
            if terminal_cards is not None:
                cards, best_frame = terminal_cards
                output["cards"] = list(cards)
                output["best_frame"] = best_frame
            reconciliation = output.setdefault("reconciliation", {})
            if isinstance(reconciliation, dict):
                reconciliation.setdefault("events", []).append({
                    "type": "terminal_last_turn_preserved",
                    "reason": "last_current_player_before_terminal",
                    "terminal_frame": first_terminal_frame,
                })
        kept.append(output)
        actor = str(output.get("actor", ""))
        if actor in remaining_cards and not bool(output.get("is_pass", False)):
            remaining_cards[actor] -= len(_cards(output.get("cards")))

    # With a dense, clean signal stream the observed turns are authoritative:
    # re-seat the actions onto that spine so the counter-clockwise rules can
    # explain every step, and a badge cloned onto a neighbouring turn cannot
    # fabricate a duplicate PASS.
    if turn_spine_mode:
        kept = _anchor_actions_to_turn_spine(kept, signal_runs)
    kept = _resolve_paired_unknown_suits(kept)

    dropped_ids = [item for event in dropped for item in event["action_ids"]]
    for index, output in enumerate(kept, 1):
        output["action_id"] = index
        reconciliation = output.setdefault("reconciliation", {})
        if isinstance(reconciliation, dict):
            # Keep the global drop audit on one deterministic anchor rather
            # than duplicating the same event list on every action.
            reconciliation["dropped_action_ids"] = dropped_ids if index == 1 else []
            reconciliation["drop_events"] = deepcopy(dropped) if index == 1 else []
    return kept


def reconcile_projected_actions(
    actions: Iterable[Mapping[str, object]] | Mapping[str, object],
    frame_observations: Iterable[Mapping[str, object]],
    *,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
) -> list[dict[str, object]]:
    """Descriptive alias for integrations that call this a projection pass."""
    return reconcile_action_trace(actions, frame_observations, max_gap_frames=max_gap_frames)


reconcile_actions = reconcile_action_trace


__all__ = ["DEFAULT_MAX_GAP_FRAMES", "PASS_MAX_GAP_FRAMES", "reconcile_action_trace", "reconcile_projected_actions", "reconcile_actions"]
