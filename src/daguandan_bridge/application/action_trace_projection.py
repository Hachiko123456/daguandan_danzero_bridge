"""Project raw video-frame observations into reviewable action spans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

SEATS: tuple[str, ...] = ("self", "right", "opposite", "left")

# 我方操作按钮（不出/提示/出牌）位于屏幕中下部。左家打出 5 张及以上时，
# 牌尾会伸进按钮区，花色被遮挡，此时读数不可信（花色丢失甚至幻影多牌）。
_OCCLUDING_BUTTONS = frozenset({"play_cards", "hint", "cannot_beat"})
_BUTTON_OCCLUDED_SEATS = frozenset({"left"})


def _row_buttons(raw: dict[str, object]) -> frozenset[str]:
    value = raw.get("buttons", ())
    if not isinstance(value, (list, tuple, set)):
        return frozenset()
    return frozenset(str(item) for item in value)


def _is_button_occluded(seat: str, cards: tuple[str, ...], buttons: frozenset[str]) -> bool:
    """Return whether this card read is taken while own buttons hide the tail.

    Only card reads can be occluded; the PASS badge sits next to the avatar and
    is never covered by the button bar.
    """
    return bool(cards) and seat in _BUTTON_OCCLUDED_SEATS and bool(buttons & _OCCLUDING_BUTTONS)


def _cards(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else ()


def _rank(card: str) -> str:
    if card in {"small_joker", "big_joker"}:
        return card
    return card[:-1] if card.endswith(("S", "H", "C", "D", "?")) else card

def _unknown_suit(cards: tuple[str, ...]) -> bool:
    return any(card.endswith("?") for card in cards)


def _unknown_suit_count(cards: tuple[str, ...]) -> int:
    return sum(card.endswith("?") for card in cards)


def _rank_counts(cards: tuple[str, ...]) -> Counter[str]:
    return Counter(_rank(card) for card in cards)


def _is_rank_subset(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Return whether left can be an incomplete reading of right."""
    left_counts = _rank_counts(left)
    right_counts = _rank_counts(right)
    return all(count <= right_counts[rank] for rank, count in left_counts.items())


def _same_cards(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return Counter(left) == Counter(right)


def _same_display_variant(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    """Whether adjacent reads can describe one visible play interval.

    This deliberately is not a multi-frame confirmation mechanism. It
    groups a partial animation reading with a later complete reading while
    treating two different, complete card sets as a replacement display.
    """
    if _same_cards(left, right):
        return True
    # In the scan-only path an ``?`` means a button or animation currently
    # hides a suit. It is not evidence that a different hand was played.
    # Keep the continuous display pending until a later frame exposes the
    # suits (or the region becomes empty). This intentionally avoids turn
    # inference: it is only an evidence-window rule for one seat.
    if _unknown_suit(left) or _unknown_suit(right):
        return True
    if _is_rank_subset(left, right) or _is_rank_subset(right, left):
        return len(left) != len(right) or _unknown_suit(left) or _unknown_suit(right)
    return False


def _candidate_key(cards: tuple[str, ...], confidence: float, frame: int) -> tuple[int, int, float, int]:
    """Pick the least ambiguous, then most complete, highest-confidence reading.

    Suit completeness must outrank card count: an occluded frame (own buttons
    covering the tail of a long play) can hallucinate an extra ghost card, and
    the old count-first rule let that phantom read beat the later clean one.
    """
    return (-_unknown_suit_count(cards), len(cards), confidence, -frame)


@dataclass
class _OpenAction:
    actor: str
    is_pass: bool
    cards: tuple[str, ...]
    start_frame: int
    start_timestamp_ms: int
    last_frame: int
    last_timestamp_ms: int
    evidence_frames: list[int]
    observed_variants: list[dict[str, object]]
    uncertainty: list[str]
    saw_empty: bool
    best_confidence: float
    best_frame: int
    first_unknown_frame: int | None = None
    first_unknown_cards: tuple[str, ...] = ()
    occluded_reads: int = 0
    saw_clean_read: bool = False

    def observe(self, frame: int, timestamp: int, cards: tuple[str, ...], confidence: float, source: str, *, occluded: bool = False) -> None:
        self.last_frame = frame
        self.last_timestamp_ms = timestamp
        self.evidence_frames.append(frame)
        variant = {"frame_index": frame, "cards": list(cards), "confidence": confidence, "source": source}
        if occluded:
            variant["button_occluded"] = True
        if not self.observed_variants or self.observed_variants[-1] != variant:
            self.observed_variants.append(variant)
        if occluded:
            # 按钮遮挡帧只证明"牌面仍在"，既不更新最佳候选，也不计入花色存疑
            # 证据——遮挡本身就是花色读不出来的原因。
            self.occluded_reads += 1
            if "button_occluded" not in self.uncertainty:
                self.uncertainty.append("button_occluded")
            return
        self.saw_clean_read = True
        if _unknown_suit(cards) and "unknown_suit" not in self.uncertainty:
            self.uncertainty.append("unknown_suit")
        if _unknown_suit(cards):
            if self.first_unknown_frame is None:
                self.first_unknown_frame = frame
                self.first_unknown_cards = cards
        if not self.is_pass and _candidate_key(cards, confidence, frame) > _candidate_key(
            self.cards, self.best_confidence, self.best_frame
        ):
            self.cards = cards
            self.best_confidence = confidence
            self.best_frame = frame

    def to_dict(self, action_id: int) -> dict[str, object]:
        uncertainty = list(self.uncertainty)
        # The batch projector only serializes closed action windows, so a
        # window with no unknown suit is already resolved (there is no
        # separate "not_needed" state for consumers to handle).
        repair_status = "resolved"
        repair_frame: int | None = None
        repair_reason = "no_unknown_suit_evidence"
        cards_before_repair: list[str] = []
        if self.saw_clean_read and "button_occluded" in uncertainty:
            # 按钮消失后的干净读数已经成为候选，遮挡本身不再是存疑理由。
            uncertainty = [item for item in uncertainty if item != "button_occluded"]
        if self.first_unknown_frame is not None:
            cards_before_repair = sorted(self.first_unknown_cards)
            resolved = not _unknown_suit(self.cards) and self.best_frame > self.first_unknown_frame
            if resolved:
                repair_status = "resolved"
                repair_frame = self.best_frame
                repair_reason = "later_frame_resolved_unknown_suit"
                uncertainty = [item for item in uncertainty if item != "unknown_suit"]
            else:
                repair_status = "unresolved"
                repair_reason = "no_complete_suit_evidence_before_display_end"
        if not self.saw_empty:
            uncertainty.append("display_present_at_scan_start")
        return {
            "schema": "guandan.video-action-trace/1",
            "action_id": action_id,
            "actor": self.actor,
            "is_pass": self.is_pass,
            # Action cards are semantic data; canonicalize their order so a
            # visually left-to-right template result can compare directly with
            # a TruthLog. Raw ordering remains intact in observed_variants.
            "cards": sorted(self.cards),
            "cards_before_repair": cards_before_repair,
            "repair_status": repair_status,
            "repair_frame": repair_frame,
            "repair_reason": repair_reason,
            "best_frame": self.best_frame,
            "frame_start": self.start_frame,
            "frame_end": self.last_frame,
            "timestamp_start_ms": self.start_timestamp_ms,
            "timestamp_end_ms": self.last_timestamp_ms,
            "evidence_frames": list(dict.fromkeys(self.evidence_frames)),
            "observed_variants": self.observed_variants,
            "uncertainty": list(dict.fromkeys(uncertainty)),
            "review_status": "needs_review" if uncertainty else "unverified",
            "source": "video_scan",
        }


class ActionTraceProjector:
    """Project every frame observation without two-frame acceptance."""

    def project(self, observations: Iterable[dict[str, object]]) -> dict[str, object]:
        active: dict[str, _OpenAction | None] = {seat: None for seat in SEATS}
        saw_empty = {seat: False for seat in SEATS}
        gap_seen = {seat: False for seat in SEATS}
        completed: list[_OpenAction] = []
        for raw in observations:
            if not raw.get("decode_ok", False):
                continue
            try:
                frame = int(raw["frame_index"])
                timestamp = int(raw["timestamp_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            regions = raw.get("regions") if isinstance(raw.get("regions"), dict) else {}
            buttons = _row_buttons(raw)
            for seat in SEATS:
                region = regions.get(seat) if isinstance(regions, dict) else None
                region = region if isinstance(region, dict) else {}
                cards = _cards(region.get("cards"))
                passed = bool(region.get("is_pass", False))
                visible = bool(cards or passed)
                if not visible:
                    saw_empty[seat] = True
                    if active[seat] is not None:
                        gap_seen[seat] = True
                    continue
                occluded = _is_button_occluded(seat, cards, buttons)
                current = active[seat]
                same = current is not None and not gap_seen[seat] and (current.is_pass == passed) and (
                    passed or _same_display_variant(current.cards, cards)
                )
                if same:
                    current.observe(frame, timestamp, cards, _float(region.get("confidence")), str(region.get("source", "")), occluded=occluded)
                    continue
                if current is not None:
                    completed.append(current)
                action = _OpenAction(
                    actor=seat, is_pass=passed, cards=cards,
                    start_frame=frame, start_timestamp_ms=timestamp,
                    last_frame=frame, last_timestamp_ms=timestamp,
                    evidence_frames=[], observed_variants=[], uncertainty=[],
                    saw_empty=saw_empty[seat], best_confidence=0.0, best_frame=frame,
                )
                action.observe(frame, timestamp, cards, _float(region.get("confidence")), str(region.get("source", "")), occluded=occluded)
                active[seat] = action
                gap_seen[seat] = False
        completed.extend(item for item in active.values() if item is not None)
        completed.sort(key=lambda item: (item.start_frame, SEATS.index(item.actor)))
        # PASS before any card play is not a legal game action.  Keep those
        # observations in frame_observations, but exclude template noise from
        # the reviewable action trace.
        first_card_index = next((index for index, item in enumerate(completed) if not item.is_pass), None)
        if first_card_index is not None:
            completed = completed[first_card_index:]
        else:
            completed = []
        actions = [item.to_dict(index + 1) for index, item in enumerate(completed)]
        return {"schema": "guandan.video-action-trace/1", "actions": actions,
                "opening": self._opening(actions), "needs_review": self._needs_review(actions)}

    @staticmethod
    def _opening(actions: list[dict[str, object]]) -> dict[str, object]:
        candidates = [{"seat": item["actor"], "first_action_frame": item["frame_start"],
                       "cards": item["cards"], "is_pass": item["is_pass"],
                       "evidence_frames": item["evidence_frames"]}
                      for item in actions if not item["is_pass"]]
        if not candidates:
            return {"status": "needs_review", "lead_player": None, "candidates": [],
                    "reason": "no_visible_card_play"}
        earliest = min(int(item["first_action_frame"]) for item in candidates)
        first = [item for item in candidates if int(item["first_action_frame"]) == earliest]
        return {"status": "needs_review", "lead_player": first[0]["seat"] if len(first) == 1 else None,
                "candidates": first, "reason": "video_only_opening_candidate"}

    @staticmethod
    def _needs_review(actions: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{"type": "action_uncertainty", "action_id": item["action_id"],
                 "actor": item["actor"], "frames": item["evidence_frames"],
                 "reasons": item["uncertainty"]}
                for item in actions if item.get("uncertainty")]


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["ActionTraceProjector", "SEATS", "reconcile_action_trace", "reconcile_projected_actions", "reconcile_actions"]


# Post-projection canonicalization is kept separate so persisted raw evidence remains immutable.
from .action_trace_reconciliation import (
    reconcile_action_trace,
    reconcile_projected_actions,
    reconcile_actions,
)
