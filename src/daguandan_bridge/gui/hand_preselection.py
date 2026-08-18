"""Safe, UI-side planning for selecting recommended hand cards.

This module deliberately contains no Win32 calls and knows nothing about the
live reducer.  It turns a fresh full-hand recognition result into a plan of
screen coordinates only when every physical card can be accounted for.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from ..capture_service import FrameSnapshot
from ..domain.recognition import RecognitionAnnotation, RecognitionResult
from ..models import ClientRect


@dataclass(frozen=True)
class PreselectionPlan:
    """An all-or-nothing, hand-only input plan in physical screen pixels."""

    request_id: str
    expected_client_rect: ClientRect
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PreselectionResult:
    """The visible outcome of planning or injecting a hand preselection."""

    request_id: str
    status: Literal["planned", "preselected", "rejected", "failed"]
    detail: str
    plan: PreselectionPlan | None = None


class HandPreselectionPlanner:
    """Build a preselection plan only from a complete, exact hand read."""

    def plan(
        self,
        *,
        request_id: str,
        advice_cards: tuple[str, ...],
        expected_hand: tuple[str, ...],
        recognition: RecognitionResult,
        frame: FrameSnapshot,
    ) -> PreselectionResult:
        request_id = str(request_id)
        expected = tuple(str(card) for card in expected_hand)
        requested = tuple(str(card) for card in advice_cards)
        if not requested:
            return self._reject(request_id, "推荐没有可预选的牌")
        if any(_is_unknown_card(card) for card in (*expected, *requested)):
            return self._reject(request_id, "手牌或推荐含未知花色，已拒绝预选")
        if Counter(requested) - Counter(expected):
            return self._reject(request_id, "推荐牌不属于当前手牌，已拒绝预选")

        recognized_hand = tuple(str(card) for card in recognition.my_hand)
        if any(_is_unknown_card(card) for card in recognized_hand):
            return self._reject(request_id, "最新手牌识别含未知花色，已拒绝预选")
        if Counter(recognized_hand) != Counter(expected):
            return self._reject(request_id, "最新手牌识别与当前状态不一致，已拒绝预选")

        annotations = tuple(
            annotation
            for annotation in recognition.annotations
            if annotation.category == "hand"
        )
        locations = _hand_locations(annotations)
        located_counts = Counter(
            {card: len(choices) for card, choices in locations.items()}
        )
        if located_counts != Counter(expected):
            return self._reject(request_id, "手牌定位不完整或重复牌不足，已拒绝预选")

        remaining = {card: list(items) for card, items in locations.items()}
        points: list[tuple[int, int]] = []
        for card in requested:
            choices = remaining.get(card, [])
            if not choices:
                return self._reject(request_id, "推荐牌定位不完整或重复牌不足，已拒绝预选")
            annotation = choices.pop(0)
            point = _annotation_center_on_screen(annotation, frame)
            if point is None:
                return self._reject(request_id, "手牌坐标无法安全映射到客户区，已拒绝预选")
            points.append(point)

        plan = PreselectionPlan(
            request_id=request_id,
            expected_client_rect=frame.frame.rect,
            points=tuple(points),
        )
        return PreselectionResult(
            request_id=request_id,
            status="planned",
            detail="已定位推荐手牌，正在预选",
            plan=plan,
        )

    @staticmethod
    def _reject(request_id: str, detail: str) -> PreselectionResult:
        return PreselectionResult(
            request_id=request_id,
            status="rejected",
            detail=str(detail),
        )


def _hand_locations(
    annotations: tuple[RecognitionAnnotation, ...],
) -> dict[str, list[RecognitionAnnotation]]:
    locations: dict[str, list[RecognitionAnnotation]] = defaultdict(list)
    for annotation in annotations:
        card = str(annotation.label)
        if not card or _is_unknown_card(card):
            continue
        locations[card].append(annotation)
    for choices in locations.values():
        choices.sort(key=lambda item: (item.box[0], item.box[1], item.box[2], item.box[3]))
    return dict(locations)


def _annotation_center_on_screen(
    annotation: RecognitionAnnotation,
    frame: FrameSnapshot,
) -> tuple[int, int] | None:
    """Map one standardized hand box back to a physical client coordinate."""

    x, y, width, height = (int(value) for value in annotation.box)
    if width <= 0 or height <= 0:
        return None
    standardized = frame.frame.standardization
    content = standardized.content_box
    scale = float(standardized.scale)
    if scale <= 0:
        return None
    center_x = x + width // 2
    center_y = y + height // 2
    if not (
        content.x <= center_x < content.x + content.w
        and content.y <= center_y < content.y + content.h
    ):
        return None

    source_x = standardized.source_viewport.x + (center_x - content.x) / scale
    source_y = standardized.source_viewport.y + (center_y - content.y) / scale
    source_width, source_height = standardized.source_size
    if not (0 <= source_x < source_width and 0 <= source_y < source_height):
        return None
    local_x = int(round(source_x))
    local_y = int(round(source_y))
    rect = frame.frame.rect
    if not (0 <= local_x < rect.width and 0 <= local_y < rect.height):
        return None
    return rect.left + local_x, rect.top + local_y


def _is_unknown_card(card: str) -> bool:
    return not card or "?" in card
