from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np

from .annotation_service import AnnotationService, RegionRecord
from .danzero.state import Seat
from .image_io import read_image_unicode
from .models import Box
from .template_service import TemplateService
from .live.turns import TURN_ORDER


PLAY_REGION_TO_SEAT: dict[str, Seat] = {
    "my_play": "self",
    "left_play": "left",
    "opposite_play": "opposite",
    "right_play": "right",
}
SEATS_IN_ORDER: tuple[Seat, ...] = TURN_ORDER
BUTTON_LABELS: dict[str, str] = {
    "pass": "不出",
    "play_cards": "出牌",
    "hint": "提示",
    "cannot_beat": "要不起",
    "double": "加倍",
    "super_double": "超级加倍",
    "arrange": "整理",
    "quick_arrange": "一键整理",
    "chat": "聊天",
    "more": "更多",
    "rules": "规则",
    "change_table": "换桌",
    "continue_game": "继续游戏",
}


@dataclass(frozen=True)
class RecognitionAnnotation:
    label: str
    box: tuple[int, int, int, int]
    confidence: float
    category: str


@dataclass(frozen=True)
class RecognizedEvent:
    player: Seat
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    source: str


@dataclass(frozen=True)
class RecognitionResult:
    round_level: str | None
    wild_rank: str | None
    current_player: Seat | None
    lead_player: Seat | None
    my_hand: tuple[str, ...]
    events: tuple[RecognizedEvent, ...]
    field_confidences: dict[str, float]
    sources: dict[str, str]
    unresolved_fields: tuple[str, ...]
    diagnostics: tuple[str, ...]
    annotations: tuple[RecognitionAnnotation, ...] = ()
    buttons: tuple[str, ...] = ()
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class _TemplateMatch:
    label: str
    kind: str
    source_role: str
    source: str
    score: float
    x: int
    y: int
    w: int
    h: int

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2


class ScreenshotRecognitionService:
    """Recognize visible GuanDan fields from configured regions and templates.

    This deliberately uses only project-owned data: region boxes provide the
    search areas and template samples provide the card/status vocabulary. A
    field is left unresolved when its confidence is below the conservative
    threshold, so the UI can ask the user to confirm instead of guessing.
    """

    _HAND_RANK_THRESHOLD = 0.68
    _HAND_SUIT_THRESHOLD = 0.68
    # Play regions overlap UI buttons in some Tencent screenshots.  A lower
    # threshold turns button glyphs into fake cards (for example, ``AC``).
    # Real rank/suit templates still score near 1.0 at the native resolution.
    _PLAY_RANK_THRESHOLD = 0.60
    _PLAY_SUIT_THRESHOLD = 0.60
    _LEVEL_THRESHOLD = 0.60
    _STATUS_THRESHOLD = 0.62
    _TIMER_THRESHOLD = 0.45
    # A hand can contain more than four cards of the same suit.  Keep enough
    # candidates for a complete suit while using the centered suppression
    # window below to remove repeated peaks from the same glyph.
    _MAX_MATCHES_PER_TEMPLATE = 16

    def __init__(
        self,
        annotation_service: AnnotationService | None = None,
        template_service: TemplateService | None = None,
    ) -> None:
        self.annotation_service = annotation_service or AnnotationService()
        self.template_service = template_service or TemplateService(
            self.annotation_service.profiles_root,
            self.annotation_service.profile_name,
        )

    def recognize(self, image: np.ndarray | Path) -> RecognitionResult:
        started = perf_counter()
        source_image = read_image_unicode(image) if isinstance(image, Path) else image
        if not isinstance(source_image, np.ndarray) or source_image.ndim not in {2, 3}:
            raise ValueError("识别输入必须是有效的 OpenCV 图片")
        if source_image.shape[0] <= 0 or source_image.shape[1] <= 0:
            raise ValueError("识别图片尺寸必须大于 0")

        regions = {region.name: region for region in self.annotation_service.list_regions()}
        templates = self._load_templates()
        diagnostics: list[str] = []
        confidences: dict[str, float] = {}
        sources: dict[str, str] = {}
        annotations: list[RecognitionAnnotation] = []

        level, level_score, level_source, level_match = self._recognize_level(
            source_image, regions.get("level_rank"), templates
        )
        if level is not None:
            confidences["round_level"] = level_score
            confidences["wild_rank"] = level_score
            sources["round_level"] = level_source
            sources["wild_rank"] = "derived:round_level"
            if level_match is not None:
                annotations.append(
                    RecognitionAnnotation(
                        label=level_match.label,
                        box=self._box_for_matches((level_match,)),
                        confidence=level_match.score,
                        category="level",
                    )
                )
        else:
            diagnostics.append("未识别到当前级牌")

        hand, hand_score, hand_source, hand_diagnostics, hand_annotations = self._recognize_cards(
            source_image,
            regions.get("my_hand"),
            templates,
            source_roles={"hand", "hand_partial"},
            wild_rank=level,
            rank_threshold=self._HAND_RANK_THRESHOLD,
            suit_threshold=self._HAND_SUIT_THRESHOLD,
        )
        diagnostics.extend(hand_diagnostics)
        annotations.extend(hand_annotations)
        hand_needs_review = bool(hand_diagnostics)
        if hand:
            confidences["my_hand"] = hand_score
            sources["my_hand"] = hand_source
        else:
            diagnostics.append("未识别到完整手牌，请手动补全")

        current_player, current_score, current_source, current_match = self._recognize_seat_status(
            source_image, regions, templates, prefix="timer", kind="timer", label="active"
        )
        if current_player is not None:
            confidences["current_player"] = current_score
            sources["current_player"] = current_source
            if current_match is not None:
                annotations.append(
                    RecognitionAnnotation(
                        label=current_match.label,
                        box=self._box_for_matches((current_match,)),
                        confidence=current_match.score,
                        category="timer",
                    )
                )
        else:
            diagnostics.append("未识别到当前行动座位，请手动选择")

        lead_player, lead_score, lead_source, lead_match = self._recognize_seat_status(
            source_image,
            regions,
            templates,
            prefix="first_play",
            kind="status",
            label="first_play",
        )
        if lead_player is not None:
            confidences["lead_player"] = lead_score
            sources["lead_player"] = lead_source
            if lead_match is not None:
                annotations.append(
                    RecognitionAnnotation(
                        label=lead_match.label,
                        box=self._box_for_matches((lead_match,)),
                        confidence=lead_match.score,
                        category="status",
                    )
                )
        else:
            diagnostics.append("未识别到本轮首出座位，请手动选择")

        events: list[RecognizedEvent] = []
        events_need_review = False
        for region_name, seat in PLAY_REGION_TO_SEAT.items():
            region = regions.get(region_name)
            cards, score, card_source, card_diagnostics, card_annotations = self._recognize_cards(
                source_image,
                region,
                templates,
                source_roles={"play"},
                wild_rank=level,
                rank_threshold=self._PLAY_RANK_THRESHOLD,
                suit_threshold=self._PLAY_SUIT_THRESHOLD,
            )
            diagnostics.extend(f"{seat}：{item}" for item in card_diagnostics)
            annotations.extend(card_annotations)
            events_need_review = events_need_review or bool(card_diagnostics)
            if cards:
                events.append(
                    RecognizedEvent(
                        player=seat,
                        cards=cards,
                        is_pass=False,
                        confidence=score,
                        source=card_source,
                    )
                )
                continue
            passed, pass_score, pass_source, pass_match = self._recognize_status(
                source_image,
                regions.get(f"passed_{seat}"),
                templates,
                label="passed",
            )
            if passed:
                if pass_match is not None:
                    annotations.append(
                        RecognitionAnnotation(
                            label=pass_match.label,
                            box=self._box_for_matches((pass_match,)),
                            confidence=pass_match.score,
                            category="status",
                        )
                    )
                events.append(
                    RecognizedEvent(
                        player=seat,
                        cards=(),
                        is_pass=True,
                        confidence=pass_score,
                        source=pass_source,
                    )
                )

        events = self._order_events(events, lead_player)
        if events:
            confidences["events"] = min(event.confidence for event in events)
            sources["events"] = "template:play/status"

        buttons, button_score, button_source, button_annotations = self._recognize_buttons(
            source_image,
            regions.get("button_actions"),
            templates,
        )
        annotations.extend(button_annotations)
        if buttons:
            confidences["buttons"] = button_score
            sources["buttons"] = button_source

        unresolved = tuple(
            field
            for field, value in (
                ("round_level", level),
                ("wild_rank", level),
                ("current_player", current_player),
                ("lead_player", lead_player),
                ("my_hand", hand),
                ("events", events),
            )
            if (
                field != "events"
                and (
                    value is None
                    or value == ()
                    or (field == "my_hand" and hand_needs_review)
                )
            ) or (field == "events" and events_need_review)
        )
        return RecognitionResult(
            round_level=level,
            wild_rank=level,
            current_player=current_player,
            lead_player=lead_player,
            my_hand=hand,
            events=tuple(events),
            field_confidences=confidences,
            sources=sources,
            unresolved_fields=unresolved,
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            annotations=tuple(annotations),
            buttons=buttons,
            elapsed_ms=(perf_counter() - started) * 1000,
        )

    def _load_templates(self) -> tuple[tuple[dict[str, object], np.ndarray], ...]:
        loaded: list[tuple[dict[str, object], np.ndarray]] = []
        for raw in self.template_service.list_templates():
            relative = Path(str(raw.get("file", "")))
            path = self.template_service.profile_root / relative
            if not path.is_file():
                continue
            try:
                loaded.append((raw, read_image_unicode(path)))
            except (OSError, RuntimeError, ValueError):
                continue
        return tuple(loaded)

    def _recognize_level(
        self,
        image: np.ndarray,
        region: RegionRecord | None,
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
    ) -> tuple[str | None, float, str, _TemplateMatch | None]:
        matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("source_role") == "level"
            and raw.get("kind") == "rank",
            threshold=self._LEVEL_THRESHOLD,
            limit=1,
        )
        if not matches:
            return None, 0.0, "", None
        match = matches[0]
        return match.label, match.score, match.source, match

    def _recognize_seat_status(
        self,
        image: np.ndarray,
        regions: dict[str, RegionRecord],
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
        *,
        prefix: str,
        kind: str,
        label: str,
    ) -> tuple[Seat | None, float, str, _TemplateMatch | None]:
        best: tuple[Seat, _TemplateMatch] | None = None
        for seat in SEATS_IN_ORDER:
            match = self._recognize_status(
                image,
                regions.get(f"{prefix}_{seat}"),
                templates,
                kind=kind,
                label=label,
            )
            if match[0] and match[3] is not None and (
                best is None or match[1] > best[1].score
            ):
                best = (seat, match[3])
        if best is None:
            if prefix == "first_play":
                full_region = RegionRecord(
                    name="first_play_global_fallback",
                    role="generic",
                    abs_box=Box(0, 0, image.shape[1], image.shape[0]),
                    ratio_box=(0.0, 0.0, 0.0, 0.0),
                )
                matches = self._matches_for_region(
                    image,
                    full_region,
                    templates,
                    predicate=lambda raw: raw.get("kind") == kind
                    and raw.get("label") == label,
                    threshold=self._STATUS_THRESHOLD,
                    limit=1,
                )
                if matches:
                    match = matches[0]
                    seat = self._infer_seat_from_position(match, image)
                    if seat is None:
                        return None, 0.0, "", None
                    return (
                        seat,
                        match.score,
                        match.source,
                        match,
                    )
            return None, 0.0, "", None
        return best[0], best[1].score, best[1].source, best[1]

    @staticmethod
    def _infer_seat_from_position(match: _TemplateMatch, image: np.ndarray) -> Seat | None:
        """Infer the player zone when a first-play marker is outside old ROIs."""

        height, width = image.shape[:2]
        if match.center_y >= height * 0.70:
            return "self"
        if match.center_y <= height * 0.30:
            return "opposite"
        if match.center_x <= width * 0.30:
            return "left"
        if match.center_x >= width * 0.70:
            return "right"
        return None

    def _recognize_status(
        self,
        image: np.ndarray,
        region: RegionRecord | None,
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
        *,
        label: str,
        kind: str = "status",
    ) -> tuple[bool, float, str, _TemplateMatch | None]:
        matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("kind") == kind and raw.get("label") == label,
            threshold=(
                self._TIMER_THRESHOLD
                if kind == "timer"
                else self._STATUS_THRESHOLD
            ),
            limit=1,
        )
        if not matches:
            return False, 0.0, "", None
        match = matches[0]
        return True, match.score, match.source, match

    def _recognize_cards(
        self,
        image: np.ndarray,
        region: RegionRecord | None,
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
        *,
        source_roles: set[str],
        wild_rank: str | None = None,
        rank_threshold: float,
        suit_threshold: float,
    ) -> tuple[
        tuple[str, ...],
        float,
        str,
        tuple[str, ...],
        tuple[RecognitionAnnotation, ...],
    ]:
        if region is None:
            return (), 0.0, "", ("未配置区域",), ()
        rank_matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("source_role") in source_roles
            and raw.get("kind") == "rank",
            threshold=rank_threshold,
            limit=self._MAX_MATCHES_PER_TEMPLATE,
        )
        suit_source_roles = set(source_roles)
        if wild_rank is not None:
            suit_source_roles.add("level")
        suit_matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("source_role") in suit_source_roles
            and raw.get("kind") == "suit",
            threshold=suit_threshold,
            limit=self._MAX_MATCHES_PER_TEMPLATE,
        )
        rank_matches = self._deduplicate(rank_matches)
        suit_matches = self._deduplicate(suit_matches)
        diagnostics: list[str] = []
        used_suits: set[int] = set()
        cards: list[
            tuple[float, str, float, str, tuple[int, int, int, int]]
        ] = []
        category = "hand" if "hand" in source_roles else "play"
        for rank in sorted(rank_matches, key=lambda item: (item.center_x, item.center_y)):
            if rank.label in {"small_joker", "big_joker"}:
                card_box = self._box_for_matches((rank,))
                cards.append((rank.center_x, rank.label, rank.score, rank.source, card_box))
                continue
            possible = [
                (index, suit)
                for index, suit in enumerate(suit_matches)
                if index not in used_suits
                and (
                    suit.source_role != "level"
                    or str(rank.label) == str(wild_rank)
                )
                and abs(suit.center_x - rank.center_x)
                <= max(24.0, rank.w * 0.8, suit.w * 0.8)
                and self._suit_is_below_rank(rank, suit)
            ]
            if not possible and str(rank.label) == str(wild_rank):
                special_suit = self._infer_special_level_suit(image, rank)
                if special_suit is not None:
                    possible = [(len(suit_matches), special_suit)]
            if not possible:
                diagnostics.append(f"{rank.label} 未匹配到花色")
                continue
            suit_index, suit = min(
                possible,
                key=lambda item: (
                    abs(item[1].center_x - rank.center_x),
                    -item[1].score,
                ),
            )
            used_suits.add(suit_index)
            card_code = f"{rank.label}{self._suit_code(suit.label)}"
            cards.append(
                (
                    rank.center_x,
                    card_code,
                    min(rank.score, suit.score),
                    f"{rank.source}+{suit.source}",
                    self._box_for_matches((rank, suit)),
                )
            )
        cards.sort(key=lambda item: item[0])
        limited_cards: list[
            tuple[float, str, float, str, tuple[int, int, int, int]]
        ] = []
        counts: Counter[str] = Counter()
        for item in cards:
            if counts[item[1]] >= 2:
                diagnostics.append(f"{item[1]} 超过双副牌限制，已忽略低优先级候选")
                continue
            counts[item[1]] += 1
            limited_cards.append(item)
        cards = limited_cards
        if not cards:
            return (), 0.0, "", tuple(diagnostics), ()
        return (
            tuple(item[1] for item in cards),
            min(item[2] for item in cards),
            "template:cards",
            tuple(diagnostics),
            tuple(
                RecognitionAnnotation(item[1], item[4], item[2], category)
                for item in cards
            ),
        )

    @staticmethod
    def _suit_is_below_rank(rank: _TemplateMatch, suit: _TemplateMatch) -> bool:
        vertical_gap = suit.center_y - rank.center_y
        minimum_gap = max(8.0, rank.h * 0.35)
        maximum_gap = max(48.0, rank.h * 1.8)
        return minimum_gap <= vertical_gap <= maximum_gap

    @staticmethod
    def _infer_special_level_suit(
        image: np.ndarray,
        rank: _TemplateMatch,
    ) -> _TemplateMatch | None:
        """Recognize Tencent's colored outlined heart used for a wild rank."""

        left = max(0, int(round(rank.center_x - rank.w * 0.75)))
        top = max(0, rank.y + rank.h - 2)
        right = min(image.shape[1], int(round(rank.center_x + rank.w * 0.75)))
        bottom = min(image.shape[0], top + max(32, rank.h))
        if right <= left or bottom <= top:
            return None
        roi = image[top:bottom, left:right]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        warm = (
            ((hsv[:, :, 0] <= 30) | (hsv[:, :, 0] >= 170))
            & (hsv[:, :, 1] >= 45)
            & (hsv[:, :, 2] >= 80)
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(warm, 8)
        candidates = [
            index
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= 20
        ]
        if not candidates:
            return None
        index = max(candidates, key=lambda item: int(stats[item, cv2.CC_STAT_AREA]))
        x, y, width, height, _area = (
            int(stats[index, offset]) for offset in range(5)
        )
        if width < 10 or height < 14:
            return None
        component = (labels[y : y + height, x : x + width] == index).astype(np.uint8)
        row_widths = [int(np.count_nonzero(row)) for row in component]
        non_empty = [value for value in row_widths if value]
        if not non_empty:
            return None
        if non_empty[0] < max(4, int(max(non_empty) * 0.55)):
            return None
        if non_empty[-1] > int(max(non_empty) * 0.45):
            return None
        return _TemplateMatch(
            label="heart",
            kind="suit",
            source_role="level",
            source="heuristic:special-level-heart",
            score=0.72,
            x=left + x,
            y=top + y,
            w=width,
            h=height,
        )

    def _recognize_buttons(
        self,
        image: np.ndarray,
        region: RegionRecord | None,
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
    ) -> tuple[str, float, str, tuple[RecognitionAnnotation, ...]]:
        if region is None:
            return (), 0.0, "", ()
        matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("kind") == "button",
            threshold=0.72,
            limit=3,
        )
        matches = self._deduplicate(matches)
        best_by_label: dict[str, _TemplateMatch] = {}
        for match in matches:
            current = best_by_label.get(match.label)
            if current is None or match.score > current.score:
                best_by_label[match.label] = match
        selected = sorted(
            best_by_label.values(),
            key=lambda item: (item.center_x, item.center_y),
        )
        if not selected:
            return (), 0.0, "", ()
        return (
            tuple(match.label for match in selected),
            min(match.score for match in selected),
            "template:buttons",
            tuple(
                RecognitionAnnotation(
                    match.label,
                    self._box_for_matches((match,)),
                    match.score,
                    "button",
                )
                for match in selected
            ),
        )

    @staticmethod
    def _box_for_matches(matches: Iterable[_TemplateMatch]) -> tuple[int, int, int, int]:
        items = tuple(matches)
        left = min(item.x for item in items)
        top = min(item.y for item in items)
        right = max(item.x + item.w for item in items)
        bottom = max(item.y + item.h for item in items)
        return left, top, right - left, bottom - top

    @staticmethod
    def _suit_code(label: str) -> str:
        return {
            "spade": "S",
            "heart": "H",
            "club": "C",
            "diamond": "D",
        }.get(str(label), "")

    def _matches_for_region(
        self,
        image: np.ndarray,
        region: RegionRecord | None,
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
        *,
        predicate,
        threshold: float,
        limit: int,
    ) -> list[_TemplateMatch]:
        if region is None:
            return []
        box = AnnotationService._box_for_image(region, image)
        if not box.fits_within((image.shape[1], image.shape[0])):
            return []
        search = image[box.y : box.y + box.h, box.x : box.x + box.w]
        matches: list[_TemplateMatch] = []
        for raw, template in templates:
            if not predicate(raw):
                continue
            template_height, template_width = template.shape[:2]
            if template_height > search.shape[0] or template_width > search.shape[1]:
                continue
            gray_search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            scores = cv2.matchTemplate(
                gray_search,
                gray_template,
                cv2.TM_CCOEFF_NORMED,
            )
            for _ in range(limit):
                _, score, _, location = cv2.minMaxLoc(scores)
                if score < threshold:
                    break
                x, y = location
                matches.append(
                    _TemplateMatch(
                        label=str(raw.get("label", "")),
                        kind=str(raw.get("kind", "")),
                        source_role=str(raw.get("source_role", "")),
                        source=f"template:{raw.get('file', '')}",
                        score=float(score),
                        x=box.x + x,
                        y=box.y + y,
                        w=template_width,
                        h=template_height,
                    )
                )
                radius_x = max(6, min(16, template_width // 3))
                radius_y = max(6, min(16, template_height // 3))
                left = max(0, x - radius_x)
                top = max(0, y - radius_y)
                right = min(scores.shape[1], x + radius_x + 1)
                bottom = min(scores.shape[0], y + radius_y + 1)
                scores[top:bottom, left:right] = -1.0
        return sorted(matches, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _deduplicate(matches: Iterable[_TemplateMatch]) -> list[_TemplateMatch]:
        selected: list[_TemplateMatch] = []
        for candidate in sorted(matches, key=lambda item: item.score, reverse=True):
            overlaps = False
            for current in selected:
                center_distance = abs(candidate.center_x - current.center_x)
                max_distance = max(8.0, min(candidate.w, current.w) * 0.45)
                if center_distance <= max_distance and abs(candidate.center_y - current.center_y) <= max_distance:
                    overlaps = True
                    break
            if not overlaps:
                selected.append(candidate)
        return sorted(selected, key=lambda item: (item.center_x, item.center_y))

    @staticmethod
    def _order_events(
        events: list[RecognizedEvent],
        lead_player: Seat | None,
    ) -> list[RecognizedEvent]:
        if lead_player is None:
            return sorted(events, key=lambda event: SEATS_IN_ORDER.index(event.player))
        lead_index = SEATS_IN_ORDER.index(lead_player)
        order = {
            seat: index
            for index, seat in enumerate(
                SEATS_IN_ORDER[lead_index:] + SEATS_IN_ORDER[:lead_index]
            )
        }
        return sorted(events, key=lambda event: order[event.player])
