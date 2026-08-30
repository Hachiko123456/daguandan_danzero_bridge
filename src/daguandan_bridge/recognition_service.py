from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from threading import RLock, local
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np

from .annotation_service import AnnotationService, RegionRecord
from .danzero.state import Seat
from .domain.recognition import (
    PLAY_REGION_TO_SEAT,
    SEATS_IN_ORDER,
    FastSignalResult,
    OpeningSignal,
    PlacementSignal,
    PlayRegionResult,
    RecognitionAnnotation,
    RecognitionResult,
    RecognizedEvent,
)
from .image_io import read_image_unicode
from .models import Box
from .template_service import TemplateService


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


@dataclass(frozen=True)
class _RecognizedCard:
    center_x: float
    code: str
    confidence: float
    source: str
    box: tuple[int, int, int, int]
    suit_options: tuple[str, ...]
    candidate_suits: tuple[str, ...]


class ScreenshotRecognitionService:
    """Recognize visible GuanDan fields from configured regions and templates.

    This deliberately uses only project-owned data: region boxes provide the
    search areas and template samples provide the card/status vocabulary. A
    field is left unresolved when its confidence is below the conservative
    threshold, so the UI can ask the user to confirm instead of guessing.
    """

    _HAND_RANK_THRESHOLD = 0.68
    _HAND_SUIT_THRESHOLD = 0.68
    # Joker art is distinctive (crown/star), and full-card templates score
    # ~0.95 on real jokers; the partial (top sliver) templates instead match
    # ornate card edges at ~0.5, so they are excluded below.
    _JOKER_RANK_THRESHOLD = 0.60
    # Play regions overlap UI buttons in some Tencent screenshots.  A lower
    # threshold turns button glyphs into fake cards (for example, ``AC``).
    # Real rank/suit templates still score near 1.0 at the native resolution.
    _PLAY_RANK_THRESHOLD = 0.60
    _PLAY_SUIT_THRESHOLD = 0.60
    # Spade/club and heart/diamond glyphs can be close under partial
    # occlusion.  Preserve colour-constrained uncertainty when their template
    # scores are effectively tied instead of committing one exact suit.
    _SUIT_AMBIGUITY_MARGIN = 0.06
    # Gray-scale correlation gives black spades and clubs similarly high
    # scores because both are compact black glyphs.  When that first pass is
    # tied, compare their normalized silhouettes with HOG instead of turning
    # a clear card into ``?`` solely because of a fixed score margin.
    _BLACK_SUIT_HOG_SIZE = 32
    _BLACK_SUIT_HOG_MIN_SCORE = 0.82
    _BLACK_SUIT_HOG_MIN_MARGIN = 0.06
    _LEVEL_THRESHOLD = 0.60
    _STATUS_THRESHOLD = 0.62
    # Placement badges mutate the player lifecycle and can end the round.
    # Historical recordings show persistent decorative false matches below
    # 0.88, while real head/second/third badges match at 0.94-1.00.
    _PLACEMENT_THRESHOLD = 0.90
    # Card-type overlays are intentionally larger than the cards they
    # describe.  Tencent can place their lower edge beyond the configured
    # play ROI (the right-side straight overlay, for example, extends below
    # ``right_play``).  Keep this expansion exclusive to effect detection:
    # cards, buttons and status markers must still obey their own strict ROIs.
    _EFFECT_SEARCH_MARGIN_X = 96
    _EFFECT_SEARCH_MARGIN_Y = 96
    # A false first-play confirmation corrupts every later turn.  Require a
    # stronger, clearly better match than ordinary transient status markers.
    _FIRST_PLAY_THRESHOLD = 0.80
    _FIRST_PLAY_MIN_MARGIN = 0.08
    # The table anchor is solely a listener/recording gate.  It deliberately
    # has a stricter score than ordinary UI glyphs, but it is a one-frame
    # readiness signal rather than a three-frame card-recognition consensus.
    _TABLE_ANCHOR_THRESHOLD = 0.85
    _TABLE_ANCHOR_SEARCH_PADDING_RATIO = 0.08
    _TIMER_THRESHOLD = 0.45
    # A hand can contain more than four cards of the same suit.  Keep enough
    # candidates for a complete suit while using the centered suppression
    # window below to remove repeated peaks from the same glyph.
    _MAX_MATCHES_PER_TEMPLATE = 16
    _WILD_HEART_FALLBACK_SCORE = 0.50

    def __init__(
        self,
        annotation_service: AnnotationService | None = None,
        template_service: TemplateService | None = None,
        *,
        diagnostic_tracing: bool = True,
    ) -> None:
        self.annotation_service = annotation_service or AnnotationService()
        self.template_service = template_service or TemplateService(
            self.annotation_service.profiles_root,
            self.annotation_service.profile_name,
        )
        self._template_lock = RLock()
        self._template_cache: tuple[
            tuple[dict[str, object], np.ndarray], ...
        ] | None = None
        self._black_suit_hog_cache: tuple[tuple[str, np.ndarray], ...] | None = None
        self._diagnostic_tracing = bool(diagnostic_tracing)
        self._diagnostic_local = local()

    def set_diagnostic_tracing_enabled(self, enabled: bool) -> None:
        """Toggle read-only traces without changing recognition decisions."""

        self._diagnostic_tracing = bool(enabled)

    def get_last_diagnostic_trace(self) -> dict[str, object] | None:
        """Return a defensive copy of the trace produced on this thread."""

        value = getattr(self._diagnostic_local, "last_trace", None)
        if not isinstance(value, dict):
            return None
        return {
            **value,
            "candidates": [dict(item) for item in value.get("candidates", [])],
            "result": dict(value.get("result", {})),
        }

    def recognize_table_anchor(self, image: np.ndarray | Path) -> float:
        """Return the best ``table_anchor_1`` score on the current table page.

        Anchor samples live in the template manifest rather than in the
        annotation-region document, so they must not be routed through the
        ordinary region recognizer.  This is intentionally a narrow direct
        match: the controller uses one score at or above 0.85 to begin
        recording, while all game-state fields retain their own stability
        requirements.
        """

        source_image = self._source_image(image)
        image_height, image_width = source_image.shape[:2]
        best_score = 0.0
        for raw, template in self._templates():
            if (
                raw.get("kind") != "anchor"
                or raw.get("label") != "table_anchor_1"
            ):
                continue
            try:
                ratio_box = tuple(float(value) for value in raw["ratio_box"])
            except (KeyError, TypeError, ValueError):
                continue
            if len(ratio_box) != 4:
                continue
            ratio_x, ratio_y, ratio_w, ratio_h = ratio_box
            box = Box(
                round(ratio_x * image_width),
                round(ratio_y * image_height),
                max(1, round(ratio_w * image_width)),
                max(1, round(ratio_h * image_height)),
            )
            padding_x = max(1, round(box.w * self._TABLE_ANCHOR_SEARCH_PADDING_RATIO))
            padding_y = max(1, round(box.h * self._TABLE_ANCHOR_SEARCH_PADDING_RATIO))
            left = max(0, box.x - padding_x)
            top = max(0, box.y - padding_y)
            right = min(image_width, box.x + box.w + padding_x)
            bottom = min(image_height, box.y + box.h + padding_y)
            search = source_image[top:bottom, left:right]
            if search.size == 0:
                continue
            template_height, template_width = template.shape[:2]
            if template_height != box.h or template_width != box.w:
                template = cv2.resize(
                    template,
                    (box.w, box.h),
                    interpolation=cv2.INTER_AREA,
                )
                template_height, template_width = template.shape[:2]
            if (
                template_height > search.shape[0]
                or template_width > search.shape[1]
            ):
                continue
            if search.ndim == 3:
                search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
            else:
                search_gray = search
            if template.ndim == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template
            _min_score, max_score, _min_location, _max_location = cv2.minMaxLoc(
                cv2.matchTemplate(
                    search_gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED,
                )
            )
            best_score = max(best_score, float(max_score))
        return best_score

    def play_roi(self, image: np.ndarray, seat: Seat) -> np.ndarray:
        """Return the configured play-region crop without exposing annotation infrastructure."""

        region_name = next(
            name for name, mapped_seat in PLAY_REGION_TO_SEAT.items()
            if mapped_seat == seat
        )
        region = next(
            (
                item
                for item in self.annotation_service.list_regions()
                if item.name == region_name
            ),
            None,
        )
        if region is None:
            return image
        box = self.annotation_service._box_for_image(region, image)
        return image[box.y : box.y + box.h, box.x : box.x + box.w]

    def recognize(
        self,
        image: np.ndarray | Path,
        *,
        allow_unknown_suit: bool = False,
    ) -> RecognitionResult:
        """Run one unchanged recognition pass and expose its read-only trace."""

        if not self._diagnostic_tracing:
            self._diagnostic_local.collector = None
            self._diagnostic_local.last_trace = None
            return self._recognize_impl(
                image,
                allow_unknown_suit=allow_unknown_suit,
            )
        collector: list[dict[str, object]] = []
        self._diagnostic_local.collector = collector
        self._diagnostic_local.last_trace = None
        try:
            result = self._recognize_impl(
                image,
                allow_unknown_suit=allow_unknown_suit,
            )
            source_image = read_image_unicode(image) if isinstance(image, Path) else image
            input_hash = (
                hashlib.sha256(memoryview(np.ascontiguousarray(source_image))).hexdigest()
                if isinstance(source_image, np.ndarray)
                else None
            )
            ordered = sorted(
                collector,
                key=lambda item: (
                    str(item.get("field", "")),
                    -float(item.get("score", -1.0)),
                    str(item.get("label", "")),
                    str(item.get("source", "")),
                ),
            )
            self._diagnostic_local.last_trace = {
                "schema": "guandan.recognition-trace/1",
                "input_sha256": input_hash,
                "input_shape": list(source_image.shape)
                if isinstance(source_image, np.ndarray)
                else None,
                "threshold_policy": "production-unchanged",
                "candidates": ordered,
                "result": {
                    "round_level": result.round_level,
                    "my_hand": list(result.my_hand),
                    "hand_count": len(result.my_hand),
                    "lead_player": result.lead_player,
                    "current_player": result.current_player,
                    "field_confidences": dict(result.field_confidences),
                    "unresolved_fields": list(result.unresolved_fields),
                    "diagnostics": list(result.diagnostics),
                },
            }
            return result
        finally:
            self._diagnostic_local.collector = None

    def _recognize_impl(
        self,
        image: np.ndarray | Path,
        *,
        allow_unknown_suit: bool = False,
    ) -> RecognitionResult:
        """Recognize one image across the configured regions.

        The annotation and replay single-frame tools pass
        ``allow_unknown_suit=True`` so a matched rank whose suit is hidden is
        surfaced as ``5?`` instead of being silently removed.  The default
        remains strict for older callers that use this broad diagnostic scan.
        """
        started = perf_counter()
        source_image = read_image_unicode(image) if isinstance(image, Path) else image
        if not isinstance(source_image, np.ndarray) or source_image.ndim not in {2, 3}:
            raise ValueError("识别输入必须是有效的 OpenCV 图片")
        if source_image.shape[0] <= 0 or source_image.shape[1] <= 0:
            raise ValueError("识别图片尺寸必须大于 0")

        regions = {region.name: region for region in self.annotation_service.list_regions()}
        templates = self._templates()
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

        hand, hand_score, hand_source, hand_diagnostics, hand_annotations, _hand_suit_options = self._recognize_cards(
            source_image,
            regions.get("my_hand"),
            templates,
            source_roles={"hand", "hand_partial"},
            wild_rank=level,
            rank_threshold=self._HAND_RANK_THRESHOLD,
            suit_threshold=self._HAND_SUIT_THRESHOLD,
            allow_unknown_suit=allow_unknown_suit,
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
            cards, score, card_source, card_diagnostics, card_annotations, _event_suit_options = self._recognize_cards(
                source_image,
                region,
                templates,
                source_roles={"play"},
                wild_rank=level,
                rank_threshold=self._PLAY_RANK_THRESHOLD,
                suit_threshold=self._PLAY_SUIT_THRESHOLD,
                allow_unknown_suit=allow_unknown_suit,
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

        buttons, button_score, button_source, button_annotations = self._recognize_buttons_in_regions(
            source_image,
            (regions.get("button_actions"), regions.get("game_end_controls")),
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

    def recognize_play_region(
        self,
        image: np.ndarray | Path,
        seat: Seat,
        *,
        wild_rank: str | None,
        allow_unknown_suit: bool = False,
        allow_pass: bool = True,
    ) -> PlayRegionResult:
        """Run the expensive card matcher only in the expected action zone.

        ``allow_unknown_suit`` keeps a rank-only card (e.g. ``5?``) when the
        suit glyph is occluded.  Live play uses it so card count/state flow is
        preserved without inventing a permanent exact suit.
        """

        if seat not in SEATS_IN_ORDER:
            raise ValueError("待识别座位无效")
        source_image = self._source_image(image)
        regions = {region.name: region for region in self.annotation_service.list_regions()}
        templates = self._templates()
        region_name = next(
            name for name, mapped_seat in PLAY_REGION_TO_SEAT.items() if mapped_seat == seat
        )
        cards, score, source, diagnostics, annotations, suit_options = self._recognize_cards(
            source_image,
            regions.get(region_name),
            templates,
            source_roles={"play"},
            wild_rank=wild_rank,
            rank_threshold=self._PLAY_RANK_THRESHOLD,
            suit_threshold=self._PLAY_SUIT_THRESHOLD,
            allow_unknown_suit=allow_unknown_suit,
        )
        # The play-zone result is authoritative for this action.  Keep the
        # legacy fields empty for log compatibility; do not re-scan my_hand.
        post_hand: tuple[str, ...] = ()
        post_hand_score = 0.0
        if cards:
            return PlayRegionResult(
                player=seat,
                cards=cards,
                is_pass=False,
                confidence=score,
                diagnostics=diagnostics,
                annotations=annotations,
                source=source,
                post_hand=post_hand,
                post_hand_confidence=post_hand_score,
                suit_options=suit_options,
            )
        if not allow_pass:
            return PlayRegionResult(
                player=seat,
                cards=(),
                is_pass=False,
                confidence=0.0,
                diagnostics=diagnostics,
                annotations=(),
                source="no_play_detected",
                post_hand=post_hand,
                post_hand_confidence=post_hand_score,
                suit_options=suit_options,
            )
        passed, pass_score, pass_source, pass_match = self._recognize_status(
            source_image,
            regions.get(f"passed_{seat}"),
            templates,
            label="passed",
        )
        pass_annotations: tuple[RecognitionAnnotation, ...] = ()
        if passed and pass_match is not None:
            pass_annotations = (
                RecognitionAnnotation(
                    label=pass_match.label,
                    box=self._box_for_matches((pass_match,)),
                    confidence=pass_match.score,
                    category="status",
                ),
            )
        return PlayRegionResult(
            player=seat,
            cards=(),
            is_pass=passed,
            confidence=pass_score,
            diagnostics=diagnostics,
            annotations=pass_annotations,
            source=pass_source,
            post_hand=post_hand,
            post_hand_confidence=post_hand_score,
            suit_options=suit_options,
        )

    def recognize_lead_player(self, image: np.ndarray | Path) -> Seat | None:
        """Recognize the first-play marker seat, if it is currently visible."""
        source_image = self._source_image(image)
        regions = {region.name: region for region in self.annotation_service.list_regions()}
        templates = self._templates()
        lead, _, _, _ = self._recognize_seat_status(
            source_image,
            regions,
            templates,
            prefix="first_play",
            kind="status",
            label="first_play",
        )
        return lead

    def recognize_opening_signal(self, image: np.ndarray | Path) -> OpeningSignal:
        """Collect the opening-only signals without deciding who leads.

        The pre-game doubling controls can share screen space with seat
        markers.  Returning raw evidence here lets the orchestrator suppress
        that transient UI and require consistent seat evidence before it
        enters the first turn.
        """

        source_image = self._source_image(image)
        regions = {region.name: region for region in self.annotation_service.list_regions()}
        templates = self._templates()
        marker_player, _, _, _ = self._recognize_seat_status(
            source_image,
            regions,
            templates,
            prefix="first_play",
            kind="status",
            label="first_play",
        )
        active_player, _, _, _ = self._recognize_seat_status(
            source_image,
            regions,
            templates,
            prefix="timer",
            kind="timer",
            label="active",
        )
        buttons, _, _, _ = self._recognize_buttons_in_regions(
            source_image,
            (regions.get("button_actions"), regions.get("game_end_controls")),
            templates,
        )
        button_set = set(buttons)
        game_end_control = next(
            (
                label
                for label in ("continue_game", "change_table")
                if label in button_set
            ),
            None,
        )
        return OpeningSignal(
            # Either doubling control means the opening screen is still
            # transient.  Keep the historical field name for callers, but
            # normal \"加倍×2\" blocks lead commitment just like 超级加倍.
            super_double_visible=bool(button_set & {"super_double", "double"}),
            marker_player=marker_player,
            active_player=active_player,
            self_action_buttons_visible=bool(
                button_set & {"play_cards", "hint", "pass", "cannot_beat"}
            ),
            game_end_control=game_end_control,
        )

    def recognize_super_double_visible(self, image: np.ndarray | Path) -> bool:
        """Check pre-game doubling controls without inspecting any seat.

        The public name is kept for compatibility; both \"超级加倍\" and the
        normal \"加倍×2\" button report ``True`` because either one means the
        lead marker must not be committed yet.
        """
        source_image = self._source_image(image)
        regions = {region.name: region for region in self.annotation_service.list_regions()}
        buttons, _, _, _ = self._recognize_buttons(
            source_image,
            regions.get("button_actions"),
            self._templates(),
        )
        return bool(set(buttons) & {"super_double", "double"})

    def recognize_fast_signals(
        self,
        image: np.ndarray | Path,
        expected_player: Seat,
        *,
        allow_pass: bool = True,
    ) -> FastSignalResult:
        """Read only turn/pass/button/effect signals for one capture frame."""

        if expected_player not in SEATS_IN_ORDER:
            raise ValueError("待识别座位无效")
        source_image = self._source_image(image)
        regions = {region.name: region for region in self.annotation_service.list_regions()}
        templates = self._templates()
        active_player, _, _, _ = self._recognize_seat_status(
            source_image,
            regions,
            templates,
            prefix="timer",
            kind="timer",
            label="active",
        )
        pass_marker_players: list[Seat] = []
        if allow_pass:
            for player in SEATS_IN_ORDER:
                marker_visible, _, _, _ = self._recognize_status(
                    source_image,
                    regions.get(f"passed_{player}"),
                    templates,
                    label="passed",
                )
                if marker_visible:
                    pass_marker_players.append(player)
        pass_visible = expected_player in pass_marker_players
        buttons, _, _, _ = self._recognize_buttons_in_regions(
            source_image,
            (regions.get("button_actions"), regions.get("game_end_controls")),
            templates,
        )
        game_end_control = next(
            (
                label
                for label in ("continue_game", "change_table")
                if label in buttons
            ),
            None,
        )
        play_region_name = next(
            name
            for name, mapped_seat in PLAY_REGION_TO_SEAT.items()
            if mapped_seat == expected_player
        )
        effects = self._matches_for_region(
            source_image,
            regions.get(play_region_name),
            templates,
            predicate=lambda raw: raw.get("kind") == "effect",
            threshold=self._STATUS_THRESHOLD,
            limit=1,
            search_margin=(
                self._EFFECT_SEARCH_MARGIN_X,
                self._EFFECT_SEARCH_MARGIN_Y,
            ),
        )
        placements = self._recognize_placements(
            source_image,
            regions,
            templates,
        )
        return FastSignalResult(
            expected_player=expected_player,
            active_player=active_player,
            pass_visible=pass_visible,
            self_action_buttons_visible=expected_player == "self" and bool(buttons),
            effect_visible=bool(effects),
            pass_marker_player=expected_player if pass_visible else None,
            pass_marker_players=tuple(pass_marker_players),
            super_double_visible="super_double" in buttons,
            game_end_control=game_end_control,
            placements=placements,
        )

    def recognize_placements(
        self,
        image: np.ndarray | Path,
    ) -> tuple[PlacementSignal, ...]:
        """Read persistent 头游/二游/三游 labels independently of card counts."""

        source_image = self._source_image(image)
        regions = {
            region.name: region for region in self.annotation_service.list_regions()
        }
        return self._recognize_placements(
            source_image,
            regions,
            self._templates(),
        )

    def _recognize_placements(
        self,
        image: np.ndarray,
        regions: dict[str, RegionRecord],
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
    ) -> tuple[PlacementSignal, ...]:
        detected: list[PlacementSignal] = []
        for player in SEATS_IN_ORDER:
            candidates: list[_TemplateMatch] = []
            for placement in ("head", "second", "third", "last"):
                matched, _score, _source, match = self._recognize_status(
                    image,
                    regions.get(f"placement_{player}"),
                    templates,
                    label=placement,
                    threshold=self._PLACEMENT_THRESHOLD,
                )
                if matched and match is not None:
                    candidates.append(match)
            if not candidates:
                continue
            best = max(candidates, key=lambda item: item.score)
            detected.append(
                PlacementSignal(
                    player=player,
                    placement=best.label,
                    confidence=best.score,
                    source=best.source,
                )
            )
        return tuple(detected)

    @staticmethod
    def _source_image(image: np.ndarray | Path) -> np.ndarray:
        source_image = read_image_unicode(image) if isinstance(image, Path) else image
        if not isinstance(source_image, np.ndarray) or source_image.ndim not in {2, 3}:
            raise ValueError("识别输入必须是有效的 OpenCV 图片")
        if source_image.shape[0] <= 0 or source_image.shape[1] <= 0:
            raise ValueError("识别图片尺寸必须大于 0")
        return source_image

    def _templates(self) -> tuple[tuple[dict[str, object], np.ndarray], ...]:
        with self._template_lock:
            if self._template_cache is None:
                self._template_cache = self._load_templates()
            return self._template_cache

    def reload_templates(self) -> tuple[tuple[dict[str, object], np.ndarray], ...]:
        """Replace cached files after an explicit template mutation."""

        with self._template_lock:
            loaded = self._load_templates()
            self._template_cache = loaded
            return loaded

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
        candidates: list[tuple[Seat, _TemplateMatch]] = []
        threshold = (
            self._FIRST_PLAY_THRESHOLD
            if prefix == "first_play" and label == "first_play"
            else self._STATUS_THRESHOLD
        )
        for seat in SEATS_IN_ORDER:
            match = self._recognize_status(
                image,
                regions.get(f"{prefix}_{seat}"),
                templates,
                kind=kind,
                label=label,
                threshold=threshold,
            )
            if match[0] and match[3] is not None:
                candidates.append((seat, match[3]))
        if not candidates:
            return None, 0.0, "", None
        candidates.sort(key=lambda item: item[1].score, reverse=True)
        best_seat, best_match = candidates[0]
        if (
            prefix == "first_play"
            and len(candidates) > 1
            and best_match.score - candidates[1][1].score < self._FIRST_PLAY_MIN_MARGIN
        ):
            return None, 0.0, "", None
        return best_seat, best_match.score, best_match.source, best_match

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
        threshold: float | None = None,
    ) -> tuple[bool, float, str, _TemplateMatch | None]:
        matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("kind") == kind and raw.get("label") == label,
            threshold=(
                threshold
                if threshold is not None
                else self._TIMER_THRESHOLD
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
        allow_unknown_suit: bool = False,
    ) -> tuple[
        tuple[str, ...],
        float,
        str,
        tuple[str, ...],
        tuple[RecognitionAnnotation, ...],
        tuple[tuple[str, ...], ...],
    ]:
        if region is None:
            return (), 0.0, "", ("未配置识别区域",), (), ()
        rank_matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("source_role") in source_roles
            and raw.get("kind") == "rank"
            and raw.get("label") not in {"small_joker", "big_joker"},
            threshold=rank_threshold,
            limit=self._MAX_MATCHES_PER_TEMPLATE,
        )
        joker_matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("source_role") in source_roles
            and raw.get("kind") == "rank"
            and raw.get("label") in {"small_joker", "big_joker"}
            and raw.get("source_role") != "hand_partial",
            threshold=self._JOKER_RANK_THRESHOLD,
            limit=self._MAX_MATCHES_PER_TEMPLATE,
            # 大小王靠颜色区分（小王/大王花色不同），灰度匹配会混淆两者，
            # 因 joker 模板用彩色匹配。
            use_color=True,
        )
        rank_matches = self._deduplicate(rank_matches)
        joker_matches = self._deduplicate(joker_matches)
        suit_source_roles = set(source_roles)
        if wild_rank is not None:
            suit_source_roles.add("level")
        raw_suit_matches = self._matches_for_region(
            image,
            region,
            templates,
            predicate=lambda raw: raw.get("source_role") in suit_source_roles
            and raw.get("kind") == "suit",
            threshold=suit_threshold,
            limit=self._MAX_MATCHES_PER_TEMPLATE,
        )
        diagnostics: list[str] = []
        used_suits: set[int] = set()
        cards: list[_RecognizedCard] = []
        joker_cards: list[_RecognizedCard] = []
        category = "hand" if "hand" in source_roles else "play"
        for joker in joker_matches:
            card_box = self._box_for_matches((joker,))
            joker_cards.append(
                _RecognizedCard(
                    joker.center_x,
                    joker.label,
                    joker.score,
                    joker.source,
                    card_box,
                    (),
                    (),
                )
            )
        for rank in sorted(rank_matches, key=lambda item: (item.center_x, item.center_y)):
            color_candidates = self._unknown_suit_options(image, rank)
            possible = [
                (index, suit)
                for index, suit in enumerate(raw_suit_matches)
                if index not in used_suits
                and (
                    suit.source_role != "level"
                    or str(rank.label) == str(wild_rank)
                )
                and abs(suit.center_x - rank.center_x)
                <= max(24.0, rank.w * 0.8, suit.w * 0.8)
                and self._suit_is_below_rank(rank, suit)
            ]
            color_matched = [
                (index, suit)
                for index, suit in possible
                if self._suit_code(suit.label) in color_candidates
            ]
            # Older template sets include a few rank-only glyph samples whose
            # colour no longer matches their accompanying suit sample. Prefer
            # colour-compatible matches, but do not erase the only spatially
            # valid candidate when that legacy data is encountered.
            preferred = color_matched or possible
            preferred_ids = {
                id(candidate)
                for candidate in self._deduplicate(
                    candidate for _index, candidate in preferred
                )
            }
            possible = [
                (index, candidate)
                for index, candidate in preferred
                if id(candidate) in preferred_ids
            ]
            if not possible and str(rank.label) == str(wild_rank):
                special_suit = self._infer_special_level_suit(image, rank)
                if special_suit is not None:
                    possible = [(len(raw_suit_matches), special_suit)]
                else:
                    possible = [
                        (
                            len(raw_suit_matches),
                            _TemplateMatch(
                                label="heart",
                                kind="suit",
                                source_role="level",
                                source="default:wild-heart",
                                score=self._WILD_HEART_FALLBACK_SCORE,
                                x=rank.x,
                                y=rank.y + max(8, int(rank.h * 0.35)),
                                w=rank.w,
                                h=max(8, int(rank.h * 0.4)),
                            ),
                        )
                    ]
            if not possible:
                if allow_unknown_suit:
                    card_box = self._box_for_matches((rank,))
                    cards.append(
                        _RecognizedCard(
                            rank.center_x,
                            f"{rank.label}?",
                            rank.score,
                            rank.source,
                            card_box,
                            color_candidates,
                            color_candidates,
                        )
                    )
                    diagnostics.append(f"{rank.label} 花色被遮挡，按未知花色")
                    continue
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
            uncertain_options = (
                self._uncertain_suit_options(
                    image,
                    rank,
                    suit,
                    raw_suit_matches,
                )
                if category == "play"
                else ()
            )
            shape_suit_code = (
                self._black_suit_shape_code(image, suit)
                if (
                    color_candidates == ("S", "C")
                    and self._suit_code(suit.label) in {"S", "C"}
                )
                else None
            )
            if shape_suit_code is not None:
                uncertain_options = ()
            elif (
                color_candidates == ("S", "C")
                and self._suit_code(suit.label) in {"S", "C"}
            ):
                uncertain_options = color_candidates
            if uncertain_options:
                card_code = f"{rank.label}?"
                diagnostics.append(
                    f"{rank.label} 花色模板相近，按未知花色保留"
                )
            else:
                card_code = f"{rank.label}{shape_suit_code or self._suit_code(suit.label)}"
            cards.append(
                _RecognizedCard(
                    rank.center_x,
                    card_code,
                    min(rank.score, suit.score),
                    (
                        f"{rank.source}+{suit.source}"
                        + (f"+shape:{shape_suit_code}" if shape_suit_code else "")
                    ),
                    self._box_for_matches((rank, suit)),
                    uncertain_options
                    or (shape_suit_code or self._suit_code(suit.label),),
                    (
                        uncertain_options
                        or (shape_suit_code,)
                        if shape_suit_code is not None
                        else (
                            color_candidates
                            if self._suit_code(suit.label) in color_candidates
                            else (self._suit_code(suit.label),)
                        )
                    ),
                )
            )
        cards.sort(key=lambda item: item.center_x)
        if joker_cards:
            # 大小王图案相似，同一张牌可能同时命中两个模板：按距离分组，
            # 每组只保留分数最高的候选。
            competed: list[_RecognizedCard] = []
            for joker in sorted(joker_cards, key=lambda item: -item.confidence):
                jx, jy, jw, jh = joker.box
                jcx, jcy = jx + jw / 2, jy + jh / 2
                near_kept = False
                for kept in competed:
                    kx, ky, kw, kh = kept.box
                    kcx, kcy = kx + kw / 2, ky + kh / 2
                    max_distance = max(10.0, min(jw, kw) * 0.6)
                    if (
                        abs(jcx - kcx) <= max_distance
                        and abs(jcy - kcy) <= max_distance
                    ):
                        near_kept = True
                        break
                if not near_kept:
                    competed.append(joker)
            joker_cards = competed
            # A relaxed joker threshold also catches ornate regular cards
            # (wild/level cards with gold art). A real joker never overlaps
            # another recognized card, so drop joker boxes that sit on one.
            kept_jokers: list[_RecognizedCard] = []
            for joker in joker_cards:
                if any(self._boxes_overlap(joker.box, card.box) for card in cards):
                    diagnostics.append(f"{joker.code} 疑似误报，已忽略重叠候选")
                    continue
                kept_jokers.append(joker)
            cards.extend(kept_jokers)
            cards.sort(key=lambda item: item.center_x)
        cards = self._limit_deck_copies(
            cards,
            diagnostics,
            allow_unknown_suit=allow_unknown_suit,
        )
        if not cards:
            return (), 0.0, "", tuple(diagnostics), (), ()
        return (
            tuple(item.code for item in cards),
            min(item.confidence for item in cards),
            "template:cards",
            tuple(diagnostics),
            tuple(
                RecognitionAnnotation(item.code, item.box, item.confidence, category)
                for item in cards
            ),
            tuple(item.suit_options for item in cards),
        )

    @staticmethod
    def _limit_deck_copies(
        cards: list[_RecognizedCard],
        diagnostics: list[str],
        *,
        allow_unknown_suit: bool,
    ) -> list[_RecognizedCard]:
        """Retain an over-counted suit as bounded uncertainty before dropping it."""

        limited = list(cards)
        while True:
            counts = Counter(
                card.code for card in limited if not card.code.endswith("?")
            )
            overflow = next(
                (code for code, count in counts.items() if count > 2),
                None,
            )
            if overflow is None:
                return limited
            matching = [
                (index, card)
                for index, card in enumerate(limited)
                if card.code == overflow
            ]
            repairable = [
                (index, card)
                for index, card in matching
                if len(card.candidate_suits) > 1
            ]
            if allow_unknown_suit and repairable and len(overflow) >= 2:
                index, card = min(repairable, key=lambda item: item[1].confidence)
                limited[index] = replace(
                    card,
                    code=f"{overflow[:-1]}?",
                    source=f"{card.source}+deck-limit:unknown",
                    suit_options=card.candidate_suits,
                )
                diagnostics.append(
                    f"{overflow} 超过双副牌限制，按候选花色保留低置信度牌"
                )
                continue
            index, _card = min(matching, key=lambda item: item[1].confidence)
            limited.pop(index)
            diagnostics.append(f"{overflow} 超过双副牌限制，已忽略低优先级候选")

    @staticmethod
    def _unknown_suit_options(
        image: np.ndarray,
        rank: _TemplateMatch,
    ) -> tuple[str, str]:
        """Use the visible rank glyph to retain red/black suit candidates."""

        left = max(0, rank.x)
        top = max(0, rank.y)
        right = min(image.shape[1], rank.x + rank.w)
        bottom = min(image.shape[0], rank.y + rank.h)
        if right <= left or bottom <= top:
            return "S", "C"
        roi = image[top:bottom, left:right]
        if roi.ndim != 3:
            return "S", "C"
        blue, green, red = cv2.split(roi)
        red_pixels = (red.astype(np.int16) - np.maximum(blue, green).astype(np.int16) >= 35) & (red >= 90)
        return ("H", "D") if int(np.count_nonzero(red_pixels)) >= 4 else ("S", "C")

    def _uncertain_suit_options(
        self,
        image: np.ndarray,
        rank: _TemplateMatch,
        selected: _TemplateMatch,
        raw_matches: Iterable[_TemplateMatch],
    ) -> tuple[str, ...]:
        """Return colour candidates when a same-colour suit match is tied."""

        color_candidates = self._unknown_suit_options(image, rank)
        selected_code = self._suit_code(selected.label)
        if selected_code not in color_candidates:
            return color_candidates
        scores: dict[str, float] = {}
        for candidate in raw_matches:
            code = self._suit_code(candidate.label)
            if code not in color_candidates:
                continue
            if (
                abs(candidate.center_x - rank.center_x)
                > max(24.0, rank.w * 0.8, candidate.w * 0.8)
                or not self._suit_is_below_rank(rank, candidate)
            ):
                continue
            scores[code] = max(scores.get(code, 0.0), candidate.score)
        selected_score = scores.get(selected_code, selected.score)
        competing = [
            score for code, score in scores.items() if code != selected_code
        ]
        if competing and max(competing) >= selected_score - self._SUIT_AMBIGUITY_MARGIN:
            return color_candidates
        return ()

    @staticmethod
    def _normalized_dark_suit_mask(image: np.ndarray) -> np.ndarray | None:
        """Center a suit silhouette and remove its card background."""

        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            return None
        _threshold, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        points = cv2.findNonZero(mask)
        if points is None:
            return None
        x, y, width, height = cv2.boundingRect(points)
        if width < 4 or height < 4:
            return None
        glyph = mask[y : y + height, x : x + width]
        side = max(width, height) + 6
        canvas = np.zeros((side, side), dtype=np.uint8)
        top = (side - height) // 2
        left = (side - width) // 2
        canvas[top : top + height, left : left + width] = glyph
        return cv2.resize(
            canvas,
            (ScreenshotRecognitionService._BLACK_SUIT_HOG_SIZE,) * 2,
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _black_suit_hog(mask: np.ndarray) -> np.ndarray:
        size = ScreenshotRecognitionService._BLACK_SUIT_HOG_SIZE
        descriptor = cv2.HOGDescriptor(
            (size, size),
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )
        return descriptor.compute(mask).reshape(-1)

    def _black_suit_template_descriptors(self) -> tuple[tuple[str, np.ndarray], ...]:
        with self._template_lock:
            if self._black_suit_hog_cache is not None:
                return self._black_suit_hog_cache
            descriptors: list[tuple[str, np.ndarray]] = []
            for raw, template in self._templates():
                if (
                    raw.get("kind") != "suit"
                    or raw.get("source_role") not in {"hand", "hand_partial", "play"}
                    or raw.get("label") not in {"spade", "club"}
                ):
                    continue
                mask = self._normalized_dark_suit_mask(template)
                if mask is None:
                    continue
                descriptors.append((self._suit_code(str(raw["label"])), self._black_suit_hog(mask)))
            self._black_suit_hog_cache = tuple(descriptors)
            return self._black_suit_hog_cache

    def _black_suit_shape_code(
        self,
        image: np.ndarray,
        selected: _TemplateMatch,
    ) -> str | None:
        """Confirm a tied black suit with normalized silhouette features."""

        # Match locations already cover the suit glyph.  Expanding upward can
        # pull the descender of an adjacent rank (notably ``A``) into the
        # silhouette and erase the very shape distinction we need.
        padding = 0
        left = max(0, selected.x - padding)
        top = max(0, selected.y - padding)
        right = min(image.shape[1], selected.x + selected.w + padding)
        bottom = min(image.shape[0], selected.y + selected.h + padding)
        if right <= left or bottom <= top:
            return None
        mask = self._normalized_dark_suit_mask(image[top:bottom, left:right])
        if mask is None:
            return None
        query = self._black_suit_hog(mask)
        scores: dict[str, float] = {}
        for code, template in self._black_suit_template_descriptors():
            denominator = float(np.linalg.norm(query) * np.linalg.norm(template))
            if denominator <= 0:
                continue
            score = float(np.dot(query, template) / denominator)
            scores[code] = max(scores.get(code, -1.0), score)
        if set(scores) != {"S", "C"}:
            return None
        winner, winner_score = max(scores.items(), key=lambda item: item[1])
        runner_score = scores["C" if winner == "S" else "S"]
        if (
            winner_score < self._BLACK_SUIT_HOG_MIN_SCORE
            or winner_score - runner_score < self._BLACK_SUIT_HOG_MIN_MARGIN
        ):
            return None
        return winner

    @staticmethod
    def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        """Return True when the smaller box is mostly covered by the other."""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
        intersection = inter_w * inter_h
        smaller = min(aw * ah, bw * bh)
        return smaller > 0 and intersection / smaller >= 0.35

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

    def _recognize_buttons_in_regions(
        self,
        image: np.ndarray,
        regions: Iterable[RegionRecord | None],
        templates: tuple[tuple[dict[str, object], np.ndarray], ...],
    ) -> tuple[tuple[str, ...], float, str, tuple[RecognitionAnnotation, ...]]:
        """Merge normal action buttons with the bottom terminal controls."""

        best: dict[str, tuple[float, str, RecognitionAnnotation]] = {}
        for region in regions:
            labels, score, source, annotations = self._recognize_buttons(
                image,
                region,
                templates,
            )
            for label, annotation in zip(labels, annotations):
                existing = best.get(label)
                if existing is None or annotation.confidence > existing[0]:
                    best[label] = (annotation.confidence, source, annotation)
        if not best:
            return (), 0.0, "", ()
        selected = sorted(best.items(), key=lambda item: (item[1][2].box[0], item[1][2].box[1]))
        return (
            tuple(label for label, _value in selected),
            min(value[0] for _label, value in selected),
            "template:buttons",
            tuple(value[2] for _label, value in selected),
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
        use_color: bool = False,
        search_margin: tuple[int, int] = (0, 0),
    ) -> list[_TemplateMatch]:
        if region is None:
            return []
        box = AnnotationService._box_for_image(region, image)
        if not box.fits_within((image.shape[1], image.shape[0])):
            return []
        candidates = [
            (raw, template)
            for raw, template in templates
            if predicate(raw)
            and template.shape[0] <= image.shape[0]
            and template.shape[1] <= image.shape[1]
        ]
        if not candidates:
            return []
        # Whole-card templates (jokers, wild cards) can be taller than the
        # region box; widen the search window to fit them.  Effect overlays
        # additionally receive their explicit outer margin, since their
        # centre may be just outside a player's card ROI.
        margin_x = max(0, int(search_margin[0]))
        margin_y = max(0, int(search_margin[1]))
        max_height = max(template.shape[0] for _, template in candidates)
        max_width = max(template.shape[1] for _, template in candidates)
        top = max(0, box.y - margin_y - max(0, max_height - box.h))
        left = max(0, box.x - margin_x - max(0, max_width - box.w))
        bottom = min(
            image.shape[0],
            box.y + box.h + margin_y + max(0, max_height - box.h),
        )
        right = min(
            image.shape[1],
            box.x + box.w + margin_x + max(0, max_width - box.w),
        )
        search = image[top:bottom, left:right]
        matches: list[_TemplateMatch] = []
        for raw, template in candidates:
            template_height, template_width = template.shape[:2]
            if template_height > search.shape[0] or template_width > search.shape[1]:
                continue
            if use_color:
                gray_search = search
                gray_template = template
            else:
                gray_search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            scores = cv2.matchTemplate(
                gray_search,
                gray_template,
                cv2.TM_CCOEFF_NORMED,
            )
            for iteration in range(limit):
                _, score, _, location = cv2.minMaxLoc(scores)
                x, y = location
                collector = getattr(self._diagnostic_local, "collector", None)
                diagnostic_record: dict[str, object] | None = None
                if isinstance(collector, list) and len(collector) < 4096:
                    diagnostic_record = {
                        "field": str(getattr(region, "name", "unknown")),
                        "label": str(raw.get("label", "")),
                        "kind": str(raw.get("kind", "")),
                        "source_role": str(raw.get("source_role", "")),
                        "source": f"template:{raw.get('file', '')}",
                        "peak_index": int(iteration),
                        "score": float(score),
                        "threshold": float(threshold),
                        "search_box": [
                            int(left),
                            int(top),
                            int(right - left),
                            int(bottom - top),
                        ],
                        "roi_box": [int(box.x), int(box.y), int(box.w), int(box.h)],
                        "peak_location": [int(left + x), int(top + y)],
                        "match_box": [
                            int(left + x),
                            int(top + y),
                            int(template_width),
                            int(template_height),
                        ],
                        "accepted": False,
                        "rejection_reason": "pending_evaluation",
                    }
                    collector.append(diagnostic_record)
                if score < threshold:
                    if diagnostic_record is not None:
                        diagnostic_record["rejection_reason"] = "below_threshold"
                    break
                candidate = search[y : y + template_height, x : x + template_width]
                if (
                    use_color
                    and str(raw.get("kind", "")) == "rank"
                    and str(raw.get("label", "")) in {"small_joker", "big_joker"}
                    and not self._joker_colors_compatible(template, candidate)
                ):
                    if diagnostic_record is not None:
                        diagnostic_record["rejection_reason"] = "joker_color_mismatch"
                    self._suppress_match_score(
                        scores,
                        x,
                        y,
                        template_width,
                        template_height,
                    )
                    continue
                center_x = left + x + template_width / 2
                center_y = top + y + template_height / 2
                if not (
                    box.x - margin_x <= center_x <= box.x + box.w + margin_x
                    and box.y - margin_y <= center_y <= box.y + box.h + margin_y
                ):
                    if diagnostic_record is not None:
                        diagnostic_record["rejection_reason"] = "center_outside_roi"
                    self._suppress_match_score(
                        scores,
                        x,
                        y,
                        template_width,
                        template_height,
                    )
                    continue
                matches.append(
                    _TemplateMatch(
                        label=str(raw.get("label", "")),
                        kind=str(raw.get("kind", "")),
                        source_role=str(raw.get("source_role", "")),
                        source=f"template:{raw.get('file', '')}",
                        score=float(score),
                        x=left + x,
                        y=top + y,
                        w=template_width,
                        h=template_height,
                    )
                )
                if diagnostic_record is not None:
                    diagnostic_record["accepted"] = True
                    diagnostic_record["rejection_reason"] = None
                self._suppress_match_score(
                    scores,
                    x,
                    y,
                    template_width,
                    template_height,
                )
        return sorted(matches, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _suppress_match_score(
        scores: np.ndarray,
        x: int,
        y: int,
        template_width: int,
        template_height: int,
    ) -> None:
        radius_x = max(6, min(16, template_width // 3))
        radius_y = max(6, min(16, template_height // 3))
        sup_left = max(0, x - radius_x)
        sup_top = max(0, y - radius_y)
        sup_right = min(scores.shape[1], x + radius_x + 1)
        sup_bottom = min(scores.shape[0], y + radius_y + 1)
        scores[sup_top:sup_bottom, sup_left:sup_right] = -1.0

    @staticmethod
    def _joker_colors_compatible(template: np.ndarray, candidate: np.ndarray) -> bool:
        """Reject a structurally similar Joker template with the wrong colour class."""
        if (
            template.ndim != 3
            or candidate.ndim != 3
            or template.shape[2] < 3
            or candidate.shape[2] < 3
            or candidate.size == 0
        ):
            return True

        def profile(image: np.ndarray) -> tuple[float, float]:
            blue, green, red = cv2.split(image[:, :, :3])
            red_pixels = (
                (red > 80)
                & (red > green * 1.35)
                & (red > blue * 1.35)
            )
            dark_pixels = np.maximum(np.maximum(blue, green), red) < 100
            return float(np.mean(red_pixels)), float(np.mean(dark_pixels))

        template_red, template_dark = profile(template)
        candidate_red, candidate_dark = profile(candidate)
        if template_red >= 0.04 and template_red > template_dark:
            return candidate_red >= max(0.02, template_red * 0.25)
        if template_dark >= 0.04:
            # Some black Joker artwork contains a small red decorative mark.
            # Keep darkness as the primary class signal and allow that
            # decoration, while still rejecting the mostly-red big Joker.
            return (
                candidate_red <= 0.10
                and candidate_dark >= max(0.03, template_dark * 0.25)
            )
        return True

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
