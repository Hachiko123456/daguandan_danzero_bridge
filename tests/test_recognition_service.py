from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.image_io import read_image_unicode
from daguandan_bridge.models import Box
from daguandan_bridge.recognition_service import (
    RecognizedEvent,
    ScreenshotRecognitionService,
)
from daguandan_bridge.template_service import TemplateService


PROFILE_ROOT = PROFILES_ROOT / "tencent_daguandan"


def _paste_template(canvas: np.ndarray, relative_file: str, x: int, y: int) -> None:
    template = read_image_unicode(PROFILE_ROOT / relative_file)
    height, width = template.shape[:2]
    canvas[y : y + height, x : x + width] = template


def test_template_recognizer_reads_level_hand_timer_and_lead(tmp_path):
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/rank/2_level.png", 80, 35)
    _paste_template(image, "templates/rank/2_hand.png", 40, 510)
    _paste_template(image, "templates/suit/spade_hand.png", 43, 550)
    _paste_template(image, "templates/rank/3_hand.png", 110, 510)
    _paste_template(image, "templates/suit/heart_hand.png", 113, 550)
    _paste_template(image, "templates/timer/active.png", 450, 220)
    _paste_template(image, "templates/status/first_play.png", 580, 240)

    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    result = service.recognize(image)

    assert result.round_level == "2"
    assert result.wild_rank == "2"
    assert result.current_player == "self"
    assert result.lead_player == "self"
    assert "2S" in result.my_hand
    assert "3H" in result.my_hand
    assert result.sources["my_hand"].startswith("template:")
    assert any(annotation.label == "2S" for annotation in result.annotations)


def test_template_recognizer_leaves_unknown_fields_for_manual_confirmation():
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    result = service.recognize(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert result.my_hand == ()
    assert result.round_level is None
    assert result.current_player is None
    assert result.unresolved_fields


def test_real_screenshot_recognizes_full_hand_and_finds_first_play_marker():
    image_path = PROFILE_ROOT / "screenshots" / "game_20260804_005737" / "000011.png"
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    expected_cards = (
        "big_joker", "big_joker", "AH", "AD", "KH", "KD", "KC", "KC",
        "QH", "QD", "QS", "JC", "JS", "10H", "10S", "10S", "9H",
        "8D", "8C", "7D", "7S", "5H", "5H", "5D", "5D", "4H", "4C",
    )
    assert Counter(result.my_hand) == Counter(expected_cards)
    assert result.lead_player == "self"
    assert any(annotation.label == "AH" for annotation in result.annotations)


def test_real_play_screenshot_keeps_spade_four_and_wild_heart_two():
    image_path = PROFILE_ROOT / "screenshots" / "game_20260804_005737" / "000031.png"
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    left_event = next(event for event in result.events if event.player == "left")
    assert left_event.cards == ("2S", "3S", "4S", "2H", "6S")


def test_real_screenshot_adds_boxes_for_all_detected_statuses():
    image_path = PROFILE_ROOT / "screenshots" / "game_20260804_005737" / "000031.png"
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    labels = [annotation.label for annotation in result.annotations]
    categories = {annotation.category for annotation in result.annotations}
    assert "2" in labels
    assert "active" in labels
    assert "first_play" in labels
    assert labels.count("passed") >= 2
    assert {"level", "timer", "status"}.issubset(categories)
    assert all(
        width > 0 and height > 0
        for _x, _y, width, height in (
            annotation.box for annotation in result.annotations
        )
    )


def test_level_suit_template_is_used_for_wild_card_in_play_region(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("screenshots"),
    )
    image_path = PROFILE_ROOT / "screenshots" / "game_20260804_005737" / "000031.png"
    image = read_image_unicode(image_path)
    template_service = TemplateService(root)
    template_service.save_template(
        image,
        kind="suit",
        label="heart",
        box=Box(267, 236, 28, 30),
        source_image=image_path.name,
        source_role="level",
    )

    result = ScreenshotRecognitionService(
        AnnotationService(root),
        template_service,
    ).recognize(image)

    left_event = next(event for event in result.events if event.player == "left")
    assert "2H" in left_event.cards


def test_first_play_global_fallback_rejects_center_screen_match():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/status/first_play.png", 600, 300)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image)

    assert result.lead_player is None
    assert "lead_player" in result.unresolved_fields


def test_real_first_play_screenshot_does_not_create_a_self_play_event():
    image_path = PROFILE_ROOT / "screenshots" / "game_20260804_005737" / "000011.png"
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    assert result.current_player == "self"
    assert result.lead_player == "self"
    assert result.events == ()


def test_template_recognizer_reports_image_elapsed_time():
    image_path = PROFILE_ROOT / "screenshots" / "game_20260804_005737" / "000011.png"
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    assert result.elapsed_ms > 0


def test_template_recognizer_detects_action_button_values_and_boxes():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/button/cannot_beat.png", 600, 240)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image)

    assert "cannot_beat" in result.buttons
    assert any(
        annotation.category == "button" and annotation.label == "cannot_beat"
        for annotation in result.annotations
    )


def test_play_templates_still_recognize_a_real_card_in_play_region():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/rank/A_play.png", 520, 236)
    _paste_template(image, "templates/suit/club_play.png", 523, 270)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image)

    assert any(
        event.player == "self" and event.cards == ("AC",) and not event.is_pass
        for event in result.events
    )


def test_recognized_events_follow_counter_clockwise_order_from_right_lead():
    events = [
        RecognizedEvent("self", ("3S",), False, 0.9, "test"),
        RecognizedEvent("left", (), True, 0.9, "test"),
        RecognizedEvent("right", ("4S",), False, 0.9, "test"),
        RecognizedEvent("opposite", (), True, 0.9, "test"),
    ]

    ordered = ScreenshotRecognitionService._order_events(events, "right")

    assert [event.player for event in ordered] == [
        "right",
        "opposite",
        "left",
        "self",
    ]


def test_templates_are_loaded_once_until_explicit_reload(monkeypatch):
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    calls = 0
    original = service.template_service.list_templates

    def counted():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(service.template_service, "list_templates", counted)

    service.recognize(image)
    service.recognize(image)
    service.reload_templates()
    service.recognize(image)

    assert calls == 2


def test_targeted_play_recognition_only_returns_expected_seat():
    image_path = (
        PROFILE_ROOT / "screenshots" / "game_20260804_005737" / "000031.png"
    )
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize_play_region(image_path, "left", wild_rank="2")

    assert result.player == "left"
    assert result.cards == ("2S", "3S", "4S", "2H", "6S")
    assert not result.is_pass
    assert all(annotation.category == "play" for annotation in result.annotations)


def test_fast_signals_are_narrow_and_keep_expected_player():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/timer/active.png", 120, 180)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    signals = service.recognize_fast_signals(image, "left")

    assert signals.expected_player == "left"
    assert signals.active_player == "left"
    assert not signals.pass_visible
