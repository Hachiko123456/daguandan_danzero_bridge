from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.image_io import read_image_unicode
from daguandan_bridge.live.card_uncertainty import normalized_suit_options
from daguandan_bridge.live.consensus import BurstConsensus, ConsensusContext
from daguandan_bridge.models import Box
from daguandan_bridge.recognition_service import (
    OpeningSignal,
    RecognizedEvent,
    ScreenshotRecognitionService,
    _TemplateMatch,
)
from daguandan_bridge.template_service import TemplateService


PROFILE_ROOT = PROFILES_ROOT / "tencent_daguandan"


def _paste_template(canvas: np.ndarray, relative_file: str, x: int, y: int) -> None:
    template = read_image_unicode(PROFILE_ROOT / relative_file)
    height, width = template.shape[:2]
    canvas[y : y + height, x : x + width] = template


def _required_screenshot(session: str, filename: str) -> Path:
    path = PROFILE_ROOT / "screenshots" / session / filename
    if not path.is_file():
        pytest.skip(f"缺少真实截图样本：{path}")
    return path


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


def test_table_anchor_uses_one_frame_at_the_configured_085_threshold():
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _paste_template(image, "templates/anchor/table_anchor_1.png", 1150, 2)

    assert service.recognize_table_anchor(image) >= 0.85
    assert service.recognize_table_anchor(
        np.zeros((720, 1280, 3), dtype=np.uint8)
    ) < 0.85


def test_initial_hand_keeps_a_rank_when_its_suit_is_occluded():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/rank/2_hand.png", 40, 510)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image, allow_unknown_suit=True)

    assert "2?" in result.my_hand


def test_hand_color_gate_keeps_a_black_suit_when_diamond_scores_higher(monkeypatch):
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    rank = _TemplateMatch("9", "rank", "hand", "test:9", 0.92, 40, 510, 35, 50)
    wrong_diamond = _TemplateMatch(
        "diamond", "suit", "hand", "test:diamond", 0.99, 43, 550, 30, 32
    )
    spade = _TemplateMatch("spade", "suit", "hand", "test:spade", 0.84, 43, 550, 31, 31)
    responses = iter(([rank], [], [wrong_diamond, spade]))
    monkeypatch.setattr(service, "_matches_for_region", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(service, "_unknown_suit_options", lambda *_args: ("S", "C"))
    monkeypatch.setattr(service, "_black_suit_shape_code", lambda *_args: "S")

    cards, *_rest = service._recognize_cards(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        object(),
        (),
        source_roles={"hand"},
        rank_threshold=service._HAND_RANK_THRESHOLD,
        suit_threshold=service._HAND_SUIT_THRESHOLD,
        allow_unknown_suit=True,
    )

    assert cards == ("9S",)


def test_hand_duplicate_suit_is_preserved_as_bounded_unknown_before_drop(monkeypatch):
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    ranks = [
        _TemplateMatch("9", "rank", "hand", "test:9", score, x, 510, 35, 50)
        for x, score in ((40, 0.92), (90, 0.88), (140, 0.73))
    ]
    diamonds = [
        _TemplateMatch("diamond", "suit", "hand", "test:diamond", 0.90, x + 3, 550, 30, 32)
        for x, _score in ((40, 0.92), (90, 0.88), (140, 0.73))
    ]
    responses = iter((ranks, [], diamonds))
    monkeypatch.setattr(service, "_matches_for_region", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(service, "_unknown_suit_options", lambda *_args: ("H", "D"))

    cards, _score, _source, diagnostics, _annotations, suit_options = service._recognize_cards(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        object(),
        (),
        source_roles={"hand"},
        rank_threshold=service._HAND_RANK_THRESHOLD,
        suit_threshold=service._HAND_SUIT_THRESHOLD,
        allow_unknown_suit=True,
    )

    assert cards == ("9D", "9D", "9?")
    assert suit_options == (("D",), ("D",), ("H", "D"))
    assert any("按候选花色保留" in item for item in diagnostics)


def test_effect_overlay_just_outside_play_roi_still_gates_the_action():
    """Effects may overhang the cards, unlike ordinary play templates."""

    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    # ``right_play`` ends at y=271.  This overlay's centre is below that
    # boundary, mirroring the real straight effect recorded in 3ccf81/F258.
    _paste_template(image, "templates/effect/consecutive_pairs.png", 800, 250)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    signal = service.recognize_fast_signals(image, "right")

    assert signal.effect_visible


def test_jokers_use_lower_rank_threshold(monkeypatch):
    import cv2 as cv2_module

    from daguandan_bridge import recognition_service as rs_module

    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    def fake_match(search, template, method):
        return np.full((3, 3), 0.62, dtype=np.float32)

    monkeypatch.setattr(rs_module.cv2, "matchTemplate", fake_match)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image)

    assert any(
        card in {"small_joker", "big_joker"} for card in result.my_hand
    )


def test_joker_color_gate_rejects_a_red_template_on_a_black_joker():
    """A structural template hit must not turn a black small Joker into a big Joker."""
    small_template = np.full((24, 18, 3), 255, dtype=np.uint8)
    small_template[4:20, 4:14] = (20, 20, 20)
    big_template = np.full((24, 18, 3), 255, dtype=np.uint8)
    big_template[4:20, 4:14] = (25, 25, 230)
    black_joker = small_template.copy()
    red_joker = big_template.copy()

    assert ScreenshotRecognitionService._joker_colors_compatible(
        small_template,
        black_joker,
    )
    assert not ScreenshotRecognitionService._joker_colors_compatible(
        big_template,
        black_joker,
    )
    assert ScreenshotRecognitionService._joker_colors_compatible(
        big_template,
        red_joker,
    )


def test_black_suit_shape_resolver_overrides_a_tied_gray_template_match():
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    cases = (
        ("spade_play.png", "S"),
        ("club_play.png", "C"),
    )
    for filename, expected in cases:
        image = read_image_unicode(PROFILE_ROOT / "templates" / "suit" / filename)
        selected = _TemplateMatch(
            label="club" if expected == "S" else "spade",
            kind="suit",
            source_role="play",
            source="synthetic:tied-gray-match",
            score=0.90,
            x=0,
            y=0,
            w=image.shape[1],
            h=image.shape[0],
        )

        assert service._black_suit_shape_code(image, selected) == expected


def test_latest_game_clear_black_suits_do_not_downgrade_to_unknown():
    """Regression for game_20260810_011602_61367b frame 82, if retained locally."""
    video_path = (
        PROFILE_ROOT
        / "sessions"
        / "game_20260810_011602_61367b"
        / "video"
        / "game.avi"
    )
    if not video_path.is_file():
        pytest.skip(f"missing local regression video: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 82)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        pytest.skip("unable to decode latest local regression frame")

    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    result = service.recognize_play_region(
        frame,
        "self",
        wild_rank="10",
        allow_unknown_suit=True,
        allow_pass=False,
    )

    assert result.cards == ("AS", "2H", "3C", "4C", "5S")
    assert result.suit_options == (("S",), ("H",), ("C",), ("C",), ("S",))


def test_regression_black_small_jokers_are_not_labeled_as_big_jokers():
    """Keep the recorded 2026-08-09 opening hand as a no-write regression case."""
    import cv2

    video_path = (
        PROFILE_ROOT
        / "sessions"
        / "game_20260809_164356_9abb9a"
        / "video"
        / "game.avi"
    )
    if not video_path.is_file():
        pytest.skip(f"缺少本地回归录像：{video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        pytest.skip("本地回归录像无法读取首帧")

    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    result = service.recognize(frame)

    assert result.my_hand.count("small_joker") == 2
    assert result.my_hand.count("big_joker") == 0


def test_first_play_uses_the_matching_seat_region():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/status/first_play.png", 1110, 220)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    assert service.recognize_lead_player(image) == "right"


def test_first_play_marker_requires_at_least_080_confidence(monkeypatch):
    """A sub-0.8 first-play hit must not select a lead seat."""
    from daguandan_bridge import recognition_service as rs_module

    def fake_match(search, _template, _method):
        # ``first_play_right`` is 163 x 156 in the Tencent profile.  Make
        # only that ROI look like the known pre-doubling false positive.
        score = 0.79 if search.shape[:2] == (156, 163) else 0.0
        return np.full((1, 1), score, dtype=np.float32)

    monkeypatch.setattr(rs_module.cv2, "matchTemplate", fake_match)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    assert service.recognize_lead_player(np.zeros((720, 1280, 3), dtype=np.uint8)) is None


def test_template_recognizer_leaves_unknown_fields_unresolved():
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    result = service.recognize(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert result.my_hand == ()
    assert result.round_level is None
    assert result.current_player is None
    assert result.unresolved_fields


def test_opening_signal_collects_marker_timer_and_super_double_without_committing_lead():
    """Opening recognition exposes raw evidence; the state machine owns the decision."""
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/status/first_play.png", 580, 240)
    _paste_template(image, "templates/timer/active.png", 450, 220)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    signal = service.recognize_opening_signal(image)

    assert isinstance(signal, OpeningSignal)
    assert signal.marker_player == "self"
    assert signal.active_player == "self"

    double_image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(double_image, "templates/button/super_double.png", 520, 250)
    double_signal = service.recognize_opening_signal(double_image)

    assert double_signal.super_double_visible is True

    normal_double_image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(normal_double_image, "templates/button/double.png", 520, 250)

    normal_double_signal = service.recognize_opening_signal(normal_double_image)

    assert normal_double_signal.super_double_visible is True
    assert service.recognize_super_double_visible(normal_double_image) is True


def test_fast_signals_recognize_continue_game_as_an_end_control():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/button/continue_game.png", 520, 250)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    signals = service.recognize_fast_signals(image, "self")

    assert signals.game_end_control == "continue_game"


def test_fast_signals_recognize_change_table_as_an_end_control():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/button/change_table.png", 520, 250)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    signals = service.recognize_fast_signals(image, "self")

    assert signals.game_end_control == "change_table"


@pytest.mark.parametrize(
    ("template", "position", "expected"),
    (
        ("templates/button/continue_game.png", (698, 586), "continue_game"),
        ("templates/button/change_table.png", (355, 577), "change_table"),
    ),
)
def test_fast_signals_recognize_bottom_end_controls(template, position, expected):
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, template, *position)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    signals = service.recognize_fast_signals(image, "self")

    assert signals.game_end_control == expected


def test_latest_first_play_recovers_clear_black_suits_and_keeps_occluded_rank():
    video_path = (
        PROFILE_ROOT
        / "sessions"
        / "game_20260810_000418_3ccf81"
        / "video"
        / "game.avi"
    )
    if not video_path.is_file():
        pytest.skip(f"缺少最新对局录像：{video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 72)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        pytest.skip("无法读取最新对局的首手帧")

    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )
    result = service.recognize_play_region(
        frame,
        "left",
        wild_rank="7",
        allow_unknown_suit=True,
        allow_pass=False,
    )

    assert result.cards == ("3H", "3S", "4D", "4S", "5?", "7H")
    assert len(result.cards) == 6
    assert result.suit_options[1] == ("S",)
    assert result.suit_options[3] == ("S",)
    assert result.suit_options[4] == ("H", "D")
    initial_hand = (
        "10D", "2D", "2H", "2S", "3C", "3C", "3H", "3S", "4H", "4S",
        "5D", "5H", "6D", "6H", "6S", "7D", "8H", "8S", "9C", "AC", "JC",
        "JH", "KS", "KS", "QH", "big_joker", "big_joker",
    )
    assert BurstConsensus.validate_candidate(
        False,
        result.cards,
        ConsensusContext(
            level_rank="7",
            remaining_cards=27,
            allow_pass=False,
            known_cards=initial_hand,
            known_suit_options=normalized_suit_options(initial_hand),
        ),
        suit_options=result.suit_options,
    ) == ""

    # The annotation/replay single-frame path must retain the same rank-only
    # card instead of showing five cards merely because the suit is covered.
    single_frame = service.recognize(frame, allow_unknown_suit=True)
    left_event = next(event for event in single_frame.events if event.player == "left")
    assert left_event.cards == ("3H", "3S", "4D", "4S", "5?", "7H")


def test_latest_final_screen_recognizes_bottom_end_control():
    video_path = (
        PROFILE_ROOT
        / "sessions"
        / "game_20260810_000418_3ccf81"
        / "video"
        / "game.avi"
    )
    if not video_path.is_file():
        pytest.skip(f"缺少最新对局录像：{video_path}")
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 1312)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        pytest.skip("无法读取最新对局的结算帧")

    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    assert service.recognize_fast_signals(frame, "self").game_end_control == "continue_game"


def test_opening_signal_carries_change_table_end_control():
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/button/change_table.png", 520, 250)
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    signal = service.recognize_opening_signal(image)

    assert signal.game_end_control == "change_table"


def test_real_screenshot_recognizes_full_hand_and_finds_first_play_marker():
    image_path = _required_screenshot("game_20260804_005737", "000011.png")
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
    image_path = _required_screenshot("game_20260804_005737", "000031.png")
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    left_event = next(event for event in result.events if event.player == "left")
    assert left_event.cards == ("2S", "3S", "4S", "2H", "6S")


def test_real_opposite_big_joker_play_is_recognized():
    image_path = (
        PROFILE_ROOT / "screenshots" / "game_20260804_214822" / "000033.png"
    )
    if not image_path.is_file():
        pytest.skip("缺少真实截图样本")
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    opposite_event = next(
        event for event in result.events if event.player == "opposite"
    )
    assert opposite_event.cards == ("big_joker",)


def test_real_screenshot_adds_boxes_for_all_detected_statuses():
    image_path = _required_screenshot("game_20260804_005737", "000031.png")
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
        ignore=shutil.ignore_patterns("screenshots", "sessions"),
    )
    image_path = _required_screenshot("game_20260804_005737", "000031.png")
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


def test_level_card_without_matched_suit_defaults_to_wild_heart(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("screenshots", "sessions"),
    )
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _paste_template(image, "templates/rank/2_level.png", 80, 35)
    _paste_template(image, "templates/rank/2_hand.png", 40, 510)

    service = ScreenshotRecognitionService(
        AnnotationService(root),
        TemplateService(root),
    )
    result = service.recognize(image)

    assert result.round_level == "2"
    assert result.wild_rank == "2"
    assert Counter(result.my_hand) == Counter({"2H"})
    assert "my_hand" not in result.unresolved_fields


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
    image_path = _required_screenshot("game_20260804_005737", "000011.png")
    service = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT),
        TemplateService(PROFILES_ROOT),
    )

    result = service.recognize(image_path)

    assert result.current_player == "self"
    assert result.lead_player == "self"
    assert result.events == ()


def test_template_recognizer_reports_image_elapsed_time():
    image_path = _required_screenshot("game_20260804_005737", "000011.png")
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
        _required_screenshot("game_20260804_005737", "000031.png")
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
