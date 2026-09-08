"""Optional private-video regressions; no images/video copied into the repo."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.recognition_service import ScreenshotRecognitionService, _TemplateMatch


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data/profiles/tencent_daguandan"
VIDEO = PROFILE / "sessions/game_20260814_004447_aab3dc/video/game.avi"


@pytest.fixture
def joker_service():
    raw = json.loads((PROFILE / "templates_config.json").read_text("utf-8"))["templates"]
    return ScreenshotRecognitionService(
        AnnotationService(PROFILE.parent),
        SimpleNamespace(profile_root=PROFILE, list_templates=lambda: raw),
        diagnostic_tracing=False,
    )


def read_frame(index):
    if not VIDEO.is_file():
        pytest.skip("optional private historical video is not installed")
    capture = cv2.VideoCapture(str(VIDEO))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        assert ok, f"source video frame {index} must decode"
        return image
    finally:
        capture.release()


@pytest.mark.parametrize("index", [1244, 1246, 1250, 1254])
def test_real_golden_big_joker_full_word_evidence_passes_unchanged_action_gate(joker_service, index):
    result = joker_service.recognize_play_region(read_frame(index), "left", wild_rank="6", allow_unknown_suit=True)
    assert result.cards == ("big_joker",)
    assert result.confidence >= .80
    assert "small_joker" not in result.cards


@pytest.mark.parametrize("index", [1311, 1312, 1314, 1315])
def test_real_blue_small_joker_uses_its_own_full_word_at_unchanged_gate(joker_service, monkeypatch, index):
    image = read_frame(index)
    result = joker_service.recognize_play_region(image, "left", wild_rank="6", allow_unknown_suit=True)
    assert result.cards == ("small_joker",)
    assert result.confidence >= .80
    assert "big_joker" not in result.cards
    with monkeypatch.context() as patch:
        patch.setattr(joker_service, "_confirm_small_joker_words", lambda image, matches, templates: matches)
        original = joker_service.recognize_play_region(image, "left", wild_rank="6", allow_unknown_suit=True)
    assert original.cards == ("small_joker",)
    assert .60 <= original.confidence < .80
    template = next(template for raw, template in joker_service._templates()
                    if raw.get("file") == "templates/rank/small_joker_play_003.png")
    # Independently measure the full five-letter word at the observed face
    # location (149, 192), allowing only the same +/-2px alignment tolerance.
    scores = cv2.matchTemplate(cv2.cvtColor(image[194:298, 155:179], cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(template[4:104, 8:28], cv2.COLOR_BGR2GRAY),
                               cv2.TM_CCOEFF_NORMED)
    assert result.confidence == pytest.approx(float(scores.max()), abs=1e-7)


@pytest.mark.parametrize("index,seat,expected", [
    (1004, "right", "big_joker"), (1010, "right", "big_joker"),
    (1229, "self", "small_joker"), (1230, "self", "small_joker"),
    (1311, "left", "small_joker"), (1314, "left", "small_joker"),
])
def test_real_red_and_black_joker_labels_are_mutually_exclusive(joker_service, index, seat, expected):
    result = joker_service.recognize_play_region(read_frame(index), seat, wild_rank="6", allow_unknown_suit=True)
    assert result.cards == (expected,)


@pytest.mark.parametrize("index", [1178, 1191, 1206, 1219, 1230, 1235])
def test_real_left_pass_blank_and_red_suit_cards_do_not_become_jokers(joker_service, index):
    result = joker_service.recognize_play_region(read_frame(index), "left", wild_rank="6", allow_unknown_suit=True)
    assert not {"big_joker", "small_joker"}.intersection(result.cards)


@pytest.mark.parametrize("index", [1206, 1219, 1311, 1314])
def test_full_word_refinement_does_not_promote_a_fake_big_candidate_on_red_card_or_black_joker(joker_service, index):
    match = _TemplateMatch("big_joker", "rank", "play",
                           "template:templates/rank/big_joker_play_002.png", .67, 145, 192, 50, 108)
    refined = joker_service._confirm_big_joker_words(read_frame(index), [match], joker_service._templates())
    assert refined[0].score == .67


def test_word_refinement_requires_original_joker_admission_and_never_scales_score(joker_service):
    assert joker_service._JOKER_RANK_THRESHOLD == .60
    weak = _TemplateMatch("big_joker", "rank", "play",
                          "template:templates/rank/big_joker_play_002.png", .59, 145, 192, 50, 108)
    image = read_frame(1244)
    assert joker_service._confirm_big_joker_words(image, [weak], joker_service._templates())[0] == weak
    # The value is the actual full-word grayscale NCC at the admitted location,
    # not a constant .80/.95 or a multiplier applied to the weak whole-card score.
    admitted = _TemplateMatch(**{**vars(weak), "score": .67})
    refined = joker_service._confirm_big_joker_words(image, [admitted], joker_service._templates())[0]
    assert .83 < refined.score < .85
    assert refined.source.endswith("+full-joker-word")


@pytest.mark.parametrize("index,x", [
    (1244, 149), (1250, 149), (1004, 1044), (1010, 1044),
    (1206, 149), (1219, 149), (1230, 149), (1178, 149), (1191, 149),
])
def test_fake_small_candidate_on_red_golden_big_or_ordinary_card_is_not_promoted(joker_service, index, x):
    fake = _TemplateMatch("small_joker", "rank", "play",
                         "template:templates/rank/small_joker_play_003.png", .74, x, 192, 39, 108)
    refined = joker_service._confirm_small_joker_words(read_frame(index), [fake], joker_service._templates())
    assert refined[0] == fake


def test_small_word_refinement_requires_same_label_source_original_admission_and_actual_score(joker_service):
    image = read_frame(1311)
    admitted = _TemplateMatch("small_joker", "rank", "play",
                             "template:templates/rank/small_joker_play_003.png", .74, 149, 192, 39, 108)
    refined = joker_service._confirm_small_joker_words(image, [admitted], joker_service._templates())[0]
    assert refined.label == "small_joker" and .84 < refined.score < .86
    assert refined.source == admitted.source + "+full-joker-word"
    for altered in (
        _TemplateMatch(**{**vars(admitted), "score": .59}),
        _TemplateMatch(**{**vars(admitted), "score": .80}),
        _TemplateMatch(**{**vars(admitted), "score": .95}),
        _TemplateMatch(**{**vars(admitted), "label": "big_joker"}),
        _TemplateMatch(**{**vars(admitted), "source": "template:templates/rank/big_joker_play_002.png"}),
        _TemplateMatch(**{**vars(admitted), "source_role": "hand"}),
        _TemplateMatch(**{**vars(admitted), "x": 125}),
    ):
        assert joker_service._confirm_small_joker_words(image, [altered], joker_service._templates())[0] == altered
    assert joker_service._confirm_small_joker_words(image, [], joker_service._templates()) == []


@pytest.mark.parametrize("value", [0, 255])
def test_blank_image_never_promotes_an_existing_weak_joker(joker_service, value):
    image = read_frame(1311)
    image[:] = value
    for label, source, x, width in (
        ("big_joker", "big_joker_play_002.png", 145, 50),
        ("small_joker", "small_joker_play_003.png", 149, 39),
    ):
        fake = _TemplateMatch(label, "rank", "play", f"template:templates/rank/{source}",
                              .74, x, 192, width, 108)
        refined = joker_service._confirm_joker_word_matches(image, [fake], joker_service._templates(), label=label)
        assert refined == [fake]
