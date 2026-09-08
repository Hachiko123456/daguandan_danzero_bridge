from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.domain.recognition import PlayRegionResult, RecognitionAnnotation
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live.session_store import LiveSessionStore
from daguandan_bridge.live_v2.identity import FrameIdentity, Seat, VersionIdentity
from daguandan_bridge.live_v2.seat_tracker import SeatTracker
from daguandan_bridge.live_v2.vision_adapter import VisionAdapter
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService


HAND = (
    "10S", "3C", "4H", "4H", "5C", "5D", "6C", "6H", "7D",
    "8C", "8D", "8H", "9C", "9H", "9H", "9S", "AH", "AS",
    "JC", "JC", "JD", "JH", "KC", "KS", "QC", "QH", "small_joker",
)
TRUTH = (
    (Seat.LEFT, ("5C", "5H"), False, .86),
    (Seat.SELF, ("9C", "9H", "9H", "9S"), False, .89),
    (Seat.RIGHT, (), True, .99),
    (Seat.OPPOSITE, (), True, .99),
    (Seat.LEFT, (), True, .82),
    (Seat.SELF, ("3C", "4H", "4H", "5C", "5D", "6H"), False, .50),
    (Seat.RIGHT, ("KD", "KD", "KH", "KS"), False, .86),
    (Seat.OPPOSITE, (), True, .92),
    (Seat.LEFT, (), True, .82),
    (Seat.SELF, (), True, 1.0),
    (Seat.RIGHT, ("2H", "2S", "3D", "3D", "AD", "AH"), False, .89),
    (Seat.OPPOSITE, ("10H", "10H", "JD", "JS", "QC", "QD"), False, .83),
    (Seat.LEFT, (), True, 1.0),
    (Seat.SELF, ("JC", "JC", "JD", "JH"), False, .90),
    (Seat.RIGHT, (), True, .99),
    (Seat.OPPOSITE, (), True, 1.0),
    (Seat.LEFT, (), True, .82),
    (Seat.SELF, ("8C", "8D"), False, .89),
    (Seat.RIGHT, (), True, .99),
    (Seat.OPPOSITE, ("AC", "AC"), False, .89),
    (Seat.LEFT, (), True, .99),
    (Seat.SELF, (), True, 1.0),
    (Seat.RIGHT, ("10C", "6C", "6H", "7C", "9C"), False, .72),
    (Seat.OPPOSITE, (), True, .99),
    (Seat.LEFT, (), True, .99),
    (Seat.SELF, (), True, 1.0),
    (Seat.RIGHT, ("3H", "4D", "5D", "6S", "7H"), False, .87),
    (Seat.OPPOSITE, (), True, 1.0),
    (Seat.LEFT, (), True, .82),
    (Seat.SELF, (), True, 1.0),
)
RAW_FRAME235 = ("3C", "6H", "4H", "4H", "5D", "5C")


def _annotations(cards: tuple[str, ...], confidence: float):
    return tuple(
        RecognitionAnnotation(
            card, (index * 10, 2, 8, 12),
            .5 if card == "6H" and confidence == .5 else max(.8, confidence),
            "play",
        )
        for index, card in enumerate(cards)
    )


def test_first_thirty_truth_actions_include_and_commit_low_confidence_plays(tmp_path: Path) -> None:
    store = LiveSessionStore(tmp_path, "profile", session_id="turn-1-14")
    store.start({})
    rules = ProductionRuleSession(store)
    rules.initialize(
        round_level="6", hand=HAND, lead_player=Seat.LEFT,
        monotonic_ms=1, capture_generation=1,
    )
    adapter = VisionAdapter()
    trackers = {seat: SeatTracker(seat) for seat in Seat}
    sequence = 0
    low_confidence_turns = []
    turn6_candidate = None

    for turn, (seat, cards, is_pass, confidence) in enumerate(TRUTH, 1):
        candidate = None
        for _ in range(2):
            sequence += 1
            frame = FrameIdentity(
                "turn-1-14", 1, sequence, sequence * 100, "roi", "truth",
            )
            result = PlayRegionResult(
                seat.value, cards, is_pass, confidence,
                (f"truth_turn={turn}",),
                _annotations(cards, confidence),
                suit_options=tuple((card,) for card in cards),
            )
            observation = adapter.normalize_play_region(
                result, frame=frame, processing_ms=frame.captured_ms,
            )
            candidate = trackers[seat].ingest(
                observation, version=rules.version
            ) or candidate
        assert candidate is not None, f"turn {turn} produced no candidate"
        if candidate.confidence < .8:
            low_confidence_turns.append(turn)
        if turn == 6:
            turn6_candidate = candidate
        binding = rules.bind_generation(1)
        projected = binding.adapter.project(
            base_version=binding.version, candidates=(candidate,),
            processing_ms=candidate.processing_ms,
        )
        committed = binding.adapter.commit(
            expected_version=binding.version, actions=projected.actions,
        )
        assert committed.committed_actions, f"turn {turn} rule rejection"

        for _ in range(2):
            sequence += 1
            frame = FrameIdentity(
                "turn-1-14", 1, sequence, sequence * 100, "roi", "truth",
            )
            empty = adapter.normalize_play_region(
                PlayRegionResult(seat.value, (), False, 0.0, (), ()),
                frame=frame, processing_ms=frame.captured_ms,
                empty_confirmed=True,
            )
            assert trackers[seat].ingest(empty, version=rules.version) is None

    snapshot = rules.snapshot(captured_ms=sequence * 100)
    assert len(snapshot.play_history) == 30
    assert snapshot.current_seat is Seat.RIGHT
    assert low_confidence_turns == [6, 23]
    turn6 = snapshot.play_history[5]
    assert turn6.cards == TRUTH[5][1]
    assert turn6_candidate is not None
    assert "play_quality=audited_two_frame_eligible" in turn6_candidate.diagnostics
    assert "candidate_confidence=0.500" in turn6_candidate.diagnostics


def test_historical_frames_233_to_235_form_the_turn6_candidate() -> None:
    video = (
        PROFILES_ROOT
        / "tencent_daguandan/sessions/game_20260814_004447_aab3dc/video/game.avi"
    )
    if not video.is_file():
        pytest.skip(f"missing historical video: {video}")
    recognizer = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT), TemplateService(PROFILES_ROOT),
        diagnostic_tracing=False,
    )
    adapter, tracker = VisionAdapter(), SeatTracker(Seat.SELF)
    version = VersionIdentity("historical-turn6", 1, 5, 5, 5)
    capture = cv2.VideoCapture(str(video))
    candidate = None
    try:
        for index in (233, 234, 235):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            assert ok and image is not None
            result = recognizer.recognize_play_region(
                image, "self", wild_rank="6",
                allow_pass=True, allow_unknown_suit=True,
            )
            frame = FrameIdentity(
                "historical-turn6", 1, index, index * 100,
                "profile", "historical-avi",
            )
            observation = adapter.normalize_play_region(
                result, frame=frame, processing_ms=index * 100,
            )
            candidate = tracker.ingest(observation, version=version) or candidate
    finally:
        capture.release()
    assert candidate is not None
    assert candidate.cards == RAW_FRAME235
    assert candidate.confidence == .5
    assert candidate.first_frame.frame_sequence == 233
    assert candidate.last_frame.frame_sequence == 234
    assert "play_annotation_count=6" in candidate.diagnostics
