"""Scan-only unknown-suit reread regression evidence.

The examples model the left-player button occlusion seen in the manually
verified ``game_20260816_125402_1687ea`` recording.  They intentionally test
only post-projection repair: action-window construction remains owned by the
projector and the raw per-frame observations must remain immutable evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json

import cv2
import numpy as np

from daguandan_bridge.application.video_scan import (
    VideoActionScanner,
    VideoScanRequest,
    repair_scan_actions,
)


LEFT_COMPLETE = ("AC", "AS", "KC", "KS", "QD", "QS")
LEFT_OCCLUDED = ("A?", "A?", "KC", "KS", "QD", "QS")


def _observation(frame: int, cards: tuple[str, ...], *, confidence: float = 0.8) -> dict[str, object]:
    return {
        "frame_index": frame,
        "timestamp_ms": frame * 33,
        "decode_ok": True,
        "regions": {
            "left": {
                "cards": list(cards),
                "is_pass": False,
                "confidence": confidence,
            },
        },
    }


def _left_action(*frames: int) -> dict[str, object]:
    return {
        "action_id": 4,
        "actor": "left",
        "is_pass": False,
        "cards": list(LEFT_OCCLUDED),
        "best_frame": 170,
        "frame_start": 165,
        "frame_end": 202,
        "evidence_frames": list(frames),
        "uncertainty": ["unknown_suit"],
        "review_status": "needs_review",
        "source": "video_scan",
    }


def test_verified_game_left_occlusion_is_repaired_from_one_complete_frame_without_mutating_observations():
    """Frame 184 clears the button, so one complete local reread is enough."""

    observations = [
        _observation(165, ("QD", "K?", "A?", "A?"), confidence=0.71),
        _observation(170, LEFT_OCCLUDED, confidence=0.80),
        _observation(184, LEFT_COMPLETE, confidence=0.89),
        _observation(202, LEFT_COMPLETE, confidence=0.75),
    ]
    original_observations = deepcopy(observations)

    result = repair_scan_actions([_left_action(165, 170, 184, 202)], observations)

    assert observations == original_observations
    assert len(result) == 1
    action = result[0]
    assert action["action_id"] == 4
    assert action["cards_before_repair"] == list(LEFT_OCCLUDED)
    assert action["cards"] == list(LEFT_COMPLETE)
    assert action["best_frame"] == 184
    assert action["repair_status"] == "resolved"
    assert action["repair_frame"] == 184
    assert action["repair_evidence_frames"] == [165, 170, 184]
    assert action["uncertainty"] == []
    assert action["review_status"] == "unverified"


def test_unknown_suit_without_complete_same_rank_reading_stays_unresolved_and_never_guesses():
    observations = [
        _observation(165, LEFT_OCCLUDED, confidence=0.80),
        # This is a fully suited *different* rank multiset, so a suit-only
        # repair must not turn the pending A/K/Q hand into it.
        _observation(184, ("AC", "AS", "JC", "JS", "QD", "QS"), confidence=0.99),
    ]

    action = repair_scan_actions([_left_action(165, 184)], observations)[0]

    assert action["cards"] == list(LEFT_OCCLUDED)
    assert action["cards_before_repair"] == list(LEFT_OCCLUDED)
    assert action["repair_status"] == "unresolved"
    assert action["repair_frame"] is None
    assert action["repair_evidence_frames"] == [165]
    assert action["uncertainty"] == ["unknown_suit"]
    assert action["review_status"] == "needs_review"


def test_complete_reading_outside_projected_evidence_window_cannot_repair_action():
    observations = [
        _observation(165, LEFT_OCCLUDED, confidence=0.80),
        # It may be a later action.  A scanner repair must not cross the
        # projector's evidence boundary to borrow it.
        _observation(184, LEFT_COMPLETE, confidence=0.99),
    ]

    action = repair_scan_actions([_left_action(165)], observations)[0]

    assert action["cards"] == list(LEFT_OCCLUDED)
    assert action["repair_status"] == "unresolved"
    assert action["repair_frame"] is None


def test_pending_action_uses_real_second_read_with_unknown_suit_disabled():
    observations = [
        _observation(165, LEFT_OCCLUDED, confidence=0.80),
        _observation(184, LEFT_OCCLUDED, confidence=0.80),
    ]
    calls: list[tuple[str, int, bool]] = []

    def reread(seat: str, frame: int):
        calls.append((seat, frame, False))
        if frame == 184:
            return LEFT_COMPLETE, 0.91
        return LEFT_OCCLUDED, 0.80

    action = repair_scan_actions(
        [_left_action(165, 184)],
        observations,
        reread=reread,
    )[0]

    assert calls == [("left", 165, False), ("left", 184, False)]
    assert action["cards"] == list(LEFT_COMPLETE)
    assert action["repair_status"] == "resolved"
    assert action["repair_reason"] == "second_pass_complete_suits"
    assert action["repair_frame"] == 184
    assert action["suit_reread_mode"] == "allow_unknown_suit_false"
    assert action["suit_reread_attempt_frames"] == [165, 184]


def test_first_pass_resolved_action_with_unknown_history_still_runs_strict_reread():
    observations = [
        _observation(165, LEFT_OCCLUDED, confidence=0.80),
        _observation(184, LEFT_COMPLETE, confidence=0.89),
    ]
    action = _left_action(165, 184)
    action.update(
        cards=list(LEFT_COMPLETE),
        cards_before_repair=list(LEFT_OCCLUDED),
        repair_status="resolved",
        repair_frame=184,
        uncertainty=[],
        review_status="unverified",
    )
    calls: list[tuple[str, int, bool]] = []

    def reread(seat: str, frame: int):
        calls.append((seat, frame, False))
        return (LEFT_COMPLETE, 0.92) if frame == 184 else (LEFT_OCCLUDED, 0.80)

    resolved = repair_scan_actions([action], observations, reread=reread)[0]

    assert calls == [("left", 165, False), ("left", 184, False)]
    assert resolved["cards"] == list(LEFT_COMPLETE)
    assert resolved["repair_status"] == "resolved"
    assert resolved["repair_reason"] == "second_pass_complete_suits"


def test_first_pass_complete_suits_are_safe_fallback_when_strict_reread_fails():
    observations = [_observation(165, LEFT_OCCLUDED), _observation(184, LEFT_COMPLETE)]
    action = _left_action(165, 184)
    action.update(
        cards=list(LEFT_COMPLETE),
        cards_before_repair=list(LEFT_OCCLUDED),
        repair_status="resolved",
        repair_frame=184,
        uncertainty=[],
        review_status="unverified",
    )

    resolved = repair_scan_actions(
        [action], observations, reread=lambda _seat, _frame: None
    )[0]

    assert resolved["cards"] == list(LEFT_COMPLETE)
    assert resolved["repair_status"] == "resolved"
    assert resolved["repair_reason"] == "fallback_first_pass_complete_suits"


class _Capture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._position = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return 10.0
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        return 0.0

    def read(self):
        if self._position >= len(self._frames):
            return False, None
        frame = self._frames[self._position].copy()
        self._position += 1
        return True, frame

    def release(self) -> None:
        return None


class _SecondPassRecognition:
    def __init__(self) -> None:
        self.left_calls: list[tuple[int, bool]] = []

    def recognize(self, _frame, *, allow_unknown_suit: bool = False):
        return type("Opening", (), {
            "round_level": "6", "my_hand": (), "lead_player": None,
            "current_player": None, "unresolved_fields": (), "diagnostics": (), "buttons": (),
        })()

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_unknown_suit, allow_pass):
        marker = int(frame[0, 0, 0])
        if seat != "left":
            return type("Region", (), {
                "cards": (), "is_pass": False, "confidence": 0.9,
                "source": "fake", "diagnostics": (), "suit_options": (),
            })()
        self.left_calls.append((marker, bool(allow_unknown_suit)))
        if allow_unknown_suit:
            cards = LEFT_OCCLUDED if marker in {1, 2} else ()
        else:
            cards = LEFT_COMPLETE if marker == 2 else LEFT_OCCLUDED if marker == 1 else ()
        return type("Region", (), {
            "cards": cards, "is_pass": False, "confidence": 0.9,
            "source": "fake", "diagnostics": (), "suit_options": (),
        })()


def test_scanner_invokes_allow_unknown_suit_false_for_every_pending_window_frame(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "game.avi"
    video.write_bytes(b"capture-is-injected")
    frames = [np.full((4, 4, 3), marker, dtype=np.uint8) for marker in range(4)]
    recognition = _SecondPassRecognition()

    result = VideoActionScanner(
        recognition,
        video_capture_factory=lambda _path: _Capture(frames),
    ).scan(VideoScanRequest(video, tmp_path / "derived"))

    action = json.loads(result.action_trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert action["cards"] == list(LEFT_COMPLETE)
    assert action["repair_status"] == "resolved"
    assert action["repair_reason"] == "second_pass_complete_suits"
    assert action["suit_reread_attempt_frames"] == [1, 2]
    assert (1, False) in recognition.left_calls
    assert (2, False) in recognition.left_calls
