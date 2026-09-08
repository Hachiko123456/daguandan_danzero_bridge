from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.annotation_service import AnnotationService  # noqa: E402
from daguandan_bridge.config import PROFILES_ROOT  # noqa: E402
from daguandan_bridge.live.consensus import BurstConsensus, ConsensusContext  # noqa: E402
from daguandan_bridge.recognition_service import ScreenshotRecognitionService  # noqa: E402
from daguandan_bridge.storage import atomic_write_json  # noqa: E402
from daguandan_bridge.template_service import TemplateService  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read one real session frame as a fast-cycle recovery diagnostic."
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--baseline-frame", type=int, required=True)
    parser.add_argument("--expected", default="right")
    parser.add_argument("--wild-rank", default="4")
    parser.add_argument("--table-card", default="AH")
    parser.add_argument("--legal-table-card", default="3C")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    session = args.session.resolve(strict=True)
    video = session / "video" / "game.avi"
    baseline = _read_frame(video, args.baseline_frame)
    frame = _read_frame(video, args.frame)
    recognizer = ScreenshotRecognitionService(
        AnnotationService(PROFILES_ROOT, "tencent_daguandan"),
        TemplateService(PROFILES_ROOT, "tencent_daguandan"),
    )
    fast = recognizer.recognize_fast_signals(frame, args.expected)
    observations = {
        seat: recognizer.recognize_play_region(
            frame,
            seat,
            wild_rank=args.wild_rank,
            allow_pass=True,
            allow_unknown_suit=True,
        )
        for seat in ("right", "opposite", "left")
    }
    right = observations["right"]
    roi_delta = _roi_delta(recognizer, baseline, frame, "right")
    rejection = BurstConsensus.validate_candidate(
        right.is_pass,
        right.cards,
        ConsensusContext(
            level_rank=args.wild_rank,
            remaining_cards=27,
            allow_pass=True,
            table_cards=(args.table_card,),
            validate_rules=True,
        ),
        suit_options=right.suit_options,
    )
    legal_rejection = BurstConsensus.validate_candidate(
        right.is_pass,
        right.cards,
        ConsensusContext(
            level_rank=args.wild_rank,
            remaining_cards=27,
            allow_pass=True,
            table_cards=(args.legal_table_card,),
            validate_rules=True,
        ),
        suit_options=right.suit_options,
    )
    report = {
        "schema": "guandan.fast-cycle-frame-diagnostic/1",
        "source_session": str(session),
        "frame_index": args.frame,
        "baseline_frame_index": args.baseline_frame,
        "expected_player": args.expected,
        "fast": {
            "active_player": fast.active_player,
            "self_action_buttons_visible": fast.self_action_buttons_visible,
            "pass_marker_players": list(fast.pass_marker_players),
        },
        "right_roi_delta": roi_delta,
        "recovery_freshness_threshold": 0.004,
        "right_surface_fresh": roi_delta >= 0.004,
        "observations": {
            seat: {
                "cards": list(value.cards),
                "is_pass": value.is_pass,
                "confidence": value.confidence,
                "source": value.source,
            }
            for seat, value in observations.items()
        },
        "historical_table_check": {
            "table_card": args.table_card,
            "accepted": not bool(rejection),
            "rejection_reason": rejection,
        },
        "rule_valid_equivalent_check": {
            "table_card": args.legal_table_card,
            "accepted": not bool(legal_rejection),
            "rejection_reason": legal_rejection,
        },
        "would_enter_fast_scan": bool(
            fast.self_action_buttons_visible
            and fast.active_player == "self"
            and tuple(fast.pass_marker_players) == ("opposite", "left")
            and right.cards == ("10C",)
            and observations["opposite"].is_pass
            and observations["left"].is_pass
            and roi_delta >= 0.004
        ),
    }
    atomic_write_json(args.output, report)
    print(args.output.resolve())
    return 0


def _read_frame(video: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot decode frame {frame_index}")
    return frame


def _roi_delta(
    recognizer: ScreenshotRecognitionService,
    baseline: np.ndarray,
    frame: np.ndarray,
    seat: str,
) -> float:
    def fingerprint(image: np.ndarray) -> np.ndarray:
        roi = recognizer.play_roi(image, seat)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        target_height = max(8, int(round(32 * height / max(1, width))))
        return cv2.resize(gray, (32, target_height), interpolation=cv2.INTER_AREA)

    first = fingerprint(baseline).astype(np.int16)
    second = fingerprint(frame).astype(np.int16)
    return float(np.mean(np.abs(first - second)) / 255.0)


if __name__ == "__main__":
    raise SystemExit(main())
