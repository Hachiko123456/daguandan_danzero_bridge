"""Regression evidence from the manually verified game 20260816_125402_1687ea.

The fixture below is intentionally synthetic.  It encodes only the two visual
phenomena observed in that recording: a false ``cannot-beat``/PASS match before
the opening play, and the opening six-card play becoming complete over several
frames.  Keeping the fixture synthetic makes the regression stable and avoids
turning a local C: drive path into a test dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.application.action_trace_projection import ActionTraceProjector
from daguandan_bridge.application.session_corpus_validation import SessionCorpusValidationService
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn, save_truth_log


SESSION_ID = "game_20260816_125402_1687ea"
OPENING_CARDS = ("2C", "2D", "3D", "3H", "4D", "4H")


def _observation(
    frame: int,
    *,
    opposite_pass: bool = False,
    self_cards: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "frame_index": frame,
        "timestamp_ms": frame * 33,
        "decode_ok": True,
        "regions": {
            "self": {"cards": list(self_cards), "is_pass": False, "confidence": 0.95},
            "right": {"cards": [], "is_pass": False, "confidence": 0.95},
            "opposite": {"cards": [], "is_pass": opposite_pass, "confidence": 0.69},
            "left": {"cards": [], "is_pass": False, "confidence": 0.95},
        },
    }


def _trusted_truth() -> TruthLog:
    return TruthLog(
        source_session_id=SESSION_ID,
        initial_state=TruthInitialState("6", "self", OPENING_CARDS),
        turns=(
            TruthTurn(1, "self", False, OPENING_CARDS, frame_index=112),
        ),
    )


def test_verified_game_preopening_false_pass_never_becomes_first_action():
    """Frames 72-79 are a known false PASS match and must stay raw evidence."""

    observations = [_observation(frame) for frame in range(72)]
    observations.extend(_observation(frame, opposite_pass=True) for frame in range(72, 80))
    observations.extend(_observation(frame) for frame in range(80, 112))
    observations.append(_observation(112, self_cards=("2D", "2?", "3?", "3D", "4H", "4D")))
    observations.append(_observation(113, self_cards=("2D", "2C", "3?", "3D", "4H", "4D")))
    observations.append(_observation(114, self_cards=("2D", "2C", "3H", "3D", "4H", "4D")))
    observations.extend(_observation(frame, self_cards=OPENING_CARDS) for frame in range(115, 165))
    observations.append(_observation(165))

    projection = ActionTraceProjector().project(observations)
    actions = projection["actions"]

    assert actions, "the trusted recording must produce at least its opening action"
    assert not any(action["is_pass"] for action in actions if int(action["frame_start"]) < 112)
    opening = [action for action in actions if action["actor"] == "self"]
    assert len(opening) == 1, "progressive recognition of one display must not create multiple turns"
    assert opening[0]["frame_start"] == 112
    assert opening[0]["frame_end"] == 164
    assert set(opening[0]["cards"]) == set(OPENING_CARDS)
    assert {112, 114, 164}.issubset(set(opening[0]["evidence_frames"]))
    assert len(opening[0]["observed_variants"]) >= 2


def test_verified_game_truth_is_compared_after_scan_without_becoming_scan_input(
    tmp_path: Path,
    monkeypatch,
):
    """A trusted TruthLog is an oracle for comparison, never a scanner baseline."""

    session = tmp_path / SESSION_ID
    video_dir = session / "video"
    video_dir.mkdir(parents=True)
    video = video_dir / "game.avi"
    video.write_bytes(b"synthetic avi placeholder")
    save_truth_log(session / "truth_log.json", _trusted_truth())

    # Keep this test independent of OpenCV: the scanner itself is represented
    # by an already-produced scan result.  The production scanner contract is
    # exercised separately; this regression focuses on the oracle boundary.
    service = SessionCorpusValidationService()
    def _completed_scan(selected, output, *, profile_root):
        item = selected[0]
        raw = {
            "status": "completed",
            "actions": [
                {
                    "actor": "self",
                    "is_pass": False,
                    "cards": list(OPENING_CARDS),
                    "frame_start": 112,
                }
            ],
            "frames_processed": 165,
            "action_count": 1,
        }
        return {item.session_id: service._normalise_scan_result(item, raw, output)}

    monkeypatch.setattr(service, "_run_audit", _completed_scan)
    run = service.validate(
        session,
        mode="single",
        output=tmp_path / "reports",
        trusted_sessions=(session,),
        run_id="trusted-game-regression",
    )

    assert run.passed
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    semantic = summary["sessions"][0]["semantic"]
    assert semantic["status"] == "passed"
    assert semantic["comparison"]["first_difference"] is None
    assert summary["scan_policy"]["truth_is_compared_after_scan_only"] is True
