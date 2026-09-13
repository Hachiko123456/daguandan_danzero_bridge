"""Scan-only repair regressions from game_20260816_125402_1687ea.

These observations model the known button-occlusion sequence without relying
on a local AVI path.  The trusted TruthLog is deliberately not an input to the
projector: it is only the expected outcome for this visual regression.
"""

from __future__ import annotations

from daguandan_bridge.application.action_trace_projection import ActionTraceProjector


_OPENING = ("2C", "2D", "3D", "3H", "4D", "4H")
_LEFT_COMPLETE = ("AC", "AS", "KC", "KS", "QD", "QS")


def _row(
    frame: int,
    *,
    opposite_pass: bool = False,
    self_cards: tuple[str, ...] = (),
    left_cards: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "frame_index": frame,
        "timestamp_ms": frame * 33,
        "decode_ok": True,
        "regions": {
            "self": {"cards": list(self_cards), "is_pass": False, "confidence": 0.95},
            "right": {"cards": [], "is_pass": False, "confidence": 0.95},
            "opposite": {"cards": [], "is_pass": opposite_pass, "confidence": 0.69},
            "left": {"cards": list(left_cards), "is_pass": False, "confidence": 0.95},
        },
    }


def test_verified_game_left_button_occlusion_is_one_repaired_action():
    """Unknown suits must stay pending until the clear 184th-frame reading."""

    observations = [*(_row(frame, opposite_pass=True) for frame in range(72, 80))]
    observations.extend(_row(frame, self_cards=_OPENING) for frame in range(112, 165))
    observations.append(_row(165, left_cards=("QD", "K?", "A?", "A?")))
    observations.append(_row(166, left_cards=("Q?", "K?", "K?", "A?", "A?")))
    observations.append(_row(167, left_cards=("QD", "QS", "KD", "K?", "A?")))
    observations.append(_row(168, left_cards=("QD", "QS", "K?", "K?", "A?", "A?")))
    observations.append(_row(169, left_cards=("QD", "QS", "K?", "K?", "A?", "A?")))
    observations.extend(
        _row(frame, left_cards=("QD", "QS", "KC", "KS", "A?", "A?"))
        for frame in range(170, 184)
    )
    observations.extend(_row(frame, left_cards=_LEFT_COMPLETE) for frame in range(184, 203))

    trace = ActionTraceProjector().project(observations)
    left = [action for action in trace["actions"] if action["actor"] == "left"]

    assert not any(action["is_pass"] for action in trace["actions"] if action["frame_start"] < 112)
    assert len(left) == 1
    action = left[0]
    assert action["frame_start"] == 165
    assert action["frame_end"] == 202
    assert action["best_frame"] == 184
    assert action["repair_frame"] == 184
    assert action["repair_status"] == "resolved"
    assert action["repair_reason"] == "later_frame_resolved_unknown_suit"
    assert action["cards"] == sorted(_LEFT_COMPLETE)
    assert action["cards_before_repair"] == ["A?", "A?", "K?", "QD"]
    assert action["uncertainty"] == []
    assert set(range(165, 203)).issubset(set(action["evidence_frames"]))


def test_unknown_suit_without_clear_frame_stays_one_unresolved_action():
    observations = [
        _row(0),
        _row(1, left_cards=("QD", "QS", "KC", "KS", "A?", "A?")),
        _row(2, left_cards=("QD", "QS", "KC", "KS", "A?", "A?")),
        _row(3),
    ]

    trace = ActionTraceProjector().project(observations)
    left = [action for action in trace["actions"] if action["actor"] == "left"]

    assert len(left) == 1
    action = left[0]
    assert action["cards"] == ["A?", "A?", "KC", "KS", "QD", "QS"]
    assert action["repair_status"] == "unresolved"
    assert action["repair_frame"] is None
    assert action["repair_reason"] == "no_complete_suit_evidence_before_display_end"
    assert action["uncertainty"] == ["unknown_suit"]
