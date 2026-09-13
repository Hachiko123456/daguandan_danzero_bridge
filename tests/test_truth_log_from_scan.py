from __future__ import annotations

from copy import deepcopy

from daguandan_bridge.application.truth_log_from_scan import (
    DRAFT_PROVENANCE_SOURCE,
    TruthLogFromScanResult,
    build_truth_log_from_scan,
)
from daguandan_bridge.domain.truth import LabelProvenance, TruthOutcome
from daguandan_bridge.live.truth_log import TruthInitialState, TruthLog, TruthTurn


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _baseline() -> TruthLog:
    return TruthLog(
        source_session_id="session-baseline",
        initial_state=TruthInitialState("2", "left", HAND),
        turns=(TruthTurn(1, "left", False, ("5H",), trick_id=4),),
        source_video="captures/game.avi",
        frame_index_path="captures/frame_index.jsonl",
        label_status="verified",
        provenance=LabelProvenance(source="human_review", annotator="reviewer"),
        outcome=TruthOutcome(),
    )


def test_builds_fresh_draft_with_stable_consecutive_turns_and_source_metadata():
    baseline = _baseline()
    baseline_before = deepcopy(baseline.to_dict())

    result = build_truth_log_from_scan(
        baseline,
        [
            {
                "action_id": 41,
                "actor": "self",
                "is_pass": False,
                "cards": ["5H", "5S"],
                "frame_index": 120,
                "evidence_frames": [119, 120],
                "monotonic_ms": 4000,
                "trick_id": 7,
                "trick_id_reliable": True,
                "repair_status": "resolved",
                "review_status": "unverified",
            },
            {
                "action_id": 42,
                "actor": "right",
                "is_pass": True,
                "cards": [],
                "frame_index": 121,
            },
        ],
    )

    assert isinstance(result, TruthLogFromScanResult)
    log = result.truth_log
    assert [turn.index for turn in log.turns] == [1, 2]
    assert [turn.actor for turn in log.turns] == ["self", "right"]
    assert log.turns[0].cards == ("5H", "5S")
    assert log.turns[1].is_pass is True
    assert log.turns[1].cards == ()
    assert log.turns[1].trick_id is None
    assert log.to_dict()["turns"][1]["trick_id"] is None
    assert log.label_status == "draft"
    assert log.provenance.source == DRAFT_PROVENANCE_SOURCE
    assert "video_scan" in log.provenance.source
    assert "canonical_reconciliation" in log.provenance.source
    assert log.source_session_id == baseline.source_session_id
    assert log.initial_state == baseline.initial_state
    assert log.source_video == baseline.source_video
    assert log.frame_index_path == baseline.frame_index_path
    assert log.outcome == TruthOutcome()
    assert baseline.to_dict() == baseline_before
    assert result.review_items == ()


def test_unresolved_needs_review_and_unknown_suit_are_kept_and_auditable():
    result = build_truth_log_from_scan(
        _baseline(),
        [
            {
                "action_id": "occluded-1",
                "actor": "left",
                "is_pass": False,
                "cards": ["A?", "KS"],
                "frame_start": 165,
                "evidence_frames": [165, 170],
                "repair_status": "unresolved",
                "review_status": "needs_review",
            }
        ],
    )

    assert len(result.truth_log.turns) == 1
    turn = result.truth_log.turns[0]
    assert turn.cards == ("A?", "KS")
    assert turn.uncertainty == ("unknown_suit",)
    assert turn.label_status == "draft"
    assert turn.provenance.source == DRAFT_PROVENANCE_SOURCE
    assert len(result.review_items) == 1
    review = result.review_items[0]
    assert review["action_id"] == "occluded-1"
    assert review["output_turn_id"] == 1
    assert review["uncertainty"] == ["unknown_suit"]
    assert review["provenance"] == {"source": DRAFT_PROVENANCE_SOURCE}
    assert "repair_status:unresolved" in review["statuses"]
    assert "review_status:needs_review" in review["statuses"]
    assert review["source_action"]["cards"] == ["A?", "KS"]


def test_pass_cards_are_normalized_but_original_is_in_review_and_invalid_actions_are_not_silent():
    result = build_truth_log_from_scan(
        _baseline(),
        [
            {
                "action_id": 1,
                "actor": "self",
                "is_pass": True,
                "cards": ["5H"],
            },
            {
                "action_id": 2,
                "actor": "right",
                "is_pass": False,
                "cards": [],
                "uncertainty": ["unresolved"],
            },
            {
                "action_id": 3,
                "actor": "opposite",
                "is_pass": False,
                "cards": ["6D"],
            },
        ],
    )

    assert [turn.index for turn in result.truth_log.turns] == [1, 2]
    assert [turn.actor for turn in result.truth_log.turns] == ["self", "opposite"]
    assert result.truth_log.turns[0].cards == ()
    assert result.truth_log.turns[1].cards == ("6D",)
    assert len(result.review_items) == 2
    pass_review = result.review_items[0]
    assert pass_review["action_id"] == 1
    assert pass_review["cards"] == ["5H"]
    assert "pass_cards_present_normalized_to_empty" in pass_review["reasons"]
    assert pass_review["provenance"] == {"source": DRAFT_PROVENANCE_SOURCE}
    missing_review = result.review_items[1]
    assert missing_review["action_id"] == 2
    assert "missing_cards" in missing_review["reasons"]
    assert missing_review["uncertainty"] == ["unresolved"]
    assert missing_review["provenance"] == {"source": DRAFT_PROVENANCE_SOURCE}


def test_only_reliable_explicit_trick_id_is_used_unreliable_value_is_reviewed():
    result = build_truth_log_from_scan(
        _baseline(),
        [
            {
                "action_id": "reliable",
                "actor": "self",
                "cards": ["5H"],
                "trick_id": 7,
                "trick_id_confidence": 1.0,
            },
            {
                "action_id": "not-reliable",
                "actor": "right",
                "is_pass": True,
                "cards": [],
                "trick_id": 99,
                "trick_id_reliable": False,
            },
        ],
    )

    assert result.truth_log.turns[0].trick_id == 7
    # The explicit 99 is never copied.  The conversion restores None after
    # TruthLog normalization, so an unreliable source value cannot become a
    # draft trick id.
    assert result.truth_log.turns[1].trick_id is None
    assert result.truth_log.to_dict()["turns"][1]["trick_id"] is None
    reviewed = [item for item in result.review_items if item["action_id"] == "not-reliable"]
    assert len(reviewed) == 1
    assert "unreliable_trick_id_ignored" in reviewed[0]["reasons"]


def test_accepts_canonical_trace_envelope_and_returns_detached_review_snapshot():
    source = {
        "schema": "guandan.canonical-action-trace/1",
        "actions": [
            {"action_id": 8, "actor": "self", "cards": ["7C"], "status": "needs_review"}
        ],
    }
    result = build_truth_log_from_scan(_baseline(), source)
    source["actions"][0]["cards"].append("7D")

    assert result.truth_log.turns[0].cards == ("7C",)
    assert result.review_items[0]["source_action"]["cards"] == ["7C"]
