from __future__ import annotations

import pytest
import json

from daguandan_bridge.application.opportunity_truth import (
    ManualOpportunityEvidence,
    OpportunityResponseRecord,
    OpportunityTruth,
    OpportunityTruthLog,
    evaluate_opportunity_responses,
    load_opportunity_truth,
    save_opportunity_truth,
    derive_self_opportunities,
)


def _opportunity(
    index: int,
    expected: str = "model_advice",
    *,
    determinable: bool = True,
) -> OpportunityTruth:
    start = index * 1_000
    return OpportunityTruth(
        opportunity_id=f"OPP-{index}",
        start_source_frame=index * 10,
        end_source_frame=index * 10 + 9,
        start_source_monotonic_ms=start,
        end_source_monotonic_ms=start + 900,
        expected_response=expected,  # type: ignore[arg-type]
        determinable=determinable,
        manual_evidence=ManualOpportunityEvidence(
            source="human_video_review",
            frame_indices=(index * 10, index * 10 + 1),
            note=f"visible self opportunity {index}",
            annotator="reviewer",
            annotated_at="2026-09-06T12:00:00+08:00",
        ),
    )


def test_truth_round_trip_keeps_source_window_and_manual_evidence(tmp_path):
    truth = OpportunityTruthLog("game-1", (_opportunity(1),))
    path = tmp_path / "opportunity_truth.json"

    save_opportunity_truth(path, truth)

    loaded = load_opportunity_truth(path, session_id="game-1")
    assert loaded == truth
    document = loaded.to_dict()
    assert document["schema"] == "guandan.opportunity-truth/1"
    row = document["opportunities"][0]
    assert row["source_window"] == {
        "start_frame": 10,
        "end_frame": 19,
        "start_monotonic_ms": 1_000,
        "end_monotonic_ms": 1_900,
    }
    assert row["manual_evidence"]["source"] == "human_video_review"


def test_self_opportunity_denominator_is_derived_from_truth_actions(tmp_path):
    session = tmp_path / "game-source"
    (session / "video").mkdir(parents=True)
    (session / "truth_log.json").write_text(json.dumps({
        "source_session_id": "game-source",
        "turns": [
            {"turn_id": 1, "actor": "right", "is_pass": False,
             "cards": ["3D"], "evidence": {"frame_indices": [2]}},
            {"turn_id": 2, "actor": "self", "is_pass": False,
             "cards": ["4D"], "evidence": {"frame_indices": [5]}},
            {"turn_id": 3, "actor": "right", "is_pass": True,
             "cards": [], "evidence": {"frame_indices": [7]}},
            {"turn_id": 4, "actor": "self", "is_pass": True,
             "cards": [], "evidence": {"frame_indices": [9]}},
        ],
    }), encoding="utf-8")
    (session / "video" / "frame_index.jsonl").write_text("".join(
        json.dumps({"frame_index": index, "monotonic_ms": 1_000 + index * 100}) + "\n"
        for index in range(10)
    ), encoding="utf-8")

    truth = derive_self_opportunities(session)

    assert [item.opportunity_id for item in truth.opportunities] == [
        "truth-self-0002", "truth-self-0004"
    ]
    assert [item.start_source_frame for item in truth.opportunities] == [2, 7]
    assert [item.expected_response for item in truth.opportunities] == [
        "model_advice", "model_or_local_pass"
    ]


def test_missed_opportunity_remains_in_all_denominators_and_rows():
    truth = OpportunityTruthLog(
        "game-1",
        (
            _opportunity(1),
            _opportunity(2, "local_pass"),
            _opportunity(3),
        ),
    )
    result = evaluate_opportunity_responses(
        truth,
        (
            OpportunityResponseRecord("OPP-1", "model_advice", 1_500),
            OpportunityResponseRecord("OPP-2", "local_pass", 2_700),
        ),
    )

    assert result["denominators"] == {
        "all_opportunities": 3,
        "determinable_opportunities": 3,
        "indeterminate_opportunities": 0,
    }
    assert result["counts"]["no_result"] == 1
    assert result["counts"]["successful_recommendations"] == 2
    assert result["coverage"]["successful_all"] == 2 / 3
    assert [row["outcome"] for row in result["rows"]] == [
        "valid_model_advice",
        "valid_local_pass",
        "no_result",
    ]


def test_two_second_explicit_error_is_a_response_but_not_success():
    truth = OpportunityTruthLog("game-1", (_opportunity(1),))

    result = evaluate_opportunity_responses(
        truth,
        (OpportunityResponseRecord("OPP-1", "explicit_unrecoverable", 3_000, detail="history gap"),),
        response_deadline_ms=2_000,
    )

    row = result["rows"][0]
    assert row["latency_ms"] == 2_000
    assert row["outcome"] == "explicit_unrecoverable"
    assert row["responded"] is True and row["on_time"] is True
    assert row["successful"] is False
    assert result["coverage"] == {
        "response_all": 1.0,
        "on_time_response_all": 1.0,
        "successful_all": 0.0,
        "successful_determinable": 0.0,
    }


def test_latency_starts_at_mapped_truth_opportunity_not_advisor_request():
    truth = OpportunityTruthLog(
        "game-1",
        (_opportunity(1), _opportunity(2), _opportunity(3)),
    )
    result = evaluate_opportunity_responses(
        truth,
        (
            OpportunityResponseRecord("OPP-1", "model_advice", 10_100),
            OpportunityResponseRecord("OPP-2", "model_advice", 11_500),
            OpportunityResponseRecord("OPP-3", "model_advice", 13_000),
        ),
        source_to_response_clock=lambda source_ms: source_ms + 9_000,
        response_deadline_ms=2_000,
    )

    assert [row["latency_ms"] for row in result["rows"]] == [100, 500, 1_000]
    latency = result["latency_ms"]["all_responses"]
    assert latency["count"] == 3
    assert latency["p50"] == 500
    assert latency["p95"] == pytest.approx(950)
    assert latency["max"] == 1_000


def test_compatible_response_after_deadline_is_late_not_success():
    truth = OpportunityTruthLog("game-1", (_opportunity(1),))
    result = evaluate_opportunity_responses(
        truth,
        (OpportunityResponseRecord("OPP-1", "model_advice", 3_001),),
    )

    assert result["rows"][0]["outcome"] == "late"
    assert result["counts"]["successful_recommendations"] == 0
    assert result["counts"]["responded"] == 1


def test_indeterminate_truth_is_retained_but_not_claimed_as_success():
    truth = OpportunityTruthLog(
        "game-1",
        (_opportunity(1), _opportunity(2, determinable=False)),
    )
    result = evaluate_opportunity_responses(
        truth,
        (
            OpportunityResponseRecord("OPP-1", "model_advice", 1_100),
            OpportunityResponseRecord("OPP-2", "model_advice", 2_100),
        ),
    )

    assert result["denominators"] == {
        "all_opportunities": 2,
        "determinable_opportunities": 1,
        "indeterminate_opportunities": 1,
    }
    assert result["counts"]["responded"] == 2
    assert result["counts"]["successful_recommendations"] == 1
    assert result["coverage"]["successful_all"] == 0.5
    assert result["coverage"]["successful_determinable"] == 1.0
