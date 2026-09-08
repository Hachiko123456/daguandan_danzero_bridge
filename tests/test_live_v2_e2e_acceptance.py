from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.application.live_v2_e2e_acceptance import (
    audit_live_v2_runtime,
    evaluate_window_opportunities,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _source(tmp_path: Path) -> Path:
    session = tmp_path / "source"
    (session / "video").mkdir(parents=True)
    turns = [
        (1, "right", False, 5),
        (2, "self", False, 10),
        (3, "right", True, 12),
        (4, "opposite", True, 15),
        (5, "left", False, 20),
        (6, "self", True, 25),
    ]
    (session / "truth_log.json").write_text(json.dumps({
        "source_session_id": "source",
        "turns": [
            {
                "turn_id": turn,
                "actor": actor,
                "is_pass": is_pass,
                "cards": [] if is_pass else ["3D"],
                "evidence": {"frame_indices": [frame]},
                "uncertainty": [],
            }
            for turn, actor, is_pass, frame in turns
        ],
    }), encoding="utf-8")
    _write_jsonl(session / "video" / "frame_index.jsonl", [
        {"frame_index": index, "monotonic_ms": 1_000 + index * 100}
        for index in range(30)
    ])
    return session


def _capture(path: Path) -> None:
    _write_jsonl(path, [
        {"source_frame_index": 5, "captured_monotonic_ms": 10_500},
        {"source_frame_index": 20, "captured_monotonic_ms": 12_000},
        {"source_frame_index": 29, "captured_monotonic_ms": 12_900},
    ])


def test_durable_model_advice_and_local_pass_cover_truth_denominator(tmp_path):
    source = _source(tmp_path)
    runtime = tmp_path / "runtime"
    capture = tmp_path / "capture.jsonl"
    updates = tmp_path / "updates.jsonl"
    _capture(capture)
    _write_jsonl(runtime / "advice.jsonl", [
        {"status": "requested", "turn_id": 2, "request_id": "r2"},
        {"status": "ready", "turn_id": 2, "request_id": "r2",
         "finished_processing_ms": 11_100},
        {"status": "requested", "turn_id": 6, "request_id": "r6"},
        {"status": "local_pass", "turn_id": 6, "request_id": "r6",
         "finished_processing_ms": 12_500},
    ])
    _write_jsonl(updates, [
        {"turn_id": 6, "local_pass_response": True,
         "observed_monotonic_ms": 12_500,
         "local_pass_evidence_id": "cannot-beat:1:2:12500"},
    ])

    result = evaluate_window_opportunities(
        source_session=source,
        runtime_directory=runtime,
        capture_log=capture,
        update_log=updates,
        qualification_required=True,
    )

    assert result["passed"] is True
    assert result["denominators"]["all_opportunities"] == 2
    assert result["counts"]["valid_model_advice"] == 1
    assert result["counts"]["valid_local_pass"] == 1
    assert result["response_record_count"] == 2
    assert result["rows"][1]["observed_response_count"] == 1
    assert [row["latency_ms"] for row in result["rows"]] == [600, 500]


def test_advice_only_local_pass_is_still_a_valid_response(tmp_path):
    source = _source(tmp_path)
    runtime = tmp_path / "runtime"
    capture = tmp_path / "capture.jsonl"
    updates = tmp_path / "updates.jsonl"
    _capture(capture)
    _write_jsonl(runtime / "advice.jsonl", [
        {"status": "requested", "turn_id": 2, "request_id": "r2"},
        {"status": "ready", "turn_id": 2, "request_id": "r2",
         "finished_processing_ms": 11_100},
        {"status": "requested", "turn_id": 6, "request_id": "r6"},
        {"status": "local_pass", "turn_id": 6, "request_id": "r6",
         "finished_processing_ms": 12_400},
    ])
    _write_jsonl(updates, [])

    result = evaluate_window_opportunities(
        source_session=source, runtime_directory=runtime,
        capture_log=capture, update_log=updates,
        qualification_required=True,
    )

    assert result["passed"] is True
    assert result["counts"]["valid_local_pass"] == 1
    assert result["rows"][1]["selected_response"]["evidence_id"] == "r6"


def test_earliest_legal_durable_response_wins_once_per_turn(tmp_path):
    source = _source(tmp_path)
    runtime = tmp_path / "runtime"
    capture = tmp_path / "capture.jsonl"
    updates = tmp_path / "updates.jsonl"
    _capture(capture)
    _write_jsonl(runtime / "advice.jsonl", [
        {"status": "requested", "turn_id": 2, "request_id": "r2"},
        {"status": "ready", "turn_id": 2, "request_id": "r2",
         "finished_processing_ms": 11_100},
        {"status": "requested", "turn_id": 6, "request_id": "r6"},
        {"status": "local_pass", "turn_id": 6, "request_id": "too-early",
         "finished_processing_ms": 11_900},
        {"status": "local_pass", "turn_id": 6, "request_id": "first-legal",
         "finished_processing_ms": 12_350},
        {"status": "local_pass", "turn_id": 6, "request_id": "later",
         "finished_processing_ms": 12_600},
    ])
    _write_jsonl(updates, [
        {"turn_id": 6, "local_pass_response": True,
         "observed_monotonic_ms": 12_300,
         "local_pass_evidence_id": "ui-duplicate"},
    ])

    result = evaluate_window_opportunities(
        source_session=source, runtime_directory=runtime,
        capture_log=capture, update_log=updates,
        qualification_required=True,
    )

    selected = result["rows"][1]["selected_response"]
    assert result["response_record_count"] == 2
    assert result["rows"][1]["observed_response_count"] == 1
    assert selected["evidence_id"] == "first-legal"
    assert result["rows"][1]["latency_ms"] == 350


def test_stale_timeout_and_cancelled_are_not_success_even_with_ui_hint(tmp_path):
    source = _source(tmp_path)
    runtime = tmp_path / "runtime"
    capture = tmp_path / "capture.jsonl"
    updates = tmp_path / "updates.jsonl"
    _capture(capture)
    _write_jsonl(runtime / "advice.jsonl", [
        {"status": "requested", "turn_id": 2, "request_id": "r2"},
        {"status": "ready", "turn_id": 2, "request_id": "r2",
         "finished_processing_ms": 11_100},
        {"status": "requested", "turn_id": 6, "request_id": "r6"},
        {"status": "stale", "turn_id": 6, "request_id": "r6",
         "finished_processing_ms": 12_200},
        {"status": "timeout", "turn_id": 6, "request_id": "r6",
         "finished_processing_ms": 12_300},
        {"status": "cancelled", "turn_id": 6, "request_id": "r6",
         "finished_processing_ms": 12_400},
    ])
    _write_jsonl(updates, [
        {"turn_id": 6, "local_pass_response": True,
         "observed_monotonic_ms": 12_250,
         "local_pass_evidence_id": "ui-is-not-authoritative"},
    ])

    result = evaluate_window_opportunities(
        source_session=source, runtime_directory=runtime,
        capture_log=capture, update_log=updates,
        qualification_required=True,
    )

    assert result["passed"] is False
    assert result["counts"]["valid_local_pass"] == 0
    assert result["counts"]["no_result"] == 1
    assert result["rows"][1]["selected_response"] is None


def test_late_and_no_result_remain_in_truth_denominator(tmp_path):
    source = _source(tmp_path)
    runtime = tmp_path / "runtime"
    capture = tmp_path / "capture.jsonl"
    updates = tmp_path / "updates.jsonl"
    _capture(capture)
    _write_jsonl(runtime / "advice.jsonl", [
        {"status": "requested", "turn_id": 2, "request_id": "r2"},
        {"status": "ready", "turn_id": 2, "request_id": "r2",
         "finished_processing_ms": 14_000},
    ])
    _write_jsonl(updates, [])

    result = evaluate_window_opportunities(
        source_session=source, runtime_directory=runtime,
        capture_log=capture, update_log=updates,
        qualification_required=True, response_deadline_ms=2_000,
    )

    assert result["passed"] is False
    assert result["counts"]["late"] == 1
    assert result["counts"]["no_result"] == 1
    assert result["denominators"]["all_opportunities"] == 2


def test_zero_model_requests_cannot_pass_even_with_local_pass(tmp_path):
    source = _source(tmp_path)
    runtime = tmp_path / "runtime"
    capture = tmp_path / "capture.jsonl"
    updates = tmp_path / "updates.jsonl"
    _capture(capture)
    _write_jsonl(runtime / "advice.jsonl", [])
    _write_jsonl(updates, [
        {"turn_id": 6, "local_pass_response": True,
         "observed_monotonic_ms": 12_300, "local_pass_evidence_id": "pass"},
    ])

    result = evaluate_window_opportunities(
        source_session=source, runtime_directory=runtime,
        capture_log=capture, update_log=updates,
        qualification_required=True,
    )

    assert result["model_requested"] == 0
    assert result["zero_requested_rejected"] is True
    assert result["passed"] is False


def test_development_fragment_limits_opportunities_to_complete_truth_prefix(tmp_path):
    source = _source(tmp_path)
    runtime = tmp_path / "runtime"
    capture = tmp_path / "capture.jsonl"
    updates = tmp_path / "updates.jsonl"
    _capture(capture)
    _write_jsonl(runtime / "advice.jsonl", [
        {"status": "requested", "turn_id": 2, "request_id": "r2"},
        {"status": "ready", "turn_id": 2, "request_id": "r2",
         "finished_processing_ms": 11_100},
        {"status": "requested", "turn_id": 6, "request_id": "r6"},
    ])
    _write_jsonl(updates, [])

    result = evaluate_window_opportunities(
        source_session=source,
        runtime_directory=runtime,
        capture_log=capture,
        update_log=updates,
        qualification_required=False,
        max_source_frame=20,
    )

    assert result["development_fragment"] is True
    assert result["passed"] is True
    assert result["denominators"]["all_opportunities"] == 1
    assert result["model_requested"] == 1
    assert result["prefix_expected_count"] == 1
    assert result["prefix_actual_count"] == 1
    assert result["first_divergence"] is None


def test_missing_truth_is_unavailable_not_a_fake_pass(tmp_path):
    runtime = tmp_path / "runtime"
    _write_jsonl(runtime / "advice.jsonl", [])
    result = evaluate_window_opportunities(
        source_session=tmp_path / "missing",
        runtime_directory=runtime,
        capture_log=tmp_path / "capture.jsonl",
        update_log=tmp_path / "updates.jsonl",
        qualification_required=False,
    )
    assert result["available"] is False
    assert result["passed"] is False
    assert result["denominators"]["all_opportunities"] == 0


def test_old_runtime_is_rejected_even_with_live_v2_manifest(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "manifest.json").write_text(
        json.dumps({"runtime": "live_v2"}), encoding="utf-8"
    )

    result = audit_live_v2_runtime(object(), runtime)

    assert result["runtime_type_ok"] is False
    assert result["manifest_runtime"] == "live_v2"
    assert result["passed"] is False
