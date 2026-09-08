"""Opportunity-denominator and response audit for production window replay."""

from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path

from .opportunity_truth import (
    OpportunityResponseRecord,
    OpportunityTruthLog,
    derive_self_opportunities,
    evaluate_opportunity_responses,
    unavailable_opportunity_acceptance,
)
from .live_v2_session_runtime import LiveV2SessionRuntime


def audit_live_v2_runtime(runtime: object, runtime_directory: Path) -> dict[str, object]:
    manifest_path = Path(runtime_directory) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {}
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = ""
    runtime_type_ok = isinstance(runtime, LiveV2SessionRuntime)
    manifest_runtime = str(manifest.get("runtime", ""))
    return {
        "passed": runtime_type_ok and manifest_runtime == "live_v2",
        "runtime_type_ok": runtime_type_ok,
        "runtime_type": f"{type(runtime).__module__}.{type(runtime).__name__}",
        "manifest_runtime": manifest_runtime,
        "manifest_path": str(manifest_path),
        "error": error,
    }


def evaluate_window_opportunities(
    *,
    source_session: Path,
    runtime_directory: Path,
    capture_log: Path,
    update_log: Path,
    qualification_required: bool,
    max_source_frame: int | None = None,
    response_deadline_ms: int = 3_000,
) -> dict[str, object]:
    """Evaluate every truth self-turn, including opportunities never requested."""

    try:
        truth = derive_self_opportunities(source_session)
        if max_source_frame is not None:
            included_turns = _truth_prefix_turn_ids(source_session, max_source_frame)
            truth = OpportunityTruthLog(
                truth.source_session_id,
                tuple(
                    opportunity
                    for opportunity in truth.opportunities
                    if _opportunity_turn_id(opportunity.opportunity_id) in included_turns
                ),
            )
    except Exception as exc:
        result = unavailable_opportunity_acceptance(
            f"truth opportunity denominator unavailable: {type(exc).__name__}: {exc}"
        )
        result["qualification_required"] = qualification_required
        return result
    if not truth.opportunities:
        if max_source_frame is not None:
            return {
                "schema": "guandan.opportunity-acceptance/1",
                "available": True,
                "passed": True,
                "qualification_required": qualification_required,
                "development_fragment": True,
                "max_source_frame": max_source_frame,
                "denominators": {
                    "all_opportunities": 0,
                    "determinable_opportunities": 0,
                    "indeterminate_opportunities": 0,
                },
                "counts": {},
                "outcomes": {},
                "latency_ms": {},
                "rows": [],
                "model_requested": 0,
                "response_record_count": 0,
                "every_determinable_succeeded": True,
                "zero_requested_rejected": False,
                "prefix_expected_count": 0,
                "prefix_actual_count": 0,
                "first_divergence": None,
            }
        result = unavailable_opportunity_acceptance(
            "truth action sequence contains no self opportunities"
        )
        result["qualification_required"] = qualification_required
        return result
    try:
        mapper = _source_to_runtime_clock(source_session, capture_log)
    except Exception as exc:
        result = unavailable_opportunity_acceptance(
            f"source/runtime clock mapping unavailable: {type(exc).__name__}: {exc}"
        )
        result["qualification_required"] = qualification_required
        return result
    advice_rows = _json_lines(runtime_directory / "advice.jsonl")
    update_rows = _json_lines(update_log)
    responses = _response_records(
        truth,
        advice_rows,
        update_rows,
        source_to_response_clock=mapper,
    )
    result = evaluate_opportunity_responses(
        truth,
        responses,
        source_to_response_clock=mapper,
        response_deadline_ms=response_deadline_ms,
    )
    truth_turns = {
        _opportunity_turn_id(opportunity.opportunity_id)
        for opportunity in truth.opportunities
    }
    requested = sum(
        row.get("status") == "requested"
        and _positive_int(row.get("turn_id")) in truth_turns
        for row in advice_rows
    )
    determinable_rows = [row for row in result["rows"] if row["determinable"]]
    every_determinable_succeeded = bool(determinable_rows) and all(
        row["successful"] and row["on_time"] for row in determinable_rows
    )
    passed = bool(requested > 0 and every_determinable_succeeded)
    response = {
        **result,
        "available": True,
        "passed": passed,
        "qualification_required": qualification_required,
        "model_requested": requested,
        "response_record_count": len(responses),
        "every_determinable_succeeded": every_determinable_succeeded,
        "zero_requested_rejected": requested == 0,
    }
    if max_source_frame is not None:
        first = next(
            (
                {
                    "kind": "opportunity_response",
                    "opportunity_id": row.get("opportunity_id"),
                    "expected": row.get("expected_response"),
                    "actual": row.get("outcome"),
                }
                for row in determinable_rows
                if not (row["successful"] and row["on_time"])
            ),
            None,
        )
        response.update(
            {
                "development_fragment": True,
                "max_source_frame": max_source_frame,
                "prefix_expected_count": len(determinable_rows),
                "prefix_actual_count": sum(
                    bool(row["successful"] and row["on_time"])
                    for row in determinable_rows
                ),
                "first_divergence": first,
            }
        )
    return response


def _truth_prefix_turn_ids(source_session: Path, max_source_frame: int) -> set[int]:
    raw = json.loads((Path(source_session) / "truth_log.json").read_text(encoding="utf-8"))
    turns = raw.get("turns", ())
    if not isinstance(turns, list):
        raise ValueError("truth_log turns must be an array")
    included: set[int] = set()
    for position, turn in enumerate(turns):
        if not isinstance(turn, dict):
            break
        frames = _truth_frames(turn)
        if not frames or any(frame > max_source_frame for frame in frames):
            break
        included.add(int(turn.get("turn_id", position + 1)))
    return included


def _truth_frames(turn: dict[str, object]) -> tuple[int, ...]:
    evidence = turn.get("evidence")
    if isinstance(evidence, dict):
        frames = evidence.get("frame_indices")
        if isinstance(frames, list):
            return tuple(int(frame) for frame in frames)
    frame = turn.get("frame_index")
    return () if frame is None else (int(frame),)


def _opportunity_turn_id(opportunity_id: str) -> int:
    try:
        return int(str(opportunity_id).rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _response_records(
    truth,
    advice_rows,
    update_rows,
    *,
    source_to_response_clock,
):
    by_turn = {
        _opportunity_turn_id(opportunity.opportunity_id): opportunity
        for opportunity in truth.opportunities
    }
    corroborated_turns = {
        _positive_int(row.get("turn_id"))
        for row in update_rows
        if row.get("local_pass_response") is True
    }
    earliest: dict[int, tuple[int, int, str, dict[str, object]]] = {}
    for position, row in enumerate(advice_rows):
        status = str(row.get("status", ""))
        response = {"ready": "model_advice", "local_pass": "local_pass"}.get(status)
        turn_id = _positive_int(row.get("turn_id"))
        opportunity = by_turn.get(turn_id)
        response_ms = _non_negative(row.get("finished_processing_ms"))
        if response is None or opportunity is None or response_ms is None:
            continue
        reference_ms = float(
            source_to_response_clock(opportunity.start_source_monotonic_ms)
        )
        if response_ms < reference_ms or not _matches_expected(
            opportunity.expected_response, response
        ):
            continue
        candidate = (response_ms, position, response, row)
        previous = earliest.get(turn_id)
        if previous is None or candidate[:2] < previous[:2]:
            earliest[turn_id] = candidate

    records: list[OpportunityResponseRecord] = []
    for turn_id, opportunity in by_turn.items():
        selected = earliest.get(turn_id)
        if selected is None:
            continue
        response_ms, _position, response, row = selected
        corroborated = response == "local_pass" and turn_id in corroborated_turns
        detail = (
            "durable live-v2 local-pass response"
            + (" corroborated by live update" if corroborated else "")
            if response == "local_pass"
            else "validated durable live-v2 advice worker response"
        )
        records.append(OpportunityResponseRecord(
            opportunity.opportunity_id,
            response,
            response_ms,
            valid=True,
            evidence_id=str(row.get("request_id", "")),
            detail=detail,
        ))
    return tuple(records)


def _matches_expected(expected: object, response: str) -> bool:
    return bool(expected == response or expected == "model_or_local_pass")


def _source_to_runtime_clock(source_session: Path, capture_log: Path):
    source_times = {
        int(row["frame_index"]): int(row["monotonic_ms"])
        for row in _json_lines(source_session / "video" / "frame_index.jsonl")
    }
    anchors: dict[int, int] = {}
    for row in _json_lines(capture_log):
        frame = _non_negative(row.get("source_frame_index"))
        captured = _non_negative(row.get("captured_monotonic_ms"))
        if frame is None or captured is None or frame not in source_times:
            continue
        source_ms = source_times[frame]
        anchors[source_ms] = min(captured, anchors.get(source_ms, captured))
    if not anchors:
        raise ValueError("capture log has no source-frame clock anchors")
    ordered = sorted(anchors.items())
    source_axis = [item[0] for item in ordered]

    def mapper(source_ms: int) -> float:
        position = bisect_left(source_axis, source_ms)
        if position == 0:
            base_source, base_runtime = ordered[0]
        elif position == len(ordered):
            base_source, base_runtime = ordered[-1]
        else:
            left, right = ordered[position - 1], ordered[position]
            base_source, base_runtime = min(
                (left, right), key=lambda item: abs(item[0] - source_ms)
            )
        return float(base_runtime + source_ms - base_source)

    return mapper


def _json_lines(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    result: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            result.append(value)
    return result


def _positive_int(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return -1
    return parsed if parsed > 0 else -1


def _non_negative(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


__all__ = ["audit_live_v2_runtime", "evaluate_window_opportunities"]
