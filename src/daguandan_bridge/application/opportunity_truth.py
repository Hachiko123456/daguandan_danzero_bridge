"""Independent ground truth and acceptance metrics for local opportunities.

Truth is annotated from source recordings before replay. A program that never
notices an opportunity therefore cannot remove it from the denominator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

from ..storage import atomic_write_json


_SCHEMA = "guandan.opportunity-truth/1"
ExpectedResponse = Literal["model_advice", "local_pass", "model_or_local_pass"]
ObservedResponse = Literal["model_advice", "local_pass", "explicit_unrecoverable"]
OpportunityOutcome = Literal[
    "valid_model_advice",
    "valid_local_pass",
    "late",
    "no_result",
    "explicit_unrecoverable",
]
_EXPECTED = frozenset({"model_advice", "local_pass", "model_or_local_pass"})
_OBSERVED = frozenset({"model_advice", "local_pass", "explicit_unrecoverable"})


@dataclass(frozen=True)
class ManualOpportunityEvidence:
    """Human evidence establishing the opportunity, never program output."""

    source: str
    frame_indices: tuple[int, ...]
    note: str
    annotator: str = ""
    annotated_at: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("manual evidence source is required")
        for index in self.frame_indices:
            _non_negative_int(index, "manual evidence frame")
        if not self.frame_indices and not self.note.strip():
            raise ValueError("manual evidence requires a frame or a note")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "frame_indices": list(self.frame_indices),
            "note": self.note,
            "annotator": self.annotator,
            "annotated_at": self.annotated_at,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "ManualOpportunityEvidence":
        if not isinstance(raw, dict):
            raise ValueError("manual evidence must be an object")
        frames = raw.get("frame_indices", ())
        if not isinstance(frames, (list, tuple)):
            raise ValueError("manual evidence frame_indices must be an array")
        return cls(
            source=str(raw.get("source", "")),
            frame_indices=tuple(_non_negative_int(item, "evidence frame") for item in frames),
            note=str(raw.get("note", "")),
            annotator=str(raw.get("annotator", "")),
            annotated_at=str(raw.get("annotated_at", "")),
        )


@dataclass(frozen=True)
class OpportunityTruth:
    opportunity_id: str
    start_source_frame: int
    end_source_frame: int
    start_source_monotonic_ms: int
    end_source_monotonic_ms: int
    expected_response: ExpectedResponse
    determinable: bool
    manual_evidence: ManualOpportunityEvidence

    def __post_init__(self) -> None:
        if not self.opportunity_id.strip():
            raise ValueError("opportunity_id is required")
        for value, label in (
            (self.start_source_frame, "start_source_frame"),
            (self.end_source_frame, "end_source_frame"),
            (self.start_source_monotonic_ms, "start_source_monotonic_ms"),
            (self.end_source_monotonic_ms, "end_source_monotonic_ms"),
        ):
            _non_negative_int(value, label)
        if self.end_source_frame < self.start_source_frame:
            raise ValueError("opportunity end frame precedes start frame")
        if self.end_source_monotonic_ms < self.start_source_monotonic_ms:
            raise ValueError("opportunity end time precedes start time")
        if self.expected_response not in _EXPECTED:
            raise ValueError(f"unsupported expected response: {self.expected_response}")

    def to_dict(self) -> dict[str, object]:
        return {
            "opportunity_id": self.opportunity_id,
            "source_window": {
                "start_frame": self.start_source_frame,
                "end_frame": self.end_source_frame,
                "start_monotonic_ms": self.start_source_monotonic_ms,
                "end_monotonic_ms": self.end_source_monotonic_ms,
            },
            "expected_response": self.expected_response,
            "determinable": self.determinable,
            "manual_evidence": self.manual_evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> "OpportunityTruth":
        if not isinstance(raw, dict):
            raise ValueError("opportunity must be an object")
        window = raw.get("source_window", {})
        if not isinstance(window, dict):
            raise ValueError("source_window must be an object")
        return cls(
            opportunity_id=str(raw.get("opportunity_id", "")),
            start_source_frame=_non_negative_int(window.get("start_frame"), "start_source_frame"),
            end_source_frame=_non_negative_int(window.get("end_frame"), "end_source_frame"),
            start_source_monotonic_ms=_non_negative_int(window.get("start_monotonic_ms"), "start_source_monotonic_ms"),
            end_source_monotonic_ms=_non_negative_int(window.get("end_monotonic_ms"), "end_source_monotonic_ms"),
            expected_response=str(raw.get("expected_response", "")),  # type: ignore[arg-type]
            determinable=_required_bool(raw.get("determinable"), "determinable"),
            manual_evidence=ManualOpportunityEvidence.from_dict(raw.get("manual_evidence")),
        )


@dataclass(frozen=True)
class OpportunityTruthLog:
    source_session_id: str
    opportunities: tuple[OpportunityTruth, ...]

    def __post_init__(self) -> None:
        if not self.source_session_id.strip():
            raise ValueError("source_session_id is required")
        identifiers = [item.opportunity_id for item in self.opportunities]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("opportunity ids must be unique")
        ordered = sorted(self.opportunities, key=lambda item: (item.start_source_monotonic_ms, item.start_source_frame))
        if list(self.opportunities) != ordered:
            raise ValueError("opportunities must be ordered by source start time")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "source_session_id": self.source_session_id,
            "opportunities": [item.to_dict() for item in self.opportunities],
        }

    @classmethod
    def from_dict(cls, raw: object) -> "OpportunityTruthLog":
        if not isinstance(raw, dict) or raw.get("schema") != _SCHEMA:
            raise ValueError("unsupported opportunity truth document")
        values = raw.get("opportunities", ())
        if not isinstance(values, (list, tuple)):
            raise ValueError("opportunities must be an array")
        return cls(
            source_session_id=str(raw.get("source_session_id", "")),
            opportunities=tuple(OpportunityTruth.from_dict(item) for item in values),
        )


@dataclass(frozen=True)
class OpportunityResponseRecord:
    """One externally observed program response attributed to truth by id."""

    opportunity_id: str
    response: ObservedResponse
    response_monotonic_ms: float
    valid: bool = True
    evidence_id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.opportunity_id.strip():
            raise ValueError("response opportunity_id is required")
        if self.response not in _OBSERVED:
            raise ValueError(f"unsupported observed response: {self.response}")
        if isinstance(self.response_monotonic_ms, bool) or not math.isfinite(self.response_monotonic_ms):
            raise ValueError("response_monotonic_ms must be finite")
        if self.response_monotonic_ms < 0:
            raise ValueError("response_monotonic_ms must be non-negative")


def evaluate_opportunity_responses(
    truth: OpportunityTruthLog,
    responses: Iterable[OpportunityResponseRecord],
    *,
    source_to_response_clock: Callable[[int], float] | None = None,
    response_deadline_ms: int = 2_000,
) -> dict[str, object]:
    """Classify every truth opportunity without shrinking its denominator.

    ``source_to_response_clock`` maps source timestamps onto the clock used by
    response records. Latency starts at the human-labelled opportunity, never
    at an advisor request the program may issue late or fail to issue.
    """

    if isinstance(response_deadline_ms, bool) or response_deadline_ms <= 0:
        raise ValueError("response_deadline_ms must be a positive integer")
    mapper = source_to_response_clock or float
    by_id: dict[str, list[OpportunityResponseRecord]] = {}
    for response in responses:
        by_id.setdefault(response.opportunity_id, []).append(response)
    known = {item.opportunity_id for item in truth.opportunities}
    unattributed = sorted(identifier for identifier in by_id if identifier not in known)
    rows: list[dict[str, object]] = []
    measured_latencies: list[float] = []
    successful_latencies: list[float] = []
    outcome_counts = {
        name: 0 for name in (
            "valid_model_advice",
            "valid_local_pass",
            "late",
            "no_result",
            "explicit_unrecoverable",
        )
    }
    for opportunity in truth.opportunities:
        reference_ms = float(mapper(opportunity.start_source_monotonic_ms))
        if not math.isfinite(reference_ms):
            raise ValueError(f"mapped start time is not finite: {opportunity.opportunity_id}")
        candidates = sorted(
            by_id.get(opportunity.opportunity_id, ()),
            key=lambda item: item.response_monotonic_ms,
        )
        valid_candidates = [
            item for item in candidates
            if item.valid and item.response_monotonic_ms >= reference_ms
        ]
        compatible = [
            item for item in valid_candidates
            if item.response != "explicit_unrecoverable"
            and _response_matches(opportunity.expected_response, item.response)
        ]
        explicit = [item for item in valid_candidates if item.response == "explicit_unrecoverable"]
        selected = compatible[0] if compatible else (explicit[0] if explicit else None)
        latency = None if selected is None else selected.response_monotonic_ms - reference_ms
        if latency is not None:
            measured_latencies.append(latency)
        on_time = latency is not None and 0 <= latency <= response_deadline_ms
        if selected is None:
            outcome: OpportunityOutcome = "no_result"
        elif not on_time:
            outcome = "late"
        elif selected.response == "explicit_unrecoverable":
            outcome = "explicit_unrecoverable"
        elif selected.response == "model_advice":
            outcome = "valid_model_advice"
        else:
            outcome = "valid_local_pass"
        successful = bool(
            opportunity.determinable
            and outcome in {"valid_model_advice", "valid_local_pass"}
        )
        if successful and latency is not None:
            successful_latencies.append(latency)
        outcome_counts[outcome] += 1
        rows.append(
            {
                **opportunity.to_dict(),
                "reference_response_clock_ms": reference_ms,
                "outcome": outcome,
                "responded": selected is not None,
                "on_time": on_time,
                "successful": successful,
                "latency_ms": latency,
                "selected_response": _response_dict(selected),
                "observed_response_count": len(candidates),
                "ignored_invalid_or_incompatible_count": len(candidates) - int(selected is not None),
            }
        )
    total = len(truth.opportunities)
    determinable = sum(item.determinable for item in truth.opportunities)
    responded = sum(bool(row["responded"]) for row in rows)
    on_time_responses = sum(bool(row["on_time"]) for row in rows)
    successful = sum(bool(row["successful"]) for row in rows)
    return {
        "schema": "guandan.opportunity-acceptance/1",
        "source_session_id": truth.source_session_id,
        "response_deadline_ms": response_deadline_ms,
        "denominators": {
            "all_opportunities": total,
            "determinable_opportunities": determinable,
            "indeterminate_opportunities": total - determinable,
        },
        "counts": {
            **outcome_counts,
            "responded": responded,
            "on_time_responses": on_time_responses,
            "successful_recommendations": successful,
        },
        "outcomes": dict(outcome_counts),
        "coverage": {
            "response_all": _ratio(responded, total),
            "on_time_response_all": _ratio(on_time_responses, total),
            "successful_all": _ratio(successful, total),
            "successful_determinable": _ratio(successful, determinable),
        },
        "latency_ms": {
            "all_responses": _summarize(measured_latencies),
            "successful_recommendations": _summarize(successful_latencies),
        },
        "unattributed_response_ids": unattributed,
        "rows": rows,
    }


def save_opportunity_truth(path: Path, truth: OpportunityTruthLog) -> None:
    atomic_write_json(Path(path), truth.to_dict())


def load_opportunity_truth(path: Path, *, session_id: str | None = None) -> OpportunityTruthLog:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load opportunity truth: {exc}") from exc
    truth = OpportunityTruthLog.from_dict(raw)
    if session_id is not None and truth.source_session_id != session_id:
        raise ValueError("opportunity truth does not belong to the selected session")
    return truth


def derive_self_opportunities(
    session: Path,
) -> OpportunityTruthLog:
    """Derive the denominator from source truth actions, never runtime requests."""

    session = Path(session)
    raw = json.loads((session / "truth_log.json").read_text(encoding="utf-8"))
    session_id = str(raw.get("source_session_id", session.name))
    turns = raw.get("turns", ())
    if not isinstance(turns, list):
        raise ValueError("truth_log turns must be an array")
    frame_times = _frame_time_index(session / "video" / "frame_index.jsonl")
    opportunities: list[OpportunityTruth] = []
    for position, item in enumerate(turns):
        if not isinstance(item, dict) or str(item.get("actor", "")) != "self":
            continue
        frame_indices = _truth_frames(item)
        if not frame_indices:
            raise ValueError("self opportunity truth requires source frame evidence")
        action_frame = min(frame_indices)
        previous = turns[position - 1] if position else None
        previous_frames = _truth_frames(previous) if isinstance(previous, dict) else ()
        start_frame = max(previous_frames) if previous_frames else min(frame_times)
        start_ms = frame_times.get(start_frame)
        action_ms = frame_times.get(action_frame)
        if start_ms is None or action_ms is None:
            raise ValueError("truth frame is absent from source frame index")
        determinable = not bool(item.get("uncertainty"))
        is_pass = bool(item.get("is_pass", False))
        opportunities.append(OpportunityTruth(
            opportunity_id=f"truth-self-{int(item.get('turn_id', position + 1)):04d}",
            start_source_frame=start_frame,
            end_source_frame=action_frame,
            start_source_monotonic_ms=start_ms,
            end_source_monotonic_ms=action_ms,
            expected_response="model_or_local_pass" if is_pass else "model_advice",
            determinable=determinable,
            manual_evidence=ManualOpportunityEvidence(
                source="truth_log_action_chain",
                frame_indices=tuple(sorted(set((*previous_frames, *frame_indices)))),
                note="derived from the trusted source action sequence",
            ),
        ))
    return OpportunityTruthLog(session_id, tuple(opportunities))


def unavailable_opportunity_acceptance(reason: str) -> dict[str, object]:
    return {
        "schema": "guandan.opportunity-acceptance/1",
        "available": False,
        "passed": False,
        "reason": str(reason),
        "denominators": {
            "all_opportunities": 0,
            "determinable_opportunities": 0,
            "indeterminate_opportunities": 0,
        },
        "counts": {},
        "outcomes": {},
        "latency_ms": {},
        "rows": [],
    }


def _frame_time_index(path: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        result[int(item["frame_index"])] = int(item["monotonic_ms"])
    return result


def _truth_frames(item: dict[str, object] | None) -> tuple[int, ...]:
    if not item:
        return ()
    evidence = item.get("evidence", {})
    if isinstance(evidence, dict):
        frames = evidence.get("frame_indices", ())
        if isinstance(frames, list):
            return tuple(int(value) for value in frames)
    frame = item.get("frame_index")
    return () if frame is None else (int(frame),)


def _response_matches(expected: ExpectedResponse, observed: ObservedResponse) -> bool:
    return bool(
        (expected == "model_advice" and observed == "model_advice")
        or (expected == "local_pass" and observed == "local_pass")
        or (expected == "model_or_local_pass" and observed in {"model_advice", "local_pass"})
    )


def _response_dict(response: OpportunityResponseRecord | None) -> dict[str, object] | None:
    if response is None:
        return None
    return {
        "opportunity_id": response.opportunity_id,
        "response": response.response,
        "response_monotonic_ms": response.response_monotonic_ms,
        "valid": response.valid,
        "evidence_id": response.evidence_id,
        "detail": response.detail,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not samples:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": len(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "max": samples[-1],
    }


def _percentile(samples: list[float], quantile: float) -> float:
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _required_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


__all__ = [
    "ExpectedResponse",
    "ManualOpportunityEvidence",
    "ObservedResponse",
    "OpportunityOutcome",
    "OpportunityResponseRecord",
    "OpportunityTruth",
    "OpportunityTruthLog",
    "evaluate_opportunity_responses",
    "derive_self_opportunities",
    "load_opportunity_truth",
    "save_opportunity_truth",
    "unavailable_opportunity_acceptance",
]
