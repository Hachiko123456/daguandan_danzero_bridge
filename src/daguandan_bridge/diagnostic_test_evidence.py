"""Small, deterministic test evidence derived from a sealed session.

The evidence is intentionally metadata-only by default.  It is a diagnostic
projection, never a source of truth: values produced by the running program
are labelled ``unverified`` even when a human-maintained ``truth_log.json``
is available alongside the session.
"""

from __future__ import annotations

import hashlib
import json
import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


TEST_EVIDENCE_SCHEMA = "guandan.test-evidence/1"
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_RECORDS = 2_000
DEFAULT_MAX_ANOMALIES = 256

_FORMAL_EVENTS = frozenset({"player_played", "player_passed"})
_REJECTION_EVENTS = frozenset(
    {"rule_rejection", "turn_rejected", "advice_withheld", "turn_desynchronized"}
)
_RECOVERY_EVENTS = frozenset(
    {
        "recovery_budget_exceeded",
        "turn_recovery_budget_exceeded",
        "advice_recovery_target_exceeded",
        "terminal_history_gap",
    }
)
_SAFE_SEATS = frozenset({"self", "left", "right", "opposite"})


@dataclass(frozen=True)
class TestEvidenceBundle:
    """Bounded virtual files to add to a diagnostic archive."""

    files: Mapping[str, bytes]
    record_counts: Mapping[str, int]
    truncated: bool = False


def build_test_evidence(
    session_directory: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_records: int = DEFAULT_MAX_RECORDS,
    max_anomalies: int = DEFAULT_MAX_ANOMALIES,
) -> TestEvidenceBundle:
    """Build bounded metadata files without modifying ``session_directory``."""

    session = Path(session_directory).resolve(strict=True)
    if not session.is_dir():
        raise ValueError("session directory must be a directory")
    max_bytes = max(16 * 1024, int(max_bytes))
    max_records = max(1, int(max_records))
    max_anomalies = max(1, int(max_anomalies))
    manifest = _read_object(session / "manifest.json")
    timeline = _read_jsonl(session / "timeline.jsonl", limit=max_records * 4)
    observations_path = session / "observations.jsonl.gz"
    if not observations_path.is_file():
        observations_path = session / "observations.jsonl.part"
    raw_observations = _read_observations(
        observations_path, limit=max_records * 4
    )
    observations = _flatten_observations(raw_observations, max_records=max_records * 8)
    advice = _read_jsonl(session / "advice.jsonl", limit=max_records * 2)
    decisions = _read_jsonl(session / "decisions.jsonl", limit=max_records * 2)
    health = _read_object(session / "health_audit.json")
    truth = _read_object(session / "truth_log.json")

    formal = [item for item in timeline if item.get("event_type") in _FORMAL_EVENTS]
    truth_turns = _truth_turns(truth)
    evidence = {
        "test_evidence/session_facts.json": _json_bytes(
            _session_facts(
                session,
                manifest=manifest,
                timeline=timeline,
                observations=observations,
                advice=advice,
                decisions=decisions,
                health=health,
                truth=truth,
            )
        ),
        "test_evidence/opening_evidence.jsonl": _jsonl_bytes(
            _opening_records(timeline, observations), max_records=max_records
        ),
        "test_evidence/proposed_actions.jsonl": _jsonl_bytes(
            _proposed_actions(formal, truth_turns), max_records=max_records
        ),
        "test_evidence/action_evidence.jsonl": _jsonl_bytes(
            _action_evidence(formal, observations), max_records=max_records
        ),
        "test_evidence/anomaly_windows.jsonl": _jsonl_bytes(
            _anomalies(
                timeline,
                observations,
                formal,
                max_anomalies=max_anomalies,
            ),
            max_records=max_anomalies,
        ),
        "test_evidence/advice_input_output.jsonl": _jsonl_bytes(
            _advice_records(advice, decisions), max_records=max_records
        ),
    }
    # Do not add a user-facing template to every archive.  It is easy to
    # generate later, and keeping it out makes the default export smaller.
    bounded, truncated = _bound_files(evidence, max_bytes=max_bytes)
    return TestEvidenceBundle(
        files=bounded,
        record_counts={name: _line_count(data) for name, data in bounded.items()},
        truncated=truncated,
    )


def _session_facts(
    session: Path,
    *,
    manifest: Mapping[str, object],
    timeline: list[dict[str, object]],
    observations: list[dict[str, object]],
    advice: list[dict[str, object]],
    decisions: list[dict[str, object]],
    health: Mapping[str, object],
    truth: Mapping[str, object],
) -> dict[str, object]:
    initial = next(
        (item for item in timeline if item.get("event_type") == "initial_state_confirmed"),
        {},
    )
    lead = next(
        (item for item in timeline if item.get("event_type") == "lead_player_confirmed"),
        {},
    )
    truth_turns = _truth_turns(truth)
    return {
        "schema": TEST_EVIDENCE_SCHEMA,
        "session_id": session.name,
        "source": "program-generated",
        "truth_status": "unverified",
        "truth_log_present": bool(truth),
        "truth_log_label_status": truth.get("label_status") if truth else None,
        "truth_log_turn_count": len(truth_turns),
        "manifest": {
            key: manifest.get(key)
            for key in ("status", "started_at", "finished_at", "frame_count", "dropped_frames")
            if key in manifest
        },
        "opening": {
            "lead_player": lead.get("actor") or _payload_value(lead, "lead_player"),
            "lead_event_id": lead.get("event_id"),
            "initial_state_event_id": initial.get("event_id"),
            "opening_evidence_refs": _string_list(lead.get("evidence_refs")),
        },
        "counts": {
            "timeline_events": len(timeline),
            "observations": len(observations),
            "formal_actions": sum(1 for item in timeline if item.get("event_type") in _FORMAL_EVENTS),
            "advice_records": len(advice),
            "decision_records": len(decisions),
        },
        "health": {
            "status": health.get("status", "UNKNOWN"),
            "issue_codes": [
                item.get("code")
                for item in health.get("issues", [])
                if isinstance(item, Mapping) and item.get("code")
            ],
        },
    }


def _opening_records(
    timeline: list[dict[str, object]], observations: list[dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for event in timeline:
        if event.get("event_type") not in {
            "initial_state_confirmed",
            "lead_player_confirmed",
            "player_played",
            "player_passed",
        }:
            continue
        if len(records) >= 32:
            break
        records.append(
            {
                "source": "program-generated",
                "truth_status": "unverified",
                "phase": "opening",
                "event_type": event.get("event_type"),
                "event_id": event.get("event_id"),
                "actor": event.get("actor"),
                "turn_id": event.get("turn_id"),
                "frame_indices": _frame_indices(event, observations),
                "evidence_refs": _string_list(event.get("evidence_refs")),
                "confidence": event.get("confidence"),
            }
        )
    return records


def _proposed_actions(
    formal: list[dict[str, object]], truth_turns: list[dict[str, object]]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, event in enumerate(formal, start=1):
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        cards = _string_list(payload.get("cards"))
        item: dict[str, object] = {
            "source": "program-generated",
            "truth_status": "unverified",
            "turn_id": event.get("turn_id") or index,
            "trick_id": event.get("trick_id"),
            "actor": event.get("actor"),
            "cards": cards,
            "is_pass": event.get("event_type") == "player_passed" or bool(payload.get("is_pass")),
            "confidence": event.get("confidence"),
            "evidence_refs": _string_list(event.get("evidence_refs")),
            "monotonic_ms": event.get("monotonic_ms"),
        }
        if index <= len(truth_turns):
            expected = truth_turns[index - 1]
            item["reference_truth"] = {
                "available": True,
                "actor": expected.get("actor"),
                "cards": _string_list(expected.get("cards")),
                "is_pass": bool(expected.get("is_pass")),
                "match": _action_matches(item, expected),
            }
        else:
            item["reference_truth"] = {"available": False}
        records.append(item)
    return records


def _action_evidence(
    formal: list[dict[str, object]], observations: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_id = {str(item.get("id")): item for item in observations if item.get("id")}
    output: list[dict[str, object]] = []
    for event in formal:
        refs = _string_list(event.get("evidence_refs"))
        linked = [by_id[ref] for ref in refs if ref in by_id]
        output.append(
            {
                "source": "program-generated",
                "truth_status": "unverified",
                "event_id": event.get("event_id"),
                "actor": event.get("actor"),
                "turn_id": event.get("turn_id"),
                "evidence_refs": refs,
                "observation_count": len(linked),
                "observation_ids": refs[:16],
                "observation_players": sorted(
                    {str(item.get("player")) for item in linked if item.get("player")}
                ),
                "observation_sources": sorted(
                    {str(item.get("source")) for item in linked if item.get("source")}
                ),
                "frame_indices": _frame_indices(event, observations),
                "first_observed_ms": _first_number(linked, "monotonic_ms"),
                "last_observed_ms": _last_number(linked, "monotonic_ms"),
                "confidence_min": _min_number(linked, "confidence"),
            }
        )
    return output


def _advice_records(
    advice: list[dict[str, object]], decisions: list[dict[str, object]]
) -> list[dict[str, object]]:
    outputs: dict[str, dict[str, object]] = {}
    for item in decisions:
        key = str(item.get("request_id") or item.get("decision_id") or len(outputs))
        outputs[key] = {
            "decision_id": item.get("decision_id"),
            "status": item.get("status"),
            "model_advice": _safe_action(item.get("model_advice")),
            "state_revision": item.get("state_revision"),
            "turn_id": item.get("turn_id"),
        }
    records: list[dict[str, object]] = []
    for item in advice:
        key = str(item.get("request_id") or item.get("decision_id") or len(records))
        record = {
            "source": "program-generated",
            "truth_status": "unverified",
            "request_id": item.get("request_id"),
            "status": item.get("status"),
            "turn_id": item.get("turn_id"),
            "state_revision": item.get("state_revision"),
            "actor": item.get("actor"),
            "reason": item.get("reason") or item.get("error"),
            "model": item.get("model") or item.get("strategy"),
            "output": _safe_action(item.get("cards") and item or None),
        }
        if key in outputs:
            record["decision"] = outputs[key]
        records.append(record)
    return records


def _anomalies(
    timeline: list[dict[str, object]],
    observations: list[dict[str, object]],
    formal: list[dict[str, object]],
    *,
    max_anomalies: int,
) -> list[dict[str, object]]:
    anomalies: list[dict[str, object]] = []
    lead_index = next(
        (
            index
            for index, item in enumerate(timeline)
            if item.get("event_type") == "lead_player_confirmed"
            or (
                item.get("event_type") == "initial_state_confirmed"
                and _payload_value(item, "lead_player") in _SAFE_SEATS
            )
        ),
        len(timeline),
    )
    first_formal_index = next(
        (index for index, item in enumerate(timeline) if item.get("event_type") in _FORMAL_EVENTS),
        len(timeline),
    )
    lead = _event_actor(timeline[lead_index]) if lead_index < len(timeline) else None
    lead_ms = (
        timeline[lead_index].get("monotonic_ms")
        if lead_index < len(timeline)
        else None
    )
    first_formal_ms = (
        timeline[first_formal_index].get("monotonic_ms")
        if first_formal_index < len(timeline)
        else None
    )
    for observation in observations:
        if len(anomalies) >= max_anomalies:
            break
        player = str(observation.get("player") or "")
        cards = _string_list(observation.get("cards"))
        source = str(observation.get("source") or "")
        observed_ms = observation.get("monotonic_ms")
        before_first_formal = (
            isinstance(observed_ms, (int, float))
            and isinstance(first_formal_ms, (int, float))
            and observed_ms < first_formal_ms
        )
        if (
            lead
            and first_formal_index > lead_index
            and before_first_formal
            and cards
            and player in _SAFE_SEATS
            and player != lead
        ):
            anomalies.append(_anomaly("preopening_foreign_candidate", observation, player=player))
            break
    frame_map = {id(event): set(_frame_indices(event, observations)) for event in formal}
    for left_index, left in enumerate(formal):
        for right in formal[left_index + 1 :]:
            left_frames = frame_map[id(left)]
            overlap = left_frames.intersection(_frame_indices(right, observations))
            if overlap and left.get("actor") != right.get("actor"):
                anomalies.append(
                    _anomaly(
                        "same_frame_multi_seat",
                        left,
                        second_event_id=right.get("event_id"),
                        actors=[left.get("actor"), right.get("actor")],
                        frame_indices=sorted(overlap)[:16],
                    )
                )
                break
        if len(anomalies) >= max_anomalies:
            break
    _append_event_anomalies(anomalies, timeline, _REJECTION_EVENTS, "rule_rejection_streak")
    _append_event_anomalies(anomalies, timeline, _RECOVERY_EVENTS, "recovery_budget_exceeded")
    _append_event_anomalies(
        anomalies,
        timeline,
        {"turn_cursor_invariant_violation", "history_integrity_failed", "turn_desynchronized"},
        "cursor_invariant_violation",
    )
    _append_event_anomalies(
        anomalies,
        timeline,
        {"advice_withheld", "advice_on_untrusted_history"},
        "untrusted_advice",
    )
    return anomalies[:max_anomalies]


def _append_event_anomalies(
    target: list[dict[str, object]],
    timeline: list[dict[str, object]],
    event_types: set[str] | frozenset[str],
    anomaly_type: str,
) -> None:
    for event in timeline:
        if event.get("event_type") in event_types:
            target.append(_anomaly(anomaly_type, event))


def _anomaly(kind: str, item: Mapping[str, object], **extra: object) -> dict[str, object]:
    return {
        "source": "program-generated",
        "truth_status": "unverified",
        "type": kind,
        "severity": "high" if kind in {"same_frame_multi_seat", "cursor_invariant_violation"} else "medium",
        "event_id": item.get("event_id") or item.get("id"),
        "actor": item.get("actor") or item.get("player"),
        "turn_id": item.get("turn_id"),
        "monotonic_ms": item.get("monotonic_ms"),
        "evidence_refs": _string_list(item.get("evidence_refs")),
        **extra,
    }


def _frame_indices(event: Mapping[str, object], observations: list[dict[str, object]]) -> list[int]:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    raw = payload.get("frame_indices") or event.get("frame_indices")
    values = [value for value in (raw or []) if isinstance(value, int)] if isinstance(raw, list) else []
    if values:
        return sorted(set(values))[:32]
    refs = set(_string_list(event.get("evidence_refs")))
    linked = [item for item in observations if str(item.get("id")) in refs]
    linked_frames = sorted(
        {
            int(item[key])
            for item in linked
            for key in ("frame_index", "frame_seq")
            if isinstance(item.get(key), int)
        }
    )[:32]
    if linked_frames:
        return linked_frames
    extracted: set[int] = set()
    for ref in refs:
        extracted.update(
            int(value)
            for value in re.findall(r":(\d+):live-v2(?:[:.]|$)", ref)
        )
    return sorted(extracted)[:32]


def _truth_turns(truth: Mapping[str, object]) -> list[dict[str, object]]:
    raw = truth.get("turns") if isinstance(truth, Mapping) else None
    return [dict(item) for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []


def _action_matches(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    return (
        actual.get("actor") == expected.get("actor")
        and bool(actual.get("is_pass")) == bool(expected.get("is_pass"))
        and _string_list(actual.get("cards")) == _string_list(expected.get("cards"))
    )


def _safe_action(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = ("cards", "is_pass", "play_type", "strategy", "status", "reason")
    return {key: value.get(key) for key in allowed if key in value}


def _payload_value(event: Mapping[str, object], key: str) -> object:
    payload = event.get("payload")
    return payload.get(key) if isinstance(payload, Mapping) else None


def _event_actor(event: Mapping[str, object]) -> str | None:
    value = event.get("actor") or _payload_value(event, "lead_player")
    return str(value) if value in _SAFE_SEATS else None


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _first_number(items: Iterable[Mapping[str, object]], key: str) -> int | float | None:
    values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
    return min(values) if values else None


def _last_number(items: Iterable[Mapping[str, object]], key: str) -> int | float | None:
    values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
    return max(values) if values else None


def _min_number(items: Iterable[Mapping[str, object]], key: str) -> float | None:
    values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
    return min(values) if values else None


def _read_object(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if len(rows) >= limit:
                    break
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, Mapping):
                    rows.append(dict(raw))
    except OSError:
        return []
    return rows


def _read_observations(path: Path, *, limit: int) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            rows: list[dict[str, object]] = []
            for line in handle:
                if len(rows) >= limit:
                    break
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, Mapping):
                    rows.append(dict(raw))
            return rows
    except (OSError, gzip.BadGzipFile):
        return []


def _flatten_observations(
    rows: Iterable[Mapping[str, object]], *, max_records: int
) -> list[dict[str, object]]:
    """Expose nested live-v2 engine observations as bounded evidence rows."""

    output: list[dict[str, object]] = []
    for row in rows:
        if len(output) >= max_records:
            break
        if row.get("id") or row.get("player") or row.get("cards"):
            output.append(dict(row))
        update_id = str(row.get("update_sequence", ""))
        for index, nested in enumerate(row.get("observations", ())):
            if len(output) >= max_records:
                break
            if not isinstance(nested, Mapping):
                continue
            item = dict(nested)
            item.setdefault("id", f"update:{update_id}:observation:{index}")
            item.setdefault("player", item.get("seat"))
            item.setdefault("frame_index", item.get("frame_sequence"))
            item.setdefault("monotonic_ms", row.get("captured_ms"))
            item.setdefault("source", "live-v2-engine-observation")
            output.append(item)
        for index, nested in enumerate(row.get("candidates", ())):
            if len(output) >= max_records:
                break
            if not isinstance(nested, Mapping):
                continue
            item = dict(nested)
            item.setdefault("id", f"update:{update_id}:candidate:{index}")
            item.setdefault("player", item.get("seat"))
            item.setdefault("frame_index", item.get("last_frame", {}).get("frame_sequence") if isinstance(item.get("last_frame"), Mapping) else None)
            item.setdefault("monotonic_ms", row.get("captured_ms"))
            item.setdefault("source", "live-v2-candidate")
            output.append(item)
    return output


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: Iterable[Mapping[str, object]], *, max_records: int) -> bytes:
    output = bytearray()
    for index, row in enumerate(rows):
        if index >= max_records:
            break
        output.extend(_json_bytes(row))
    return bytes(output)


def _bound_files(files: Mapping[str, bytes], *, max_bytes: int) -> tuple[dict[str, bytes], bool]:
    # Keep every named stream represented, but truncate records at line
    # boundaries if an unusually noisy old session exceeds the package cap.
    remaining = max_bytes
    output: dict[str, bytes] = {}
    truncated = False
    for name, data in files.items():
        if remaining <= 0:
            output[name] = b""
            truncated = True
            continue
        if len(data) <= remaining:
            output[name] = data
            remaining -= len(data)
            continue
        output[name] = _fit_lines(data, remaining)
        remaining -= len(output[name])
        truncated = True
    return output, truncated


def _fit_lines(data: bytes, limit: int) -> bytes:
    if limit <= 0:
        return b""
    lines = data.splitlines(keepends=True)
    output = bytearray()
    for line in lines:
        if len(output) + len(line) > limit:
            break
        output.extend(line)
    return bytes(output)


def _line_count(data: bytes) -> int:
    return data.count(b"\n")


__all__ = ["DEFAULT_MAX_ANOMALIES", "DEFAULT_MAX_BYTES", "DEFAULT_MAX_RECORDS", "TEST_EVIDENCE_SCHEMA", "TestEvidenceBundle", "build_test_evidence"]
