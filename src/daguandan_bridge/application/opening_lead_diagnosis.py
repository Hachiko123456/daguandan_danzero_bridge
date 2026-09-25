from __future__ import annotations

"""Read-only opening-lead diagnosis for recorded sessions.

This module deliberately sits beside the live opening gate.  It reads the
recorded trace and, when a frame is requested, runs the existing recognition
service without changing thresholds, opening state, or the source session.
"""

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "guandan.opening-lead-diagnosis/v1"
DEFAULT_PLAY_THRESHOLD = 0.80
DEFAULT_MARKER_THRESHOLD = 0.80


@dataclass(frozen=True)
class _FrameEvidence:
    frame_index: int
    frame_index_record: dict[str, Any] | None
    recognition: dict[str, Any]
    trace: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not a JSON object")
            rows.append(value)
    return rows


def _json_value(value: object) -> object:
    """Convert recognition dataclasses/enums to stable JSON primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value"):
        return _json_value(getattr(value, "value"))
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return str(value)


def _first_value(rows: Iterable[dict[str, Any]], *keys: str) -> object:
    for row in rows:
        for key in keys:
            if key in row and row[key] not in (None, ""):
                return row[key]
    return None


def _is_opening_unrecognized(row: dict[str, Any]) -> bool:
    diagnostics = row.get("diagnostics") or ()
    text = " ".join(str(item) for item in diagnostics).lower()
    if any(token in text for token in ("首出", "opening", "lead_player", "first_play")):
        return True
    if "lead_player" in row:
        return row.get("lead_player") in (None, "", "unknown")
    return False


def _trace_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hand27_rows = [row for row in rows if row.get("hand_count") == 27]
    opening_unrecognized_rows = [row for row in rows if _is_opening_unrecognized(row)]
    actions: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        for action in row.get("events") or row.get("actions") or ():
            if not isinstance(action, dict):
                continue
            cards = list(action.get("cards") or ())
            confidence = _as_float(action.get("confidence"))
            if cards and not action.get("is_pass", False):
                actions.append({
                    "source": "recognition_trace",
                    "trace_row": row_number,
                    "actor": action.get("player") or action.get("actor"),
                    "cards": cards,
                    "confidence": confidence,
                    "reliable": confidence is not None and confidence >= DEFAULT_PLAY_THRESHOLD,
                })
    reliable = next((item for item in actions if item["reliable"]), None)
    return {
        "trace_rows": len(rows),
        "hand27_count": len(hand27_rows),
        "hand_count_histogram": {
            str(key): count
            for key, count in sorted(Counter(row.get("hand_count") for row in rows).items(), key=lambda item: str(item[0]))
        },
        "opening_unrecognized_count": len(opening_unrecognized_rows),
        "first_reliable_play_action": reliable,
        "last_trace": rows[-1] if rows else None,
    }


def _as_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _candidate_digest(candidates: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    selected = [item for item in candidates if str(item.get("field", "")).startswith(prefix)]
    best = max(selected, key=lambda item: _as_float(item.get("score")) or 0.0, default=None)
    result: dict[str, Any] = {
        "candidate_count": len(selected),
        "accepted_count": sum(1 for item in selected if item.get("accepted")),
    }
    if best is None:
        return result
    score = _as_float(best.get("score"))
    threshold = _as_float(best.get("threshold"))
    below_threshold = (
        score is not None and threshold is not None and score < threshold
    )
    result.update({
        "score": score,
        "threshold": threshold,
        "accepted": bool(best.get("accepted")),
        "below_threshold": below_threshold,
        "reason": best.get("rejection_reason"),
        "source": best.get("source"),
        "match_box": best.get("match_box"),
    })
    if below_threshold:
        result["explanation"] = (
            f"首出标志低于阈值（{score:.4f} < {threshold:.4f}），因此首出座位未确认。"
            if prefix.startswith("first_play_")
            else f"候选低于阈值（{score:.4f} < {threshold:.4f}）。"
        )
    return result


def _best_prefixed_digest(
    candidates: list[dict[str, Any]], prefixes: tuple[str, ...]
) -> dict[str, Any]:
    per_prefix = {
        prefix: _candidate_digest(candidates, prefix=prefix)
        for prefix in prefixes
        if any(str(item.get("field", "")).startswith(prefix) for item in candidates)
    }
    best_prefix = max(
        per_prefix,
        key=lambda prefix: per_prefix[prefix].get("score") or 0.0,
        default=None,
    )
    if best_prefix is None:
        return {"candidate_player": None, "by_player": {}}
    return {
        "candidate_player": best_prefix.rsplit("_", 1)[-1],
        **per_prefix[best_prefix],
        "by_player": per_prefix,
    }


def _recognition_dict(result: object, trace: dict[str, Any] | None) -> dict[str, Any]:
    events = []
    for event in getattr(result, "events", ()) or ():
        events.append({
            "player": _json_value(getattr(event, "player", None)),
            "cards": list(getattr(event, "cards", ()) or ()),
            "is_pass": bool(getattr(event, "is_pass", False)),
            "confidence": _as_float(getattr(event, "confidence", None)),
            "source": str(getattr(event, "source", "") or ""),
        })
    candidates = list((trace or {}).get("candidates") or ())
    marker = _best_prefixed_digest(
        candidates,
        ("first_play_self", "first_play_left", "first_play_opposite", "first_play_right"),
    )
    timer = _best_prefixed_digest(
        candidates,
        ("timer_self", "timer_left", "timer_opposite", "timer_right"),
    )
    first_reliable = next(
        (
            {
                **event,
                "reliable": True,
            }
            for event in events
            if event["cards"] and not event["is_pass"]
            and event["confidence"] is not None
            and event["confidence"] >= DEFAULT_PLAY_THRESHOLD
        ),
        None,
    )
    accepted_play_candidates = [
        item for item in candidates
        if str(item.get("field", "")).endswith("_play") and item.get("accepted")
    ]
    return {
        "round_level": _json_value(getattr(result, "round_level", None)),
        "hand": list(getattr(result, "my_hand", ()) or ()),
        "hand_count": len(getattr(result, "my_hand", ()) or ()),
        "current_player": _json_value(getattr(result, "current_player", None)),
        "lead_player": _json_value(getattr(result, "lead_player", None)),
        "field_confidences": _json_value(getattr(result, "field_confidences", {})),
        "unresolved_fields": list(getattr(result, "unresolved_fields", ()) or ()),
        "diagnostics": list(getattr(result, "diagnostics", ()) or ()),
        "events": events,
        "first_reliable_play_action": first_reliable,
        "marker": marker,
        "timer": timer,
        "card_evidence": {
            "accepted_candidate_count": len(accepted_play_candidates),
            "actions": events,
        },
        "trace_schema": (trace or {}).get("schema"),
    }


def _locate_profile_root(session_dir: Path) -> Path | None:
    candidate = session_dir.parent.parent.parent
    if (candidate / "regions_config.json").is_file() and (candidate / "templates_config.json").is_file():
        return candidate
    return None


def _analyze_frame(session_dir: Path, frame_index: int) -> _FrameEvidence:
    import cv2

    video_path = session_dir / "video" / "game.avi"
    if not video_path.is_file():
        raise FileNotFoundError(f"recorded video not found: {video_path}")
    index_rows = _read_jsonl(session_dir / "video" / "frame_index.jsonl")
    index_record = next((row for row in index_rows if row.get("frame_index") == frame_index), None)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"unable to open recorded video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise ValueError(f"frame {frame_index} is not available in {video_path}")

    profile_root = _locate_profile_root(session_dir)
    if profile_root is None:
        raise FileNotFoundError(
            "cannot infer recognition profile from session path; expected profile regions_config.json"
        )
    import sys
    from daguandan_bridge.annotation_service import AnnotationService
    from daguandan_bridge.recognition_service import ScreenshotRecognitionService
    from daguandan_bridge.template_service import TemplateService

    profile_name = profile_root.name
    profiles_root = profile_root.parent
    service = ScreenshotRecognitionService(
        AnnotationService(profiles_root, profile_name),
        TemplateService(profiles_root, profile_name),
        diagnostic_tracing=True,
    )
    result = service.recognize(image, allow_unknown_suit=True)
    trace = service.get_last_diagnostic_trace() or {}
    return _FrameEvidence(
        frame_index=frame_index,
        frame_index_record=index_record,
        recognition=_recognition_dict(result, trace),
        trace=trace,
    )


def diagnose_session(session_dir: str | Path, frame: int | None = None) -> dict[str, Any]:
    """Return a JSON-compatible, read-only diagnosis for one session directory."""

    root = Path(session_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"session directory not found: {root}")
    trace_path = root / "recognition_trace.jsonl"
    frame_index_path = root / "video" / "frame_index.jsonl"
    video_path = root / "video" / "game.avi"
    for path in (trace_path, frame_index_path):
        if not path.is_file():
            raise FileNotFoundError(f"required session evidence not found: {path}")
    trace_rows = _read_jsonl(trace_path)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "session_dir": str(root),
        "source_files": {
            "game_avi": str(video_path),
            "frame_index": str(frame_index_path),
            "recognition_trace": str(trace_path),
        },
        "read_only": True,
        "summary": _trace_summary(trace_rows),
        "frame": None,
    }
    if frame is not None:
        if frame < 0:
            raise ValueError("frame must be non-negative")
        evidence = _analyze_frame(root, int(frame))
        result["frame"] = {
            "frame_index": evidence.frame_index,
            "frame_index_record": evidence.frame_index_record,
            "recognition": evidence.recognition,
        }
        if result["summary"]["first_reliable_play_action"] is None:
            result["summary"]["first_reliable_play_action"] = evidence.recognition["first_reliable_play_action"]
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Read-only diagnosis of opening lead evidence in a recorded session."
    )
    parser.add_argument("session", type=Path, help="session directory containing recognition_trace.jsonl and video/")
    parser.add_argument("--frame", type=int, default=None, help="optional zero-based video frame to inspect")
    args = parser.parse_args(argv)
    try:
        report = diagnose_session(args.session, frame=args.frame)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = ["SCHEMA", "diagnose_session", "main"]
