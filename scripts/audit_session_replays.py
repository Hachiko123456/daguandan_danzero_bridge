"""Replay the fixed regression sessions without writing into their stores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.advisor_strategy import build_advisor
from daguandan_bridge.live.replay import VideoReplaySource, replay_truth_through_live_advisor, replay_video_through_live_pipeline
from daguandan_bridge.live.session_store import read_json_lines
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService


PROFILE_NAME = "tencent_daguandan"
SOURCE_PROFILES_ROOT = PROJECT_ROOT / "data" / "profiles"
RELEASE_PROFILES_ROOT = (
    PROJECT_ROOT / "release" / "dist" / "DaguandanAssistant" / "data" / "profiles"
)
TARGETS = (
    ("20260823_125114_dc9160", RELEASE_PROFILES_ROOT),
    ("20260823_125412_6b9274", RELEASE_PROFILES_ROOT),
    ("20260814_004447_aab3dc", SOURCE_PROFILES_ROOT),
    ("20260816_125402_1687ea", SOURCE_PROFILES_ROOT),
    ("20260820_135517_d19c22", SOURCE_PROFILES_ROOT),
    ("20260822_002135_9c2328", SOURCE_PROFILES_ROOT),
    ("20260822_142942_e356d4", SOURCE_PROFILES_ROOT),
)
ACTION_TYPES = {"player_played", "player_passed", "manual_confirmed_event"}


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _stored_initial_state(session: Path) -> dict[str, object]:
    for event in read_json_lines(session / "timeline.jsonl"):
        if event.get("event_type") == "initial_state_confirmed":
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                return {
                    "round_level": str(payload.get("round_level", "")),
                    "hand": sorted(str(card) for card in payload.get("hand", ())),
                }
    raise ValueError("timeline.jsonl lacks initial_state_confirmed")


def _opening_agreement(
    session: Path,
    recognition: ScreenshotRecognitionService,
    stored: dict[str, object],
) -> dict[str, object]:
    source = VideoReplaySource(
        session / "video" / "game.avi",
        session / "video" / "frame_index.jsonl",
    )
    frames = iter(source.frames())
    reads: list[dict[str, object]] = []
    try:
        for _ in range(2):
            item = next(frames, None)
            if item is None:
                break
            record, frame = item
            result = recognition.recognize(frame, allow_unknown_suit=True)
            level = str(getattr(result, "round_level", "") or "")
            hand = sorted(str(card) for card in getattr(result, "my_hand", ()) or ())
            reads.append(
                {
                    "frame_index": record.frame_index,
                    "monotonic_ms": record.monotonic_ms,
                    "round_level": level,
                    "hand": hand,
                    "matches_stored_level": level == stored["round_level"],
                    "matches_stored_hand": hand == stored["hand"],
                }
            )
    finally:
        close = getattr(frames, "close", None)
        if callable(close):
            close()
    two_frame_agreement = len(reads) == 2 and (
        reads[0]["round_level"], reads[0]["hand"]
    ) == (reads[1]["round_level"], reads[1]["hand"])
    return {
        "stored_initial_state": stored,
        "reads": reads,
        "two_frame_agreement": two_frame_agreement,
        "two_frame_matches_stored": bool(
            two_frame_agreement
            and all(
                item["matches_stored_level"] and item["matches_stored_hand"]
                for item in reads
            )
        ),
    }


def _replayed_events(frame_log: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in read_json_lines(frame_log):
        raw_events = record.get("events", ())
        if not isinstance(raw_events, list):
            continue
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id", ""))
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            events.append(event)
    return events


def _event_summary(events: list[dict[str, object]]) -> dict[str, object]:
    types = Counter(str(event.get("event_type", "unknown")) for event in events)
    return {
        "actions": sum(types[name] for name in ACTION_TYPES),
        "plays": types["player_played"],
        "passes": types["player_passed"],
        "lifecycle": {
            "player_finished": types["player_finished"],
            "wind_caught": types["wind_caught"],
        },
        "event_types": dict(sorted(types.items())),
    }


def _comparison_summary(comparison: object) -> dict[str, object]:
    return {
        "identical_turn_ids": list(comparison.identical_turn_ids),
        "missing": len(comparison.missing),
        "added": len(comparison.added),
        "changed": len(comparison.changed),
        "metric_deltas": len(comparison.metric_deltas),
    }


def _source_advice_statuses(session: Path) -> dict[str, int]:
    statuses = Counter(
        str(record.get("status", "unknown"))
        for record in read_json_lines(session / "advice.jsonl")
    )
    return dict(sorted(statuses.items()))


def _truth_advisor_audit(
    session: Path,
    profiles_root: Path,
    output_root: Path,
) -> dict[str, object]:
    truth_path = session / "truth_log.json"
    if not truth_path.is_file():
        return {"available": False}
    advisor = build_advisor(
        "fabledan",
        profiles_root=profiles_root,
        profile_name=PROFILE_NAME,
        fabledan_diagnostics="full",
    )
    # The replay uses a temporary LiveSessionStore.  Avoid creating the
    # advisor's optional global decision trace while auditing a read-only run.
    if hasattr(advisor, "write_decision_log"):
        advisor.write_decision_log = False
    result = replay_truth_through_live_advisor(
        session,
        advisor,
        truth_log=truth_path,
        output_root=output_root,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    audit_info = getattr(advisor, "audit_info", None)
    return {
        "available": True,
        "run_directory": result.run_directory,
        "completed": result.completed,
        "turn_count": result.turn_count,
        "processed_turn_count": result.processed_turn_count,
        "advice": {
            "requested": result.advice_requested,
            "ready": result.advice_ready,
            "failed": result.advice_failed,
            "stale": result.advice_stale,
            "timeouts": result.advice_timeouts,
            "statuses": summary.get("advice_statuses", {}),
        },
        "advisor": audit_info() if callable(audit_info) else {},
    }


def _audit_session(session_id: str, profiles_root: Path, output_root: Path) -> dict[str, object]:
    session = profiles_root / PROFILE_NAME / "sessions" / f"game_{session_id}"
    session_output = output_root / f"game_{session_id}"
    session_output.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {
        "session_id": session_id,
        "session": session,
        "profiles_root": profiles_root,
        "output_directory": session_output,
        "source_advice_statuses": _source_advice_statuses(session),
    }
    try:
        stored = _stored_initial_state(session)
        annotation = AnnotationService(profiles_root, PROFILE_NAME)
        templates = TemplateService(profiles_root, PROFILE_NAME)
        recognition = ScreenshotRecognitionService(annotation, templates)
        row["opening"] = _opening_agreement(session, recognition, stored)
        result = replay_video_through_live_pipeline(
            session,
            recognition,
            use_live_pipeline=True,
            recognition_strategy="two_valid_streak",
            output_root=session_output / "visual",
        )
        events = _replayed_events(result.output_path)
        row.update(
            {
                "status": "completed",
                "frames_processed": result.frame_count,
                "replay": _event_summary(events),
                "warnings": [
                    {"reason": warning.reason, "details": warning.details}
                    for warning in result.warnings
                ],
                "timeline_comparison": _comparison_summary(result.comparison),
            }
        )
        try:
            row["truth_advisor"] = _truth_advisor_audit(
                session,
                profiles_root,
                session_output / "truth_advisor",
            )
        except Exception as exc:
            row["truth_advisor"] = {
                "available": True,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    _write_json(session_output / "summary.json", row)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the seven fixed visual/truth replay regression sessions."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit directory for all audit artifacts; original sessions stay read-only.",
    )
    args = parser.parse_args(argv)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = [_audit_session(session_id, root, output_root) for session_id, root in TARGETS]
    summary = {
        "schema_version": 1,
        "output_root": output_root,
        "session_count": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "errors": sum(row.get("status") == "error" for row in rows),
        "elapsed_seconds": round(time.time() - started, 3),
        "sessions": rows,
    }
    path = output_root / "audit_summary.json"
    _write_json(path, summary)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
