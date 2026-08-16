from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pytest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.live.consensus import ConsensusContext, RecognitionSample
from daguandan_bridge.live.recognition_strategy import (
    RECOGNITION_STRATEGY_OPTIONS,
    decide_recognition_strategy,
)
from daguandan_bridge.live.replay import (
    VideoReplaySource,
    replay_video_through_live_pipeline,
)
from daguandan_bridge.live.session_store import read_json_lines
from daguandan_bridge.live.truth_log import TruthLog, load_truth_log
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_ROOT = PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions"
ACTION_TYPES = {"player_played", "player_passed", "manual_confirmed_event"}


def _available_truth_sessions(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Discover usable truth fixtures without binding the gate to one workstation."""

    truth_sessions: list[str] = []
    anchored_sessions: list[str] = []
    if not root.is_dir():
        return (), ()
    for session in sorted(root.iterdir(), key=lambda path: path.name):
        truth_path = session / "truth_log.json"
        if not session.is_dir() or not truth_path.is_file():
            continue
        try:
            truth = load_truth_log(truth_path, session_id=session.name)
        except (OSError, ValueError):
            continue
        video_path = session / truth.source_video
        frame_index_path = session / truth.frame_index_path
        if not truth.turns or not video_path.is_file() or not frame_index_path.is_file():
            continue
        truth_sessions.append(session.name)
        if any(turn.frame_index is not None for turn in truth.turns):
            anchored_sessions.append(session.name)
    return tuple(truth_sessions), tuple(anchored_sessions)


TRUTH_SESSION_IDS, FRAME_ANCHORED_SESSION_IDS = _available_truth_sessions(
    SESSIONS_ROOT
)
STRATEGY_VALUES = tuple(value for value, _label in RECOGNITION_STRATEGY_OPTIONS)
NO_TRUTH_SESSION_IDS = (
    tuple(
        session.name
        for session in sorted(SESSIONS_ROOT.iterdir())
        if session.is_dir() and session.name not in TRUTH_SESSION_IDS
    )
    if SESSIONS_ROOT.is_dir()
    else ()
)

pytestmark = pytest.mark.skipif(
    os.environ.get("DAGUANDAN_RUN_SESSION_STRATEGY_BENCHMARK") != "1",
    reason="real video benchmark is opt-in; set DAGUANDAN_RUN_SESSION_STRATEGY_BENCHMARK=1",
)


def test_real_sessions_compare_four_action_strategies_with_bounded_memory(
    tmp_path: Path,
):
    """Score frame-anchored truth logs and report all other data separately.

    Only the five-frame windows around edited action frames invoke template
    recognition. Video decoding remains sequential and one-frame-at-a-time.
    Truth fixtures are discovered from the current sessions root. Logs without
    frame anchors are reported as sequence-only coverage instead of being
    misused as an accuracy denominator.
    """

    assert FRAME_ANCHORED_SESSION_IDS, (
        "SESSION-FULL requires at least one readable truth_log.json with "
        "frame anchors plus its source video and frame index"
    )

    annotation = AnnotationService(PROFILES_ROOT, "tencent_daguandan")
    templates = TemplateService(PROFILES_ROOT, "tencent_daguandan")
    report: dict[str, object] = {
        "schema_version": 2,
        "strategies": list(STRATEGY_VALUES),
        "accuracy_definition": "correct strategy decisions / frame-anchored truth turns",
        "truth_sessions": list(TRUTH_SESSION_IDS),
        "frame_anchored_sessions": list(FRAME_ANCHORED_SESSION_IDS),
        "sequence_only_sessions": [
            session_id
            for session_id in TRUTH_SESSION_IDS
            if session_id not in FRAME_ANCHORED_SESSION_IDS
        ],
        "no_truth_sessions": list(NO_TRUTH_SESSION_IDS),
        "runs": [],
    }
    accuracy_rows: list[dict[str, object]] = []

    for session_id in FRAME_ANCHORED_SESSION_IDS:
        session = SESSIONS_ROOT / session_id
        truth = load_truth_log(session / "truth_log.json", session_id=session_id)
        results = _evaluate_frame_anchored_session(
            session, truth, annotation, templates
        )
        for strategy in STRATEGY_VALUES:
            result = results[strategy]
            row = {
                "kind": "truth_accuracy",
                "session": session_id,
                "strategy": strategy,
                "expected_actions": result["expected_actions"],
                "truth_actions_total": result["truth_actions_total"],
                "unanchored_actions": result["unanchored_actions"],
                "correct_actions": result["correct_actions"],
                "accuracy": result["accuracy"],
                "no_decision": result["no_decision"],
                "frames_decoded": result["frames_decoded"],
                "recognition_calls": result["recognition_calls"],
                "elapsed_seconds": round(result["elapsed_seconds"], 3),
            }
            accuracy_rows.append(row)
            report["runs"].append(row)  # type: ignore[union-attr]

    for session_id in TRUTH_SESSION_IDS:
        if session_id in FRAME_ANCHORED_SESSION_IDS:
            continue
        truth = load_truth_log(
            SESSIONS_ROOT / session_id / "truth_log.json",
            session_id=session_id,
        )
        row = {
            "kind": "truth_sequence_only",
            "session": session_id,
            "truth_actions": len(truth.turns),
            "turns_with_frame_index": sum(
                turn.frame_index is not None for turn in truth.turns
            ),
            "status": "excluded_from_accuracy_without_frame_anchors",
        }
        report["runs"].append(row)  # type: ignore[union-attr]

    for session_id in NO_TRUTH_SESSION_IDS:
        session = SESSIONS_ROOT / session_id
        frame_cap = 120 if _frame_count(session) > 10_000 else 300
        started = time.perf_counter()
        decoded = _bounded_decode(session, frame_cap)
        row = {
            "kind": "smoke_only",
            "session": session_id,
            "status": "no_truth_log",
            "frame_cap": frame_cap,
            "frames_decoded": decoded,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        report["runs"].append(row)  # type: ignore[union-attr]

    report["accuracy_summary"] = _summarize_accuracy(accuracy_rows)
    report["coverage_summary"] = {
        "truth_sequence_only_sessions": (
            len(TRUTH_SESSION_IDS) - len(FRAME_ANCHORED_SESSION_IDS)
        ),
        "no_truth_smoke_sessions": len(NO_TRUTH_SESSION_IDS),
        "no_truth_smoke_frames": sum(
            int(row["frames_decoded"])
            for row in report["runs"]  # type: ignore[union-attr]
            if row.get("kind") == "smoke_only"
        ),
    }
    report_path = Path(
        os.environ.get(
            "DAGUANDAN_STRATEGY_REPORT",
            str(tmp_path / "real_session_strategy_report.json"),
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["accuracy_summary"], ensure_ascii=False, indent=2))

    assert len(accuracy_rows) == len(FRAME_ANCHORED_SESSION_IDS) * len(STRATEGY_VALUES)
    assert all(row["frames_decoded"] > 0 for row in accuracy_rows)


def test_all_sessions_replay_every_frame_through_the_live_pipeline(tmp_path: Path):
    """Replay every frame in all sessions, or an explicit QA sample limit."""

    available_sessions = _available_live_pipeline_sessions(SESSIONS_ROOT)
    limit_text = os.environ.get("DAGUANDAN_SESSION_REPLAY_LIMIT", "").strip()
    if limit_text:
        try:
            limit = int(limit_text)
        except ValueError as exc:
            raise ValueError("DAGUANDAN_SESSION_REPLAY_LIMIT 必须是正整数") from exc
        if limit < 1:
            raise ValueError("DAGUANDAN_SESSION_REPLAY_LIMIT 必须是正整数")
        sessions = available_sessions[:limit]
    else:
        sessions = available_sessions
    assert sessions, "SESSION-FULL requires at least one complete recorded session"

    annotation = AnnotationService(PROFILES_ROOT, "tencent_daguandan")
    templates = TemplateService(PROFILES_ROOT, "tencent_daguandan")
    recognition = ScreenshotRecognitionService(annotation, templates)
    rows: list[dict[str, object]] = []
    first_session_actions: list[dict[str, object]] = []

    for session in sessions:
        frame_index = read_json_lines(session / "video" / "frame_index.jsonl")
        # Keep one per-frame event stream as a regression artifact without
        # tying the SESSION-FULL gate to a session that may no longer exist.
        persist_frame_log = session == sessions[0]
        started = time.perf_counter()
        result = replay_video_through_live_pipeline(
            session,
            recognition,
            use_live_pipeline=True,
            recognition_strategy="two_valid_streak",
            output_root=tmp_path / "session_full" / session.name,
            persist_frame_log=persist_frame_log,
        )
        warning_reasons = [warning.reason for warning in result.warnings]
        actual_actions = (
            len(result.comparison.identical_turn_ids)
            + len(result.comparison.changed)
            + len(result.comparison.added)
        )
        completed = (
            result.frame_count == len(frame_index)
            and "missing_video_frames" not in warning_reasons
        )
        rows.append(
            {
                "session": session.name,
                "frames_expected": len(frame_index),
                "frames_processed": result.frame_count,
                "actions": actual_actions,
                "status": "passed" if completed else "failed",
                "warnings": warning_reasons,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        if persist_frame_log:
            first_session_actions = _actions_from_frame_log(result.output_path)

    report = {
        "schema_version": 1,
        "qa_scope": "SESSION-SAMPLE" if limit_text else "SESSION-FULL",
        "sessions_root": str(SESSIONS_ROOT),
        "available_session_count": len(available_sessions),
        "session_count": len(sessions),
        "frames_processed": sum(int(row["frames_processed"]) for row in rows),
        "actions": sum(int(row["actions"]) for row in rows),
        "runs": rows,
    }
    report_path = Path(
        os.environ.get(
            "DAGUANDAN_SESSION_FULL_REPORT",
            str(tmp_path / "session_full_report.json"),
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert all(row["status"] == "passed" for row in rows)
    # The selected session must traverse the production recognizer and reduce
    # at least one action; exact card accuracy is covered by truth fixtures.
    assert first_session_actions


def _available_live_pipeline_sessions(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    required = (
        Path("manifest.json"),
        Path("timeline.jsonl"),
        Path("video/game.avi"),
        Path("video/frame_index.jsonl"),
    )
    sessions: list[Path] = []
    for session in sorted(root.iterdir(), key=lambda path: path.name):
        if not session.is_dir() or not all((session / item).is_file() for item in required):
            continue
        if not read_json_lines(session / "video" / "frame_index.jsonl"):
            continue
        events = read_json_lines(session / "timeline.jsonl")
        if not any(item.get("event_type") == "initial_state_confirmed" for item in events):
            continue
        sessions.append(session)
    return tuple(sessions)


def _actions_from_frame_log(path: Path) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in read_json_lines(path):
        for event in record.get("events", ()):
            event_id = str(event.get("event_id", ""))
            if event.get("event_type") not in ACTION_TYPES or event_id in seen:
                continue
            seen.add(event_id)
            actions.append(event)
    return actions


def _evaluate_frame_anchored_session(
    session: Path,
    truth: TruthLog,
    annotation: AnnotationService,
    templates: TemplateService,
) -> dict[str, dict[str, object]]:
    """Recognize only action windows, then apply all four policies to samples."""

    target_frames: dict[int, list[tuple[int, int]]] = defaultdict(list)
    anchored_turns = tuple(turn for turn in truth.turns if turn.frame_index is not None)
    for turn in anchored_turns:
        for offset in range(5):
            target_frames[int(turn.frame_index) + offset].append((turn.index, offset))
    turns = {turn.index: turn for turn in anchored_turns}
    samples_by_turn: dict[int, list[RecognitionSample]] = defaultdict(list)
    recognition = ScreenshotRecognitionService(annotation, templates)
    frames_decoded = 0
    recognition_calls = 0
    started = time.perf_counter()
    source = VideoReplaySource(
        session / truth.source_video,
        session / truth.frame_index_path,
    )
    for record, frame in source.frames():
        frames_decoded += 1
        targets = target_frames.get(record.frame_index)
        if not targets:
            continue
        for turn_index, _offset in targets:
            turn = turns[turn_index]
            result = recognition.recognize_play_region(
                frame,
                turn.actor,
                wild_rank=truth.initial_state.round_level,
                allow_pass=turn.index > 1,
            )
            recognition_calls += 1
            samples_by_turn[turn_index].append(
                RecognitionSample(
                    cards=tuple(result.cards),
                    is_pass=bool(result.is_pass),
                    confidence=float(result.confidence),
                    source=result.source,
                    evidence_ref=f"{session.name}:frame-{record.frame_index}",
                )
            )

    output: dict[str, dict[str, object]] = {}
    for strategy in STRATEGY_VALUES:
        correct = 0
        no_decision = 0
        for turn in anchored_turns:
            samples = samples_by_turn[turn.index]
            context = ConsensusContext(
                level_rank=truth.initial_state.round_level,
                remaining_cards=27,
                allow_pass=turn.index > 1,
                validate_rules=False,
            )
            decision = None
            for index, sample in enumerate(samples):
                # The first and fifth frames represent the reference and stable
                # single-shot points. Multi-sample strategies see the full
                # window. This keeps timing differences explicit in the test.
                if strategy == "reference_single_shot" and index > 0:
                    break
                if strategy == "stable_single_shot" and index < 2:
                    continue
                decision = decide_recognition_strategy(
                    strategy,
                    samples[: index + 1],
                    context=context,
                )
                if decision is not None:
                    break
            if decision is None:
                no_decision += 1
            elif decision.is_pass == turn.is_pass and (
                decision.is_pass
                or tuple(sorted(decision.cards)) == tuple(sorted(turn.cards))
            ):
                correct += 1
        expected = len(anchored_turns)
        output[strategy] = {
            "expected_actions": expected,
            "truth_actions_total": len(truth.turns),
            "unanchored_actions": len(truth.turns) - expected,
            "correct_actions": correct,
            "accuracy": correct / expected if expected else 0.0,
            "no_decision": no_decision,
            "frames_decoded": frames_decoded,
            "recognition_calls": recognition_calls,
            "elapsed_seconds": time.perf_counter() - started,
        }
    return output


def _bounded_decode(session: Path, frame_cap: int) -> int:
    source = VideoReplaySource(
        session / "video" / "game.avi",
        session / "video" / "frame_index.jsonl",
    )
    count = 0
    for _record, _frame in source.frames():
        count += 1
        if count >= frame_cap:
            break
    return count


def _frame_count(session: Path) -> int:
    try:
        raw = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
        return int(raw.get("frame_count", 0) or 0)
    except (OSError, ValueError, TypeError):
        return 0


def _summarize_accuracy(rows: list[dict[str, object]]) -> dict[str, object]:
    by_strategy: dict[str, dict[str, int]] = {}
    for row in rows:
        item = by_strategy.setdefault(
            str(row["strategy"]),
            {"expected_actions": 0, "correct_actions": 0, "no_decision": 0},
        )
        item["expected_actions"] += int(row["expected_actions"])
        item["correct_actions"] += int(row["correct_actions"])
        item["no_decision"] += int(row["no_decision"])
    ranked = []
    for strategy, item in by_strategy.items():
        ranked.append(
            {
                "strategy": strategy,
                **item,
                "accuracy": item["correct_actions"] / item["expected_actions"],
            }
        )
    ranked.sort(key=lambda item: (item["accuracy"], item["correct_actions"]), reverse=True)
    accuracies = {float(item["accuracy"]) for item in ranked}
    return {
        "frame_anchored_actions": sum(int(row["expected_actions"]) for row in rows)
        // len(STRATEGY_VALUES),
        "by_strategy": ranked,
        "strategies_are_discriminated": len(accuracies) > 1,
        "best_strategy": ranked[0]["strategy"] if len(accuracies) > 1 else None,
    }
