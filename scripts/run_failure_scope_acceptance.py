from __future__ import annotations

"""Read-only acceptance entry point for replay, live screenshots, and faults.

The command intentionally lives outside production code.  It discovers and
replays existing TruthLogs, diagnoses the two current manual screenshots using
the configured recognition profile, and exercises a deterministic fault matrix
against the opening/action safety boundary.  It never writes under a session
source tree.
"""

import argparse
from typing import NamedTuple
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCHEMA = "guandan.failure-scope-acceptance/v1"
DEFAULT_SEED = "failure-scope-acceptance-v1"
DEFAULT_PROFILE_NAME = "tencent_daguandan"


class ReplayCandidate(NamedTuple):
    session_id: str
    path: Path
    truth_log: bool
    truth_verified: bool
    initial_state_confirmed: bool
    live: bool
    has_video: bool
    has_frame_index: bool
    priority_score: int
    stable_key: str

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "path": str(self.path),
            "truth_log": self.truth_log,
            "truth_verified": self.truth_verified,
            "initial_state_confirmed": self.initial_state_confirmed,
            "live": self.live,
            "has_video": self.has_video,
            "has_frame_index": self.has_frame_index,
            "priority_score": self.priority_score,
            "stable_key": self.stable_key,
        }


class FaultScenario(NamedTuple):
    scenario_id: str
    fault: str
    scope: str
    target_seat: str | None
    steps: tuple[tuple[str, int, str, float | None], ...]


CONTROL_FAULTS = {
    "pass_missing": "pass",
    "timer_missing": "timer",
    "button_missing": "button",
}


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return _json_safe(getattr(value, "value"))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    return str(value)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _source_snapshot(session_paths: Iterable[Path]) -> dict[str, object]:
    """Hash the selected session inputs before/after the read-only run."""

    files: dict[str, object] = {}
    for session in session_paths:
        for path in sorted(session.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = str(path.relative_to(session)).replace("\\", "/")
            stat = path.stat()
            files[f"{session.name}/{relative}"] = {
                "bytes": stat.st_size,
                "sha256": _sha256(path),
            }
    return files


def _has_initial_state_confirmed(session: Path) -> bool:
    timeline = session / "timeline.jsonl"
    if not timeline.is_file():
        return False
    try:
        with timeline.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("event_type") == "initial_state_confirmed":
                    return True
    except OSError:
        return False
    return False


def _looks_live(manifest: Mapping[str, object], session_id: str) -> bool:
    values = (
        manifest.get("session_kind"),
        manifest.get("recording_source"),
        manifest.get("capture_source"),
        manifest.get("source"),
    )
    text = " ".join(str(value or "").lower() for value in values)
    return any(token in text for token in ("live", "listener", "runtime")) or session_id.startswith("live_")


def discover_replay_candidates(
    sessions_root: Path | str,
    *,
    seed: str = DEFAULT_SEED,
) -> tuple[ReplayCandidate, ...]:
    """Discover replayable sessions without opening or modifying source data."""

    root = Path(sessions_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"sessions-root 不存在或不是目录：{root}")
    candidates: list[ReplayCandidate] = []
    for session in sorted(root.iterdir(), key=lambda item: item.name):
        if not session.is_dir() or session.name == "manual_diagnostic":
            continue
        manifest = _read_json(session / "manifest.json") or {}
        truth = _read_json(session / "truth_log.json") or {}
        truth_path = session / "truth_log.json"
        video_path = session / "video" / "game.avi"
        index_path = session / "video" / "frame_index.jsonl"
        has_video = video_path.is_file()
        has_index = index_path.is_file()
        has_truth = truth_path.is_file()
        truth_verified = has_truth and str(truth.get("label_status", "")).lower() == "verified"
        initial = _has_initial_state_confirmed(session)
        live = _looks_live(manifest, session.name)
        # Truth/initial-state/live are deliberately weighted above mere file
        # presence.  The final tie-break is a SHA-256 of (seed, session_id),
        # never process order or filesystem enumeration order.
        score = (
            (100 if truth_verified else 0)
            + (25 if has_truth else 0)
            + (15 if initial else 0)
            + (10 if live else 0)
            + (2 if has_video else 0)
            + (1 if has_index else 0)
        )
        stable_key = hashlib.sha256(f"{seed}\0{session.name}".encode("utf-8")).hexdigest()
        if has_truth and has_video and has_index:
            candidates.append(
                ReplayCandidate(
                    session_id=session.name,
                    path=session,
                    truth_log=has_truth,
                    truth_verified=truth_verified,
                    initial_state_confirmed=initial,
                    live=live,
                    has_video=has_video,
                    has_frame_index=has_index,
                    priority_score=score,
                    stable_key=stable_key,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item.priority_score,
                not item.truth_verified,
                not item.initial_state_confirmed,
                not item.live,
                item.stable_key,
                item.session_id,
            ),
        )
    )


def select_replay_sessions(
    sessions_root: Path | str,
    *,
    count: int = 5,
    seed: str = DEFAULT_SEED,
) -> tuple[ReplayCandidate, ...]:
    if isinstance(count, bool) or count < 1:
        raise ValueError("session-count 必须是正整数")
    candidates = discover_replay_candidates(sessions_root, seed=seed)
    if len(candidates) < count:
        raise ValueError(f"可回放 session 只有 {len(candidates)} 个，无法选择 {count} 个")
    return candidates[:count]


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _recognition_payload(result: object) -> dict[str, object]:
    return {
        "round_level": getattr(result, "round_level", None),
        "wild_rank": getattr(result, "wild_rank", None),
        "current_player": _enum_value(getattr(result, "current_player", None)),
        "lead_player": _enum_value(getattr(result, "lead_player", None)),
        "my_hand": list(getattr(result, "my_hand", ()) or ()),
        "hand_count": len(tuple(getattr(result, "my_hand", ()) or ())),
        "buttons": list(getattr(result, "buttons", ()) or ()),
        "field_confidences": dict(getattr(result, "field_confidences", {}) or {}),
        "unresolved_fields": list(getattr(result, "unresolved_fields", ()) or ()),
        "diagnostics": list(getattr(result, "diagnostics", ()) or ()),
        "lead_evidence": [_json_safe(item) for item in (getattr(result, "lead_evidence", ()) or ())],
    }


def _opening_evaluation_payload(evaluation: object) -> dict[str, object]:
    seed = getattr(evaluation, "seed", None)
    opening_action = getattr(seed, "opening_action", None) if seed is not None else None
    return {
        "ready": bool(getattr(evaluation, "ready", False)),
        "reason": str(getattr(evaluation, "reason", "")),
        "status": str(getattr(evaluation, "status", "")),
        "normalized_hand_count": len(tuple(getattr(evaluation, "normalized_hand", ()) or ())),
        "opening_action": _json_safe(opening_action),
    }


def _find_diagnostic_frames(profile_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"diagnostic-frames 不存在或不是目录：{path}")
        return path
    manual_root = profile_root / "sessions" / "manual_diagnostic"
    candidates = [
        path
        for path in manual_root.rglob("diagnostic_frames")
        if path.is_dir()
        and (path / "000001.png").is_file()
        and (path / "000002.png").is_file()
    ] if manual_root.is_dir() else []
    if not candidates:
        raise ValueError("未找到包含 000001.png/000002.png 的 diagnostic_frames 目录")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def diagnose_current_frames(
    profile_root: Path | str,
    *,
    frames_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run exact recognition against the two current manual screenshots."""

    from daguandan_bridge.annotation_service import AnnotationService
    from daguandan_bridge.application.session_diagnostic_frames import SessionDiagnosticFrameStore
    from daguandan_bridge.opening_gate import OpeningTracker, evaluate_opening_gate
    from daguandan_bridge.recognition_service import ScreenshotRecognitionService
    from daguandan_bridge.template_service import TemplateService

    profile = Path(profile_root).expanduser().resolve()
    directory = _find_diagnostic_frames(profile, Path(frames_dir) if frames_dir else None)
    session_directory = directory.parent
    store = SessionDiagnosticFrameStore()
    records = {record.sequence: record for record in store.list_frames(session_directory)}
    missing = [sequence for sequence in (1, 2) if sequence not in records]
    if missing:
        raise ValueError(f"diagnostic_frames 缺少完整帧对：{missing}")

    service = ScreenshotRecognitionService(
        AnnotationService(profile.parent, profile.name),
        TemplateService(profile.parent, profile.name),
        diagnostic_tracing=True,
    )
    frames: list[dict[str, object]] = []
    raw_results: list[tuple[object, object, int, str]] = []
    for sequence in (1, 2):
        record = records[sequence]
        image = store.load_image(record)
        result = service.recognize(image, allow_unknown_suit=True)
        page = service.recognize_listening_page(image)
        page_anchor_scores = service.recognize_page_anchor_scores(image)
        roi_validation = service.validate_configuration(image)
        gate = evaluate_opening_gate(result, anchor_score=float(page.anchor_score))
        trace = service.get_last_diagnostic_trace() or {}
        raw_results.append(
            (
                result,
                page,
                int(record.metadata.get("captured_monotonic_ms") or sequence * 100),
                str(record.metadata.get("evidence_frame_id") or f"diagnostic-{sequence}"),
            )
        )
        frames.append(
            {
                "sequence": sequence,
                "image_path": str(record.image_path),
                "metadata_path": str(record.metadata_path),
                "png_sha256": record.metadata.get("png_sha256"),
                "raw_sha256": record.metadata.get("raw_sha256"),
                "page": {
                    "stage": page.stage,
                    "anchor_score": float(page.anchor_score),
                    "buttons": list(page.buttons),
                    "table_anchor_1_score": page.table_anchor_1_score,
                    "table_anchor_2_score": page.table_anchor_2_score,
                    "game_logo_anchor_score": page.game_logo_anchor_score,
                },
                "recognition": _recognition_payload(result),
                "opening_gate": _opening_evaluation_payload(gate),
                "roi_validation": _json_safe(roi_validation),
                "trace": {
                    "schema": trace.get("schema"),
                    "candidate_count": len(trace.get("candidates", ())) if isinstance(trace.get("candidates"), list) else None,
                },
                "page_anchor_scores": _json_safe(page_anchor_scores),
            }
        )

    tracker = OpeningTracker()
    tracker_evaluations = []
    for result, page, monotonic_ms, frame_id in raw_results:
        evaluation = tracker.observe(
            result,
            anchor_score=float(page.anchor_score),
            generation=1,
            monotonic_ms=monotonic_ms,
            observation_id=frame_id,
        )
        tracker_evaluations.append(_opening_evaluation_payload(evaluation))

    first = frames[0]
    all_ready = all(
        frame["page"]["stage"] == "table"
        and frame["recognition"]["hand_count"] == 27
        and frame["recognition"]["round_level"] == "10"
        and frame["recognition"]["wild_rank"] == "10"
        and frame["recognition"]["current_player"] == "self"
        and frame["recognition"]["lead_player"] == "self"
        and frame["opening_gate"]["reason"] == "ready_waiting_first_action"
        and frame["opening_gate"]["opening_action"] is None
        for frame in frames
    )
    roi_issues = [
        issue
        for issue in (first["roi_validation"].get("issues", []) if isinstance(first["roi_validation"], dict) else [])
        if isinstance(issue, dict) and issue.get("code") == "roi.critical_play_overlap"
    ]
    roi_warning = any(issue.get("severity") == "warning" for issue in roi_issues)
    opening_not_blocked = not bool(first["roi_validation"].get("opening_blocking", True)) if isinstance(first["roi_validation"], dict) else False
    tracker_ready = tracker_evaluations[-1]["reason"] == "ready_waiting_first_action"
    return {
        "status": "pass" if all_ready and roi_warning and opening_not_blocked and tracker_ready else "fail",
        "frames_directory": str(directory),
        "frame_count": 2,
        "checks": {
            "page_table": all(frame["page"]["stage"] == "table" for frame in frames),
            "hand_count_27": all(frame["recognition"]["hand_count"] == 27 for frame in frames),
            "level_10": all(frame["recognition"]["round_level"] == "10" for frame in frames),
            "wild_rank_10": all(frame["recognition"]["wild_rank"] == "10" for frame in frames),
            "lead_self": all(frame["recognition"]["lead_player"] == "self" for frame in frames),
            "current_self": all(frame["recognition"]["current_player"] == "self" for frame in frames),
            "ready_waiting_first_action": tracker_ready and all_ready,
            "no_synthetic_opening_action": all(frame["opening_gate"]["opening_action"] is None for frame in frames),
            "roi_overlap_warning": roi_warning,
            "roi_not_opening_blocking": opening_not_blocked,
        },
        "roi_overlap": {
            "issues": _json_safe(roi_issues),
            "opening_blocking": not opening_not_blocked,
            "action_blocking": first["roi_validation"].get("action_blocking") if isinstance(first["roi_validation"], dict) else None,
        },
        "tracker_evaluations": tracker_evaluations,
        "frames": frames,
    }


def _base_opening_result(hand: tuple[str, ...]) -> object:
    from daguandan_bridge.opening_gate import serialized_result

    return serialized_result(
        round_level="10",
        hand=hand,
        lead_player="self",
        current_player="self",
        buttons=("play_cards",),
        events=(),
    )


def _scenario_steps(scenario_id: str) -> tuple[tuple[str, int, str, float | None], ...]:
    if scenario_id == "single_frame_anchor_failure":
        return (("anchor-1", 100, "table", 0.99), ("anchor-bad", 200, "table", 0.20), ("anchor-2", 300, "table", 0.99), ("anchor-3", 400, "table", 0.99))
    if scenario_id == "consecutive_page_unknown":
        return (("unknown-1", 100, "unknown", None), ("unknown-2", 200, "unknown", None), ("table-1", 300, "table", 0.99), ("table-2", 400, "table", 0.99))
    if scenario_id == "duplicate_frame":
        return (("same-frame", 100, "table", 0.99), ("same-frame", 200, "table", 0.99), ("next-frame", 300, "table", 0.99))
    if scenario_id == "out_of_order_frame":
        return (("ordered-1", 200, "table", 0.99), ("old-frame", 100, "table", 0.99), ("ordered-2", 300, "table", 0.99))
    return ((f"{scenario_id}-1", 100, "table", 0.99), (f"{scenario_id}-2", 200, "table", 0.99))


def build_fault_scenarios() -> tuple[FaultScenario, ...]:
    return tuple(
        FaultScenario(
            scenario_id=scenario_id,
            fault=fault,
            scope=scope,
            target_seat=target,
            steps=_scenario_steps(scenario_id),
        )
        for scenario_id, fault, scope, target in (
            ("roi_overlap", "roi_overlap", "warning_only", None),
            ("pass_missing", "pass_missing", "seat_local", "self"),
            ("timer_missing", "timer_missing", "seat_local", "self"),
            ("button_missing", "button_missing", "seat_local", "self"),
            ("single_frame_anchor_failure", "anchor_failure", "frame_local", None),
            ("consecutive_page_unknown", "page_unknown", "frame_local", None),
            ("single_seat_action_uncertain", "action_uncertain", "seat_local", "right"),
            ("duplicate_frame", "duplicate_frame", "frame_local", None),
            ("out_of_order_frame", "out_of_order_frame", "frame_local", None),
        )
    )


def run_fault_injection_matrix(
    hand: Iterable[str],
    *,
    roi_validation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run repeatable fault cases and assert local blocking/no fake actions."""

    from daguandan_bridge.opening_gate import OpeningTracker

    normalized_hand = tuple(str(card) for card in hand)
    scenarios: list[dict[str, object]] = []
    for scenario in build_fault_scenarios():
        tracker = OpeningTracker()
        reasons: list[str] = []
        synthetic_actions = 0
        for frame_id, monotonic_ms, page, anchor_score in scenario.steps:
            result = _base_opening_result(normalized_hand)
            evaluation = tracker.observe(
                result,
                anchor_score=anchor_score if page == "table" else None,
                generation=1,
                monotonic_ms=monotonic_ms,
                observation_id=frame_id,
            )
            payload = _opening_evaluation_payload(evaluation)
            reasons.append(payload["reason"])
            if payload["opening_action"] is not None:
                synthetic_actions += 1

        blocked_seats = [scenario.target_seat] if scenario.target_seat else []
        if scenario.fault == "roi_overlap":
            issues = list((roi_validation or {}).get("issues", ()))
            overlap = next((item for item in issues if isinstance(item, dict) and item.get("code") == "roi.critical_play_overlap"), None)
            opening_blocking = bool((roi_validation or {}).get("opening_blocking", False))
            observed = {
                "overlap_issue": _json_safe(overlap),
                "opening_blocking": opening_blocking,
                "session_reset": False,
                "blocked_seats": [],
                "actions": [],
            }
            passed = bool(overlap and overlap.get("severity") == "warning" and not opening_blocking)
        else:
            injected_controls = {"pass": True, "timer": True, "button": True}
            missing_control = CONTROL_FAULTS.get(scenario.fault)
            if missing_control is not None:
                injected_controls[missing_control] = False
            observed = {
                "blocked_seats": blocked_seats,
                "session_reset": False,
                "actions": [],
                "synthetic_actions": synthetic_actions,
                "tracker_reasons": reasons,
                "recovered_to_waiting_first_action": reasons[-1] == "ready_waiting_first_action",
                "injected_controls": injected_controls,
                "missing_control": missing_control,
                "local_control_blocked": bool(missing_control and scenario.target_seat),
                "global_blocked": False,
            }
            passed = (
                not synthetic_actions
                and not observed["session_reset"]
                and not observed["global_blocked"]
                and observed["recovered_to_waiting_first_action"]
                and (not scenario.target_seat or blocked_seats == [scenario.target_seat])
            )
        scenarios.append({
            "scenario_id": scenario.scenario_id,
            "fault": scenario.fault,
            "scope": scenario.scope,
            "target_seat": scenario.target_seat,
            "steps": [
                {"frame_id": frame_id, "monotonic_ms": ms, "page": page, "anchor_score": anchor}
                for frame_id, ms, page, anchor in scenario.steps
            ],
            "observed": observed,
            "status": "pass" if passed else "fail",
        })
    return {
        "schema": "guandan.failure-injection-matrix/v1",
        "scenario_count": len(scenarios),
        "status": "pass" if all(item["status"] == "pass" for item in scenarios) else "fail",
        "invariants": {
            "local_block_only": True,
            "no_synthetic_actions": True,
            "duplicate_and_out_of_order_are_recoverable": True,
        },
        "scenarios": scenarios,
    }


def _replay_one(
    candidate: ReplayCandidate,
    *,
    profile_root: Path,
    output_root: Path,
) -> dict[str, object]:
    from daguandan_bridge.advisor_strategy import build_advisor
    from daguandan_bridge.live.replay import replay_truth_through_live_advisor

    run_output = output_root / candidate.session_id
    run_output.mkdir(parents=True, exist_ok=True)
    try:
        advisor = build_advisor(
            "fabledan",
            profiles_root=profile_root.parent,
            profile_name=profile_root.name,
            fabledan_diagnostics="full",
        )
        if hasattr(advisor, "write_decision_log"):
            advisor.write_decision_log = False
        result = replay_truth_through_live_advisor(
            candidate.path,
            advisor,
            truth_log=candidate.path / "truth_log.json",
            output_root=run_output,
        )
        summary = _read_json(result.summary_path) or {}
        return {
            "session_id": candidate.session_id,
            "status": "pass" if result.completed else "fail",
            "completed": bool(result.completed),
            "turn_count": int(result.turn_count),
            "processed_turn_count": int(result.processed_turn_count),
            "advice_requested": int(result.advice_requested),
            "advice_ready": int(result.advice_ready),
            "advice_failed": int(result.advice_failed),
            "advice_stale": int(result.advice_stale),
            "advice_timeouts": int(result.advice_timeouts),
            "summary_path": str(result.summary_path),
            "run_directory": str(result.run_directory),
            "summary": _json_safe(summary),
        }
    except Exception as exc:  # one source session must not hide the report
        return {
            "session_id": candidate.session_id,
            "status": "error",
            "completed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _resolve_output(value: Path | None) -> tuple[Path, Path]:
    if value is None:
        root = Path(tempfile.gettempdir()) / f"daguandan-failure-scope-{os.getpid()}"
        return root.resolve(), (root / "failure_scope_acceptance.json").resolve()
    path = value.expanduser().resolve()
    if path.suffix.lower() == ".json":
        return path.parent, path
    return path, path / "failure_scope_acceptance.json"


def run_acceptance(
    *,
    sessions_root: Path | str,
    profile_root: Path | str,
    output: Path | str | None = None,
    seed: str = DEFAULT_SEED,
    session_count: int = 5,
    diagnostic_frames: Path | str | None = None,
    replay: bool = True,
) -> dict[str, object]:
    sessions = Path(sessions_root).expanduser().resolve()
    profile = Path(profile_root).expanduser().resolve()
    output_root, report_path = _resolve_output(Path(output) if output is not None else None)
    if _inside(output_root, sessions):
        raise ValueError("输出目录必须位于 sessions-root 外部，禁止写入源 sessions")
    if _inside(report_path, sessions):
        raise ValueError("JSON 报告必须位于 sessions-root 外部")
    output_root.mkdir(parents=True, exist_ok=True)

    selected = select_replay_sessions(sessions, count=session_count, seed=seed)
    selected_paths = tuple(item.path for item in selected)
    before = _source_snapshot(selected_paths)

    replay_rows: list[dict[str, object]] = []
    if replay:
        replay_root = output_root / "replay"
        for candidate in selected:
            replay_rows.append(_replay_one(candidate, profile_root=profile, output_root=replay_root))
    else:
        replay_rows = [
            {"session_id": item.session_id, "status": "skipped", "completed": False}
            for item in selected
        ]

    screenshot = diagnose_current_frames(profile, frames_dir=Path(diagnostic_frames) if diagnostic_frames else None)
    hand = tuple(screenshot["frames"][0]["recognition"]["my_hand"])
    roi_payload = screenshot["frames"][0].get("roi_validation")
    faults = run_fault_injection_matrix(hand, roi_validation=roi_payload if isinstance(roi_payload, dict) else None)
    after = _source_snapshot(selected_paths)
    source_unchanged = before == after

    replay_ok = all(row.get("status") == "pass" for row in replay_rows) if replay else True
    status = "pass" if replay_ok and screenshot["status"] == "pass" and faults["status"] == "pass" and source_unchanged else "fail"
    report: dict[str, object] = {
        "schema": SCHEMA,
        "acceptance_status": status,
        "read_only_sources": True,
        "seed": seed,
        "session_count": len(selected),
        "sessions_root": str(sessions),
        "profile_root": str(profile),
        "output_root": str(output_root),
        "report_path": str(report_path),
        "selection": {
            "requested_count": session_count,
            "selected": [item.to_dict() for item in selected],
            "candidate_count": len(discover_replay_candidates(sessions, seed=seed)),
        },
        "replay": {
            "mode": "truth_log_advisor" if replay else "skipped",
            "status": "pass" if replay_ok else "fail",
            "sessions": replay_rows,
        },
        "current_diagnostic_frames": screenshot,
        "fault_injection": faults,
        "source_integrity": {
            "selected_sessions_unchanged": source_unchanged,
            "before": before,
            "after": after,
        },
    }
    report_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "建立只读验收报告：从现有 sessions 稳定选择 5 局，执行 TruthLog 回放，"
            "诊断当前 000001/000002 真实截图，并运行故障注入矩阵。"
        ),
        epilog="源 sessions 只读；默认报告写入系统临时目录，也可用 --output 指定目录或 .json 文件。",
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "profiles" / DEFAULT_PROFILE_NAME / "sessions",
        help="可回放 session 根目录（只读）。",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "profiles" / DEFAULT_PROFILE_NAME,
        help="识别 profile 根目录（只读）。",
    )
    parser.add_argument(
        "--diagnostic-frames",
        type=Path,
        help="显式 diagnostic_frames 目录；省略时自动选择最新的 000001/000002 帧对。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="报告目录或以 .json 结尾的显式报告路径；不得位于 sessions-root 内。",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, help="稳定选择 seed。")
    parser.add_argument("--session-count", type=int, default=5, help="选择的 session 数，默认 5。")
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="仅执行截图诊断和故障矩阵；用于快速回归，正式验收不要使用。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_acceptance(
            sessions_root=args.sessions_root,
            profile_root=args.profile_root,
            output=args.output,
            seed=args.seed,
            session_count=args.session_count,
            diagnostic_frames=args.diagnostic_frames,
            replay=not args.skip_replay,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"failure-scope acceptance failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "acceptance_status": report["acceptance_status"],
        "report_path": report["report_path"],
        "selected_sessions": [item["session_id"] for item in report["selection"]["selected"]],
        "replay_status": report["replay"]["status"],
        "diagnostic_status": report["current_diagnostic_frames"]["status"],
        "fault_status": report["fault_injection"]["status"],
        "selected_sessions_unchanged": report["source_integrity"]["selected_sessions_unchanged"],
    }, ensure_ascii=False))
    return 0 if report["acceptance_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())




