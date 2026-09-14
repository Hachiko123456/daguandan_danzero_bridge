"""Read-only production-pipeline audit for recorded sessions."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Literal
from uuid import uuid4

import cv2

from ..advisor_strategy import build_advisor
from ..annotation_service import AnnotationService
from ..application.live_v2_recorded_replay import replay_video_through_production_live_v2
from ..live.replay import (
    ReplayComparison,
    TrustedAdviceReplayResult,
    VideoReplaySource,
    VisualPipelineReplayResult,
    replay_truth_through_live_advisor,
)
from ..live.session_store import read_json_lines
from ..live.truth_log import TruthLog, load_truth_log
from ..recognition_service import ScreenshotRecognitionService
from ..storage import atomic_write_json
from ..template_service import TemplateService

_ACTION_TYPES = frozenset({"player_played", "player_passed", "manual_confirmed_event"})
_GAP_TYPES = frozenset({"terminal_history_gap", "history_gap_detected"})
_FRAME_WARNINGS = frozenset({"missing_video_frames", "extra_video_frames"})
_SEATS = ("self", "right", "opposite", "left")
ProgressCallback = Callable[[str, str, int, int, str], None]


@dataclass(frozen=True)
class TruthAuditReference:
    kind: Literal["canonical", "staged"]
    path: Path


@dataclass(frozen=True)
class _Session:
    source: Path
    root: Path
    store_id: str
    session_id: str


@dataclass(frozen=True)
class SessionReplayAuditRun:
    run_directory: Path
    summary_path: Path
    sessions: tuple[dict[str, object], ...]
    inventory_path: Path | None = None
    verification_path: Path | None = None
    execution_ok: bool = False


def resolve_truth_audit_reference(
    session: Path | str, *, scan_run_id: str | Iterable[str] | None = None
) -> TruthAuditReference | None:
    """Canonical always wins; staged truth is never selected implicitly."""

    root = Path(session)
    canonical = root / "truth_log.json"
    if canonical.is_file():
        return TruthAuditReference("canonical", canonical)
    for selected_run_id in _scan_run_ids(scan_run_id):
        staged = root / "derived" / "truth_scan_drafts" / selected_run_id / "truth_log.json"
        if staged.is_file():
            return TruthAuditReference("staged", staged)
    return None


def summarize_visual_events(events: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = tuple(events)
    actions = _effective_actions(rows)
    event_counts = Counter(str(row.get("event_type", "unknown")) for row in rows)
    expected_turn_id = 1
    issues: list[dict[str, int]] = []
    for action in actions:
        turn_id = _int_or_none(action.get("turn_id"))
        if turn_id is not None:
            if turn_id != expected_turn_id:
                issues.append({"expected_turn_id": expected_turn_id, "actual_turn_id": turn_id})
            expected_turn_id = turn_id + 1
    leads: list[dict[str, object]] = []
    winds: list[dict[str, object]] = []
    rankings: list[dict[str, object]] = []
    gaps: list[dict[str, object]] = []
    for row in rows:
        event_type = str(row.get("event_type", ""))
        payload = row.get("payload", {})
        payload = payload if isinstance(payload, dict) else {}
        if event_type == "lead_player_confirmed":
            leads.append(
                {
                    "actor": row.get("actor"),
                    "lead_player": payload.get("lead_player", row.get("actor")),
                    "event_id": row.get("event_id"),
                    "frame_index": row.get("_replay_frame_index"),
                }
            )
        if event_type == "wind_caught":
            winds.append(_event_digest(row))
        if event_type == "player_finished":
            rankings.append(_event_digest(row))
        if event_type in _GAP_TYPES or "history_gap" in event_type:
            gaps.append(_event_digest(row))
    return {
        "actions": {
            "count": len(actions),
            "plays": sum(not bool(row["is_pass"]) for row in actions),
            "passes": sum(bool(row["is_pass"]) for row in actions),
            "rows": actions,
        },
        "lead": {"confirmations": leads},
        "turn_order": {"contiguous_turn_ids": not issues, "issues": issues},
        "wind_catch_chain": winds,
        "rankings": rankings,
        "visual_gaps": gaps,
        "recognized_result_categories": {"event_types": dict(sorted(event_counts.items()))},
    }


def compare_truth_visual_fields(
    truth: TruthLog,
    opening: dict[str, object],
    visual: dict[str, object],
) -> dict[str, object]:
    """Return ordered field metrics without collapsing duplicate turn IDs."""

    return _field_metrics(truth, opening, visual)


class SessionReplayAuditService:
    """Run every session independently and emit one verifiable report directory."""

    def __init__(
        self,
        *,
        recognition_factory: Callable[[Path], ScreenshotRecognitionService] | None = None,
        opening_probe: Callable[[Path, ScreenshotRecognitionService, dict[str, object]], dict[str, object]] | None = None,
        visual_replay: Callable[..., VisualPipelineReplayResult] | None = None,
        advisor_factory: Callable[[Path, str], Any] | None = None,
        advisor_replay: Callable[..., TrustedAdviceReplayResult] = replay_truth_through_live_advisor,
        profile_root: Path | str | None = None,
        trusted_session_ids: Iterable[str] = (),
    ) -> None:
        self._profile_root = Path(profile_root).resolve() if profile_root is not None else None
        self._recognition_factory = recognition_factory or (
            _recognition_for_profile(self._profile_root)
            if self._profile_root is not None
            else _recognition_for_session
        )
        # ``opening_probe`` is retained only for legacy/unit-test injection.
        # The production audit deliberately has no pre-replay TruthLog probe.
        self._opening_probe = opening_probe
        self._visual_replay = visual_replay or replay_video_through_production_live_v2
        self._advisor_factory = advisor_factory or _fabledan_for_profile
        self._advisor_replay = advisor_replay
        self._trusted_session_ids = frozenset(
            str(item).strip() for item in trusted_session_ids if str(item).strip()
        )

    def audit(
        self,
        session_roots: Iterable[Path | str],
        *,
        output: Path | str,
        scan_run_id: str | Iterable[str] | None = None,
        run_id: str | None = None,
        command: Iterable[str] | None = None,
        session_paths: Iterable[Path | str] | None = None,
        on_progress: ProgressCallback | None = None,
        max_workers: int = 1,
    ) -> SessionReplayAuditRun:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if not 1 <= max_workers <= 3:
            raise ValueError("max_workers must be between 1 and 3")
        roots = _normalize_roots(session_roots)
        scan_run_ids = tuple(_scan_run_ids(scan_run_id))
        command_values = tuple(command) if command is not None else None
        explicit_sessions = _normalize_explicit_sessions(session_paths)
        source_roots = _unique_paths((*roots, *(item.source for item in explicit_sessions)))
        output_root = Path(output).resolve()
        _reject_internal_output(output_root, source_roots)
        run_dir = output_root / _safe_name(run_id or _new_run_id())
        _reject_internal_output(run_dir, source_roots)
        run_dir.mkdir(parents=True, exist_ok=False)

        sessions = _discover(roots, explicit_sessions=explicit_sessions)
        before = _source_snapshot(source_roots)
        before_path = run_dir / "source_snapshot_before.json"
        atomic_write_json(before_path, before)
        inventory = _inventory(
            sessions, source_roots, scan_run_ids, profile_root=self._profile_root
        )
        inventory_path = run_dir / "inventory.json"
        atomic_write_json(inventory_path, inventory)
        # Each session owns its VideoCapture, recognizer, runtime workers and
        # temporary output.  Bound the executor so only a small number of
        # frame streams and model workers exist at once; never preload all AVI
        # frames or keep completed session objects in memory.
        rows_by_index: list[dict[str, object] | None] = [None] * len(sessions)
        progress_lock = Lock()

        def emit(phase: str, session_id: str, processed: int, total: int, detail: str) -> None:
            if on_progress is None:
                return
            # Progress callbacks are application-owned and are often not
            # thread-safe (the CLI writes one terminal line).  Serialize them
            # without serializing the actual session work.
            with progress_lock:
                on_progress(phase, session_id, processed, total, detail)

        worker_count = min(max_workers, max(1, len(sessions)))
        previous_cv_threads: int | None = None
        if worker_count > 1:
            # One OpenCV thread per session worker avoids nested native thread
            # pools multiplying CPU and memory usage.
            previous_cv_threads = int(cv2.getNumThreads())
            cv2.setNumThreads(1)
        def audit_one(
            session: _Session,
            session_dir: Path,
            inventory_row: dict[str, object],
        ) -> dict[str, object]:
            emit("session_start", session.session_id, 0, len(sessions), "准备会话")
            return self._audit_session(
                session,
                session_dir,
                scan_run_ids,
                inventory_row,
                on_progress=emit,
            )

        try:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="listener-regression",
            ) as executor:
                futures = {}
                for index, session in enumerate(sessions):
                    session_dir = (
                        run_dir
                        / "sessions"
                        / session.store_id
                        / _safe_name(session.session_id)
                    )
                    future = executor.submit(
                        audit_one,
                        session,
                        session_dir,
                        _inventory_row(inventory, session),
                    )
                    futures[future] = (index, session, session_dir)

                completed_count = 0
                for future in as_completed(futures):
                    index, session, session_dir = futures.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        # One unexpected session error must not discard results
                        # from other workers or prevent the run from producing a
                        # machine-readable failure report.
                        row = _unexpected_session_row(
                            session, session_dir, exc, _inventory_row(inventory, session)
                        )
                    rows_by_index[index] = row
                    completed_count += 1
                    emit(
                        "session_done", session.session_id, completed_count,
                        len(sessions), "会话完成"
                    )
                    # Release cyclic references and large temporary arrays as
                    # soon as a worker result is collected.  The executor still
                    # bounds concurrent memory even if the Python allocator
                    # retains arenas.
                    gc.collect()
        finally:
            if previous_cv_threads is not None:
                cv2.setNumThreads(previous_cv_threads)

        rows = [row for row in rows_by_index if row is not None]
        after = _source_snapshot(source_roots)
        after_path = run_dir / "source_snapshot_after.json"
        atomic_write_json(after_path, after)
        summary = _make_summary(
            run_id=run_dir.name,
            rows=rows,
            inventory=inventory,
            scan_run_id=scan_run_ids,
            command=command_values,
            source_changes=_source_changes(before, after),
            source_snapshot_paths=(before_path, after_path),
            max_workers=worker_count,
        )
        summary_path = run_dir / "all_session_audit.json"
        atomic_write_json(summary_path, summary)
        _write_sessions_csv(run_dir / "sessions.csv", rows)
        _write_failures_csv(run_dir / "failures.csv", rows)
        _write_markdown(run_dir / "summary.md", summary)
        verification = _verify(run_dir, summary, before, after)
        verification_path = run_dir / "verification.json"
        atomic_write_json(verification_path, verification)
        return SessionReplayAuditRun(
            run_dir,
            summary_path,
            tuple(rows),
            inventory_path,
            verification_path,
            bool(verification["passed"]),
        )

    def _audit_session(
        self,
        item: _Session,
        output: Path,
        scan_run_id: str | Iterable[str] | None,
        inventory: dict[str, object],
        *,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, object]:
        """Run the visual audit first; load TruthLog only as post-run evidence.

        The default visual path intentionally starts with no TruthLog object and
        no saved level/hand/lead.  A custom ``opening_probe`` is supported for
        old unit tests only and is never selected by the production CLI.
        """
        output.mkdir(parents=True, exist_ok=True)
        reference: TruthAuditReference | None = None
        initial_frame_inventory = inventory.get("frame_index", {})
        initial_frame_inventory = (
            initial_frame_inventory if isinstance(initial_frame_inventory, dict) else {}
        )

        def report(phase: str, processed: int = 0, total: int = 0, detail: str = "") -> None:
            if on_progress is not None:
                on_progress(phase, item.session_id, int(processed), int(total), detail)

        row: dict[str, object] = {
            "source": str(item.source),
            "sessions_root": str(item.root),
            "store_id": item.store_id,
            "session_id": item.session_id,
            "status": "error",
            "execution_status": "error",
            "truth_quality": "not_available",
            "visual_quality": "not_evaluated",
            "fabledan_quality": "not_evaluated",
            "frames_processed": 0,
            "indexed_frames": int(initial_frame_inventory.get("row_count", 0) or 0),
            "truth_log": {
                "kind": str(inventory.get("truth_kind", "none")),
                "path": inventory.get("truth_path"),
                "loaded_before_visual_replay": False,
            },
            "inventory": inventory,
            "evidence_paths": {"session_audit": str(output)},
        }
        truth: TruthLog | None = None
        opening: dict[str, object] = {}
        try:
            recognition = self._recognition_factory(item.source)
            legacy_probe = self._opening_probe
            if legacy_probe is not None:
                # Compatibility path for injected test doubles.  The default
                # CLI never enters this branch.
                # Legacy injected probes may inspect the recorded timeline,
                # but they must not receive TruthLog data before visual replay.
                expected_initial = _stored_initial(item.source)
                report("opening", 0, 0, "检查级牌、手牌和首出（兼容测试注入）")
                opening = legacy_probe(item.source, recognition, expected_initial)
            else:
                report("opening", 0, 0, "生产开局链路：页面/牌桌/级牌/手牌/首出")

            visual_advisor = self._advisor(item.source)
            report("visual", 0, int(initial_frame_inventory.get("row_count", 0) or 0), "开始 LiveV2 视觉监听回放")
            visual_kwargs: dict[str, object] = {}
            if on_progress is not None:
                visual_kwargs["on_progress"] = (
                    lambda processed, total, frame: report(
                        "visual", processed, total, f"帧 {frame}"
                    )
                )
            if legacy_probe is None and self._visual_replay is replay_video_through_production_live_v2:
                visual_kwargs["profile_root"] = self._profile_root or item.source.parent.parent
                visual_kwargs["advisor"] = visual_advisor
            else:
                # Existing injected doubles use the historical signature, but
                # the visual replay must never receive TruthLog as an input.
                # TruthLog is resolved and loaded only after this call returns.
                visual_kwargs["use_live_pipeline"] = True
                visual_kwargs["recognition_strategy"] = "two_valid_streak"
                visual_kwargs["advisor"] = visual_advisor
            result = self._visual_replay(
                item.source,
                recognition,
                output_root=output / "visual_driven",
                **visual_kwargs,
            )

            # Resolve, load and hash TruthLog only after visual replay has
            # completed, solely for comparison and isolated advice.
            reference = resolve_truth_audit_reference(item.source, scan_run_id=scan_run_id)
            truth = _load_truth(item.source, reference)
            row["truth_log"]["loaded_before_visual_replay"] = False  # type: ignore[index]
            if legacy_probe is None:
                opening = dict(getattr(result, "opening", {}) or {})

            if truth is not None:
                row["truth_log"] = _truth_metadata(
                    reference,
                    truth,
                    trusted=item.session_id in self._trusted_session_ids,
                )
                row["truth_log"]["loaded_before_visual_replay"] = False  # type: ignore[index]
            events = _read_replay_events(result.output_path)
            visual = summarize_visual_events(events)
            frame_inventory = inventory.get("frame_index", {})
            frame_inventory = frame_inventory if isinstance(frame_inventory, dict) else {}
            indexed = int(frame_inventory.get("row_count", 0))
            complete = (
                indexed > 0
                and result.frame_count == indexed
                and bool(frame_inventory.get("continuous"))
                and not frame_inventory.get("parse_error")
                and not any(w.reason in _FRAME_WARNINGS for w in result.warnings)
            )
            listener_status, listener_status_reason = _listener_completion(
                result, visual, complete, legacy_compat=legacy_probe is not None
            )
            metrics = _field_metrics(truth, opening, visual) if truth else _na_metrics()
            divergence = _first_divergence(truth, opening, visual)
            evidence = (
                _write_evidence(
                    item.source,
                    output / "first_divergence",
                    divergence,
                    truth,
                    visual_frame_log=result.output_path,
                    runtime_directory=result.run_directory,
                )
                if divergence
                else None
            )
            visual_fabledan = _visual_advice_summary(
                result, visual_advisor, listener_status=listener_status
            )
            report("fabledan_truth", 0, len(truth.turns) if truth is not None else 0, "完整 TruthLog 驱动 FableDan")
            truth_fabledan = self._truth_advice(
                item.source,
                output,
                reference,
                on_progress=(
                    lambda processed, total, detail: report(
                        "fabledan_truth", processed, total, detail
                    )
                )
                if on_progress is not None
                else None,
            )
            truth_tool_complete = (
                not truth_fabledan.get("available")
                or bool(truth_fabledan.get("completed"))
            )
            execution_complete = listener_status == "complete" and truth_tool_complete
            strict = (
                _strict_quality(metrics)
                if reference
                and reference.kind == "canonical"
                and truth is not None
                and (
                    truth.label_status == "verified"
                    or item.session_id in self._trusted_session_ids
                )
                else "diagnostic"
            )
            row.update(
                {
                    "status": "completed" if execution_complete else "error",
                    "execution_status": "completed" if execution_complete else "incomplete",
                    "frame_replay_status": "complete" if complete else "incomplete",
                    "opening_status": str(opening.get("status", "recognized" if opening else "not_observable")),
                    "listener_status": listener_status,
                    "listener_status_reason": listener_status_reason,
                    "comparison_status": strict,
                    "truth_quality": strict,
                    "truth_qualification": (
                        "trusted_for_run"
                        if item.session_id in self._trusted_session_ids
                        else "verified_label"
                        if truth is not None and truth.label_status == "verified"
                        else "reference_only"
                    ),
                    "frames_processed": result.frame_count,
                    "indexed_frames": indexed,
                    "processed_turns": result.processed_turn_count,
                    "visual_quality": _visual_quality(
                        metrics, listener_status == "complete"
                    ),
                    "fabledan_quality": _advice_quality(
                        truth_fabledan, visual_fabledan
                    ),
                    "lineage": {
                        "runtime": "live_v2" if not legacy_probe else "legacy_injected_test_double",
                        "legacy_orchestrator_used": bool(legacy_probe is not None),
                        "truth_log_used_as_visual_input": False if legacy_probe is None else True,
                    },
                    "field_metrics": metrics,
                    "warnings": [{"reason": w.reason, "details": w.details} for w in result.warnings],
                    "timeline_comparison": _comparison_summary(result.comparison),
                    "first_divergence": evidence,
                    "fabledan": {"truth_driven": truth_fabledan, "visual_driven": visual_fabledan},
                    "evidence_paths": {
                        "session_audit": str(output),
                        "visual_frame_log": str(result.output_path),
                        "visual_comparison": str(result.comparison_path),
                        "first_divergence": evidence.get("directory") if evidence else None,
                    },
                }
            )
            if not execution_complete:
                row["error"] = (
                    f"frame/listener replay incomplete: processed={result.frame_count}, indexed={indexed}, status={result.status}"
                    if listener_status != "complete"
                    else str(truth_fabledan.get("error", "truth-driven advisor replay incomplete"))
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(output / "summary.json", row)
        return row

    def _advisor(self, session: Path) -> Any:
        profile = self._profile_root or session.parent.parent
        advisor = self._advisor_factory(profile.parent, profile.name)
        if hasattr(advisor, "write_decision_log"):
            advisor.write_decision_log = False
        return advisor

    def _truth_advice(
        self,
        session: Path,
        output: Path,
        reference: TruthAuditReference | None,
        *,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, object]:
        if reference is None:
            return {"available": False, "quality": "not_available", "reason": "no explicitly selected truth log"}
        try:
            advisor = self._advisor(session)
            load_truth_log(reference.path, session_id=_session_id(session))
            advice_kwargs: dict[str, object] = {}
            if on_progress is not None:
                advice_kwargs["on_progress"] = (
                    lambda processed, total, turn_id: on_progress(
                        processed,
                        total,
                        f"TruthLog 回合 {turn_id}/{total}",
                    )
                )
            result = self._advisor_replay(
                session,
                advisor,
                truth_log=reference.path,
                output_root=output / "truth_driven",
                **advice_kwargs,
            )
            if on_progress is not None:
                on_progress(
                    int(result.processed_turn_count),
                    int(result.turn_count),
                    f"推荐处理 {result.processed_turn_count}/{result.turn_count}",
                )
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            statuses = summary.get("advice_statuses", {})
            advice = {
                "requested": int(result.advice_requested),
                "ready": int(result.advice_ready),
                "failed": int(result.advice_failed),
                "stale": int(result.advice_stale),
                "timeout": int(result.advice_timeouts),
                "timeouts": int(result.advice_timeouts),
                "withheld": int(statuses.get("withheld", 0)),
                "statuses": statuses,
            }
            if advice["failed"] or advice["timeout"]:
                status = "failed"
                status_reason = "advice_failure_or_timeout"
            elif not bool(result.completed):
                status = "incomplete"
                status_reason = "truth_replay_incomplete"
            else:
                # A completed trusted replay with no failed/timeout requests
                # is a valid TruthLog-driven FableDan result. Stale counts,
                # if any, remain visible in ``advice`` and do not turn a
                # completed trusted channel into a false failure.
                status = "passed"
                status_reason = "truth_replay_completed"
            return {
                "available": True,
                "status": status,
                "status_reason": status_reason,
                "quality": status,
                "truth_log_kind": reference.kind,
                "run_directory": str(result.run_directory),
                "completed": bool(result.completed),
                "turn_count": result.turn_count,
                "processed_turn_count": result.processed_turn_count,
                "advice": advice,
                "advisor": _advisor_info(advisor),
                "artifacts": _artifact_map(result.run_directory),
            }
        except Exception as exc:
            return {"available": True, "completed": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _stored_initial(session: Path) -> dict[str, object]:
    for event in read_json_lines(session / "timeline.jsonl"):
        if event.get("event_type") == "initial_state_confirmed":
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                return {
                    "round_level": str(payload.get("round_level", "")),
                    "lead_player": payload.get("lead_player", event.get("actor")),
                    "hand": sorted(str(card) for card in payload.get("hand", ())),
                }
    raise ValueError("timeline.jsonl lacks initial_state_confirmed")


def _expected_initial(session: Path, truth: TruthLog | None) -> dict[str, object]:
    if truth is None:
        return _stored_initial(session)
    return {
        "round_level": truth.initial_state.round_level,
        "lead_player": truth.initial_state.lead_player,
        "hand": sorted(truth.initial_state.my_hand),
    }


def _opening_agreement(session: Path, recognition: Any, stored: dict[str, object]) -> dict[str, object]:
    source = VideoReplaySource(session / "video" / "game.avi", session / "video" / "frame_index.jsonl")
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
    stable = len(reads) == 2 and (reads[0]["round_level"], reads[0]["hand"]) == (reads[1]["round_level"], reads[1]["hand"])
    return {
        "stored_initial_state": stored,
        "reads": reads,
        "two_frame_agreement": stable,
        "two_frame_matches_stored": bool(stable and all(r["matches_stored_level"] and r["matches_stored_hand"] for r in reads)),
    }


def _read_replay_events(path: Path) -> tuple[dict[str, object], ...]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for frame in read_json_lines(path):
        events = frame.get("events", ())
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id", ""))
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            row = dict(event)
            row["_replay_frame_index"] = frame.get("frame_index")
            row["_replay_monotonic_ms"] = frame.get("monotonic_ms")
            row["_replay_state_revision"] = frame.get("state_revision")
            result.append(row)
    return tuple(result)


def _effective_actions(rows: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    for event in rows:
        event_type = str(event.get("event_type", ""))
        payload = event.get("payload", {})
        payload = payload if isinstance(payload, dict) else {}
        if event_type in _ACTION_TYPES:
            action = {
                "turn_id": _int_or_none(payload.get("turn_id", event.get("turn_id"))),
                "trick_id": _int_or_none(event.get("trick_id")),
                "actor": event.get("actor"),
                "is_pass": bool(payload.get("is_pass", event_type == "player_passed")),
                "cards": sorted(str(card) for card in payload.get("cards", ()) or ()),
                "event_id": event.get("event_id"),
                "evidence": payload.get("evidence"),
                "frame_index": event.get("_replay_frame_index", payload.get("frame_index")),
                "monotonic_ms": event.get("_replay_monotonic_ms", event.get("monotonic_ms")),
                "state_revision_before": event.get("state_revision_before"),
                "state_revision_after": event.get("state_revision_after", event.get("_replay_state_revision")),
            }
            actions.append(action)
            if event.get("event_id"):
                by_id[str(event["event_id"])] = action
        elif event_type in {"event_correction", "suit_corrected"}:
            target = by_id.get(str(payload.get("target_event_id", "")))
            if target:
                if "cards" in payload:
                    target["cards"] = sorted(str(card) for card in payload.get("cards", ()) or ())
                if "is_pass" in payload:
                    target["is_pass"] = bool(payload.get("is_pass"))
    return actions


def _opening_evidence(opening: dict[str, object]) -> dict[str, object]:
    confirmed = opening.get("confirmed")
    if isinstance(confirmed, dict):
        return confirmed
    reads = opening.get("reads", ())
    if isinstance(reads, list):
        for row in reversed(reads):
            if isinstance(row, dict) and row.get("candidate_ready"):
                return row
        for row in reversed(reads):
            if isinstance(row, dict) and row.get("round_level") and row.get("hand"):
                return row
    return {}


def _field_metrics(truth: TruthLog, opening: dict[str, object], visual: dict[str, object]) -> dict[str, object]:
    actual_open = _opening_evidence(opening)
    expected_hand = Counter(truth.initial_state.my_hand)
    actual_hand = Counter(str(card) for card in actual_open.get("hand", ()) or ())
    hand_matches = sum((expected_hand & actual_hand).values())
    expected = [
        {"turn_id": t.index, "trick_id": t.trick_id, "actor": t.actor, "is_pass": t.is_pass, "cards": sorted(t.cards), "frame_index": t.frame_index, "monotonic_ms": t.monotonic_ms}
        for t in truth.turns
    ]
    actual = list(visual.get("actions", {}).get("rows", ()))
    compared = min(len(expected), len(actual))
    actor_ok = pass_ok = cards_ok = identical = 0
    changed: list[dict[str, object]] = []
    for index in range(compared):
        exp, act = expected[index], actual[index]
        fields: list[str] = []
        if exp["actor"] == act.get("actor"):
            actor_ok += 1
        else:
            fields.append("actor")
        if exp["is_pass"] == act.get("is_pass"):
            pass_ok += 1
        else:
            fields.append("pass")
        if Counter(exp["cards"]) == Counter(act.get("cards", ())):
            cards_ok += 1
        else:
            fields.append("cards")
        if not fields:
            identical += 1
        else:
            changed.append({"position": index + 1, "fields": fields, "expected": exp, "actual": act})
    denominator = len(expected)
    confirmations = visual.get("lead", {}).get("confirmations", ())
    actual_lead = confirmations[0].get("lead_player") if confirmations else None
    finish_expected = list(truth.outcome.finish_order)
    finish_actual = [row.get("actor") for row in visual.get("rankings", ())]
    ranking_comparable = bool(truth.outcome.complete)
    ranking_status = (
        "strict" if ranking_comparable and finish_expected
        else "diagnostic_only_outcome_incomplete" if finish_expected
        else "not_available"
    )
    ranking_expected_for_metrics = finish_expected if ranking_comparable else []
    return {
        "strict": True,
        "level": _metric(1, int(actual_open.get("round_level") == truth.initial_state.round_level), truth.initial_state.round_level, actual_open.get("round_level")),
        "hand_multiset": {
            "expected_count": sum(expected_hand.values()), "actual_count": sum(actual_hand.values()), "matched_count": hand_matches,
            "precision": _ratio(hand_matches, sum(actual_hand.values())), "recall": _ratio(hand_matches, sum(expected_hand.values())),
            "exact": expected_hand == actual_hand, "missing": list((expected_hand - actual_hand).elements()), "added": list((actual_hand - expected_hand).elements()),
        },
        "lead_player": _metric(1, int(actual_lead == truth.initial_state.lead_player), truth.initial_state.lead_player, actual_lead),
        "actions": {
            "expected": len(expected), "actual": len(actual), "identical": identical,
            "missing": max(0, len(expected) - len(actual)), "added": max(0, len(actual) - len(expected)), "changed": len(changed),
            "accuracy": _ratio(identical, denominator), "actor": _metric(denominator, actor_ok), "pass": _metric(denominator, pass_ok), "cards": _metric(denominator, cards_ok),
            "order": {**_metric(denominator, identical), "duplicate_expected_turn_ids": _duplicates(expected), "duplicate_actual_turn_ids": _duplicates(actual)},
            "mismatches": changed, "missing_rows": expected[compared:], "added_rows": actual[compared:],
        },
        "wind_catch_chain": {"denominator": 0, "correct": 0, "errors": 0, "accuracy": None, "status": "diagnostic_only_truth_schema_has_no_wind_chain", "actual": visual.get("wind_catch_chain", ())},
        "ranking": {
            **_metric(
                len(ranking_expected_for_metrics),
                sum(a == b for a, b in zip(ranking_expected_for_metrics, finish_actual)),
            ),
            "expected": finish_expected,
            "actual": finish_actual,
            "status": ranking_status,
        },
    }


def _na_metrics() -> dict[str, object]:
    return {"strict": False, "status": "not_available_without_explicit_truth", "level": None, "hand_multiset": None, "lead_player": None, "actions": None, "wind_catch_chain": None, "ranking": None}


def _first_divergence(truth: TruthLog | None, opening: dict[str, object], visual: dict[str, object]) -> dict[str, object] | None:
    if truth is None:
        return None
    actual_open = _opening_evidence(opening)
    expected_open = {"round_level": truth.initial_state.round_level, "hand": sorted(truth.initial_state.my_hand), "lead_player": truth.initial_state.lead_player}
    if expected_open["round_level"] != actual_open.get("round_level") or Counter(expected_open["hand"]) != Counter(actual_open.get("hand", ())):
        return {"kind": "initial_state", "field": "round_level_or_hand", "frame_index": _int_or_none(actual_open.get("frame_index")) or 0, "monotonic_ms": _int_or_none(actual_open.get("monotonic_ms")), "expected": expected_open, "actual": actual_open}
    actions = list(visual.get("actions", {}).get("rows", ()))
    for index in range(max(len(truth.turns), len(actions))):
        exp = truth.turns[index] if index < len(truth.turns) else None
        act = actions[index] if index < len(actions) else None
        if exp and act and exp.actor == act.get("actor") and exp.is_pass == act.get("is_pass") and Counter(exp.cards) == Counter(act.get("cards", ())):
            continue
        exp_raw = {"turn_id": exp.index, "trick_id": exp.trick_id, "actor": exp.actor, "is_pass": exp.is_pass, "cards": list(exp.cards), "frame_index": exp.frame_index, "monotonic_ms": exp.monotonic_ms} if exp else None
        frame = _int_or_none(act.get("frame_index")) if act else exp.frame_index if exp else 0
        return {"kind": "action", "field": "missing" if act is None else "added" if exp is None else "changed", "position": index + 1, "frame_index": frame or 0, "monotonic_ms": _int_or_none(act.get("monotonic_ms")) if act else exp.monotonic_ms if exp else None, "expected": exp_raw, "actual": act}
    leads = visual.get("lead", {}).get("confirmations", ())
    actual_lead = leads[0].get("lead_player") if leads else None
    if actual_lead != truth.initial_state.lead_player:
        return {"kind": "lead_player", "field": "lead_player", "frame_index": _int_or_none(leads[0].get("frame_index")) if leads else 0, "monotonic_ms": None, "expected": {"lead_player": truth.initial_state.lead_player}, "actual": {"lead_player": actual_lead}}
    expected_ranking = list(truth.outcome.finish_order)
    actual_ranking_rows = list(visual.get("rankings", ()))
    actual_ranking = [row.get("actor") for row in actual_ranking_rows]
    if truth.outcome.complete and expected_ranking and expected_ranking != actual_ranking:
        first_rank = actual_ranking_rows[0] if actual_ranking_rows else {}
        fallback_frame = truth.turns[-1].frame_index if truth.turns else 0
        return {
            "kind": "ranking",
            "field": "finish_order",
            "frame_index": _int_or_none(first_rank.get("frame_index")) or fallback_frame or 0,
            "monotonic_ms": _int_or_none(first_rank.get("monotonic_ms")),
            "expected": {"finish_order": expected_ranking},
            "actual": {"finish_order": actual_ranking},
        }
    return None


def _write_evidence(
    session: Path,
    output: Path,
    divergence: dict[str, object],
    truth: TruthLog | None,
    *,
    visual_frame_log: Path | None = None,
    runtime_directory: Path | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    expected_path, actual_path, context_path = output / "expected.json", output / "actual.json", output / "context.json"
    atomic_write_json(expected_path, divergence.get("expected"))
    atomic_write_json(actual_path, divergence.get("actual"))
    expected = divergence.get("expected") if isinstance(divergence.get("expected"), dict) else {}
    actual = divergence.get("actual") if isinstance(divergence.get("actual"), dict) else {}
    fallback = _resolve_evidence_context(
        divergence,
        actual,
        visual_frame_log=visual_frame_log,
        runtime_directory=runtime_directory,
    )
    revision_before = actual.get("state_revision_before")
    revision_after = actual.get("state_revision_after")
    if revision_before is None:
        revision_before = fallback.get("state_revision_before")
    if revision_after is None:
        revision_after = fallback.get("state_revision_after")
    monotonic_ms = divergence.get("monotonic_ms")
    if monotonic_ms is None:
        monotonic_ms = fallback.get("monotonic_ms")
    event_id = actual.get("event_id") or fallback.get("event_id")
    reducer_before = fallback.get("reducer_state_before")
    reducer_after = fallback.get("reducer_state_after")
    if not isinstance(reducer_before, dict):
        reducer_before = {"revision": revision_before, "available": revision_before is not None}
    if not isinstance(reducer_after, dict):
        reducer_after = {"revision": revision_after, "available": revision_after is not None}
    reducer_before["revision"] = revision_before
    reducer_after["revision"] = revision_after
    decisions_path = (
        runtime_directory / "decisions.jsonl"
        if runtime_directory is not None
        else output.parent / "visual_driven" / "runtime" / "decisions.jsonl"
    )
    context = {
        "kind": divergence.get("kind"), "field": divergence.get("field"), "position": divergence.get("position"),
        "frame_index": divergence.get("frame_index"), "monotonic_ms": monotonic_ms,
        "event_id": event_id, "turn_id": actual.get("turn_id", expected.get("turn_id")), "trick_id": actual.get("trick_id", expected.get("trick_id")),
        "state_revision_before": revision_before, "state_revision_after": revision_after,
        "reducer_state_before": reducer_before,
        "reducer_state_after": reducer_after,
        "reducer_state_diff": _reducer_state_diff(reducer_before, reducer_after),
        "engine_input": actual.get("engine_input", fallback.get("engine_input")),
        "engine_input_evidence": {
            "path": str(decisions_path),
            "available": decisions_path.is_file(),
        },
        "fallback": fallback.get("fallback"),
        "evidence_unavailable": fallback.get("evidence_unavailable", []),
        "truth_source_session_id": truth.source_session_id if truth else None,
    }
    atomic_write_json(context_path, context)
    description = output / "description.md"
    description.write_text(f"# 首个质量分歧\n\n- 类型：{divergence.get('kind')}\n- 字段：{divergence.get('field')}\n- 帧：{divergence.get('frame_index')}\n- 单调时间：{monotonic_ms}\n- 事件：{event_id}\n- 状态版本：{revision_before} → {revision_after}\n", encoding="utf-8")
    images = _save_frame_rois(session / "video" / "game.avi", int(divergence.get("frame_index", 0) or 0), output, str(actual.get("actor", expected.get("actor", "")) or ""))
    return {"directory": str(output), "kind": divergence.get("kind"), "field": divergence.get("field"), "frame_index": divergence.get("frame_index"), "monotonic_ms": monotonic_ms, "event_id": event_id, "state_revision_before": revision_before, "state_revision_after": revision_after, "expected": str(expected_path), "actual": str(actual_path), "context": str(context_path), "description": str(description), "images": images}


def _resolve_evidence_context(
    divergence: dict[str, object],
    actual: dict[str, object],
    *,
    visual_frame_log: Path | None,
    runtime_directory: Path | None,
) -> dict[str, object]:
    """Resolve honest nearby evidence when a missing action has no event object."""

    target_frame = _int_or_none(divergence.get("frame_index"))
    target_ms = _int_or_none(divergence.get("monotonic_ms"))
    frame_rows = _read_existing_json_lines(visual_frame_log)
    indexed_rows = [
        row for row in frame_rows if _int_or_none(row.get("frame_index")) is not None
    ]
    indexed_rows.sort(key=lambda row: int(row["frame_index"]))
    nearest_row = _nearest_frame_row(indexed_rows, target_frame)
    before_row, after_row = _surrounding_frame_rows(indexed_rows, target_frame)
    if target_ms is None and nearest_row is not None:
        target_ms = _int_or_none(nearest_row.get("monotonic_ms"))

    nearby_events: list[dict[str, object]] = []
    for row in indexed_rows:
        raw_events = row.get("events", ())
        if not isinstance(raw_events, list):
            continue
        for event in raw_events:
            if not isinstance(event, dict) or not event.get("event_id"):
                continue
            nearby_events.append(
                {
                    **event,
                    "_frame_index": row.get("frame_index"),
                    "_frame_monotonic_ms": row.get("monotonic_ms"),
                    "_frame_state_revision": row.get("state_revision"),
                }
            )
    nearest_event = _nearest_event(nearby_events, target_frame, target_ms)

    runtime_timeline = (
        runtime_directory / "timeline.jsonl"
        if runtime_directory is not None
        else None
    )
    timeline_rows = _read_existing_json_lines(runtime_timeline)
    nearest_timeline_event = _nearest_event(timeline_rows, None, target_ms)
    selected_event = nearest_event or nearest_timeline_event

    before_revision = _row_revision(before_row)
    after_revision = _row_revision(after_row)
    if before_revision is None and selected_event is not None:
        before_revision = _int_or_none(selected_event.get("state_revision_before"))
    if after_revision is None and selected_event is not None:
        after_revision = _int_or_none(selected_event.get("state_revision_after"))
    if before_revision is None:
        before_revision = _int_or_none(actual.get("state_revision_before"))
    if after_revision is None:
        after_revision = _int_or_none(actual.get("state_revision_after"))

    before_state = _frame_state_summary(before_row, "nearest_frame_before_or_at")
    after_state = _frame_state_summary(after_row, "nearest_frame_after_or_at")
    if before_state is not None:
        before_state["revision"] = before_revision
    if after_state is not None:
        after_state["revision"] = after_revision

    decisions_path = runtime_directory / "decisions.jsonl" if runtime_directory else None
    decisions = _read_existing_json_lines(decisions_path)
    decision = _nearest_decision(decisions, divergence)
    engine_input = None
    if decision is not None:
        engine_input = decision.get("engine_input")
        if engine_input is None:
            engine_input = {
                "available_in_decision": False,
                "decision_id": decision.get("decision_id"),
                "request_id": decision.get("request_id"),
                "state_revision": decision.get("state_revision"),
            }

    unavailable: list[str] = []
    values = {
        "monotonic_ms": target_ms,
        "event_id": selected_event.get("event_id") if selected_event else None,
        "state_revision_before": before_revision,
        "state_revision_after": after_revision,
    }
    unavailable.extend(name for name, value in values.items() if value is None)
    return {
        **values,
        "reducer_state_before": before_state,
        "reducer_state_after": after_state,
        "engine_input": engine_input,
        "fallback": {
            "monotonic_ms_source": (
                "divergence" if divergence.get("monotonic_ms") is not None
                else "nearest_visual_frame" if target_ms is not None
                else "unavailable"
            ),
            "event_id_source": (
                "nearest_actual_event_in_visual_frame_log" if nearest_event is not None
                else "nearest_actual_event_in_runtime_timeline" if nearest_timeline_event is not None
                else "unavailable"
            ),
            "state_source": "surrounding_visual_frame_rows",
            "target_frame_index": target_frame,
            "before_frame_index": before_row.get("frame_index") if before_row else None,
            "after_frame_index": after_row.get("frame_index") if after_row else None,
        },
        "evidence_unavailable": unavailable,
    }


def _read_existing_json_lines(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.is_file():
        return []
    try:
        return list(read_json_lines(path))
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _nearest_frame_row(
    rows: list[dict[str, object]], target_frame: int | None
) -> dict[str, object] | None:
    if not rows:
        return None
    if target_frame is None:
        return rows[0]
    return min(
        rows,
        key=lambda row: abs(int(row["frame_index"]) - target_frame),
    )


def _surrounding_frame_rows(
    rows: list[dict[str, object]], target_frame: int | None
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not rows:
        return None, None
    if target_frame is None:
        return rows[0], rows[0]
    before = [row for row in rows if int(row["frame_index"]) < target_frame]
    after = [row for row in rows if int(row["frame_index"]) >= target_frame]
    return (
        before[-1] if before else rows[0],
        after[0] if after else rows[-1],
    )


def _nearest_event(
    events: list[dict[str, object]],
    target_frame: int | None,
    target_ms: int | None,
) -> dict[str, object] | None:
    if not events:
        return None

    def score(event: dict[str, object]) -> tuple[int, int]:
        frame = _int_or_none(event.get("_frame_index"))
        monotonic = _int_or_none(
            event.get("_frame_monotonic_ms", event.get("monotonic_ms"))
        )
        frame_delta = abs(frame - target_frame) if frame is not None and target_frame is not None else 10**12
        ms_delta = abs(monotonic - target_ms) if monotonic is not None and target_ms is not None else 10**12
        return frame_delta, ms_delta

    return min(events, key=score)


def _row_revision(row: dict[str, object] | None) -> int | None:
    return _int_or_none(row.get("state_revision")) if row is not None else None


def _frame_state_summary(
    row: dict[str, object] | None, source: str
) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "available": True,
        "source": source,
        "frame_index": row.get("frame_index"),
        "monotonic_ms": row.get("monotonic_ms"),
        "status": row.get("status"),
        "current_player": row.get("current_player"),
        "revision": row.get("state_revision"),
    }


def _reducer_state_diff(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    before_revision = _int_or_none(before.get("revision"))
    after_revision = _int_or_none(after.get("revision"))
    return {
        "before_revision": before_revision,
        "after_revision": after_revision,
        "revision_delta": (
            after_revision - before_revision
            if before_revision is not None and after_revision is not None
            else None
        ),
        "current_player_before": before.get("current_player"),
        "current_player_after": after.get("current_player"),
        "current_player_changed": before.get("current_player") != after.get("current_player"),
        "source": "surrounding_visual_frame_rows",
        "scope": "summary_only_not_full_reducer_state",
    }


def _nearest_decision(
    decisions: list[dict[str, object]], divergence: dict[str, object]
) -> dict[str, object] | None:
    if not decisions:
        return None
    expected = divergence.get("expected")
    expected = expected if isinstance(expected, dict) else {}
    turn_id = _int_or_none(expected.get("turn_id"))
    if turn_id is not None:
        same_turn = [row for row in decisions if _int_or_none(row.get("turn_id")) == turn_id]
        if same_turn:
            return same_turn[-1]
    return decisions[-1]


def _save_frame_rois(video: Path, index: int, output: Path, actor: str) -> dict[str, str]:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            return {}
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = capture.read()
        if not ok:
            return {}
    finally:
        capture.release()
    height, width = frame.shape[:2]
    crops = {
        "trigger": frame,
        "level_roi": frame[: max(1, height // 4), width // 3 : max(width // 3 + 1, 2 * width // 3)],
        "hand_roi": frame[2 * height // 3 :, :],
    }
    actor_boxes = {"self": (0, height // 2, width, height), "right": (width // 2, height // 4, width, 3 * height // 4), "opposite": (0, 0, width, height // 2), "left": (0, height // 4, width // 2, 3 * height // 4)}
    if actor in actor_boxes:
        x1, y1, x2, y2 = actor_boxes[actor]
        crops[f"play_{actor}_roi"] = frame[y1:y2, x1:x2]
    saved: dict[str, str] = {}
    for name, image in crops.items():
        path = output / f"{name}.png"
        if image.size and cv2.imwrite(str(path), image):
            saved[name] = str(path)
    return saved


def _visual_advice_summary(
    result: VisualPipelineReplayResult,
    advisor: Any,
    *,
    listener_status: str | None = None,
) -> dict[str, object]:
    """Summarize visual advice without treating an empty run as success.

    Replay completion means the input frames were consumed; it does not prove
    that the listener reached an advice opportunity.  The status below keeps
    "not exercised", "withheld because the listener is incomplete", and a real
    successful advice run distinct.
    """

    requested = int(result.advice_requested or 0)
    ready = int(result.advice_ready or 0)
    failed = int(result.advice_failed or 0)
    stale = int(result.advice_stale or 0)
    timeouts = int(result.advice_timeouts or 0)
    withheld = int(result.advice_withheld or 0)
    identity = getattr(result, "runtime_identity", {}) or {}
    identity = identity if isinstance(identity, dict) else {}
    effective_listener_status = listener_status or identity.get("listener_status")
    result_status = str(getattr(result, "status", "") or "")
    listener_gap = (
        effective_listener_status in {
            "incomplete", "blocked", "review_required", "error", "not_started"
        }
        or result_status in {"incomplete", "blocked", "review_required", "error"}
        or withheld > 0
    )

    if failed > 0 or timeouts > 0:
        status = "failed"
        status_reason = "advice_failure_or_timeout"
    elif listener_gap:
        status = "withheld_due_listener_gap"
        status_reason = (
            "advice_withheld"
            if withheld > 0
            else f"listener_status={effective_listener_status or result_status or 'unknown'}"
        )
    elif requested == 0:
        status = "not_exercised"
        status_reason = "no_advice_requests"
    elif bool(getattr(result, "completed", False)) and ready == requested and stale == 0:
        status = "passed"
        status_reason = "advice_requests_completed"
    elif bool(getattr(result, "completed", False)) and ready + stale >= requested:
        # ``stale`` means the production advice result was superseded before
        # consumption. It is an explicit visual-channel advisory, not a
        # FableDan failure and must never be hidden or relabeled as passed.
        status = "completed_with_stale"
        status_reason = "advice_requests_completed_with_stale"
    else:
        status = "incomplete"
        status_reason = "visual_replay_not_completed_or_advice_incomplete"

    return {
        "available": True,
        "status": status,
        "status_reason": status_reason,
        "completed": bool(result.completed),
        "listener_status": effective_listener_status,
        "processed_turn_count": result.processed_turn_count,
        "run_directory": str(result.run_directory) if result.run_directory else None,
        "advice": {
            "requested": requested,
            "ready": ready,
            "failed": failed,
            "stale": stale,
            "timeout": timeouts,
            "timeouts": timeouts,
            "withheld": withheld,
            "statuses": dict(result.advice_statuses),
        },
        "advisor": _advisor_info(advisor),
        "artifacts": {name: str(path) for name, path in result.artifact_paths.items()},
    }


def _comparison_summary(value: ReplayComparison) -> dict[str, object]:
    return {"identical_turn_ids": list(value.identical_turn_ids), "identical": len(value.identical_turn_ids), "missing": len(value.missing), "added": len(value.added), "changed": len(value.changed), "metric_deltas": len(value.metric_deltas)}


def _normalize_roots(values: Iterable[Path | str]) -> tuple[Path, ...]:
    result: list[Path] = []
    for value in values:
        raw = Path(value).resolve()
        root = raw / "sessions" if (raw / "sessions").is_dir() else raw
        if root not in result:
            result.append(root)
    return tuple(result)


def _scan_run_ids(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    result: list[str] = []
    for item in values:
        normalized = str(item).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


def _reject_internal_output(output: Path, roots: tuple[Path, ...]) -> None:
    if any(_relative_to(output, root) for root in roots):
        raise ValueError("audit output must be outside every sessions root")


def _discover(
    roots: tuple[Path, ...],
    *,
    explicit_sessions: tuple[_Session, ...] = (),
) -> tuple[_Session, ...]:
    result: list[_Session] = []
    used: set[str] = set()
    result.extend(explicit_sessions)
    used.update(item.store_id for item in explicit_sessions)
    for index, root in enumerate(roots, start=1):
        label = root.parent.name if root.name.lower() == "sessions" else root.name
        store_id = f"{_safe_name(label or f'store_{index}')}_{hashlib.sha256(str(root).casefold().encode()).hexdigest()[:8]}"
        while store_id in used:
            store_id += f"_{index}"
        used.add(store_id)
        if not root.is_dir():
            continue
        for session in sorted(root.iterdir(), key=lambda path: path.name):
            if session.is_dir() and (session / "manifest.json").is_file():
                result.append(_Session(session.resolve(), root, store_id, _session_id(session)))
    return tuple(result)


def _normalize_explicit_sessions(values: Iterable[Path | str] | None) -> tuple[_Session, ...]:
    if values is None:
        return ()
    result: list[_Session] = []
    used: set[tuple[Path, str]] = set()
    for value in values:
        source = Path(value).resolve()
        if not source.is_dir() or not (source / "manifest.json").is_file():
            raise ValueError(f"explicit session must contain manifest.json: {source}")
        root = source.parent
        store_id = f"{_safe_name(root.name or 'session')}_{hashlib.sha256(str(root).casefold().encode()).hexdigest()[:8]}"
        key = (source, store_id)
        if key not in used:
            result.append(_Session(source, root, store_id, _session_id(source)))
            used.add(key)
    return tuple(result)


def _unique_paths(values: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    for value in values:
        path = Path(value).resolve()
        if path not in result:
            result.append(path)
    return tuple(result)


def _inventory(
    sessions: tuple[_Session, ...],
    roots: tuple[Path, ...],
    scan_run_id: str | Iterable[str] | None,
    *,
    profile_root: Path | None = None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    profile_cache: dict[Path, dict[str, object]] = {}
    scan_ids = _scan_run_ids(scan_run_id)
    for item in sessions:
        # Presence-only metadata is intentional. Do not call
        # resolve_truth_audit_reference() here: reference selection, parsing and
        # hashing belong to the post-visual comparison phase.
        canonical_truth = item.source / "truth_log.json"
        staged_truth = [
            item.source / "derived" / "truth_scan_drafts" / run_id / "truth_log.json"
            for run_id in scan_ids
        ]
        selected_truth = canonical_truth if canonical_truth.is_file() else next(
            (path for path in staged_truth if path.is_file()), None
        )
        truth_kind = (
            "canonical" if canonical_truth.is_file()
            else "staged" if selected_truth is not None
            else "none"
        )
        frame_path = item.source / "video" / "frame_index.jsonl"
        frame_rows, frame_error = _read_lines_safe(frame_path)
        indices = [_int_or_none(row.get("frame_index")) for row in frame_rows]
        values = [value for value in indices if value is not None]
        profile = profile_root or item.source.parent.parent
        profile_data = profile_cache.setdefault(profile, _profile_inventory(profile))
        rows.append(
            {
                "source": str(item.source), "sessions_root": str(item.root), "store_id": item.store_id, "session_id": item.session_id,
                "truth_kind": truth_kind,
                "truth_path": str(selected_truth) if selected_truth is not None else None,
                "files": {
                    "manifest": _file_info(item.source / "manifest.json"), "video": _file_info(item.source / "video" / "game.avi"),
                    "frame_index": _file_info(frame_path),
                    # TruthLog is presence metadata only during inventory;
                    # its bytes/hash are post-visual evidence.
                    "truth": _presence_file_info(selected_truth),
                    "canonical_truth": _presence_file_info(canonical_truth),
                    "recognition_trace": _file_info(item.source / "recognition_trace.jsonl"), "timeline": _file_info(item.source / "timeline.jsonl"),
                },
                "frame_index": {"row_count": len(frame_rows), "continuous": bool(values) and all(b == a + 1 for a, b in zip(values, values[1:])), "first": values[0] if values else None, "last": values[-1] if values else None, "duplicates": sorted(v for v, count in Counter(values).items() if count > 1), "parse_error": frame_error},
                "profile_artifacts": profile_data,
            }
        )
    return {"schema": "guandan.session-replay-inventory/1", "created_at": datetime.now().astimezone().isoformat(), "roots": [str(root) for root in roots], "session_count": len(rows), "indexed_frame_count": sum(int(row["frame_index"]["row_count"]) for row in rows), "sessions": rows, "environment": _environment()}


def _profile_inventory(profile: Path) -> dict[str, object]:
    configs = [path for path in (profile / "profile.json", profile / "regions_config.json", profile / "templates_config.json") if path.is_file()]
    template_root, model_root = profile / "templates", profile / "models"
    templates = sorted(path for path in template_root.rglob("*") if path.is_file()) if template_root.is_dir() else []
    models = sorted(path for path in model_root.rglob("*") if path.is_file()) if model_root.is_dir() else []
    preferred = next((path for path in models if path.name == "best.npz"), None)
    return {
        "profile": str(profile), "configuration_files": [_file_info(path) for path in configs], "configuration_hash": _tree_hash(configs, profile),
        "template_manifest_hash": _tree_hash(templates, template_root), "models": [_file_info(path) for path in models],
        "model_digest": _sha_file(preferred) if preferred else None, "model_path": str(preferred) if preferred else None, "backend": "numpy" if preferred and preferred.suffix == ".npz" else None,
    }


def _source_snapshot(roots: tuple[Path, ...]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            key = f"{root}::{path.relative_to(root).as_posix()}"
            data: dict[str, object] = {"relative_path": path.relative_to(root).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if path.name in {"manifest.json", "timeline.jsonl", "recognition_trace.jsonl", "game.avi", "frame_index.jsonl"}:
                data["sha256"] = _sha_file(path)
            result[key] = data
    return result


def _make_summary(*, run_id: str, rows: list[dict[str, object]], inventory: dict[str, object], scan_run_id: str | Iterable[str] | None, command: Iterable[str] | None, source_changes: list[dict[str, object]], source_snapshot_paths: tuple[Path, Path], max_workers: int = 1) -> dict[str, object]:
    completed = sum(row.get("execution_status") == "completed" for row in rows)
    return {
        "schema": "guandan.session-replay-audit/2", "run_id": run_id, "created_at": datetime.now().astimezone().isoformat(),
        "parameters": {
            "sessions_roots": inventory.get("roots", ()),
            "scan_run_ids": list(_scan_run_ids(scan_run_id)),
            "command": list(command) if command else None,
            "max_workers": int(max_workers),
            "execution_mode": "bounded_thread_pool",
            "memory_policy": "at most three sequential frame streams; one OpenCV native thread per worker; runtime workers cleaned per session",
            "report_order": "original session selection order",
            "report_merge": "top-level files are generated once after every worker finishes",
        },
        "environment": inventory.get("environment", {}), "session_count": len(rows), "completed": completed, "errors": len(rows) - completed,
        "frames_processed": sum(int(row.get("frames_processed", 0) or 0) for row in rows), "indexed_frames": sum(int(row.get("indexed_frames", 0) or 0) for row in rows),
        "source_integrity": {"unchanged": not source_changes, "changes": source_changes, "before_snapshot": str(source_snapshot_paths[0]), "after_snapshot": str(source_snapshot_paths[1])},
        "truth_levels": dict(Counter(str(row.get("truth_log", {}).get("kind", "none")) for row in rows)), "fabledan": _aggregate_advice(rows), "sessions": rows,
    }


def _aggregate_advice(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for channel in ("truth_driven", "visual_driven"):
        totals = Counter()
        status_counts = Counter()
        available = completed = 0
        for row in rows:
            fabledan = row.get("fabledan", {})
            data = fabledan.get(channel, {}) if isinstance(fabledan, dict) else {}
            if not isinstance(data, dict) or not data.get("available"):
                continue
            available += 1
            completed += bool(data.get("completed"))
            status = data.get("status")
            if status:
                status_counts[str(status)] += 1
            advice = data.get("advice", {})
            if isinstance(advice, dict):
                for name in ("requested", "ready", "failed", "stale", "timeout", "withheld"):
                    totals[name] += int(advice.get(name, 0) or 0)
        result[channel] = {
            "available_sessions": available,
            "completed_sessions": completed,
            "status_counts": dict(sorted(status_counts.items())),
            **dict(totals),
        }
    return result


def _write_sessions_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fields = ("store_id", "session_id", "source", "truth_kind", "execution_status", "truth_quality", "visual_quality", "fabledan_quality", "visual_advice_status", "frames_processed", "indexed_frames", "visual_requested", "visual_ready", "truth_requested", "truth_ready")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            fabledan = row.get("fabledan", {})
            visual = fabledan.get("visual_driven", {}) if isinstance(fabledan, dict) else {}
            truth = fabledan.get("truth_driven", {}) if isinstance(fabledan, dict) else {}
            va = visual.get("advice", {}) if isinstance(visual, dict) else {}
            ta = truth.get("advice", {}) if isinstance(truth, dict) else {}
            writer.writerow({"store_id": row.get("store_id"), "session_id": row.get("session_id"), "source": row.get("source"), "truth_kind": row.get("truth_log", {}).get("kind"), "execution_status": row.get("execution_status"), "truth_quality": row.get("truth_quality"), "visual_quality": row.get("visual_quality"), "fabledan_quality": row.get("fabledan_quality"), "visual_advice_status": visual.get("status"), "frames_processed": row.get("frames_processed", 0), "indexed_frames": row.get("indexed_frames", 0), "visual_requested": va.get("requested", 0), "visual_ready": va.get("ready", 0), "truth_requested": ta.get("requested", 0), "truth_ready": ta.get("ready", 0)})


def _write_failures_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fields = ("store_id", "session_id", "category", "status", "detail", "evidence")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            failures: list[tuple[str, str, str]] = []
            if row.get("execution_status") != "completed":
                failures.append(("execution", str(row.get("execution_status")), str(row.get("error", ""))))
            for category in ("truth_quality", "visual_quality", "fabledan_quality"):
                if row.get(category) in {"failed", "error", "incomplete"}:
                    failures.append((category, str(row.get(category)), "quality mismatch"))
            divergence = row.get("first_divergence")
            evidence = divergence.get("directory") if isinstance(divergence, dict) else ""
            for category, status, detail in failures:
                writer.writerow({"store_id": row.get("store_id"), "session_id": row.get("session_id"), "category": category, "status": status, "detail": detail, "evidence": evidence})


def _write_markdown(path: Path, summary: dict[str, object]) -> None:
    integrity = summary.get("source_integrity", {})
    lines = ["# 第一阶段：离线全量审计报告", "", f"- 对局：{summary.get('completed', 0)}/{summary.get('session_count', 0)} 完整执行", f"- 帧：{summary.get('frames_processed', 0)}/{summary.get('indexed_frames', 0)}", f"- 工具错误：{summary.get('errors', 0)}", f"- 源数据未变化：{'是' if integrity.get('unchanged') else '否'}", f"- 真值等级：{json.dumps(summary.get('truth_levels', {}), ensure_ascii=False)}", "", "## FableDan", ""]
    for channel, data in summary.get("fabledan", {}).items():
        lines.append(f"- {channel}：{json.dumps(data, ensure_ascii=False)}")
    lines.extend(["", "## 会话", ""])
    lines.extend([
        "## 主链运行时",
        "",
        "主视觉验收链：LiveV2SessionRuntime -> LiveV2 Vision Runtime -> "
        "LiveEngine -> ProductionRuleSession -> LiveV2 Advice/FableDan。",
        "旧 LiveOrchestrator 仅属于兼容/单元测试路径，不是主链。",
        "",
        "## 会话",
        "",
    ])
    for row in summary.get("sessions", ()):
        lines.append(
            f"- `{row.get('store_id')}/{row.get('session_id')}`："
            f"执行={row.get('execution_status')}，监听={row.get('listener_status')} "
            f"({row.get('listener_status_reason')})，视觉={row.get('visual_quality')}，"
            f"FableDan={row.get('fabledan_quality')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify(run_dir: Path, summary: dict[str, object], before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]) -> dict[str, object]:
    rows = tuple(summary.get("sessions", ()))
    csv_count = _csv_count(run_dir / "sessions.csv")
    count = int(summary.get("session_count", 0))
    checks = {
        "nonempty_session_set": count > 0,
        "session_counts_match": count == len(rows) == csv_count,
        "frames_complete": int(summary.get("frames_processed", 0)) == int(summary.get("indexed_frames", 0)),
        "no_tool_errors": int(summary.get("errors", 0)) == 0,
        "source_unchanged": before == after,
        "required_artifacts_exist": all((run_dir / name).is_file() for name in ("all_session_audit.json", "inventory.json", "sessions.csv", "failures.csv", "summary.md", "source_snapshot_before.json", "source_snapshot_after.json")),
        "per_session_summaries_exist": all((run_dir / "sessions" / str(row.get("store_id")) / _safe_name(str(row.get("session_id"))) / "summary.json").is_file() for row in rows),
        "per_session_quality": all(
            not _is_strict_row(row) or not _row_quality_failures(row)
            for row in rows if isinstance(row, dict)
        ),
    }
    return {"schema": "guandan.session-replay-verification/1", "created_at": datetime.now().astimezone().isoformat(), "passed": all(checks.values()), "checks": checks, "counts": {"json_sessions": count, "csv_sessions": csv_count, "frames_processed": summary.get("frames_processed", 0), "indexed_frames": summary.get("indexed_frames", 0), "tool_errors": summary.get("errors", 0), "source_changes": len(_source_changes(before, after))}}


def _is_strict_row(row: dict[str, object]) -> bool:
    truth = row.get("truth_log")
    return (
        isinstance(truth, dict)
        and truth.get("kind") in {"canonical", "verified"}
        and row.get("truth_qualification") in {
            "verified_label", "verified", "trusted_for_run"
        }
    )


def _row_quality_failures(row: dict[str, object]) -> list[str]:
    """Return strict verification failures for one canonical verified row.

    ``fabledan_quality`` is an aggregate, not a replacement for the two
    channel reports.  ``advisory`` is therefore valid when the trusted
    TruthLog channel completed successfully and the visual channel only has
    stale responses.  The nested channel checks remain authoritative for
    incomplete TruthLog advice and real failed/timeout requests.
    """

    failures: list[str] = []
    expected = {
        "execution_status": "completed",
        "frame_replay_status": "complete",
        "listener_status": "complete",
        "opening_status": "recognized",
        "truth_quality": "passed",
        "visual_quality": "passed",
        "comparison_status": "passed",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            failures.append(f"{key}={row.get(key)!r}")

    # A stale visual response is an advisory finding and remains visible in
    # fabledan.visual_driven.advice.stale.  Only an explicit overall failure
    # (or an unknown/missing aggregate) fails here; trusted-channel and
    # request-level failures are handled by the nested checks below.
    overall_quality = str(row.get("fabledan_quality", "") or "")
    if overall_quality not in {"passed", "advisory"}:
        failures.append(f"fabledan_quality={overall_quality!r}")

    failures.extend(_fabledan_blocking_reasons(row, strict=True))
    return failures

def _unexpected_session_row(
    item: _Session,
    output: Path,
    exc: Exception,
    inventory: dict[str, object],
) -> dict[str, object]:
    """Return a failure row if a worker escapes the normal audit guard."""

    frame_inventory = inventory.get("frame_index", {})
    indexed = (
        int(frame_inventory.get("row_count", 0) or 0)
        if isinstance(frame_inventory, dict)
        else 0
    )
    row = {
        "source": str(item.source),
        "sessions_root": str(item.root),
        "store_id": item.store_id,
        "session_id": item.session_id,
        "status": "error",
        "execution_status": "error",
        "frame_replay_status": "incomplete",
        "opening_status": "not_observable",
        "listener_status": "incomplete",
        "listener_status_reason": "worker_exception",
        "comparison_status": "failed",
        "truth_quality": "not_available",
        "visual_quality": "incomplete",
        "fabledan_quality": "not_evaluated",
        "frames_processed": 0,
        "indexed_frames": indexed,
        "processed_turns": 0,
        "truth_log": {
            "kind": str(inventory.get("truth_kind", "none")),
            "path": inventory.get("truth_path"),
            "loaded_before_visual_replay": False,
        },
        "inventory": inventory,
        "lineage": {
            "runtime": "live_v2",
            "legacy_orchestrator_used": False,
            "truth_log_used_as_visual_input": False,
        },
        "warnings": [],
        "first_divergence": None,
        "timeline_comparison": {},
        "fabledan": {},
        "evidence_paths": {},
        "error": f"{type(exc).__name__}: {exc}",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", row)
    return row


def _recognition_for_session(session: Path) -> ScreenshotRecognitionService:
    profile = session.parent.parent
    return ScreenshotRecognitionService(AnnotationService(profile.parent, profile.name), TemplateService(profile.parent, profile.name))


def _recognition_for_profile(profile_root: Path | None) -> Callable[[Path], ScreenshotRecognitionService]:
    if profile_root is None:
        return _recognition_for_session
    profile = Path(profile_root).resolve()
    return lambda _session: ScreenshotRecognitionService(
        AnnotationService(profile.parent, profile.name),
        TemplateService(profile.parent, profile.name),
    )


def _fabledan_for_profile(profiles_root: Path, profile_name: str) -> Any:
    return build_advisor("fabledan", profiles_root=profiles_root, profile_name=profile_name, fabledan_diagnostics="full")


def _load_truth(session: Path, reference: TruthAuditReference | None) -> TruthLog | None:
    return load_truth_log(reference.path, session_id=_session_id(session)) if reference else None


def _truth_metadata(
    reference: TruthAuditReference | None,
    truth: TruthLog | None,
    *,
    trusted: bool = False,
) -> dict[str, object]:
    if not reference or not truth:
        return {"kind": "none", "path": None, "sha256": None, "schema": None, "provenance": None}
    raw = json.loads(reference.path.read_text(encoding="utf-8"))
    return {
        # ``kind`` describes where the reference came from.  Do not replace
        # canonical with ``verified`` here: the CLI uses the separate
        # ``truth_qualification`` field to decide whether it is strict.
        "kind": reference.kind,
        "reference_kind": reference.kind,
        "trusted_session": trusted,
        "path": str(reference.path),
        "sha256": _sha_file(reference.path),
        "schema": raw.get("schema", raw.get("schema_version")),
        "provenance": truth.provenance.to_dict(),
        "label_status": truth.label_status,
    }


def _session_id(session: Path) -> str:
    try:
        raw = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    return str(raw.get("session_id", session.name))


def _presence_file_info(path: Path | None) -> dict[str, object]:
    """Return TruthLog presence metadata without touching file contents."""

    return _file_info(path, include_hash=False)


def _file_info(path: Path | None, *, include_hash: bool = True) -> dict[str, object]:
    if path is None or not path.is_file():
        result: dict[str, object] = {
            "path": str(path) if path else None, "exists": False,
            "size": None, "mtime_ns": None,
        }
    else:
        stat = path.stat()
        result = {
            "path": str(path), "exists": True, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    if include_hash:
        result["sha256"] = _sha_file(path) if path is not None and path.is_file() else None
    return result


def _tree_hash(paths: Iterable[Path], base: Path) -> str | None:
    digest, found = hashlib.sha256(), False
    for path in sorted(paths, key=str):
        if path.is_file():
            found = True
            try:
                name = path.relative_to(base).as_posix()
            except ValueError:
                name = str(path)
            digest.update(name.encode())
            digest.update(bytes.fromhex(_sha_file(path)))
    return digest.hexdigest() if found else None


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, object]:
    status = _git("status", "--short")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(status), "git_status": status.splitlines()}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], check=False, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    except OSError:
        return ""


def _artifact_map(directory: Path) -> dict[str, str]:
    names = ("manifest.json", "timeline.jsonl", "timeline.md", "advice.jsonl", "decisions.jsonl", "recognition_trace.jsonl", "observations.jsonl.part", "observations.jsonl.gz", "summary.json")
    return {name: str(directory / name) for name in names if (directory / name).is_file()}


def _advisor_info(advisor: Any) -> dict[str, object]:
    func = getattr(advisor, "audit_info", None)
    return dict(func()) if callable(func) else {}


def _inventory_row(inventory: dict[str, object], session: _Session) -> dict[str, object]:
    for row in inventory.get("sessions", ()):
        if row.get("source") == str(session.source) and row.get("store_id") == session.store_id:
            return dict(row)
    return {}


def _event_digest(row: dict[str, object]) -> dict[str, object]:
    return {"event_id": row.get("event_id"), "actor": row.get("actor"), "payload": row.get("payload", {}), "frame_index": row.get("_replay_frame_index"), "monotonic_ms": row.get("_replay_monotonic_ms", row.get("monotonic_ms")), "state_revision": row.get("_replay_state_revision", row.get("state_revision_after"))}


def _metric(denominator: int, correct: int, expected: object = None, actual: object = None) -> dict[str, object]:
    result: dict[str, object] = {"denominator": denominator, "correct": correct, "errors": max(0, denominator - correct), "accuracy": _ratio(correct, denominator)}
    if expected is not None or actual is not None:
        result.update({"expected": expected, "actual": actual})
    return result


def _strict_quality(metrics: dict[str, object]) -> str:
    if not metrics.get("strict"):
        return "not_available"
    actions = metrics.get("actions", {})
    ranking = metrics.get("ranking", {})
    ranking_ok = not ranking.get("denominator") or ranking.get("errors") == 0
    passed = metrics.get("level", {}).get("errors") == 0 and metrics.get("hand_multiset", {}).get("exact") is True and metrics.get("lead_player", {}).get("errors") == 0 and actions.get("missing") == actions.get("added") == actions.get("changed") == 0 and not actions.get("order", {}).get("duplicate_actual_turn_ids") and ranking_ok
    return "passed" if passed else "failed"


_TERMINAL_LISTENER_REASONS = frozenset({
    "listener_terminal",
    "terminal_reached",
    "game_finished",
})


def _last_replay_frame(path: Path) -> dict[str, object]:
    """Return the last replay row with nested runtime identity flattened.

    LiveV2 writes the authoritative terminal/current-player fields under the
    replay summary's ``runtime`` object.  Older artifacts may have written
    those fields at the top level, so normalize both shapes for diagnostics.
    """

    last: dict[str, object] = {}
    try:
        for row in read_json_lines(path):
            if isinstance(row, dict):
                last = row
    except Exception:
        return {}
    runtime = last.get("runtime")
    if isinstance(runtime, dict):
        normalized = dict(last)
        for key in (
            "runtime", "current_player", "listener_terminal",
            "listener_status", "listener_status_reason", "status_reason",
        ):
            if key in runtime and key not in normalized:
                normalized[key] = runtime[key]
        return normalized
    return last


def _listener_completion(
    result: VisualPipelineReplayResult,
    visual: dict[str, object],
    frame_complete: bool,
    *,
    legacy_compat: bool = False,
) -> tuple[str, str]:
    """Decide completion from production listener evidence, not sealing."""

    if legacy_compat:
        if frame_complete and str(getattr(result, "status", "")) == "complete":
            return "complete", "legacy_result_complete"
        return "incomplete", "legacy_result_incomplete"
    if not frame_complete:
        return "incomplete", "frame_replay_incomplete"

    result_status = str(getattr(result, "status", "") or "")
    if result_status != "complete":
        return "incomplete", f"result_status={result_status or 'missing'}"

    identity = getattr(result, "runtime_identity", {}) or {}
    if not isinstance(identity, dict):
        return "incomplete", "runtime_identity_invalid"
    if identity.get("runtime") != "live_v2":
        return "incomplete", "runtime_identity_not_live_v2"
    identity_listener_status = identity.get("listener_status")
    if identity_listener_status not in (None, "", "complete"):
        return "incomplete", f"runtime_listener_status={identity_listener_status}"

    actions = visual.get("actions", {})
    action_count = int(actions.get("count", 0) or 0) if isinstance(actions, dict) else 0
    if action_count <= 0:
        return "incomplete", "no_listener_actions"

    if "current_player" in identity:
        current_player = identity.get("current_player")
    else:
        # Backward-compatible fallback for old replay artifacts only; the
        # runtime identity always wins when it provides the field.
        current_player = _last_replay_frame(result.output_path).get("current_player")
    if current_player not in (None, ""):
        return "incomplete", f"current_player={current_player}"

    reason = str(
        getattr(result, "status_reason", "")
        or identity.get("listener_status_reason", "")
        or identity.get("status_reason", "")
        or ""
    )
    if reason not in _TERMINAL_LISTENER_REASONS:
        return "incomplete", f"non_terminal_status_reason={reason or 'missing'}"
    if identity.get("listener_terminal") is not True:
        return "incomplete", "runtime_reports_non_terminal_or_missing"
    return "complete", reason


def _visual_quality(metrics: dict[str, object], complete: bool) -> str:
    if not complete:
        return "incomplete"
    return _strict_quality(metrics) if metrics.get("strict") else "diagnostic"


def _advice_quality(truth: dict[str, object], visual: dict[str, object]) -> str:
    """Return overall advice quality without hiding visual-channel status."""

    visual_status = str(visual.get("status", "") or "")
    if visual_status == "failed":
        return "failed"

    truth_available = bool(truth.get("available"))
    if truth_available:
        truth_status = str(truth.get("status", "") or "")
        if truth_status == "failed" or not bool(truth.get("completed")):
            return "failed"
        advice = truth.get("advice", {})
        advice = advice if isinstance(advice, dict) else {}
        if int(advice.get("failed", 0) or 0) or int(advice.get("timeout", advice.get("timeouts", 0)) or 0):
            return "failed"
        # TruthLog-driven advice is the authoritative overall channel.  A
        # completed visual run with stale responses remains an advisory issue,
        # not an overall failure; an unexercised/withheld visual path remains
        # visible through its nested status and is not called a visual pass.
        if visual_status in {"", "passed", "completed_with_stale"}:
            return "passed"
        return "advisory"

    if visual_status == "passed":
        return "passed"
    if visual_status in {"completed_with_stale", "withheld_due_listener_gap", "not_exercised"}:
        return "advisory"
    return "not_available"


def _fabledan_blocking_reasons(
    row: dict[str, object], *, strict: bool
) -> list[str]:
    """Return only FableDan conditions that are allowed to block a run.

    Visual stale/withheld/incomplete states are intentionally advisory. The
    blocking rules are limited to actual failed/timeout requests and an
    incomplete trusted TruthLog channel, as required by the regression
    contract.
    """

    fabledan = row.get("fabledan")
    channels = fabledan if isinstance(fabledan, dict) else {}
    reasons: list[str] = []
    truth = channels.get("truth_driven")
    visual = channels.get("visual_driven")

    if strict:
        if isinstance(truth, dict):
            if not truth.get("available"):
                reasons.append("fabledan_truth_channel_not_available")
            else:
                truth_status = str(truth.get("status", "") or "")
                if truth_status in {"failed", "incomplete", "timeout"} or not bool(
                    truth.get("completed")
                ):
                    reasons.append("fabledan_truth_channel_incomplete")
        elif channels:
            # A structured FableDan report is present but has no trusted
            # channel, so a strict session cannot claim trusted completion.
            reasons.append("fabledan_truth_channel_not_available")
        else:
            # Compatibility for older summaries that contain only the overall
            # quality. ``advisory`` is a valid outcome: it means the trusted
            # channel passed while the visual channel has explainable stale
            # results. It must not be treated as a verification failure.
            overall = str(row.get("fabledan_quality", "") or "")
            if overall not in {"passed", "advisory"}:
                reasons.append("fabledan_truth_channel_not_available")

    for name, channel in (("truth", truth), ("visual", visual)):
        if not isinstance(channel, dict) or not channel.get("available"):
            continue
        status = str(channel.get("status", "") or "")
        if status in {"failed", "timeout"}:
            reasons.append(f"fabledan_{name}_{status}")
        if name == "truth" and (
            status == "incomplete" or not bool(channel.get("completed"))
        ):
            reasons.append("fabledan_truth_channel_incomplete")
        advice = channel.get("advice", {})
        advice = advice if isinstance(advice, dict) else {}
        failed = int(advice.get("failed", 0) or 0)
        timeouts = int(advice.get("timeout", advice.get("timeouts", 0)) or 0)
        if failed:
            reasons.append(f"fabledan_{name}_failed={failed}")
        if timeouts:
            reasons.append(f"fabledan_{name}_timeout={timeouts}")

    # Explicit overall failures remain blocking even when a producer omitted
    # one of the nested channel details. Advisory is intentionally excluded.
    overall = str(row.get("fabledan_quality", "") or "")
    if overall in {"failed", "timeout"}:
        reasons.append(f"fabledan_quality={overall}")
    elif strict and overall in {"not_evaluated", "incomplete"} and not channels:
        reasons.append(f"fabledan_quality={overall}")
    return reasons


def _advice_has_failures(advice: dict[str, object]) -> bool:
    return bool(
        int(advice.get("failed", 0) or 0)
        or int(advice.get("timeout", advice.get("timeouts", 0)) or 0)
    )


def _duplicates(rows: Iterable[dict[str, object]]) -> list[int]:
    values = [_int_or_none(row.get("turn_id")) for row in rows]
    return sorted(value for value, count in Counter(value for value in values if value is not None).items() if count > 1)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _read_lines_safe(path: Path) -> tuple[list[dict[str, object]], str | None]:
    try:
        return list(read_json_lines(path)), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def _source_changes(before: dict[str, object], after: dict[str, object]) -> list[dict[str, object]]:
    return [{"path": key, "before": before.get(key), "after": after.get(key)} for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]


def _csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_name(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._")
    if not result or result in {".", ".."}:
        raise ValueError(f"invalid path component: {value!r}")
    return result


def _new_run_id() -> str:
    return datetime.now().astimezone().strftime("all_sessions_%Y%m%dT%H%M%S_") + uuid4().hex[:12]


__all__ = ["SessionReplayAuditRun", "SessionReplayAuditService", "TruthAuditReference", "compare_truth_visual_fields", "resolve_truth_audit_reference", "summarize_visual_events"]
