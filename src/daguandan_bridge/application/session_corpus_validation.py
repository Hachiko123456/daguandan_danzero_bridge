"""AVI-first validation orchestration for recorded Guandan sessions.

The recording is the factual input to a scan.  Timeline, diagnostics and
TruthLog files are never supplied to the scanner.  A TruthLog is loaded only
after an independent video scan has finished, for semantic comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping
from uuid import uuid4

from ..live.truth_log import TruthLog, load_truth_log
from .session_dataset_catalog import SessionDatasetCatalogBuilder
from .session_locator import SessionDescriptor, SessionLocator, SessionSelection
from .truth_log_from_scan import DRAFT_PROVENANCE_SOURCE, build_truth_log_from_scan


ValidationMode = Literal["smoke", "single", "corpus", "qualify"]

# The user confirmed this particular draft manually.  It is an in-memory
# qualification override; the source TruthLog is never silently promoted.
USER_TRUSTED_SESSION_IDS = frozenset({"game_20260816_125402_1687ea"})


@dataclass(frozen=True)
class SessionCorpusValidationRun:
    mode: ValidationMode
    output_directory: Path
    summary_path: Path
    selection: tuple[SessionDescriptor, ...]
    structural_passed: int
    structural_failed: int
    scan_requested: bool
    scan_failed: int
    semantic_eligible: int
    semantic_failed: int = 0
    semantic_status: str = "not_eligible"
    scan_error: str | None = None

    @property
    def replay_requested(self) -> bool:
        """Compatibility alias: it means AVI scan, never legacy replay."""

        return self.scan_requested

    @property
    def passed(self) -> bool:
        if not self.scan_requested:
            return bool(self.selection) and self.structural_failed == 0
        return (
            bool(self.selection)
            and self.structural_failed == 0
            and self.scan_failed == 0
            and self.scan_error is None
            and self.semantic_failed == 0
            and self.semantic_status not in {"failed", "not_run"}
        )


class SessionCorpusValidationService:
    """Run one or more sessions through the pure video scanner.

    The scanner factory is injected in tests.  The production scanner is loaded
    lazily so merely opening the workbench cannot use the old replay pipeline.
    """

    schema = "guandan.session-corpus-validation/2"

    def __init__(
        self,
        *,
        locator: SessionLocator | None = None,
        catalog_builder: SessionDatasetCatalogBuilder | None = None,
        scan_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.locator = locator or SessionLocator()
        self.catalog_builder = catalog_builder or SessionDatasetCatalogBuilder()
        self.scan_factory = scan_factory

    def validate(
        self,
        target: Path | str,
        *,
        mode: ValidationMode = "corpus",
        output: Path | str,
        run_id: str | None = None,
        profile_root: Path | str | None = None,
        trusted_sessions: Iterable[Path | str] = (),
        scan_run_id: str | Iterable[str] | None = None,
        execute_replay: bool = True,
        command: Iterable[str] | None = None,
    ) -> SessionCorpusValidationRun:
        """Scan AVI files read-only and only then compare TruthLog actions.

        ``scan_run_id`` and ``command`` remain accepted for CLI compatibility,
        but cannot reach the scanner: old staged results must not seed a new
        scan.
        """

        del scan_run_id, command
        if mode not in {"smoke", "single", "corpus", "qualify"}:
            raise ValueError(f"unsupported validation mode: {mode}")
        # SessionLocator.discover validates TruthLogs as part of its normal
        # workbench descriptor. A scan must not do that before AVI decoding,
        # so scanning uses metadata-only descriptors and defers TruthLog reads
        # until the scanner has returned.
        selection = (
            self._discover_scan_safe(target)
            if execute_replay
            else self.locator.discover(target)
        )
        requested_trusted = self._trusted_ids(trusted_sessions, selection.sessions)
        selected = self._select(
            selection,
            mode,
            priority_ids=requested_trusted | USER_TRUSTED_SESSION_IDS,
        )
        if mode == "single" and len(selected) != 1:
            raise ValueError("single mode requires a directory containing exactly one session")
        if not selected:
            raise ValueError(f"no recorded sessions found below: {selection.target}")

        out_root = Path(output).expanduser().resolve()
        self._reject_output(out_root, selection)
        run_dir = out_root / self._safe_name(run_id or self._new_run_id(mode))
        self._reject_output(run_dir, selection)
        run_dir.mkdir(parents=True, exist_ok=False)

        trusted = requested_trusted | (
            USER_TRUSTED_SESSION_IDS & {item.session_id for item in selected}
        )
        structural_rows = [self._structural_row(item) for item in selected]
        structural_passed = sum(bool(row["passed"]) for row in structural_rows)
        structural_failed = len(structural_rows) - structural_passed

        scan_requested = bool(execute_replay and any(item.has_video for item in selected))
        scan_error: str | None = None
        scan_rows: dict[str, dict[str, object]] = {}
        if scan_requested:
            try:
                # Keep the narrow method name as a compatibility seam for the
                # previous test doubles.  Its production implementation below
                # only calls the pure AVI scanner; it never invokes old replay.
                scan_result = self._run_audit(
                    selected,
                    run_dir / "scan",
                    profile_root=profile_root,
                )
                scan_rows = self._coerce_scan_rows(scan_result, selected, run_dir / "scan")
            except Exception as exc:
                scan_error = f"{type(exc).__name__}: {exc}"
        for item, structural in zip(selected, structural_rows, strict=True):
            scan_rows.setdefault(
                item.session_id,
                self._not_run_scan_row(
                    item,
                    reason=(
                        scan_error
                        or ("video_missing" if not structural["passed"] else "scan_not_requested")
                    ),
                ),
            )

        if scan_requested:
            selected = self._with_post_scan_truth_status(selected)
            for item in selected:
                row = scan_rows.get(item.session_id)
                if row is not None:
                    row["truth_status"] = item.truth_status
                    row["truth_missing"] = item.truth_status == "missing"
        # Catalog generation reads TruthLogs for evidence metadata, so it must
        # also remain on the post-scan side of the input boundary.
        catalog = self._write_catalog(selection, run_dir / "catalog")

        # The scanner has finished before a baseline TruthLog is opened.  This
        # draft-only post-processing consumes only its canonical action trace;
        # it never passes old TruthLog data into the AVI scanner.
        self._write_truth_drafts(selected, scan_rows, run_dir / "scan")

        semantic_rows, semantic_failed, semantic_status = self._semantic_results(
            selected,
            trusted,
            scan_rows,
        )
        semantic_eligible = sum(bool(row["eligible"]) for row in semantic_rows.values())
        scan_failed = sum(row.get("status") == "failed" for row in scan_rows.values())
        scan_manifest = self._write_scan_manifest(
            selected,
            run_dir,
            scan_rows,
            scan_requested=scan_requested,
        )

        summary = {
            "schema": self.schema,
            "created_at": datetime.now().astimezone().isoformat(),
            "mode": mode,
            "target": str(selection.target),
            "output_directory": str(run_dir),
            "catalog": catalog,
            "scan_manifest": str(scan_manifest),
            "scan_policy": {
                "video_is_only_scan_factual_input": True,
                "optional_frame_index": True,
                "forbidden_scan_inputs": [
                    "timeline.jsonl",
                    "truth_log.json",
                    "recognition_trace.jsonl",
                    "observations.jsonl",
                    "advice.jsonl",
                    "decisions.jsonl",
                ],
                "truth_is_compared_after_scan_only": True,
            },
            "truth_policy": {
                "verified_only_for_release": True,
                "trusted_session_ids": sorted(trusted),
                "draft_is_not_gold_unless_explicitly_trusted_for_this_run": True,
            },
            "counts": {
                "discovered": len(selection.sessions),
                "selected": len(selected),
                "structural_passed": structural_passed,
                "structural_failed": structural_failed,
                "semantic_eligible": semantic_eligible,
                "truth_missing": sum(item.truth_status == "missing" for item in selected),
                "truth_draft": sum(item.truth_status == "draft" for item in selected),
                "truth_verified": sum(item.truth_status == "verified" for item in selected),
                "truth_invalid": sum(item.truth_status == "invalid" for item in selected),
                "semantic_failed": semantic_failed,
            },
            "sessions": [
                {
                    **item.to_dict(),
                    "selected": True,
                    "structural": structural,
                    "scan": scan_rows[item.session_id],
                    "semantic": semantic_rows[item.session_id],
                    "semantic_status": semantic_rows[item.session_id]["status"],
                }
                for item, structural in zip(selected, structural_rows, strict=True)
            ],
            "scan": {
                "status": "completed" if scan_requested and not scan_error else ("error" if scan_error else "not_run"),
                "requested": scan_requested,
                "error": scan_error,
            },
            # Retained as a read-only compatibility view for existing report
            # consumers.  It never denotes legacy replay anymore.
            "replay_audit": {
                "status": "completed" if scan_requested and not scan_error else ("error" if scan_error else "not_run"),
                "execution_ok": scan_requested and not scan_error,
                "error": scan_error,
            },
            "status": self._status(
                structural_failed,
                scan_failed,
                scan_error,
                semantic_failed,
                semantic_status,
                scan_requested=scan_requested,
            ),
            "semantic_status": semantic_status,
            "semantic_failed": semantic_failed,
        }
        summary_path = run_dir / "corpus_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return SessionCorpusValidationRun(
            mode=mode,
            output_directory=run_dir,
            summary_path=summary_path,
            selection=tuple(selected),
            structural_passed=structural_passed,
            structural_failed=structural_failed,
            scan_requested=scan_requested,
            scan_failed=scan_failed,
            semantic_eligible=semantic_eligible,
            semantic_failed=semantic_failed,
            semantic_status=semantic_status,
            scan_error=scan_error,
        )

    @staticmethod
    def _discover_scan_safe(target: Path | str) -> SessionSelection:
        """Discover scan metadata without opening a source TruthLog."""

        from .session_workbench import discover_session_paths

        root = Path(target).expanduser().resolve()
        sessions: list[SessionDescriptor] = []
        for path in discover_session_paths(root):
            manifest_path = path / "manifest.json"
            manifest = _read_json_object(manifest_path) or {}
            frame_index_path = path / "video" / "frame_index.jsonl"
            raw_frame_count = manifest.get("frame_count")
            frame_count = (
                raw_frame_count
                if isinstance(raw_frame_count, int) and raw_frame_count >= 0
                else _count_nonempty_lines(frame_index_path)
            )
            timeline_path = path / "timeline.jsonl"
            sessions.append(
                SessionDescriptor(
                    root=path,
                    session_id=str(manifest.get("session_id") or path.name),
                    manifest_path=manifest_path,
                    video_path=path / "video" / "game.avi",
                    frame_index_path=frame_index_path,
                    timeline_path=timeline_path,
                    truth_log_path=path / "truth_log.json",
                    manifest_readable=bool(manifest),
                    has_video=(path / "video" / "game.avi").is_file(),
                    has_frame_index=frame_index_path.is_file(),
                    has_timeline=timeline_path.is_file(),
                    # Presence is deliberately not parsed until post-scan.
                    truth_status="uninspected",
                    truth_error="",
                    frame_count=frame_count,
                    timeline_event_count=_count_nonempty_lines(timeline_path),
                )
            )
        return SessionSelection(root, tuple(sessions))

    @staticmethod
    def _with_post_scan_truth_status(
        selected: tuple[SessionDescriptor, ...],
    ) -> tuple[SessionDescriptor, ...]:
        """Populate descriptor label state only after AVI scanning has ended."""

        resolved: list[SessionDescriptor] = []
        for item in selected:
            if not item.truth_log_path.is_file():
                resolved.append(replace(item, truth_status="missing", truth_error=""))
                continue
            try:
                truth = load_truth_log(item.truth_log_path, session_id=item.session_id)
            except Exception as exc:
                resolved.append(
                    replace(
                        item,
                        truth_status="invalid",
                        truth_error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                resolved.append(
                    replace(
                        item,
                        truth_status=(
                            "verified" if truth.label_status == "verified" else "draft"
                        ),
                        truth_error="",
                    )
                )
        return tuple(resolved)

    def _run_scans(
        self,
        selected: tuple[SessionDescriptor, ...],
        output: Path,
        *,
        profile_root: Path | str | None,
    ) -> dict[str, dict[str, object]]:
        scanner = self._make_scanner(profile_root)
        rows: dict[str, dict[str, object]] = {}
        for item in selected:
            session_output = output / self._safe_name(item.session_id)
            session_output.mkdir(parents=True, exist_ok=False)
            if not item.has_video:
                rows[item.session_id] = self._not_run_scan_row(item, reason="video_missing")
                continue
            try:
                raw_result = self._scan_video(scanner, item, session_output, profile_root)
                rows[item.session_id] = self._normalise_scan_result(item, raw_result, session_output)
            except Exception as exc:
                rows[item.session_id] = {
                    **self._not_run_scan_row(item, reason=f"{type(exc).__name__}: {exc}"),
                    "status": "failed",
                }
        return rows

    def _run_audit(
        self,
        selected: tuple[SessionDescriptor, ...],
        output: Path,
        *,
        profile_root: Path | str | None,
    ) -> dict[str, dict[str, object]]:
        """Compatibility seam whose implementation is the pure video scan."""

        return self._run_scans(selected, output, profile_root=profile_root)

    @staticmethod
    def _coerce_scan_rows(
        value: Any,
        selected: tuple[SessionDescriptor, ...],
        output: Path,
    ) -> dict[str, dict[str, object]]:
        """Accept old test doubles without making legacy replay production code."""

        if isinstance(value, Mapping):
            return {str(key): dict(row) for key, row in value.items() if isinstance(row, Mapping)}
        legacy_rows = getattr(value, "sessions", None)
        if not isinstance(legacy_rows, (list, tuple)):
            raise TypeError("video scanner must return per-session mappings")
        by_id = {item.session_id: item for item in selected}
        result: dict[str, dict[str, object]] = {}
        for raw in legacy_rows:
            if not isinstance(raw, Mapping):
                continue
            session_id = str(raw.get("session_id", ""))
            item = by_id.get(session_id)
            if item is None:
                continue
            visual = raw.get("visual") if isinstance(raw.get("visual"), Mapping) else {}
            actions = visual.get("actions") if isinstance(visual.get("actions"), Mapping) else {}
            result[session_id] = {
                **SessionCorpusValidationService._not_run_scan_row(item, reason="compatibility_scan_result"),
                "status": "completed" if raw.get("execution_status") == "completed" else "failed",
                "frames_processed": _as_int(raw.get("frames_processed"), 0),
                "indexed_frames": _as_int(raw.get("indexed_frames"), item.frame_count or 0),
                "action_event_count": _as_int(actions.get("count"), 0),
                "action_candidate_count": _as_int(actions.get("count"), 0),
                "actions": _as_list(actions.get("rows")),
                "semantic_override": raw.get("truth_quality"),
                "evidence_paths": [str(path) for path in raw.get("evidence_paths", {}).values()] if isinstance(raw.get("evidence_paths"), Mapping) else [],
            }
        return result

    def _make_scanner(self, profile_root: Path | str | None) -> Any:
        if self.scan_factory is not None:
            try:
                return self.scan_factory(profile_root=profile_root)
            except TypeError:
                return self.scan_factory()
        # This is the pure core.  Do not substitute replay_video_through_live_pipeline.
        from ..annotation_service import AnnotationService
        from ..recognition_service import ScreenshotRecognitionService
        from ..template_service import TemplateService
        from .session_workbench import default_profile_context
        from .video_scan import VideoActionScanner

        profile = (
            Path(profile_root).expanduser().resolve()
            if profile_root is not None
            else default_profile_context().profile_path
        )
        if not profile.is_dir():
            raise FileNotFoundError(f"找不到识别 profile：{profile}")
        recognition = ScreenshotRecognitionService(
            AnnotationService(profile.parent, profile.name),
            TemplateService(profile.parent, profile.name),
        )
        return VideoActionScanner(recognition)

    @staticmethod
    def _scan_video(
        scanner: Any,
        item: SessionDescriptor,
        output: Path,
        profile_root: Path | str | None,
    ) -> Any:
        """Call the narrow contract without a session path or old log path."""

        from .video_scan import VideoScanRequest

        request = VideoScanRequest(
            video_path=item.video_path,
            frame_index_path=item.frame_index_path if item.has_frame_index else None,
            output_directory=output,
            session_id=item.session_id,
        )
        return scanner.scan(request)

    @staticmethod
    def _normalise_scan_result(
        item: SessionDescriptor,
        raw: Any,
        output: Path,
    ) -> dict[str, object]:
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        if not isinstance(raw, Mapping):
            raise TypeError("pure video scanner must return a mapping or an object with to_dict()")
        value = dict(raw)
        status = str(value.get("status", "completed"))
        if status not in {"completed", "needs_review", "failed"}:
            status = "completed"
        trace_path = _path_value(value.get("action_trace_path"), output)
        observations_path = _path_value(
            value.get("frame_observations_path") or value.get("observations_path"),
            output,
        )
        summary_path = _path_value(value.get("summary_path"), output)
        turn_slots_path = _path_value(value.get("turn_slots_path"), output)
        actions = _as_list(value.get("actions") or value.get("action_trace"))
        if not actions and trace_path is not None and trace_path.is_file():
            actions = _read_json_lines(trace_path)
        opening = value.get("opening") or value.get("initial_state")
        opening_path = _path_value(value.get("opening_path"), output)
        if not isinstance(opening, Mapping) and opening_path is not None and opening_path.is_file():
            opening = _read_json_object(opening_path)
        return {
            "session_id": item.session_id,
            "source": str(item.root),
            "status": status,
            "reason": value.get("reason"),
            "frames_processed": _as_int(value.get("frames_processed", value.get("decoded_frames")), 0),
            "indexed_frames": _as_int(value.get("indexed_frames"), item.frame_count or 0),
            "action_candidate_count": _as_int(value.get("action_candidate_count", value.get("action_count")), len(actions)),
            "action_event_count": _as_int(value.get("action_event_count", value.get("action_count")), len(actions)),
            "opening": opening if isinstance(opening, Mapping) else None,
            "anomalies": _as_list(value.get("anomalies") or value.get("needs_review")),
            "evidence_paths": [str(path) for path in (observations_path, trace_path, turn_slots_path, summary_path) if path is not None],
            "frame_observations_path": str(observations_path) if observations_path else None,
            "turn_slots_path": str(turn_slots_path) if turn_slots_path else None,
            "action_trace_path": str(trace_path) if trace_path else None,
            "scan_summary_path": str(summary_path) if summary_path else None,
            "actions": actions,
            "truth_status": item.truth_status,
            "truth_missing": item.truth_status == "missing",
        }

    @staticmethod
    def _not_run_scan_row(item: SessionDescriptor, *, reason: str) -> dict[str, object]:
        return {
            "session_id": item.session_id,
            "source": str(item.root),
            "status": "not_run",
            "reason": reason,
            "frames_processed": 0,
            "indexed_frames": item.frame_count or 0,
            "action_candidate_count": 0,
            "action_event_count": 0,
            "opening": None,
            "anomalies": [],
            "evidence_paths": [],
            "frame_observations_path": None,
            "turn_slots_path": None,
            "action_trace_path": None,
            "scan_summary_path": None,
            "actions": [],
            "truth_status": item.truth_status,
            "truth_missing": item.truth_status == "missing",
        }

    def _write_truth_drafts(
        self,
        selected: tuple[SessionDescriptor, ...],
        scan_rows: Mapping[str, dict[str, object]],
        scan_output: Path,
    ) -> None:
        """Publish scan-only TruthLog drafts into the external run directory.

        The canonical ``action_trace.jsonl`` is produced by the scan first.
        Only afterwards is the session's baseline TruthLog read, and only for
        its initial state and source metadata through ``build_truth_log_from_scan``.
        Failures are represented in the external run report and never change
        source session files or promote a label to verified.
        """

        for item in selected:
            row = scan_rows.get(item.session_id)
            if row is None:
                continue
            artifact_dir = scan_output / self._safe_name(item.session_id)
            status = str(row.get("status", ""))
            trace_value = row.get("action_trace_path")
            trace_path = Path(str(trace_value)) if trace_value else None
            draft_info: dict[str, object] = {
                "status": "not_generated",
                "truth_log_path": None,
                "review_path": None,
                "review_count": 0,
                "reason": None,
            }
            row["truth_draft"] = draft_info
            if status not in {"completed", "needs_review"}:
                draft_info["reason"] = "scan_not_completed"
                continue
            if trace_path is None or not trace_path.is_file():
                draft_info["reason"] = "canonical_action_trace_missing"
                continue
            if not item.truth_log_path.is_file():
                draft_info["reason"] = "baseline_truth_log_missing"
                continue

            try:
                # This read is deliberately after the scan and action-trace
                # existence checks; it must never become a scanner input.
                baseline = load_truth_log(
                    item.truth_log_path,
                    session_id=item.session_id,
                )
                canonical_actions = _read_json_lines(trace_path)
                generated = build_truth_log_from_scan(baseline, canonical_actions)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                draft_path = artifact_dir / "truth_log.draft.json"
                review_path = artifact_dir / "truth_log_review.json"
                draft_path.write_text(
                    json.dumps(
                        generated.truth_log.to_dict(),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                review_path.write_text(
                    json.dumps(
                        {
                            "schema": "guandan.truth-log-review/1",
                            "source_session_id": generated.truth_log.source_session_id,
                            "label_status": "draft",
                            "provenance": {"source": DRAFT_PROVENANCE_SOURCE},
                            "canonical_action_trace_path": str(trace_path),
                            "truth_log_draft_path": str(draft_path),
                            "review_items": list(generated.review_items),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:
                draft_info["status"] = "failed"
                draft_info["reason"] = f"{type(exc).__name__}: {exc}"
                continue

            draft_info.update(
                {
                    "status": "generated",
                    "truth_log_path": str(draft_path),
                    "review_path": str(review_path),
                    "review_count": len(generated.review_items),
                    "reason": None,
                }
            )
    def _semantic_results(
        self,
        selected: tuple[SessionDescriptor, ...],
        trusted: set[str],
        scan_rows: Mapping[str, Mapping[str, object]],
    ) -> tuple[dict[str, dict[str, object]], int, str]:
        rows: dict[str, dict[str, object]] = {}
        failed = 0
        eligible_count = 0
        not_run = 0
        for item in selected:
            eligible = item.truth_status == "verified" or item.session_id in trusted
            scan = scan_rows[item.session_id]
            if not eligible:
                rows[item.session_id] = {
                    "status": "not_eligible",
                    "eligible": False,
                    "truth_status": item.truth_status,
                    "reason": "TruthLog is missing or draft and was not explicitly trusted",
                    "first_difference": None,
                }
                continue
            eligible_count += 1
            if scan.get("status") not in {"completed", "needs_review"}:
                if scan.get("status") == "failed":
                    failed += 1
                not_run += 1
                rows[item.session_id] = {
                    "status": "not_run",
                    "eligible": True,
                    "truth_status": item.truth_status,
                    "reason": "pure AVI scan did not complete",
                    "first_difference": None,
                }
                continue
            if scan.get("semantic_override") == "failed":
                failed += 1
                rows[item.session_id] = {
                    "status": "failed",
                    "eligible": True,
                    "truth_status": item.truth_status,
                    "reason": "scanner reported semantic comparison failure",
                    "first_difference": None,
                }
                continue
            try:
                truth = load_truth_log(item.truth_log_path, session_id=item.session_id)
                comparison = self._compare_scan_to_truth(scan, truth)
            except Exception as exc:
                failed += 1
                rows[item.session_id] = {
                    "status": "failed",
                    "eligible": True,
                    "truth_status": item.truth_status,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "first_difference": None,
                }
                continue
            passed = bool(comparison["passed"])
            if not passed:
                failed += 1
            rows[item.session_id] = {
                "status": "passed" if passed else "failed",
                "eligible": True,
                "truth_status": item.truth_status,
                "reason": None if passed else "scan action trace differs from TruthLog",
                "first_difference": comparison["first_difference"],
                "comparison": comparison,
            }
        overall = "failed" if failed else "not_eligible" if not eligible_count else "not_run" if not_run else "passed"
        return rows, failed, overall

    @staticmethod
    def _compare_scan_to_truth(scan: Mapping[str, object], truth: TruthLog) -> dict[str, object]:
        actual = [_action_key(row) for row in _as_list(scan.get("actions"))]
        expected = [
            {
                "actor": turn.actor,
                "is_pass": bool(turn.is_pass),
                "cards": list(turn.cards),
                "turn_id": turn.index,
                "frame_index": turn.frame_index,
            }
            for turn in truth.turns
        ]
        first: dict[str, object] | None = None
        for index, (wanted, got) in enumerate(zip(expected, actual, strict=False), start=1):
            if _semantic_action(wanted) != _semantic_action(got):
                first = {"action_index": index, "expected": wanted, "actual": got}
                break
        if first is None and len(expected) != len(actual):
            index = min(len(expected), len(actual)) + 1
            first = {
                "action_index": index,
                "expected": expected[index - 1] if len(expected) >= index else None,
                "actual": actual[index - 1] if len(actual) >= index else None,
            }
        return {
            "passed": first is None,
            "expected_action_count": len(expected),
            "actual_action_count": len(actual),
            "first_difference": first,
        }

    def _write_catalog(self, selection: SessionSelection, output: Path) -> dict[str, object]:
        root = self._catalog_root(selection)
        if root is None:
            output.mkdir(parents=True, exist_ok=True)
            path = output / "catalog_unavailable.json"
            value = {"status": "not_available", "reason": "sessions are not direct children of a common root"}
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return value
        result = self.catalog_builder.build(root, output)
        return {
            "status": "available",
            "root": str(root),
            "catalog": str(result.catalog_path),
            "cases": str(result.cases_path),
            "coverage_gaps": str(result.gaps_path),
        }

    @staticmethod
    def _write_scan_manifest(
        selected: tuple[SessionDescriptor, ...],
        run_dir: Path,
        rows: Mapping[str, Mapping[str, object]],
        *,
        scan_requested: bool,
    ) -> Path:
        sessions = [dict(rows[item.session_id]) for item in selected]
        payload = {
            "schema": "guandan.video-scan-manifest/1",
            "status": "completed" if scan_requested else "not_run",
            "source_policy": {"video_only": True, "timeline_or_truth_used_as_scan_input": False},
            "sessions": sessions,
            "counts": {
                "selected": len(sessions),
                "completed": sum(row["status"] == "completed" for row in sessions),
                "needs_review": sum(row["status"] == "needs_review" for row in sessions),
                "failed": sum(row["status"] == "failed" for row in sessions),
                "not_run": sum(row["status"] == "not_run" for row in sessions),
                "truth_missing": sum(bool(row["truth_missing"]) for row in sessions),
                "action_events": sum(_as_int(row["action_event_count"], 0) for row in sessions),
            },
        }
        path = run_dir / "scan_manifest.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for row in sessions:
            summary = run_dir / "scan" / str(row["session_id"]) / "scan_summary.json"
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def _catalog_root(selection: SessionSelection) -> Path | None:
        if selection.is_single_session:
            return selection.target.parent
        if all(item.root.parent == selection.target for item in selection.sessions):
            return selection.target
        nested = selection.target / "sessions"
        return nested if nested.is_dir() and all(item.root.parent == nested for item in selection.sessions) else None

    @staticmethod
    def _select(
        selection: SessionSelection,
        mode: ValidationMode,
        *,
        priority_ids: Iterable[str] = (),
    ) -> tuple[SessionDescriptor, ...]:
        if mode in {"corpus", "qualify", "single"}:
            return selection.sessions
        priority = set(priority_ids)
        chosen: list[SessionDescriptor] = [item for item in selection.sessions if item.session_id in priority]
        for predicate in (
            lambda item: item.truth_status == "verified",
            lambda item: item.truth_status == "draft",
            lambda item: item.root.name.casefold().startswith("manual_"),
            lambda item: item.has_video and not item.has_frame_index,
            lambda item: item.has_video,
        ):
            candidate = next((item for item in selection.sessions if predicate(item) and item not in chosen), None)
            if candidate is not None:
                chosen.append(candidate)
            if len(chosen) >= 8:
                break
        return tuple(chosen or selection.sessions[:1])

    @staticmethod
    def _structural_row(item: SessionDescriptor) -> dict[str, object]:
        # A complete AVI is sufficient.  Manifest and frame index are optional
        # capture metadata and never block a pure-video scan.
        errors = ["video_missing"] if not item.has_video else []
        warnings: list[str] = []
        if not item.has_frame_index:
            warnings.append("frame_index_missing_timestamps_will_be_derived_from_avi")
        if not item.manifest_readable:
            warnings.append("manifest_missing_session_id_uses_directory_name")
        return {"passed": not errors, "errors": errors, "warnings": warnings}

    @staticmethod
    def _trusted_ids(values: Iterable[Path | str], selected: Iterable[SessionDescriptor]) -> set[str]:
        by_path = {item.root: item.session_id for item in selected}
        by_id = {item.session_id for item in selected}
        result: set[str] = set()
        for value in values:
            raw = str(value)
            path = Path(value).expanduser().resolve()
            if path in by_path:
                result.add(by_path[path])
            elif raw in by_id:
                result.add(raw)
        return result

    @staticmethod
    def _status(
        structural_failed: int,
        scan_failed: int,
        scan_error: str | None,
        semantic_failed: int,
        semantic_status: str,
        *,
        scan_requested: bool,
    ) -> str:
        if not scan_requested:
            return "passed" if not structural_failed else "failed"
        if structural_failed or scan_failed or scan_error or semantic_failed or semantic_status in {"failed", "not_run"}:
            return "failed"
        return "passed"

    @staticmethod
    def _reject_output(output: Path, selection: SessionSelection) -> None:
        for source in (selection.target, *(item.root for item in selection.sessions)):
            try:
                output.relative_to(source)
            except ValueError:
                continue
            raise ValueError(f"validation output must be outside source session data: {output}")

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._") or "run"

    @staticmethod
    def _new_run_id(mode: str) -> str:
        return f"{mode}_{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"



def _count_nonempty_lines(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (OSError, UnicodeError):
        return None

def _as_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        return []
    return [dict(row) for row in value]


def _as_int(value: object, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _path_value(value: object, output: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else output / path


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if isinstance(raw, Mapping):
                    result.append(dict(raw))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    return result


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(raw) if isinstance(raw, Mapping) else None


def _action_key(value: Mapping[str, object]) -> dict[str, object]:
    cards = value.get("cards")
    return {
        "actor": value.get("actor"),
        "is_pass": bool(value.get("is_pass", value.get("pass", False))),
        "cards": list(cards) if isinstance(cards, (list, tuple)) else [],
        "turn_id": value.get("turn_id", value.get("turn_index", value.get("index"))),
        "frame_index": value.get("frame_index", value.get("frame_start")),
    }


def _semantic_action(value: Mapping[str, object]) -> tuple[object, bool, tuple[str, ...]]:
    return (
        value.get("actor"),
        bool(value.get("is_pass")),
        tuple(sorted(str(card) for card in value.get("cards", ()))),
    )


__all__ = [
    "SessionCorpusValidationRun",
    "SessionCorpusValidationService",
    "USER_TRUSTED_SESSION_IDS",
    "ValidationMode",
]
