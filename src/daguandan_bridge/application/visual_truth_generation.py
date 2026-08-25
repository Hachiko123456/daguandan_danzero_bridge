"""Safe, headless generation of staged visual TruthLog candidates.

This service deliberately separates a reproducible visual scan from the one
irreversible-looking step: publishing ``session/truth_log.json``.  Every run
first writes an isolated draft under ``derived/``; publication is opt-in and
uses a no-replace atomic create.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal
from uuid import uuid4

from ..annotation_service import AnnotationService
from ..domain.truth import LabelProvenance
from ..live.models import LiveEvent
from ..live.reducer import LiveReducer
from ..live.replay import VisualPipelineReplayResult, replay_video_through_live_pipeline
from ..live.session_store import read_json_lines
from ..live.truth_log import TruthInitialState, TruthLog, load_truth_log, save_truth_log
from ..recognition_service import ScreenshotRecognitionService
from ..storage import atomic_write_json
from ..template_service import TemplateService
from .replay_turn_draft import (
    ReplayTurnDraftAssembler,
    truth_scan_draft_paths,
    validate_turn_actor_chain,
    write_truth_scan_draft_sidecars,
)


VISUAL_SCAN_PROVENANCE = "visual_scan_batch"
_ACTION_TYPES = frozenset({"player_played", "player_passed", "manual_confirmed_event"})
_CORRECTION_TYPES = frozenset({"event_correction", "suit_corrected"})
_SEATS = frozenset({"self", "right", "opposite", "left"})
_ResultStatus = Literal[
    "staged",
    "staged_existing",
    "published",
    "blocked",
    "error",
]


@dataclass(frozen=True)
class VisualTruthGenerationSession:
    session: Path
    status: _ResultStatus
    message: str
    stage_directory: Path | None = None
    receipt_path: Path | None = None
    frames_processed: int = 0
    indexed_frames: int = 0
    action_count: int = 0
    gates: tuple[dict[str, object], ...] = ()
    published: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "session": str(self.session),
            "status": self.status,
            "message": self.message,
            "stage_directory": str(self.stage_directory) if self.stage_directory else None,
            "receipt_path": str(self.receipt_path) if self.receipt_path else None,
            "frames_processed": self.frames_processed,
            "indexed_frames": self.indexed_frames,
            "action_count": self.action_count,
            "gates": list(self.gates),
            "published": self.published,
        }


@dataclass(frozen=True)
class VisualTruthGenerationRun:
    run_id: str
    publish_requested: bool
    sessions: tuple[VisualTruthGenerationSession, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "guandan.visual-truth-generation/1",
            "run_id": self.run_id,
            "publish_requested": self.publish_requested,
            "created_at": datetime.now().astimezone().isoformat(),
            "sessions": [item.to_dict() for item in self.sessions],
            "summary": {
                "discovered": len(self.sessions),
                "staged": sum(item.status == "staged" for item in self.sessions),
                "staged_existing": sum(
                    item.status == "staged_existing" for item in self.sessions
                ),
                "published": sum(item.published for item in self.sessions),
                "blocked": sum(item.status == "blocked" for item in self.sessions),
                "errors": sum(item.status == "error" for item in self.sessions),
            },
        }


@dataclass(frozen=True)
class _ExistingStagedTruth:
    truth_log: TruthLog
    sha256: str
    relative_path: str


class VisualTruthGenerationService:
    """Generate isolated visual TruthLog drafts for explicitly supplied roots."""

    def __init__(
        self,
        *,
        recognition_factory: Callable[[Path], Any] | None = None,
        replay: Callable[..., VisualPipelineReplayResult] = replay_video_through_live_pipeline,
    ) -> None:
        self._recognition_factory = recognition_factory or _recognition_for_session
        self._replay = replay

    @staticmethod
    def discover_sessions(session_roots: Iterable[Path | str]) -> tuple[Path, ...]:
        """Discover session directories at runtime without assuming session IDs."""

        discovered: dict[Path, Path] = {}
        for raw_root in session_roots:
            root = Path(raw_root).resolve()
            candidates = (root / "sessions",) if (root / "sessions").is_dir() else (root,)
            for sessions_root in candidates:
                if not sessions_root.is_dir():
                    continue
                for session in sessions_root.iterdir():
                    if not session.is_dir() or not (session / "manifest.json").is_file():
                        continue
                    resolved = session.resolve()
                    discovered.setdefault(resolved, resolved)
        return tuple(sorted(discovered.values(), key=lambda item: str(item)))

    def generate(
        self,
        session_roots: Iterable[Path | str],
        *,
        publish: bool = False,
        only_missing: bool = False,
        run_id: str | None = None,
        trusted_staged_references: Iterable[Path | str] = (),
    ) -> VisualTruthGenerationRun:
        """Stage discovered scans and publish only safe missing TruthLogs.

        ``only_missing`` is the resumable batch mode: it excludes sessions
        that already have a canonical log before any video or template work is
        started.  It never treats a staged draft as canonical.
        """

        normalized_run_id = run_id or _new_run_id()
        trusted_references = frozenset(
            Path(path).resolve() for path in trusted_staged_references
        )
        sessions = self.discover_sessions(session_roots)
        if only_missing:
            sessions = tuple(
                session
                for session in sessions
                if not (session / "truth_log.json").is_file()
            )
        return VisualTruthGenerationRun(
            run_id=normalized_run_id,
            publish_requested=publish,
            sessions=tuple(
                self._generate_session(
                    session,
                    normalized_run_id,
                    publish=publish,
                    trusted_staged_references=trusted_references,
                )
                for session in sessions
            ),
        )

    def _generate_session(
        self,
        session: Path,
        run_id: str,
        *,
        publish: bool,
        trusted_staged_references: frozenset[Path],
    ) -> VisualTruthGenerationSession:
        paths = truth_scan_draft_paths(session, run_id)
        if paths.directory.exists():
            return VisualTruthGenerationSession(
                session,
                "blocked",
                "该 run_id 的扫描草稿目录已存在，拒绝覆盖",
                stage_directory=paths.directory,
            )

        try:
            timeline = tuple(read_json_lines(session / "timeline.jsonl"))
            baseline = _baseline_from_timeline(session, timeline)
            canonical, canonical_sha256 = _existing_truth(session)
            staged_reference = _existing_staged_truth(session)
            indexed_frames = len(
                tuple(read_json_lines(session / baseline.frame_index_path))
            )
        except Exception as exc:
            return VisualTruthGenerationSession(
                session,
                "blocked",
                f"扫描基线不可用：{type(exc).__name__}: {exc}",
            )

        assembler = ReplayTurnDraftAssembler(baseline)
        stream_rejections: list[str] = []
        expected_source_turn_id = 1
        paths.directory.mkdir(parents=True, exist_ok=False)

        def collect_turn(raw: dict[str, object]) -> None:
            nonlocal expected_source_turn_id
            kind = str(raw.get("kind", "action"))
            if kind not in _CORRECTION_TYPES:
                try:
                    source_turn_id = int(raw.get("turn_id", 0) or 0)
                except (TypeError, ValueError):
                    stream_rejections.append("动作流包含无效 turn_id")
                    return
                if source_turn_id != expected_source_turn_id:
                    stream_rejections.append(
                        f"来源 turn_id 不连续：应为 {expected_source_turn_id}，实际为 {source_turn_id}"
                    )
                    return
            appended = assembler.append(dict(raw))
            if not appended.accepted:
                stream_rejections.append(appended.reason or appended.status)
                return
            if kind not in _CORRECTION_TYPES:
                expected_source_turn_id += 1

        try:
            recognition = self._recognition_factory(session)
            # The visual replay must never create artifacts in a source
            # session.  Its complete per-frame evidence is retained only in
            # this immutable stage directory for later review.
            replay_result = self._replay(
                session,
                recognition,
                truth_log=baseline,
                use_live_pipeline=True,
                recognition_strategy="two_valid_streak",
                output_root=paths.directory / "replay",
                on_turn=collect_turn,
            )
            replay_events = _read_replay_events(replay_result.output_path)
        except Exception as exc:
            return VisualTruthGenerationSession(
                session,
                "error",
                f"视觉回放失败：{type(exc).__name__}: {exc}",
                stage_directory=paths.directory,
                indexed_frames=indexed_frames,
            )

        draft = assembler.truth_log
        finish_evidence = _finish_card_accounting(draft, replay_events)
        gates = _validate_gates(
            draft,
            replay_result,
            indexed_frames=indexed_frames,
            replay_events=replay_events,
            stream_rejections=stream_rejections,
            finish_evidence=finish_evidence,
        )
        trusted_uncertainty_override = False
        if staged_reference is not None and canonical is None:
            staged_reference_is_trusted = (
                (session / staged_reference.relative_path).resolve()
                in trusted_staged_references
            )
            staged_match = _truth_action_semantics_equal(staged_reference.truth_log, draft)
            gates = (
                *gates,
                _gate(
                    "trusted_staged_draft_match",
                    staged_match,
                    (
                        "本次扫描与既有隔离草稿的首局/动作语义完全一致"
                        if staged_match
                        else "本次扫描与既有隔离草稿不一致；禁止自动发布"
                    ),
                ),
            )
            unresolved_gate = next(
                gate
                for gate in gates
                if gate["name"] == "resolved_cards_and_wildcards"
            )
            if not bool(unresolved_gate["passed"]) and staged_reference_is_trusted:
                trusted_uncertainty_override = all(
                    bool(gate["passed"])
                    for gate in gates
                    if gate["name"] != "resolved_cards_and_wildcards"
                )
            gates = (
                *gates,
                _gate(
                    "trusted_staged_draft_corroborates_uncertainty",
                    bool(unresolved_gate["passed"]) or trusted_uncertainty_override,
                    (
                        "无需不确定性例外；本次扫描的牌面和赖子语义已完整解析"
                        if bool(unresolved_gate["passed"])
                        else "既有可信隔离草稿与本次初始状态/逐手动作语义一致；"
                        "其余发布门禁均通过，允许其佐证未决识别"
                        if trusted_uncertainty_override
                        else "不确定性不能由未经显式信任的既有草稿佐证；禁止自动发布"
                    ),
                ),
            )
        approved = all(
            bool(gate["passed"])
            or (
                gate["name"] == "resolved_cards_and_wildcards"
                and trusted_uncertainty_override
            )
            for gate in gates
        )
        stage_error = ""
        try:
            save_truth_log(paths.truth_log_path, draft)
            write_truth_scan_draft_sidecars(
                paths,
                session=session,
                scan_id=run_id,
                canonical=canonical,
                canonical_sha256=canonical_sha256,
                draft=draft,
                recorded_events=timeline,
                comparison_reference=(
                    staged_reference.truth_log if staged_reference is not None else None
                ),
                comparison_reference_sha256=(
                    staged_reference.sha256 if staged_reference is not None else None
                ),
                comparison_reference_path=(
                    staged_reference.relative_path if staged_reference is not None else None
                ),
            )
        except Exception as exc:
            stage_error = f"草稿写入失败：{type(exc).__name__}: {exc}"
            approved = False
            gates = (*gates, _gate("stage_serialization", False, stage_error))

        receipt_path = paths.directory / "visual_scan_receipt.json"
        receipt = {
            "schema": "guandan.visual-truth-scan-receipt/1",
            "run_id": run_id,
            "session": session.name,
            "provenance": VISUAL_SCAN_PROVENANCE,
            "publish_requested": publish,
            "canonical": {
                "path": "truth_log.json",
                "exists_before_scan": canonical_sha256 is not None,
                "sha256_before_scan": canonical_sha256,
            },
            "comparison_reference": {
                "kind": (
                    "canonical"
                    if canonical is not None
                    else "staged_draft"
                    if staged_reference is not None
                    else None
                ),
                "path": (
                    "truth_log.json"
                    if canonical is not None
                    else staged_reference.relative_path
                    if staged_reference is not None
                    else None
                ),
                "sha256": (
                    canonical_sha256
                    if canonical is not None
                    else staged_reference.sha256
                    if staged_reference is not None
                    else None
                ),
            },
            "scan": {
                "indexed_frames": indexed_frames,
                "frames_processed": replay_result.frame_count,
                "action_count": len(draft.turns),
                "replay_warnings": [
                    {"reason": warning.reason, "details": warning.details}
                    for warning in replay_result.warnings
                ],
                "visual_finish_evidence": list(finish_evidence),
                "trusted_staged_draft_corroborates_uncertainty": {
                    "applied": trusted_uncertainty_override,
                    "comparison_reference": (
                        staged_reference.relative_path
                        if staged_reference is not None and canonical is None
                        else None
                    ),
                    "comparison_reference_trusted": (
                        staged_reference is not None
                        and (session / staged_reference.relative_path).resolve()
                        in trusted_staged_references
                        and canonical is None
                    ),
                },
            },
            "gates": list(gates),
            "stage": {
                "truth_log": "truth_log.json" if paths.truth_log_path.is_file() else None,
                "comparison": "comparison.json" if paths.comparison_path.is_file() else None,
                "manifest": "manifest.json" if paths.metadata_path.is_file() else None,
                "replay": "replay" if (paths.directory / "replay").is_dir() else None,
            },
        }
        atomic_write_json(receipt_path, receipt)

        if not approved:
            return VisualTruthGenerationSession(
                session,
                "blocked",
                stage_error or "扫描草稿未通过发布门禁",
                stage_directory=paths.directory,
                receipt_path=receipt_path,
                frames_processed=replay_result.frame_count,
                indexed_frames=indexed_frames,
                action_count=len(draft.turns),
                gates=gates,
            )
        if canonical_sha256 is not None:
            return VisualTruthGenerationSession(
                session,
                "staged_existing",
                "已有 truth_log.json；仅生成扫描草稿和比较，不会覆盖",
                stage_directory=paths.directory,
                receipt_path=receipt_path,
                frames_processed=replay_result.frame_count,
                indexed_frames=indexed_frames,
                action_count=len(draft.turns),
                gates=gates,
            )
        if not publish:
            return VisualTruthGenerationSession(
                session,
                "staged",
                "已生成隔离扫描草稿；未请求发布",
                stage_directory=paths.directory,
                receipt_path=receipt_path,
                frames_processed=replay_result.frame_count,
                indexed_frames=indexed_frames,
                action_count=len(draft.turns),
                gates=gates,
            )
        try:
            _atomic_create_json(session / "truth_log.json", draft.to_dict())
        except FileExistsError:
            return VisualTruthGenerationSession(
                session,
                "staged_existing",
                "发布前检测到 truth_log.json，拒绝覆盖",
                stage_directory=paths.directory,
                receipt_path=receipt_path,
                frames_processed=replay_result.frame_count,
                indexed_frames=indexed_frames,
                action_count=len(draft.turns),
                gates=gates,
            )
        except Exception as exc:
            return VisualTruthGenerationSession(
                session,
                "error",
                f"正式日志发布失败：{type(exc).__name__}: {exc}",
                stage_directory=paths.directory,
                receipt_path=receipt_path,
                frames_processed=replay_result.frame_count,
                indexed_frames=indexed_frames,
                action_count=len(draft.turns),
                gates=gates,
            )
        return VisualTruthGenerationSession(
            session,
            "published",
            "已原子创建缺失的 truth_log.json",
            stage_directory=paths.directory,
            receipt_path=receipt_path,
            frames_processed=replay_result.frame_count,
            indexed_frames=indexed_frames,
            action_count=len(draft.turns),
            gates=gates,
            published=True,
        )


def write_visual_truth_generation_report(
    report_root: Path | str,
    run: VisualTruthGenerationRun,
) -> Path:
    """Persist a central report outside the scanned session data."""

    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"visual_truth_generation_{run.run_id}.json"
    _atomic_create_json(path, run.to_dict())
    return path


def _baseline_from_timeline(
    session: Path,
    timeline: tuple[dict[str, object], ...],
) -> TruthLog:
    events = tuple(LiveEvent.from_dict(raw) for raw in timeline)
    initial = next(
        (event for event in events if event.event_type == "initial_state_confirmed"),
        None,
    )
    if initial is None:
        raise ValueError("timeline 缺少 initial_state_confirmed")
    lead = initial.payload.get("lead_player")
    if lead not in _SEATS:
        first_action = next(
            (
                index
                for index, event in enumerate(events)
                if event.event_type in _ACTION_TYPES
            ),
            len(events),
        )
        confirmation = next(
            (
                event
                for index, event in enumerate(events)
                if index < first_action and event.event_type == "lead_player_confirmed"
            ),
            None,
        )
        if confirmation is not None:
            candidate = confirmation.payload.get("lead_player")
            lead = candidate if candidate in _SEATS else confirmation.actor
    if lead not in _SEATS:
        raise ValueError("timeline 缺少首出玩家或首出确认")
    try:
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    return TruthLog(
        source_session_id=str(manifest.get("session_id", session.name)),
        initial_state=TruthInitialState(
            str(initial.payload.get("round_level", "")),
            lead,
            tuple(str(card) for card in initial.payload.get("hand", ())),
        ),
        turns=(),
        label_status="draft",
        provenance=LabelProvenance(source=VISUAL_SCAN_PROVENANCE),
    )


def _existing_truth(session: Path) -> tuple[TruthLog | None, str | None]:
    path = session / "truth_log.json"
    if not path.is_file():
        return None, None
    payload = path.read_bytes()
    return (
        load_truth_log(path, session_id=_session_id(session)),
        hashlib.sha256(payload).hexdigest(),
    )


def _existing_staged_truth(session: Path) -> _ExistingStagedTruth | None:
    """Return a prior isolated candidate as comparison-only protection.

    A hand-reviewed staged draft is intentionally treated more conservatively
    than a missing formal log: a later batch scan may compare against it but
    may never auto-promote over it.  Invalid/incomplete stage files are
    ignored; they cannot block a valid session indefinitely.
    """

    root = session / "derived" / "truth_scan_drafts"
    if not root.is_dir():
        return None
    for path in sorted(root.glob("*/truth_log.json")):
        try:
            payload = path.read_bytes()
            truth_log = load_truth_log(path, session_id=_session_id(session))
        except Exception:
            continue
        return _ExistingStagedTruth(
            truth_log=truth_log,
            sha256=hashlib.sha256(payload).hexdigest(),
            relative_path=path.relative_to(session).as_posix(),
        )
    return None


def _truth_action_semantics_equal(left: TruthLog, right: TruthLog) -> bool:
    """Compare the publish-relevant truth semantics, ignoring evidence noise."""

    if (
        left.initial_state.round_level != right.initial_state.round_level
        or left.initial_state.lead_player != right.initial_state.lead_player
        or sorted(left.initial_state.my_hand) != sorted(right.initial_state.my_hand)
        or len(left.turns) != len(right.turns)
    ):
        return False
    return all(
        first.actor == second.actor
        and first.is_pass == second.is_pass
        and sorted(first.cards) == sorted(second.cards)
        for first, second in zip(left.turns, right.turns)
    )


def _session_id(session: Path) -> str:
    try:
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    return str(manifest.get("session_id", session.name))


def _recognition_for_session(session: Path) -> ScreenshotRecognitionService:
    profile = session.parent.parent
    return ScreenshotRecognitionService(
        AnnotationService(profile.parent, profile.name),
        TemplateService(profile.parent, profile.name),
    )


def _read_replay_events(path: Path) -> tuple[dict[str, object], ...]:
    events: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in read_json_lines(path):
        values = row.get("events", ())
        if not isinstance(values, list):
            continue
        for raw in values:
            if not isinstance(raw, dict):
                continue
            event_id = str(raw.get("event_id", ""))
            if not event_id or event_id in seen:
                continue
            seen.add(event_id)
            events.append(dict(raw))
    return tuple(events)


def _validate_gates(
    draft: TruthLog,
    replay: VisualPipelineReplayResult,
    *,
    indexed_frames: int,
    replay_events: tuple[dict[str, object], ...],
    stream_rejections: list[str],
    finish_evidence: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    gates: list[dict[str, object]] = []
    gates.append(
        _gate(
            "full_indexed_frames",
            indexed_frames > 0 and replay.frame_count == indexed_frames,
            f"processed={replay.frame_count}, indexed={indexed_frames}",
        )
    )
    gates.append(
        _gate(
            "replay_warnings",
            not replay.warnings,
            ", ".join(warning.reason for warning in replay.warnings) or "none",
        )
    )
    has_terminal_history_gap = any(
        raw.get("event_type") == "terminal_history_gap" for raw in replay_events
    )
    gates.append(
        _gate(
            "terminal_history_gap",
            not has_terminal_history_gap,
            (
                "terminal_history_gap absent"
                if not has_terminal_history_gap
                else "terminal_history_gap detected in replay evidence"
            ),
        )
    )
    finish_mismatches = [
        item
        for item in finish_evidence
        if not bool(item.get("matches_draft_remaining"))
    ]
    gates.append(
        _gate(
            "visual_finish_card_accounting",
            not finish_mismatches,
            (
                "all visual player_finished actors have zero remaining cards"
                if not finish_mismatches
                else "; ".join(
                    f"{item.get('actor')}: remaining={item.get('draft_remaining')}"
                    for item in finish_mismatches
                )
            ),
        )
    )
    gates.append(
        _gate(
            "stream_actions",
            not stream_rejections,
            "; ".join(stream_rejections) or "all accepted",
        )
    )
    gates.append(_gate("nonempty", bool(draft.turns), f"turns={len(draft.turns)}"))
    try:
        validate_turn_actor_chain(draft)
    except Exception as exc:
        gates.append(_gate("actor_chain", False, str(exc)))
    else:
        gates.append(_gate("actor_chain", True, "contiguous actor chain"))
    try:
        reducer = LiveReducer("visual-truth-gate")
        for event in draft.to_events(session_id="visual-truth-gate"):
            reducer.apply(event)
    except Exception as exc:
        gates.append(_gate("live_reducer_replay", False, str(exc)))
    else:
        gates.append(_gate("live_reducer_replay", True, "accepted by current reducer"))
    unresolved = _unresolved_action_reasons(draft, replay_events)
    gates.append(
        _gate(
            "resolved_cards_and_wildcards",
            not unresolved,
            "; ".join(unresolved) or "no unresolved suit/wildcard action",
        )
    )
    provenance_ok = (
        draft.label_status == "draft"
        and draft.provenance.source == VISUAL_SCAN_PROVENANCE
        and all(
            turn.label_status == "draft"
            and turn.provenance.source == VISUAL_SCAN_PROVENANCE
            for turn in draft.turns
        )
    )
    gates.append(
        _gate("draft_provenance", provenance_ok, VISUAL_SCAN_PROVENANCE)
    )
    return tuple(gates)


def _finish_card_accounting(
    draft: TruthLog,
    replay_events: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Cross-check visual ranks against cards actually represented in the draft."""

    remaining = {seat: 27 for seat in _SEATS}
    remaining["self"] = len(draft.initial_state.my_hand)
    for turn in draft.turns:
        if not turn.is_pass and turn.actor in remaining:
            remaining[turn.actor] -= len(turn.cards)

    evidence: list[dict[str, object]] = []
    for event in replay_events:
        if event.get("event_type") != "player_finished":
            continue
        payload = event.get("payload", {})
        payload = payload if isinstance(payload, dict) else {}
        actor = event.get("actor") or payload.get("player")
        actor_name = str(actor) if actor is not None else ""
        cards_remaining = remaining.get(actor_name)
        evidence.append(
            {
                "event_id": event.get("event_id"),
                "actor": actor_name or None,
                "rank": payload.get("placement", payload.get("rank")),
                "frame_index": payload.get("frame_index", event.get("frame_index")),
                "draft_remaining": cards_remaining,
                "matches_draft_remaining": cards_remaining == 0,
            }
        )
    return tuple(evidence)


def _unresolved_action_reasons(
    draft: TruthLog,
    replay_events: tuple[dict[str, object], ...],
) -> list[str]:
    reasons: list[str] = []
    for turn in draft.turns:
        if any(card.endswith("?") for card in turn.cards) or turn.uncertainty:
            reasons.append(f"turn {turn.index} has unresolved card uncertainty")
    for raw in _effective_replay_actions(replay_events):
        payload = raw.get("payload", {})
        if not isinstance(payload, dict):
            continue
        semantics = payload.get("move_semantics")
        source = semantics.get("selection_source") if isinstance(semantics, dict) else None
        warnings = payload.get("integrity_warnings", ())
        warning_values = tuple(str(value) for value in warnings) if isinstance(warnings, (list, tuple)) else ()
        if (
            payload.get("interpretation_ambiguous") is True
            or source == "unresolved"
            or "wildcard_interpretation_ambiguous" in warning_values
        ):
            reasons.append(f"event {raw.get('event_id', '?')} has unresolved wildcard semantics")
    return reasons


def _effective_replay_actions(
    events: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    actions: dict[str, dict[str, object]] = {}
    order: list[str] = []
    corrections: dict[str, dict[str, object]] = {}
    for raw in events:
        event_type = str(raw.get("event_type", ""))
        event_id = str(raw.get("event_id", ""))
        if event_type in _ACTION_TYPES and event_id:
            actions[event_id] = dict(raw)
            order.append(event_id)
        elif event_type == "event_correction":
            payload = raw.get("payload", {})
            if isinstance(payload, dict):
                corrections[str(payload.get("target_event_id", ""))] = payload
    effective: list[dict[str, object]] = []
    for event_id in order:
        raw = actions[event_id]
        correction = corrections.get(event_id)
        if correction is not None:
            payload = dict(raw.get("payload", {}))
            payload["cards"] = list(correction.get("cards", ()))
            payload["is_pass"] = bool(correction.get("is_pass", False))
            raw["payload"] = payload
        effective.append(raw)
    return tuple(effective)


def _gate(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": passed, "detail": detail}


def _new_run_id() -> str:
    return datetime.now().astimezone().strftime("batch_%Y%m%dT%H%M%S_") + uuid4().hex[:12]


def _atomic_create_json(path: Path, value: object) -> None:
    """Atomically create JSON once; a concurrent target is never replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


__all__ = [
    "VISUAL_SCAN_PROVENANCE",
    "VisualTruthGenerationRun",
    "VisualTruthGenerationService",
    "VisualTruthGenerationSession",
    "write_visual_truth_generation_report",
]
