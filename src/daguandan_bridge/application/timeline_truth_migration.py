from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal
from uuid import uuid4

from ..danzero.state import SEATS, Seat
from ..application.model_evaluation import validate_evaluation_truth
from ..domain.truth import LabelProvenance, TruthEvidence
from ..live.models import LiveEvent
from ..live.reducer import LiveReducer
from ..live.session_store import read_json_lines
from ..live.suit_correction import validate_suit_correction
from ..live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    load_truth_log,
    truth_log_from_dict,
)
from ..storage import atomic_write_json


MIGRATION_SOURCE = "timeline_migration_v1"
_ACTION_TYPES = frozenset(
    {"player_played", "player_passed", "manual_confirmed_event"}
)
_CORRECTION_TYPES = frozenset({"event_correction", "suit_corrected"})
_InspectionStatus = Literal["candidate", "skipped", "blocked"]
_MigrationStatus = Literal["migrated", "skipped", "blocked", "failed"]
_RepairInspectionStatus = Literal["candidate", "blocked"]
_RepairStatus = Literal["repaired", "blocked", "failed"]


@dataclass(frozen=True)
class TimelineTruthSessionInspection:
    session: Path
    status: _InspectionStatus
    code: str
    message: str
    timeline_sha256: str = ""
    timeline_bytes: int = 0
    event_count: int = 0
    action_count: int = 0
    event_type_counts: tuple[tuple[str, int], ...] = ()
    draft: TruthLog | None = None
    context_repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimelineTruthRootInspection:
    sessions_root: Path
    sessions: tuple[TimelineTruthSessionInspection, ...]

    @property
    def candidate_count(self) -> int:
        return sum(item.status == "candidate" for item in self.sessions)

    @property
    def skipped_count(self) -> int:
        return sum(item.status == "skipped" for item in self.sessions)

    @property
    def blocked_count(self) -> int:
        return sum(item.status == "blocked" for item in self.sessions)


@dataclass(frozen=True)
class TimelineTruthSessionMigration:
    session: Path
    status: _MigrationStatus
    code: str
    message: str
    truth_log_path: Path | None = None
    receipt_path: Path | None = None
    action_count: int = 0


@dataclass(frozen=True)
class TimelineTruthRootMigration:
    sessions_root: Path
    sessions: tuple[TimelineTruthSessionMigration, ...]
    cancelled: bool = False

    @property
    def migrated_count(self) -> int:
        return sum(item.status == "migrated" for item in self.sessions)

    @property
    def skipped_count(self) -> int:
        return sum(item.status == "skipped" for item in self.sessions)

    @property
    def blocked_count(self) -> int:
        return sum(item.status == "blocked" for item in self.sessions)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.sessions)


@dataclass(frozen=True)
class ExistingTruthRepairInspection:
    session: Path
    status: _RepairInspectionStatus
    code: str
    message: str
    existing_sha256: str = ""
    timeline_sha256: str = ""
    replacement_sha256: str = ""
    turn_count: int = 0
    replacement: TruthLog | None = None
    context_repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExistingTruthRepairResult:
    session: Path
    status: _RepairStatus
    code: str
    message: str
    truth_log_path: Path | None = None
    backup_path: Path | None = None
    receipt_path: Path | None = None


class _BlockedMigration(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class TimelineTruthMigrationService:
    """Safely reconstruct draft TruthLogs from append-only live timelines."""

    def inspect_session(
        self,
        session: Path | str,
        *,
        force: bool = False,
    ) -> TimelineTruthSessionInspection:
        session_path = Path(session)
        truth_path = session_path / "truth_log.json"
        receipt_path = session_path / "truth_log.migration.json"
        if truth_path.exists() and not force:
            return TimelineTruthSessionInspection(
                session=session_path,
                status="skipped",
                code="existing_truth_log",
                message="truth_log.json already exists",
            )
        if receipt_path.exists() and not force:
            return TimelineTruthSessionInspection(
                session=session_path,
                status="blocked",
                code="orphan_audit_receipt",
                message="migration receipt exists without a truth log; inspect it manually",
            )

        timeline_path = session_path / "timeline.jsonl"
        if not timeline_path.is_file():
            return TimelineTruthSessionInspection(
                session=session_path,
                status="blocked",
                code="missing_timeline",
                message="timeline.jsonl does not exist",
            )
        try:
            timeline_bytes = timeline_path.read_bytes()
            raw_events = read_json_lines(timeline_path)
        except Exception as exc:
            return self._blocked(
                session_path,
                "timeline_read_failed",
                f"could not read timeline.jsonl: {exc}",
            )
        digest = hashlib.sha256(timeline_bytes).hexdigest()
        counts = Counter(str(item.get("event_type", "")) for item in raw_events)
        base = {
            "timeline_sha256": digest,
            "timeline_bytes": len(timeline_bytes),
            "event_count": len(raw_events),
            "action_count": sum(counts[name] for name in _ACTION_TYPES),
            "event_type_counts": tuple(sorted(counts.items())),
        }
        try:
            events = tuple(LiveEvent.from_dict(raw) for raw in raw_events)
            draft, context_repairs = self._build_and_validate(session_path.name, events)
        except _BlockedMigration as exc:
            return self._blocked(session_path, exc.code, str(exc), **base)
        except Exception as exc:
            return self._blocked(
                session_path,
                "event_parse_failed",
                f"timeline event parsing failed: {exc}",
                **base,
            )
        return TimelineTruthSessionInspection(
            session=session_path,
            status="candidate",
            code="safe_to_migrate",
            message="timeline deterministically replays as a draft TruthLog",
            draft=draft,
            context_repairs=context_repairs,
            **base,
        )

    def inspect_root(
        self,
        sessions_root: Path | str,
        *,
        force: bool = False,
    ) -> TimelineTruthRootInspection:
        root = Path(sessions_root)
        sessions = () if not root.is_dir() else tuple(
            self.inspect_session(path, force=force)
            for path in sorted(root.iterdir(), key=lambda item: item.name)
            if path.is_dir()
        )
        return TimelineTruthRootInspection(root, sessions)

    def inspect_existing_truth_repair(
        self,
        session: Path | str,
    ) -> ExistingTruthRepairInspection:
        """Inspect one legacy TruthLog without changing any session artifact."""

        session_path = Path(session).resolve()
        truth_path = session_path / "truth_log.json"
        timeline_path = session_path / "timeline.jsonl"
        receipt_path = session_path / "truth_log.repair.json"
        if not truth_path.is_file():
            return self._repair_blocked(session_path, "missing_truth_log", "truth_log.json does not exist")
        if receipt_path.exists():
            return self._repair_blocked(
                session_path,
                "repair_already_recorded",
                "truth_log.repair.json already exists",
            )
        try:
            truth_bytes = truth_path.read_bytes()
            raw_truth = json.loads(truth_bytes.decode("utf-8"))
            existing = load_truth_log(truth_path, session_id=session_path.name)
        except Exception as exc:
            return self._repair_blocked(
                session_path, "existing_truth_unreadable", f"could not read existing TruthLog: {exc}"
            )
        existing_sha = hashlib.sha256(truth_bytes).hexdigest()
        if not isinstance(raw_truth, dict):
            return self._repair_blocked(
                session_path, "existing_truth_not_object", "existing TruthLog is not an object",
                existing_sha256=existing_sha,
            )
        try:
            timeline_bytes = timeline_path.read_bytes()
            raw_events = read_json_lines(timeline_path)
        except Exception as exc:
            return self._repair_blocked(
                session_path, "timeline_read_failed", f"could not read timeline.jsonl: {exc}",
                existing_sha256=existing_sha,
            )
        timeline_sha = hashlib.sha256(timeline_bytes).hexdigest()

        # Prefer repairing the saved TruthLog itself.  The older timeline may
        # contain recognition diagnostics that are unrelated to the saved
        # action chain; replacing that chain would discard a human correction.
        # A repair is allowed only when current strict validation identifies a
        # stale trick id and replaying the exact same actions produces a fully
        # valid, trick-id-only replacement.
        existing_validation = validate_evaluation_truth(session_path, existing)
        if any(issue.code == "INPUT_TRICK_MISMATCH" for issue in existing_validation.issues):
            try:
                replacement, context_repairs = self._reindex_truth_tricks(
                    session_path.name,
                    existing,
                )
            except Exception as exc:
                return self._repair_blocked(
                    session_path,
                    "truth_reducer_replay_failed",
                    f"could not replay the saved TruthLog: {exc}",
                    existing_sha256=existing_sha,
                    timeline_sha256=timeline_sha,
                )
            strict = validate_evaluation_truth(session_path, replacement)
            if not strict.ready:
                first = strict.issues[0]
                turn = f" at turn {first.turn_id}" if first.turn_id is not None else ""
                return self._repair_blocked(
                    session_path,
                    "replacement_strict_validation_failed",
                    f"{first.code}{turn}: {first.message}",
                    existing_sha256=existing_sha,
                    timeline_sha256=timeline_sha,
                    turn_count=len(existing.turns),
                )
            replacement_document = truth_log_from_dict(replacement.to_dict()).to_dict()
            replacement_text = json.dumps(
                replacement_document, ensure_ascii=False, indent=2
            ) + "\n"
            replacement_bytes = replacement_text.replace("\n", os.linesep).encode("utf-8")
            return ExistingTruthRepairInspection(
                session=session_path,
                status="candidate",
                code="safe_to_repair_trick_context",
                message="saved TruthLog has a proven stale trick context and can be reindexed",
                existing_sha256=existing_sha,
                timeline_sha256=timeline_sha,
                replacement_sha256=hashlib.sha256(replacement_bytes).hexdigest(),
                turn_count=len(existing.turns),
                replacement=replacement,
                context_repairs=context_repairs,
            )
        schema = str(raw_truth.get("schema", ""))
        try:
            version = int(raw_truth.get("schema_version", 1))
        except (TypeError, ValueError):
            return self._repair_blocked(
                session_path,
                "invalid_legacy_schema_version",
                "existing TruthLog has an invalid schema_version",
                existing_sha256=existing_sha,
            )
        migrated_draft = (
            existing.label_status == "draft"
            and existing.provenance.source == MIGRATION_SOURCE
            and all(turn.provenance.source == MIGRATION_SOURCE for turn in existing.turns)
        )
        legacy_truth = schema != "guandan.truth/3" or version < 3
        if not legacy_truth and not migrated_draft:
            return self._repair_blocked(
                session_path,
                "not_legacy_truth_log",
                "only legacy or timeline-migrated draft TruthLogs are eligible for automatic repair",
                existing_sha256=existing_sha,
            )
        if existing.label_status == "verified" or any(
            turn.label_status == "verified" for turn in existing.turns
        ):
            return self._repair_blocked(
                session_path,
                "verified_truth_log",
                "verified TruthLogs must only be changed by a human editor",
                existing_sha256=existing_sha,
            )

        timeline = self.inspect_session(session_path, force=True)
        if timeline.status != "candidate" or timeline.draft is None:
            code = (
                timeline.code
                if timeline.code == "unsafe_timeline_integrity"
                else f"timeline_{timeline.code}"
            )
            return self._repair_blocked(
                session_path,
                code,
                timeline.message,
                existing_sha256=existing_sha,
                timeline_sha256=timeline.timeline_sha256,
            )
        replacement = timeline.draft
        existing_coverage = tuple((turn.index, turn.actor) for turn in existing.turns)
        replacement_coverage = tuple(
            (turn.index, turn.actor) for turn in replacement.turns
        )
        if existing_coverage != replacement_coverage:
            return self._repair_blocked(
                session_path,
                "action_coverage_mismatch",
                "timeline does not cover the same ordered turns and actors as the existing TruthLog",
                existing_sha256=existing_sha,
                timeline_sha256=timeline.timeline_sha256,
                turn_count=len(existing.turns),
                context_repairs=timeline.context_repairs,
            )
        strict = validate_evaluation_truth(session_path, replacement)
        if not strict.ready:
            first = strict.issues[0]
            turn = f" at turn {first.turn_id}" if first.turn_id is not None else ""
            return self._repair_blocked(
                session_path,
                "replacement_strict_validation_failed",
                f"{first.code}{turn}: {first.message}",
                existing_sha256=existing_sha,
                timeline_sha256=timeline.timeline_sha256,
                turn_count=len(existing.turns),
            )
        replacement_document = truth_log_from_dict(replacement.to_dict()).to_dict()
        replacement_text = json.dumps(
            replacement_document, ensure_ascii=False, indent=2
        ) + "\n"
        # ``atomic_write_json`` uses ``Path.write_text`` and therefore applies
        # the platform newline convention.
        replacement_bytes = replacement_text.replace("\n", os.linesep).encode("utf-8")
        return ExistingTruthRepairInspection(
            session=session_path,
            status="candidate",
            code="safe_to_repair",
            message="legacy TruthLog can be replaced from the deterministic timeline",
            existing_sha256=existing_sha,
            timeline_sha256=timeline.timeline_sha256,
            replacement_sha256=hashlib.sha256(replacement_bytes).hexdigest(),
            turn_count=len(existing.turns),
            replacement=replacement,
            context_repairs=timeline.context_repairs,
        )

    def repair_existing_truth(
        self,
        session: Path | str,
    ) -> ExistingTruthRepairResult:
        """Explicitly repair one inspected legacy TruthLog with backup and receipt."""

        inspection = self.inspect_existing_truth_repair(session)
        if inspection.status != "candidate" or inspection.replacement is None:
            return ExistingTruthRepairResult(
                inspection.session, "blocked", inspection.code, inspection.message
            )
        truth_path = inspection.session / "truth_log.json"
        receipt_path = inspection.session / "truth_log.repair.json"
        try:
            current_bytes = truth_path.read_bytes()
            timeline_bytes = (inspection.session / "timeline.jsonl").read_bytes()
        except Exception as exc:
            return ExistingTruthRepairResult(
                inspection.session, "failed", "repair_source_read_failed", str(exc)
            )
        if hashlib.sha256(current_bytes).hexdigest() != inspection.existing_sha256:
            return ExistingTruthRepairResult(
                inspection.session, "blocked", "truth_changed_after_inspection",
                "truth_log.json changed while repair was being prepared",
            )
        if hashlib.sha256(timeline_bytes).hexdigest() != inspection.timeline_sha256:
            return ExistingTruthRepairResult(
                inspection.session, "blocked", "timeline_changed_after_inspection",
                "timeline.jsonl changed while repair was being prepared",
            )

        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
        backup_path = inspection.session / f"truth_log.repair.{timestamp}.json"
        published = False
        try:
            self._atomic_create_bytes(backup_path, current_bytes)
            atomic_write_json(
                truth_path,
                truth_log_from_dict(inspection.replacement.to_dict()).to_dict(),
            )
            published = True
            output_sha = hashlib.sha256(truth_path.read_bytes()).hexdigest()
            if output_sha != inspection.replacement_sha256:
                raise OSError("published TruthLog digest does not match the inspected replacement")
            receipt = {
                "schema": "guandan.truth-log-repair/1",
                "repaired_at": datetime.now().astimezone().isoformat(),
                "session_id": inspection.session.name,
                "source_timeline": {
                    "path": "timeline.jsonl",
                    "sha256": inspection.timeline_sha256,
                },
                "previous_truth_log": {
                    "path": backup_path.name,
                    "sha256": inspection.existing_sha256,
                },
                "output_truth_log": {
                    "path": "truth_log.json",
                    "sha256": output_sha,
                    "turn_count": inspection.turn_count,
                },
                "validation": {
                    "legacy_or_migrated_draft_source": True,
                    "same_action_coverage": True,
                    "strict_evaluation_input": True,
                    "context_repairs": list(inspection.context_repairs),
                },
            }
            self._atomic_create_json(receipt_path, receipt)
        except Exception as exc:
            restore_error = ""
            if published:
                try:
                    self._atomic_replace_bytes(truth_path, current_bytes)
                except Exception as restore_exc:
                    restore_error = f"; restoring the original failed: {restore_exc}"
            return ExistingTruthRepairResult(
                inspection.session,
                "failed",
                "repair_publish_failed",
                str(exc) + restore_error,
                truth_log_path=truth_path if truth_path.exists() else None,
                backup_path=backup_path if backup_path.exists() else None,
            )
        return ExistingTruthRepairResult(
            inspection.session,
            "repaired",
            "repaired",
            "legacy TruthLog repaired; backup and audit receipt published",
            truth_log_path=truth_path,
            backup_path=backup_path,
            receipt_path=receipt_path,
        )

    def migrate_session(
        self,
        session: Path | str,
        *,
        force: bool = False,
    ) -> TimelineTruthSessionMigration:
        inspection = self.inspect_session(session, force=force)
        if inspection.status != "candidate" or inspection.draft is None:
            status: _MigrationStatus = (
                "skipped" if inspection.status == "skipped" else "blocked"
            )
            return TimelineTruthSessionMigration(
                inspection.session,
                status,
                inspection.code,
                inspection.message,
                action_count=inspection.action_count,
            )

        truth_path = inspection.session / "truth_log.json"
        receipt_path = inspection.session / "truth_log.migration.json"
        document = truth_log_from_dict(inspection.draft.to_dict()).to_dict()
        try:
            if force:
                atomic_write_json(truth_path, document)
            else:
                self._atomic_create_json(truth_path, document)
        except FileExistsError:
            return TimelineTruthSessionMigration(
                inspection.session,
                "skipped",
                "existing_truth_log",
                "truth_log.json appeared while migration was running",
                action_count=inspection.action_count,
            )
        except Exception as exc:
            return TimelineTruthSessionMigration(
                inspection.session,
                "failed",
                "truth_write_failed",
                f"could not publish truth_log.json: {exc}",
                action_count=inspection.action_count,
            )

        truth_digest = hashlib.sha256(truth_path.read_bytes()).hexdigest()
        event_type_counts = dict(inspection.event_type_counts)
        receipt = {
            "schema": "guandan.timeline-truth-migration/1",
            "migration_source": MIGRATION_SOURCE,
            "migrated_at": datetime.now().astimezone().isoformat(),
            "session_id": inspection.session.name,
            "source_timeline": {
                "path": "timeline.jsonl",
                "sha256": inspection.timeline_sha256,
                "byte_count": inspection.timeline_bytes,
                "event_count": inspection.event_count,
                "event_type_counts": dict(inspection.event_type_counts),
                "action_count": inspection.action_count,
                "player_finished_count": event_type_counts.get("player_finished", 0),
                "correction_count": sum(
                    event_type_counts.get(name, 0)
                    for name in _CORRECTION_TYPES
                ),
            },
            "output_truth_log": {
                "path": "truth_log.json",
                "sha256": truth_digest,
                "label_status": "draft",
                "turn_count": inspection.action_count,
            },
            "validation": {
                "truth_log_roundtrip": True,
                "source_reducer_replay": True,
                "truth_reducer_replay": True,
                "context_repairs": list(inspection.context_repairs),
                "player_finished_replayed_for_sequence_check": bool(
                    event_type_counts.get("player_finished", 0)
                ),
                "outcome_migrated": False,
            },
        }
        try:
            if force:
                atomic_write_json(receipt_path, receipt)
            else:
                self._atomic_create_json(receipt_path, receipt)
        except Exception as exc:
            return TimelineTruthSessionMigration(
                inspection.session,
                "failed",
                "audit_write_failed",
                f"truth_log.json was published but its audit receipt failed: {exc}",
                truth_log_path=truth_path,
                action_count=inspection.action_count,
            )
        return TimelineTruthSessionMigration(
            inspection.session,
            "migrated",
            "migrated",
            "draft TruthLog and migration receipt published",
            truth_log_path=truth_path,
            receipt_path=receipt_path,
            action_count=inspection.action_count,
        )

    def migrate_root(
        self,
        sessions_root: Path | str,
        *,
        force: bool = False,
        should_stop: Callable[[], bool] | None = None,
    ) -> TimelineTruthRootMigration:
        root = Path(sessions_root)
        results: list[TimelineTruthSessionMigration] = []
        cancelled = False
        if root.is_dir():
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                if not path.is_dir():
                    continue
                if should_stop is not None and should_stop():
                    cancelled = True
                    break
                results.append(self.migrate_session(path, force=force))
        return TimelineTruthRootMigration(root, tuple(results), cancelled)

    @staticmethod
    def _blocked(
        session: Path,
        code: str,
        message: str,
        **values: object,
    ) -> TimelineTruthSessionInspection:
        return TimelineTruthSessionInspection(
            session=session,
            status="blocked",
            code=code,
            message=message,
            **values,  # type: ignore[arg-type]
        )

    @staticmethod
    def _repair_blocked(
        session: Path,
        code: str,
        message: str,
        **values: object,
    ) -> ExistingTruthRepairInspection:
        return ExistingTruthRepairInspection(
            session=session,
            status="blocked",
            code=code,
            message=message,
            **values,  # type: ignore[arg-type]
        )

    @staticmethod
    def _reindex_truth_tricks(
        session_id: str,
        truth: TruthLog,
    ) -> tuple[TruthLog, tuple[str, ...]]:
        """Recompute trick ids while preserving every saved action field."""

        reducer = LiveReducer(session_id)
        reducer.confirm_initial_state(
            round_level=truth.initial_state.round_level,
            hand=truth.initial_state.my_hand,
            lead_player=truth.initial_state.lead_player,
            source="truth_log_repair",
        )
        repaired_turns: list[TruthTurn] = []
        context_repairs: list[str] = []
        for turn in truth.turns:
            expected_trick = reducer.snapshot().trick_id
            if turn.trick_id != expected_trick:
                context_repairs.append(
                    f"turn {turn.index}: trick {turn.trick_id} -> {expected_trick} "
                    "after replayed wind catch"
                )
            repaired_turns.append(replace(turn, trick_id=expected_trick))
            if turn.is_pass:
                reducer.record_pass(turn.actor, source="truth_log_repair")
            else:
                reducer.record_play(
                    turn.actor,
                    turn.cards,
                    source="truth_log_repair",
                )
        if not context_repairs:
            raise ValueError("saved TruthLog has no stale trick ids")
        return replace(truth, turns=tuple(repaired_turns)), tuple(context_repairs)

    def _build_and_validate(
        self,
        session_id: str,
        events: tuple[LiveEvent, ...],
    ) -> tuple[TruthLog, tuple[str, ...]]:
        if not session_id:
            raise _BlockedMigration("invalid_session_id", "session directory name is empty")
        for event in events:
            if event.session_id != session_id:
                raise _BlockedMigration(
                    "session_id_mismatch",
                    f"event {event.event_id} belongs to {event.session_id}, not {session_id}",
                )
            if event.monotonic_ms < 0:
                raise _BlockedMigration(
                    "negative_monotonic_time",
                    f"event {event.event_id} has a negative monotonic timestamp",
                )
        if len({event.event_id for event in events}) != len(events):
            raise _BlockedMigration("duplicate_event_id", "timeline contains duplicate event IDs")
        if any(
            (right.monotonic_ms, right.seq) < (left.monotonic_ms, left.seq)
            for left, right in zip(events, events[1:])
        ):
            raise _BlockedMigration(
                "timeline_not_chronological",
                "timeline records are not in monotonic/sequence order",
            )

        initial_events = [
            (index, event)
            for index, event in enumerate(events)
            if event.event_type == "initial_state_confirmed"
        ]
        if len(initial_events) != 1:
            raise _BlockedMigration(
                "initial_state_count",
                f"expected exactly one initial_state_confirmed event, found {len(initial_events)}",
            )
        initial_position, initial_event = initial_events[0]
        if initial_position != 0 and any(
            event.event_type in _ACTION_TYPES for event in events[:initial_position]
        ):
            raise _BlockedMigration(
                "initial_state_after_action",
                "initial state occurs after a recorded action",
            )
        first_action_position = next(
            (
                index
                for index, event in enumerate(events)
                if event.event_type in _ACTION_TYPES
            ),
            len(events),
        )
        lead = self._resolve_lead(events, initial_event, first_action_position)
        round_level = str(initial_event.payload.get("round_level", ""))
        wild_rank = str(initial_event.payload.get("wild_rank", round_level))
        if wild_rank != round_level:
            raise _BlockedMigration(
                "initial_wild_rank_mismatch",
                "TruthLog cannot preserve an initial wild rank different from round level",
            )
        hand = initial_event.payload.get("hand", ())
        if not isinstance(hand, (list, tuple)):
            raise _BlockedMigration("invalid_initial_hand", "initial hand is not an array")

        effective_actions = self._effective_actions(events)
        normalized_actions, context_repairs = self._normalize_source_actions(
            session_id,
            events,
            effective_actions,
        )
        turns = tuple(
            self._turn_from_event(index, event)
            for index, event in enumerate(normalized_actions, start=1)
        )
        try:
            draft = truth_log_from_dict(
                TruthLog(
                    source_session_id=session_id,
                    initial_state=TruthInitialState(
                        round_level=round_level,
                        lead_player=lead,
                        my_hand=tuple(str(card) for card in hand),
                    ),
                    turns=turns,
                    label_status="draft",
                    provenance=LabelProvenance(source=MIGRATION_SOURCE),
                ).to_dict()
            )
        except Exception as exc:
            raise _BlockedMigration(
                "truth_log_validation_failed",
                f"draft TruthLog failed schema validation: {exc}",
            ) from exc
        self._validate_truth_replay(session_id, normalized_actions, draft)
        return draft, context_repairs

    @staticmethod
    def _resolve_lead(
        events: tuple[LiveEvent, ...],
        initial_event: LiveEvent,
        first_action_position: int,
    ) -> Seat:
        initial_lead = initial_event.payload.get("lead_player")
        confirmations = [
            (index, event)
            for index, event in enumerate(events)
            if event.event_type == "lead_player_confirmed"
        ]
        if initial_lead is not None:
            if initial_lead not in SEATS:
                raise _BlockedMigration("invalid_initial_lead", "initial lead player is invalid")
            if confirmations:
                raise _BlockedMigration(
                    "redundant_lead_confirmation",
                    "timeline confirms a lead player even though the initial state already has one",
                )
            return initial_lead  # type: ignore[return-value]
        if len(confirmations) != 1:
            raise _BlockedMigration(
                "lead_confirmation_count",
                "initial state has no lead; exactly one lead confirmation is required",
            )
        position, confirmation = confirmations[0]
        lead = confirmation.payload.get("lead_player")
        if position >= first_action_position:
            raise _BlockedMigration(
                "lead_confirmation_too_late",
                "lead player confirmation does not occur before the first action",
            )
        if lead not in SEATS or confirmation.actor != lead:
            raise _BlockedMigration(
                "invalid_lead_confirmation",
                "lead confirmation actor/payload is invalid or inconsistent",
            )
        return lead  # type: ignore[return-value]

    @staticmethod
    def _effective_actions(events: tuple[LiveEvent, ...]) -> tuple[LiveEvent, ...]:
        positions = {event.event_id: index for index, event in enumerate(events)}
        action_by_id = {
            event.event_id: event for event in events if event.event_type in _ACTION_TYPES
        }
        corrections: dict[str, LiveEvent] = {}
        for position, correction in enumerate(events):
            if correction.event_type not in _CORRECTION_TYPES:
                continue
            target_id = str(correction.payload.get("target_event_id", "")).strip()
            target = action_by_id.get(target_id)
            if target is None:
                raise _BlockedMigration(
                    "correction_target_missing",
                    f"correction {correction.event_id} does not target a recorded action",
                )
            if positions[target_id] >= position:
                raise _BlockedMigration(
                    "correction_before_target",
                    f"correction {correction.event_id} occurs before its target",
                )
            if target_id in corrections:
                raise _BlockedMigration(
                    "multiple_corrections",
                    f"action {target_id} has more than one correction",
                )
            if correction.event_type == "event_correction":
                previous_actions = [
                    event
                    for event in events[:position]
                    if event.event_type in _ACTION_TYPES
                ]
                if not previous_actions or previous_actions[-1].event_id != target_id:
                    raise _BlockedMigration(
                        "nonlatest_event_correction",
                        f"event correction {correction.event_id} does not target the latest action",
                    )
            elif target.event_type == "player_passed" or bool(
                target.payload.get("is_pass", False)
            ):
                raise _BlockedMigration(
                    "suit_correction_targets_pass",
                    f"suit correction {correction.event_id} targets a pass",
                )
            elif correction.event_type == "suit_corrected":
                target_cards = tuple(str(card) for card in target.payload.get("cards", ()))
                corrected_cards = correction.payload.get("cards", ())
                if not isinstance(corrected_cards, (list, tuple)) or validate_suit_correction(
                    target_cards,
                    tuple(str(card) for card in corrected_cards),
                ) is None:
                    raise _BlockedMigration(
                        "invalid_suit_correction",
                        f"suit correction {correction.event_id} does not prove the same action",
                    )
            corrections[target_id] = correction

        effective: list[LiveEvent] = []
        for event in events:
            if event.event_type not in _ACTION_TYPES:
                continue
            correction = corrections.get(event.event_id)
            if correction is None:
                effective.append(event)
                continue
            is_pass = (
                bool(correction.payload.get("is_pass", False))
                if correction.event_type == "event_correction"
                else False
            )
            cards = correction.payload.get("cards", ())
            if not isinstance(cards, (list, tuple)):
                raise _BlockedMigration(
                    "invalid_correction_cards",
                    f"correction {correction.event_id} cards are not an array",
                )
            if correction.actor not in {None, event.actor}:
                raise _BlockedMigration(
                    "correction_actor_mismatch",
                    f"correction {correction.event_id} actor does not match its target",
                )
            effective.append(
                replace(
                    event,
                    event_type="player_passed" if is_pass else "player_played",
                    payload={"cards": list(cards), "is_pass": is_pass},
                    confidence=correction.confidence,
                    source=correction.source,
                    evidence_refs=correction.evidence_refs or event.evidence_refs,
                )
            )
        return tuple(effective)

    @staticmethod
    def _turn_from_event(index: int, event: LiveEvent) -> TruthTurn:
        if event.actor not in SEATS:
            raise _BlockedMigration(
                "invalid_action_actor", f"action {event.event_id} has an invalid actor"
            )
        if event.turn_id != index:
            raise _BlockedMigration(
                "action_turn_mismatch",
                f"action {event.event_id} has turn {event.turn_id}, expected {index}",
            )
        if event.trick_id < 1:
            raise _BlockedMigration(
                "invalid_action_trick", f"action {event.event_id} has an invalid trick ID"
            )
        is_pass = event.event_type == "player_passed" or bool(
            event.payload.get("is_pass", False)
        )
        cards = event.payload.get("cards", ())
        if not isinstance(cards, (list, tuple)):
            raise _BlockedMigration(
                "invalid_action_cards", f"action {event.event_id} cards are not an array"
            )
        return TruthTurn(
            index=index,
            actor=event.actor,
            is_pass=is_pass,
            cards=() if is_pass else tuple(str(card) for card in cards),
            monotonic_ms=event.monotonic_ms,
            trick_id=event.trick_id,
            evidence=TruthEvidence(monotonic_ms=event.monotonic_ms),
            label_status="draft",
            provenance=LabelProvenance(source=MIGRATION_SOURCE),
        )

    @staticmethod
    def _normalize_source_actions(
        session_id: str,
        events: tuple[LiveEvent, ...],
        actions: tuple[LiveEvent, ...],
    ) -> tuple[tuple[LiveEvent, ...], tuple[str, ...]]:
        """Replay source actions and canonically repair one proven wind offset.

        A historical reducer used to leave the completed play on the table when
        its owner had just finished.  The corresponding timeline therefore
        carries an old trick id, and its first legal wind-catch lead is marked
        ``observed_table_mismatch``/``beats_table=false``.  This is safe to
        normalize only when the corrected reducer proves a one-trick offset and
        the actor is already the new lead.  Any other warning remains blocking.
        """

        effective_by_id = {event.event_id: event for event in actions}
        source = LiveReducer(session_id)
        normalized: list[LiveEvent] = []
        context_repairs: list[str] = []
        accepted_offset: int | None = None
        action_index = 0
        for event in events:
            if event.event_type in _CORRECTION_TYPES:
                continue
            if event.event_type in _ACTION_TYPES:
                event = effective_by_id[event.event_id]
                action_index += 1
                snapshot = source.snapshot()
                if snapshot.turn_id != event.turn_id:
                    raise _BlockedMigration(
                        "source_replay_position_mismatch",
                        f"source action {action_index} expects turn {event.turn_id}, "
                        f"reducer has {snapshot.turn_id}",
                    )
                offset = snapshot.trick_id - event.trick_id
                payload = event.payload
                warnings = tuple(
                    str(value)
                    for value in payload.get("integrity_warnings", ())
                    if str(value)
                )
                observed_mismatch = (
                    "observed_table_mismatch" in warnings
                    and payload.get("beats_table") is False
                )
                valid_wind_origin = (
                    offset == 1
                    and observed_mismatch
                    and event.actor == snapshot.current_player
                    and event.actor == snapshot.lead_player
                    and not bool(payload.get("is_pass", False))
                )
                if accepted_offset is None and valid_wind_origin:
                    accepted_offset = offset
                    context_repairs.append(
                        f"{event.event_id}: trick {event.trick_id} -> "
                        f"{snapshot.trick_id} after proven wind catch"
                    )
                if offset != (accepted_offset or 0):
                    raise _BlockedMigration(
                        "source_replay_position_mismatch",
                        f"source action {action_index} expects trick {event.trick_id}, "
                        f"reducer has {snapshot.trick_id}",
                    )
                if warnings or payload.get("beats_table") is False or payload.get(
                    "interpretation_ambiguous"
                ) is True:
                    if not valid_wind_origin:
                        raise _BlockedMigration(
                            "unsafe_timeline_integrity",
                            f"source action {action_index} has unresolved integrity evidence",
                        )
                event = replace(event, trick_id=snapshot.trick_id)
                normalized.append(event)
            elif event.event_type not in {
                "initial_state_confirmed",
                "lead_player_confirmed",
                "player_finished",
            }:
                continue
            try:
                source.apply(event)
            except Exception as exc:
                raise _BlockedMigration(
                    "source_reducer_replay_failed",
                    f"source reducer rejected {event.event_id}: {exc}",
                ) from exc
        return tuple(normalized), tuple(context_repairs)

    @staticmethod
    def _validate_truth_replay(
        session_id: str,
        actions: tuple[LiveEvent, ...],
        draft: TruthLog,
    ) -> None:
        truth = LiveReducer(session_id)
        truth_events = draft.to_events(session_id=session_id)
        try:
            truth.apply(truth_events[0])
        except Exception as exc:
            raise _BlockedMigration(
                "truth_reducer_initial_failed",
                f"generated initial state was rejected: {exc}",
            ) from exc
        for index, (source_action, turn, generated) in enumerate(
            zip(actions, draft.turns, truth_events[1:]), start=1
        ):
            snapshot = truth.snapshot()
            if (
                snapshot.turn_id != source_action.turn_id
                or snapshot.trick_id != source_action.trick_id
                or turn.index != index
                or turn.trick_id != source_action.trick_id
                or turn.monotonic_ms != source_action.monotonic_ms
                or generated.monotonic_ms != source_action.monotonic_ms
            ):
                raise _BlockedMigration(
                    "truth_replay_position_mismatch",
                    f"generated turn {index} cannot preserve source turn/trick/timing",
                )
            try:
                truth.apply(generated)
            except Exception as exc:
                raise _BlockedMigration(
                    "truth_reducer_replay_failed",
                    f"generated action {index} was rejected: {exc}",
                ) from exc
            replayed = truth.snapshot().play_history[-1]
            if (
                replayed.player != turn.actor
                or replayed.is_pass != turn.is_pass
                or replayed.cards != turn.cards
            ):
                raise _BlockedMigration(
                    "truth_replay_action_mismatch",
                    f"generated action {index} changed actor/pass/cards semantics",
                )

    @staticmethod
    def _atomic_create_json(path: Path, document: dict[str, object]) -> None:
        """Atomically publish a new JSON file without ever replacing a peer."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        try:
            with temp_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_create_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temp_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temp_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


__all__ = [
    "MIGRATION_SOURCE",
    "ExistingTruthRepairInspection",
    "ExistingTruthRepairResult",
    "TimelineTruthMigrationService",
    "TimelineTruthRootInspection",
    "TimelineTruthRootMigration",
    "TimelineTruthSessionInspection",
    "TimelineTruthSessionMigration",
]
