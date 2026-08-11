from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal
from uuid import uuid4

from ..live.truth_log import TruthLog, load_truth_log


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    severity: Severity
    session_id: str
    path: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "session_id": self.session_id,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class DatasetValidationReport:
    sessions: tuple[str, ...]
    findings: tuple[ValidationFinding, ...]

    @property
    def error_count(self) -> int:
        return sum(item.severity == "error" for item in self.findings)

    @property
    def warning_count(self) -> int:
        return sum(item.severity == "warning" for item in self.findings)

    @property
    def valid(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "session_count": len(self.sessions),
            "sessions": list(self.sessions),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "findings": [item.to_dict() for item in self.findings],
        }


@dataclass(frozen=True)
class DatasetExportResult:
    sessions: tuple[str, ...]
    artifacts: tuple[str, ...]
    sample_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "sessions": list(self.sessions),
            "artifacts": list(self.artifacts),
            "sample_counts": dict(self.sample_counts),
        }


class DatasetUseCase:
    """Validate truth/decision data and publish deterministic derived datasets."""

    def validate_session(self, session: Path) -> DatasetValidationReport:
        session = Path(session)
        session_id = session.name
        findings: list[ValidationFinding] = []
        truth_path = session / "truth_log.json"
        if not truth_path.is_file():
            findings.append(self._finding("TRUTH_MISSING", "error", session_id, truth_path, "truth_log.json is missing"))
            return DatasetValidationReport((session_id,), tuple(findings))
        try:
            truth = load_truth_log(truth_path, session_id=session_id)
        except Exception as exc:
            findings.append(self._finding("TRUTH_INVALID", "error", session_id, truth_path, str(exc)))
            return DatasetValidationReport((session_id,), tuple(findings))
        findings.extend(self._validate_truth(session, truth))
        findings.extend(self._validate_decisions(session, truth))
        return DatasetValidationReport((session_id,), tuple(findings))

    def validate_root(self, sessions_root: Path) -> DatasetValidationReport:
        sessions = self._session_directories(sessions_root)
        findings: list[ValidationFinding] = []
        for session in sessions:
            findings.extend(self.validate_session(session).findings)
        if not sessions:
            findings.append(self._finding("SESSIONS_EMPTY", "error", "", Path(sessions_root), "no session directories found"))
        return DatasetValidationReport(tuple(path.name for path in sessions), tuple(findings))

    def export_session(self, session: Path) -> DatasetExportResult:
        session = Path(session)
        report = self.validate_session(session)
        if not report.valid:
            raise ValueError(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        truth = load_truth_log(session / "truth_log.json", session_id=session.name)
        vision = self._vision_samples(session, truth)
        policy = self._policy_samples(session, truth)
        return self._publish(session, vision, policy)

    def export_root(self, sessions_root: Path) -> DatasetExportResult:
        report = self.validate_root(sessions_root)
        if not report.valid:
            raise ValueError(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
        sessions = self._session_directories(sessions_root)
        results = [self.export_session(session) for session in sessions]
        return DatasetExportResult(
            sessions=tuple(session for result in results for session in result.sessions),
            artifacts=tuple(path for result in results for path in result.artifacts),
            sample_counts={
                "vision": sum(result.sample_counts["vision"] for result in results),
                "policy": sum(result.sample_counts["policy"] for result in results),
            },
        )

    def _validate_truth(self, session: Path, truth: TruthLog) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        previous_trick = 0
        for turn in truth.turns:
            prefix = f"turns/{turn.index}"
            if turn.trick_id < previous_trick or turn.trick_id <= 0:
                findings.append(self._finding("TRICK_ID_INVALID", "error", truth.source_session_id, session / prefix, "trick ids must be positive and nondecreasing"))
            previous_trick = turn.trick_id
            if turn.label_status == "verified" and not turn.provenance.source:
                findings.append(self._finding("PROVENANCE_MISSING", "error", truth.source_session_id, session / prefix, "verified label requires provenance.source"))
            if any(card.endswith("?") for card in turn.cards) and "unknown_suit" not in turn.uncertainty:
                findings.append(self._finding("UNCERTAINTY_MISSING", "error", truth.source_session_id, session / prefix, "unknown-suit cards require uncertainty=unknown_suit"))
            if turn.label_status == "verified" and not turn.evidence.frame_indices:
                findings.append(self._finding("EVIDENCE_MISSING", "error", truth.source_session_id, session / prefix, "verified action requires at least one evidence frame"))
        if truth.label_status == "verified" and not truth.provenance.source:
            findings.append(self._finding("PROVENANCE_MISSING", "error", truth.source_session_id, session / "truth_log.json", "verified truth log requires provenance.source"))
        return findings

    def _validate_decisions(self, session: Path, truth: TruthLog) -> list[ValidationFinding]:
        path = session / "decisions.jsonl"
        if not path.exists():
            return []
        findings: list[ValidationFinding] = []
        try:
            decisions = _read_json_lines(path)
        except Exception as exc:
            return [self._finding("DECISIONS_INVALID", "error", truth.source_session_id, path, str(exc))]
        ids: set[str] = set()
        for index, decision in enumerate(decisions, start=1):
            location = path / str(index)
            decision_id = str(decision.get("decision_id", ""))
            if not decision_id or decision_id in ids:
                findings.append(self._finding("DECISION_ID_INVALID", "error", truth.source_session_id, location, "decision_id must be nonempty and unique"))
            ids.add(decision_id)
            if decision.get("actor") != "self":
                findings.append(self._finding("DECISION_ACTOR_INVALID", "error", truth.source_session_id, location, "policy decisions must belong to self"))
            if not isinstance(decision.get("state_before"), dict):
                findings.append(self._finding("DECISION_STATE_MISSING", "error", truth.source_session_id, location, "state_before is required"))
            legal = decision.get("legal_actions")
            if not isinstance(legal, list) or not legal:
                findings.append(self._finding("LEGAL_ACTIONS_MISSING", "error", truth.source_session_id, location, "legal_actions are required"))
            features = decision.get("features_567")
            if features is not None and (
                not isinstance(features, list)
                or any(not isinstance(row, list) or len(row) != 567 for row in features)
            ):
                findings.append(self._finding("FEATURES_INVALID", "error", truth.source_session_id, location, "every feature row must contain 567 values"))
            actual = decision.get("actual_action")
            if actual is not None and not decision.get("actual_action_event_id"):
                findings.append(self._finding("ACTION_LINK_MISSING", "error", truth.source_session_id, location, "actual_action requires actual_action_event_id"))
        return findings

    def _vision_samples(self, session: Path, truth: TruthLog) -> list[dict[str, object]]:
        if truth.label_status != "verified":
            return []
        split = session_split(truth.source_session_id)
        video_path = session / truth.source_video
        video_hash = _sha256(video_path) if video_path.is_file() else ""
        samples: list[dict[str, object]] = []
        roi_names = {"self": "my_play", "right": "right_play", "opposite": "opposite_play", "left": "left_play"}
        for turn in truth.turns:
            if turn.label_status != "verified" or not turn.evidence.frame_indices or turn.uncertainty:
                continue
            for frame_index in sorted(set(turn.evidence.frame_indices)):
                samples.append({
                    "schema": "guandan.vision-sample/1",
                    "sample_id": f"{truth.source_session_id}:frame_{frame_index}:{turn.actor}:turn_{turn.index}",
                    "session_id": truth.source_session_id,
                    "split": split,
                    "split_group": truth.source_session_id,
                    "frame": {
                        "video_path": truth.source_video,
                        "video_sha256": video_hash,
                        "frame_index": frame_index,
                        "monotonic_ms": turn.evidence.monotonic_ms,
                    },
                    "roi": {"name": turn.evidence.roi_name or roi_names[turn.actor]},
                    "labels": {"cards": list(turn.cards), "is_pass": turn.is_pass},
                    "label_status": "verified",
                    "provenance": turn.provenance.to_dict(),
                })
        return samples

    def _policy_samples(self, session: Path, truth: TruthLog) -> list[dict[str, object]]:
        if truth.label_status != "verified" or not truth.outcome.complete:
            return []
        split = session_split(truth.source_session_id)
        samples: list[dict[str, object]] = []
        verified_turns = {
            turn.index: turn
            for turn in truth.turns
            if turn.actor == "self" and turn.label_status == "verified"
        }
        for raw in _read_json_lines(session / "decisions.jsonl"):
            if raw.get("actor") != "self":
                continue
            actual = raw.get("actual_action")
            if not isinstance(actual, dict) or not raw.get("actual_action_event_id"):
                continue
            truth_turn = verified_turns.get(int(raw.get("actual_turn_id", 0) or 0))
            truth_verified = bool(
                truth_turn
                and bool(actual.get("is_pass")) == truth_turn.is_pass
                and tuple(str(card) for card in actual.get("cards", ())) == truth_turn.cards
            )
            if raw.get("label_status") != "verified" and not truth_verified:
                continue
            if raw.get("features_567") is None:
                continue
            samples.append({
                "schema": "guandan.policy-decision/1",
                "feature_schema": "danzero-567/v1",
                "decision_id": str(raw["decision_id"]),
                "session_id": truth.source_session_id,
                "split": split,
                "split_group": truth.source_session_id,
                "state_before": raw["state_before"],
                "legal_actions": raw["legal_actions"],
                "features_567": raw["features_567"],
                "chosen_action": actual,
                "actual_action_event_id": raw["actual_action_event_id"],
                "choice_source": "human_verified",
                "model_advice": raw.get("model_advice"),
                "outcome": truth.outcome.to_dict(),
                "label_status": "verified",
            })
        return samples

    def _publish(self, session: Path, vision: list[dict[str, object]], policy: list[dict[str, object]]) -> DatasetExportResult:
        derived = session / "derived"
        staging = session / f".derived.{uuid4().hex}.tmp"
        backup = session / f".derived.{uuid4().hex}.bak"
        staging.mkdir()
        try:
            artifacts = {
                "vision_samples.jsonl": _jsonl_bytes(vision),
                "policy_samples.jsonl": _jsonl_bytes(policy),
            }
            for name, payload in artifacts.items():
                (staging / name).write_bytes(payload)
            manifest = {
                "schema": "guandan.dataset-manifest/1",
                "session_id": session.name,
                "split": session_split(session.name),
                "split_group": session.name,
                "artifacts": {
                    name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                    for name, payload in sorted(artifacts.items())
                },
                "sample_counts": {"vision": len(vision), "policy": len(policy)},
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if derived.exists():
                derived.replace(backup)
            staging.replace(derived)
            if backup.exists():
                shutil.rmtree(backup)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists() and not derived.exists():
                backup.replace(derived)
            raise
        paths = tuple(str(derived / name) for name in ("vision_samples.jsonl", "policy_samples.jsonl", "manifest.json"))
        return DatasetExportResult((session.name,), paths, {"vision": len(vision), "policy": len(policy)})

    @staticmethod
    def _finding(code: str, severity: Severity, session_id: str, path: Path, message: str) -> ValidationFinding:
        return ValidationFinding(code, severity, session_id, str(path), message)

    @staticmethod
    def _session_directories(root: Path) -> tuple[Path, ...]:
        root = Path(root)
        return tuple(sorted((path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")), key=lambda path: path.name)) if root.is_dir() else ()


def session_split(session_id: str) -> str:
    bucket = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _jsonl_bytes(records: Iterable[dict[str, object]]) -> bytes:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if not isinstance(value, dict):
            raise ValueError("every JSONL record must be an object")
        records.append(value)
    return records
