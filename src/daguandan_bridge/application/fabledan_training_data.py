from __future__ import annotations

"""Confirmed real-game data export for FableDan offline training.

This module deliberately stops before optimisation or model replacement.  A
real session is first sealed, then a reviewer explicitly confirms the derived
outcome in Replay.  Only then are immutable DMC-style samples published.
"""

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from ..live.session_store import read_json_lines
from ..storage import atomic_write_json


TRAINING_REVIEW_SCHEMA = "fabledan.training-review/1"
TRAINING_SAMPLE_SCHEMA = "fabledan.offline-dmc-sample/1"
FEATURE_SCHEMA = "fabledan-token48-feat80/v1"
_SEATS = ("self", "right", "opposite", "left")
_PARTNER = {"self": "opposite", "opposite": "self", "right": "left", "left": "right"}


@dataclass(frozen=True)
class FableDanTrainingReviewResult:
    session: Path
    review_path: Path | None
    status: str
    message: str
    candidate_count: int
    eligible_count: int
    skipped: tuple[dict[str, str], ...]
    payload: dict[str, object]

    @property
    def can_confirm(self) -> bool:
        return self.status == "draft" and self.eligible_count > 0


@dataclass(frozen=True)
class FableDanTrainingExportResult:
    session: Path
    review_path: Path
    samples_path: Path
    manifest_path: Path
    sample_count: int
    skipped: tuple[dict[str, str], ...]
    payload: dict[str, object]


class FableDanTrainingDataService:
    """Create auditable real-game FableDan samples without touching weights."""

    def initialize_review(self, session: Path) -> FableDanTrainingReviewResult:
        """Create/update the draft review after a session is sealed.

        Sessions without FableDan decisions are intentionally ignored so
        DanZero-only and old generic sessions gain no unrelated artifact.
        """

        return self.inspect_session(session, write=True)

    def inspect_session(
        self,
        session: Path,
        *,
        write: bool = True,
    ) -> FableDanTrainingReviewResult:
        session = Path(session)
        review_path = session / "fabledan_training_review.json"
        manifest = _read_json_object(session / "manifest.json")
        decisions = read_json_lines(session / "decisions.jsonl")
        advice = read_json_lines(session / "advice.jsonl")
        fabledan = [
            decision
            for decision in decisions
            if _is_fabledan_decision(decision, advice)
        ]
        if not fabledan:
            return FableDanTrainingReviewResult(
                session=session,
                review_path=None,
                status="not_applicable",
                message="本局没有可用于 FableDan 的模型决策记录。",
                candidate_count=0,
                eligible_count=0,
                skipped=(),
                payload={},
            )

        timeline = read_json_lines(session / "timeline.jsonl")
        outcome = _infer_outcome(timeline)
        source = _source_fingerprints(session)
        existing = _read_json_object(review_path) if review_path.exists() else {}
        existing_source = existing.get("source") if isinstance(existing, dict) else None
        existing_is_current = isinstance(existing_source, dict) and existing_source == source
        _, skipped = self._candidate_samples(
            session=session,
            decisions=fabledan,
            advice=advice,
            outcome=outcome,
            require_valid=False,
        )
        eligible_count = len(fabledan) - len(skipped)
        blockers: list[str] = []
        if manifest.get("status") != "sealed":
            blockers.append("对局尚未封局，不能固定训练标签。")
        if outcome.get("status") != "complete":
            blockers.append(str(outcome.get("reason") or "未识别到可信的终局名次。"))
        if eligible_count <= 0:
            blockers.append("没有通过编码与实际动作一致性校验的决策。")

        if existing_is_current and existing.get("status") == "verified":
            status = "verified"
            message = "该局训练数据已人工确认；源文件未变化。"
            payload = dict(existing)
        else:
            status = "draft"
            message = (
                "可在核对回放后确认并导出训练样本。"
                if not blockers
                else "；".join(blockers)
            )
            payload = {
                "schema": TRAINING_REVIEW_SCHEMA,
                "session_id": session.name,
                "status": status,
                "created_at": (
                    existing.get("created_at")
                    if isinstance(existing.get("created_at"), str)
                    else _now_text()
                ),
                "source": source,
                "outcome": outcome,
                "candidates": {
                    "fabledan_decision_count": len(fabledan),
                    "eligible_decision_count": eligible_count,
                    "skipped": list(skipped),
                },
                "eligibility": {
                    "sealed": manifest.get("status") == "sealed",
                    "outcome_complete": outcome.get("status") == "complete",
                    "can_confirm": not blockers,
                    "blockers": blockers,
                },
            }
            if write:
                atomic_write_json(review_path, payload)

        return FableDanTrainingReviewResult(
            session=session,
            review_path=review_path,
            status=status,
            message=message,
            candidate_count=len(fabledan),
            eligible_count=eligible_count,
            skipped=tuple(skipped),
            payload=payload,
        )

    def confirm_and_export(self, session: Path) -> FableDanTrainingExportResult:
        """Publish samples only after explicit human confirmation in Replay."""

        session = Path(session)
        persisted = _read_json_object(session / "fabledan_training_review.json")
        if not persisted:
            raise ValueError("请先在回放中检查训练资格，再确认导出。")
        if persisted.get("source") != _source_fingerprints(session):
            raise ValueError("检查后对局源数据已变化，请重新核对回放。")
        inspected = self.inspect_session(session, write=False)
        if inspected.status == "not_applicable":
            raise ValueError(inspected.message)
        if not inspected.can_confirm and inspected.status != "verified":
            raise ValueError(inspected.message)

        payload = dict(inspected.payload)
        outcome = payload.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("status") != "complete":
            raise ValueError("终局标签未完成，不能导出 FableDan 训练样本。")
        advice = read_json_lines(session / "advice.jsonl")
        decisions = [
            item
            for item in read_json_lines(session / "decisions.jsonl")
            if _is_fabledan_decision(item, advice)
        ]
        samples, skipped = self._candidate_samples(
            session=Path(session),
            decisions=decisions,
            advice=advice,
            outcome=outcome,
            require_valid=True,
        )
        if not samples:
            raise ValueError("没有通过质量校验的 FableDan 决策，未生成训练数据。")

        derived = session / "derived"
        samples_path = derived / "fabledan_training_samples.jsonl"
        manifest_path = derived / "fabledan_training_manifest.json"
        confirmed_at = _now_text()
        source = _source_fingerprints(session)
        if payload.get("source") != source:
            raise ValueError("对局源数据在确认期间发生变化，请重新检查回放。")
        _atomic_write_json_lines(samples_path, samples)
        export_manifest = {
            "schema": "fabledan.training-export/1",
            "session_id": session.name,
            "created_at": confirmed_at,
            "review": "fabledan_training_review.json",
            "source": source,
            "sample_schema": TRAINING_SAMPLE_SCHEMA,
            "sample_count": len(samples),
            "skipped": skipped,
            "model_replacement": "not_performed",
        }
        atomic_write_json(manifest_path, export_manifest)
        payload.update(
            {
                "status": "verified",
                "confirmed_at": confirmed_at,
                "confirmation": "manual_replay_confirmation",
                "export": {
                    "samples_path": str(samples_path.relative_to(session)),
                    "manifest_path": str(manifest_path.relative_to(session)),
                    "sample_count": len(samples),
                    "skipped": skipped,
                },
            }
        )
        review_path = session / "fabledan_training_review.json"
        atomic_write_json(review_path, payload)
        return FableDanTrainingExportResult(
            session=session,
            review_path=review_path,
            samples_path=samples_path,
            manifest_path=manifest_path,
            sample_count=len(samples),
            skipped=tuple(skipped),
            payload=payload,
        )

    def _candidate_samples(
        self,
        *,
        session: Path,
        decisions: list[dict[str, object]],
        advice: list[dict[str, object]],
        outcome: dict[str, object],
        require_valid: bool,
    ) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
        advice_by_request = {
            str(item.get("request_id")): item
            for item in advice
            if isinstance(item.get("request_id"), str)
        }
        samples: list[dict[str, object]] = []
        skipped: list[dict[str, str]] = []
        for decision in decisions:
            decision_id = str(decision.get("decision_id", ""))
            try:
                sample = _sample_from_decision(
                    session=session,
                    decision=decision,
                    advice=advice_by_request.get(str(decision.get("request_id", ""))),
                    outcome=outcome,
                )
            except ValueError as exc:
                skipped.append({"decision_id": decision_id, "reason": str(exc)})
                continue
            samples.append(sample)
        if require_valid and not samples:
            return [], skipped
        return samples, skipped


def _sample_from_decision(
    *,
    session: Path,
    decision: dict[str, object],
    advice: dict[str, object] | None,
    outcome: dict[str, object],
) -> dict[str, object]:
    decision_id = str(decision.get("decision_id", "")).strip()
    if not decision_id:
        raise ValueError("缺少 decision_id")
    actual = decision.get("actual_action")
    if not isinstance(actual, dict) or not decision.get("actual_action_event_id"):
        raise ValueError("缺少已关联的实际动作")
    legal = decision.get("legal_actions")
    if not isinstance(legal, list) or not legal or not all(isinstance(x, dict) for x in legal):
        raise ValueError("缺少 FableDan 合法动作集合")
    training_input, source_kind = _extract_training_input(decision, advice)
    tokens, features = _validate_training_input(training_input, legal_count=len(legal))
    chosen_index = _match_actual_action(
        actual,
        legal,
        _actual_event(session, str(decision["actual_action_event_id"])),
    )
    if chosen_index is None:
        raise ValueError("实际动作无法唯一映射到当时的 FableDan 合法动作")
    selected = legal[chosen_index]
    raw_reward = int(outcome["raw_team_reward"])
    if raw_reward not in {-3, -2, -1, 1, 2, 3}:
        raise ValueError("终局奖励不符合掼蛋计分规则")
    return {
        "schema": TRAINING_SAMPLE_SCHEMA,
        "sample_id": f"{session.name}:{decision_id}",
        "session_id": session.name,
        "split": _session_split(session.name),
        "split_group": session.name,
        "rules": {
            "ruleset": "fabledan-standard-no-tribute/v1",
            "player_mapping": {"self": 0, "right": 1, "opposite": 2, "left": 3},
        },
        "decision": {
            "decision_id": decision_id,
            "request_id": decision.get("request_id"),
            "actual_action_event_id": decision.get("actual_action_event_id"),
            "actual_turn_id": decision.get("actual_turn_id"),
            "actual_trick_id": decision.get("actual_trick_id"),
            "chosen_legal_index": chosen_index,
            "actual_action": {
                "cards": [str(card) for card in actual.get("cards", ())],
                "is_pass": bool(actual.get("is_pass", False)),
            },
            "legal_action": selected,
            "model_advice": decision.get("model_advice"),
        },
        "encoding": {
            "feature_schema": FEATURE_SCHEMA,
            "tokens": tokens,
            "tokens_sha256": _tokens_hash(tokens),
            "chosen_feature": features[chosen_index],
            "features_sha256": _features_hash(features),
            "legal_action_count": len(legal),
        },
        "target": {
            "raw_team_reward": raw_reward,
            "normalized_dmc_return": raw_reward / 3.0,
            "terminal_kind": outcome.get("terminal_kind"),
            "finish_order": outcome.get("finish_order"),
        },
        "provenance": {
            "label_status": "verified",
            "label_source": "manual_replay_confirmation",
            "decision_input_source": source_kind,
            "model_hash": training_input.get("model_hash"),
            "adapter_schema": training_input.get("adapter_schema"),
            "upstream_commit": training_input.get("upstream_commit"),
        },
    }


def _extract_training_input(
    decision: dict[str, object], advice: dict[str, object] | None) -> tuple[dict[str, object], str]:
    direct = decision.get("fabledan_training_input")
    if isinstance(direct, dict):
        return direct, "decision_snapshot"
    engine_input = advice.get("engine_input") if isinstance(advice, dict) else None
    if isinstance(engine_input, dict):
        snapshot = engine_input.get("fabledan_training_input")
        if isinstance(snapshot, dict):
            return snapshot, "advice_snapshot"
        trace = engine_input.get("fabledan_trace")
        if isinstance(trace, dict):
            encoding = trace.get("encoding")
            output = trace.get("model_output")
            if isinstance(encoding, dict):
                token_info = encoding.get("tokens")
                feature_info = encoding.get("features")
                if isinstance(token_info, dict) and isinstance(feature_info, dict):
                    return {
                        "schema": "fabledan-decision-input/legacy-trace",
                        "feature_schema": FEATURE_SCHEMA,
                        "tokens": token_info.get("tokens"),
                        "tokens_sha256": token_info.get("tokens_sha256"),
                        "features": feature_info.get("feats"),
                        "features_sha256": feature_info.get("feats_sha256"),
                        "legal_action_count": len(decision.get("legal_actions", ())),
                        "model_hash": output.get("model_hash") if isinstance(output, dict) else None,
                        "adapter_schema": engine_input.get("schema"),
                        "upstream_commit": engine_input.get("upstream_commit"),
                    }, "legacy_full_trace"
    raise ValueError("缺少 FableDan 编码快照；此旧记录未开启完整诊断，不能安全导出")


def _validate_training_input(
    raw: dict[str, object],
    *,
    legal_count: int,
) -> tuple[list[int], list[list[float]]]:
    if raw.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("FableDan 特征版本不兼容")
    tokens_raw = raw.get("tokens")
    features_raw = raw.get("features")
    if not isinstance(tokens_raw, list) or not tokens_raw:
        raise ValueError("FableDan token 序列缺失")
    if not isinstance(features_raw, list) or len(features_raw) != legal_count:
        raise ValueError("FableDan 特征行数与合法动作数量不一致")
    try:
        tokens = [int(value) for value in tokens_raw]
        features = [[float(value) for value in row] for row in features_raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("FableDan 编码中包含非数值") from exc
    if any(len(row) != 80 for row in features):
        raise ValueError("FableDan 每个动作特征必须为 80 维")
    if not all(math.isfinite(value) for row in features for value in row):
        raise ValueError("FableDan 特征包含无效浮点数")
    expected_tokens = raw.get("tokens_sha256")
    expected_features = raw.get("features_sha256")
    if expected_tokens and str(expected_tokens) != _tokens_hash(tokens):
        raise ValueError("FableDan token 哈希校验失败")
    if expected_features and str(expected_features) != _features_hash(features):
        raise ValueError("FableDan 特征哈希校验失败")
    return tokens, features


def _match_actual_action(
    actual: dict[str, object],
    legal: list[dict[str, object]],
    event: dict[str, object] | None,
) -> int | None:
    is_pass = bool(actual.get("is_pass", False))
    actual_cards = sorted(str(card) for card in actual.get("cards", ()))
    candidates = [
        index
        for index, item in enumerate(legal)
        if bool(item.get("is_pass", False)) == is_pass
        and sorted(str(card) for card in item.get("cards", ())) == actual_cards
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    semantic = _selected_semantic(event)
    if semantic is None:
        return None
    semantic_type = _normalise_type(semantic.get("move_type"))
    narrowed = [
        index
        for index in candidates
        if _normalise_type(legal[index].get("type")) == semantic_type
    ]
    return narrowed[0] if len(narrowed) == 1 else None


def _selected_semantic(event: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(event, dict):
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    value = payload.get("selected_interpretation")
    return value if isinstance(value, dict) else None


def _normalise_type(value: object) -> str:
    aliases = {
        "single": "single",
        "pair": "pair",
        "triple": "triple",
        "straightflush": "straightflush",
        "straight_flush": "straightflush",
        "pass": "pass",
    }
    raw = "".join(char for char in str(value or "").lower() if char.isalnum() or char == "_")
    return aliases.get(raw, raw)


def _actual_event(session: Path, event_id: str) -> dict[str, object] | None:
    for event in read_json_lines(session / "timeline.jsonl"):
        if event.get("event_id") == event_id:
            return event
    return None


def _infer_outcome(timeline: list[dict[str, object]]) -> dict[str, object]:
    placements: dict[str, str] = {}
    for event in timeline:
        if event.get("event_type") != "player_finished":
            continue
        actor = event.get("actor")
        payload = event.get("payload")
        placement = payload.get("placement") if isinstance(payload, dict) else None
        if actor not in _SEATS or placement not in {"first", "second", "third", "last"}:
            continue
        if placement in placements and placements[placement] != actor:
            return {"status": "incomplete", "reason": "终局名次记录互相冲突。"}
        placements[str(placement)] = str(actor)
    first = placements.get("first")
    second = placements.get("second")
    if first and second and _PARTNER[first] == second:
        return _outcome_payload(
            first=first,
            finish_order=[first, second],
            terminal_kind="double_down",
        )
    if all(name in placements for name in ("first", "second", "third", "last")):
        order = [placements[name] for name in ("first", "second", "third", "last")]
        return _outcome_payload(first=order[0], finish_order=order, terminal_kind="full_ranking")
    return {
        "status": "incomplete",
        "reason": "未识别到完整终局名次，或无法确认双下。",
        "recorded_placements": placements,
    }


def _outcome_payload(*, first: str, finish_order: list[str], terminal_kind: str) -> dict[str, object]:
    partner_position = finish_order.index(_PARTNER[first])
    score = {1: 3, 2: 2, 3: 1}[partner_position]
    self_won = first in {"self", "opposite"}
    raw_reward = score if self_won else -score
    return {
        "status": "complete",
        "terminal_kind": terminal_kind,
        "finish_order": finish_order,
        "winning_team": "self_team" if self_won else "opponents",
        "raw_team_reward": raw_reward,
        "normalized_dmc_return": raw_reward / 3.0,
    }


def _is_fabledan_decision(decision: dict[str, object], advice: list[dict[str, object]]) -> bool:
    if isinstance(decision.get("fabledan_training_input"), dict):
        return True
    if str(decision.get("feature_schema", "")).startswith("fabledan-"):
        return True
    model_advice = decision.get("model_advice")
    if isinstance(model_advice, dict) and str(model_advice.get("strategy", "")).startswith("fabledan"):
        return True
    request_id = decision.get("request_id")
    return any(
        item.get("request_id") == request_id
        and str(item.get("strategy", "")).startswith("fabledan")
        for item in advice
    )


def _source_fingerprints(session: Path) -> dict[str, str]:
    return {
        "timeline_sha256": _sha256_file(session / "timeline.jsonl"),
        "decisions_sha256": _sha256_file(session / "decisions.jsonl"),
        "advice_sha256": _sha256_file(session / "advice.jsonl"),
        "manifest_sha256": _sha256_file(session / "manifest.json"),
    }


def _tokens_hash(tokens: list[int]) -> str:
    return sha256(np.asarray(tokens, dtype="<i8").tobytes()).hexdigest()


def _features_hash(features: list[list[float]]) -> str:
    return sha256(np.ascontiguousarray(np.asarray(features, dtype="<f4")).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须为 JSON 对象")
    return value


def _atomic_write_json_lines(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for item in records
    )
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _session_split(session_id: str) -> Literal["train", "validation", "test"]:
    bucket = int(sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
