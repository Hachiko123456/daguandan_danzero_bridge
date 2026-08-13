from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
from time import perf_counter
from typing import Callable, Literal
from uuid import uuid4

from ..advisor_strategy import (
    EVALUATION_STRATEGY_OPTIONS,
    build_evaluation_advisor,
    normalize_evaluation_strategy,
)
from ..application.ports import AdvicePort
from ..danzero.rules import actions_for_cards, play_beats_table
from ..danzero.state import RANKS
from ..domain.advice import AdviceResult, StrategyExecutionTrace
from ..live.reducer import LiveReducer
from ..live.truth_log import TruthLog, TruthTurn, load_truth_log, truth_log_from_dict


FinalStatus = Literal[
    "completed", "completed_with_errors", "blocked", "cancelled", "failed"
]
FINAL_STATUSES = frozenset(
    {"completed", "completed_with_errors", "blocked", "cancelled", "failed"}
)
_DISPLAY_NAMES = dict(EVALUATION_STRATEGY_OPTIONS)
_BINDINGS = {
    "danzero_model": ("danzero", "danzero"),
    "fabledan_model": ("fabledan-numpy", "numpy"),
    "fabledan_rule": ("fabledan-rule", "rule"),
}


@dataclass(frozen=True)
class EvaluationIssue:
    code: str
    message: str
    turn_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        raw: dict[str, object] = {"code": self.code, "message": self.message}
        if self.turn_id is not None:
            raw["turn_id"] = self.turn_id
        return raw


@dataclass(frozen=True)
class EvaluationTruthValidation:
    """Strict replay validation shared by evaluation and guarded repair."""

    issues: tuple[EvaluationIssue, ...]
    self_contexts: tuple[str, ...]
    finish_order: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.issues and bool(self.self_contexts)


@dataclass(frozen=True)
class EvaluationProgress:
    phase: str
    completed: int
    total: int
    message: str = ""


@dataclass(frozen=True)
class PreparedEvaluation:
    session_directory: Path
    strategy_id: str
    frozen_truth: TruthLog | None
    input_sha256: str | None
    truth_source: str
    advisor: AdvicePort | None
    strategy_audit: dict[str, object]
    issues: tuple[EvaluationIssue, ...]
    eligible_self_decisions: int
    self_contexts: tuple[str, ...]
    finish_order: tuple[str, ...]
    warmup_latency_ms: float = 0.0

    @property
    def ready(self) -> bool:
        return self.frozen_truth is not None and not self.issues and self.advisor is not None


@dataclass(frozen=True)
class EvaluationInputValidation:
    """Frozen, strategy-independent eligibility result for one evaluation input."""

    session_directory: Path
    frozen_truth: TruthLog | None
    input_sha256: str | None
    truth_source: str
    issues: tuple[EvaluationIssue, ...]
    eligible_self_decisions: int
    self_contexts: tuple[str, ...]
    finish_order: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.frozen_truth is not None and not self.issues


@dataclass(frozen=True)
class EvaluationRunResult:
    run_id: str
    status: FinalStatus
    report_directory: Path
    summary: dict[str, object]
    decisions: tuple[dict[str, object], ...]


class ModelEvaluationService:
    """Freeze, validate and teacher-force one TruthLog through an advisor."""

    schema = "guandan.model-evaluation/v1"
    adapter_version = "model-evaluation/v1"

    def __init__(
        self,
        *,
        profiles_root: Path | str | None = None,
        profile_name: str = "tencent_daguandan",
        advisor_factory: Callable[[str], AdvicePort] | None = None,
    ) -> None:
        self.profiles_root = Path(profiles_root) if profiles_root is not None else None
        self.profile_name = str(profile_name)
        self._advisor_factory = advisor_factory

    def prepare(
        self,
        session_directory: Path | str,
        *,
        strategy_id: str = "danzero_model",
    ) -> PreparedEvaluation:
        validated = self.validate_input(session_directory)
        session = validated.session_directory
        strategy = normalize_evaluation_strategy(strategy_id)
        issues = list(validated.issues)
        advisor: AdvicePort | None = None
        warmup_ms = 0.0
        strategy_audit: dict[str, object] = {
            "strategy_id": strategy,
            "display_name": _DISPLAY_NAMES[strategy],
            "expected_strategy": _BINDINGS[strategy][0],
            "backend": _BINDINGS[strategy][1],
            "adapter_version": self.adapter_version,
        }
        # Loading model resources belongs to the explicit start path.  Invalid
        # input never incurs model startup and cannot obscure its real errors.
        if validated.ready:
            try:
                advisor = self._build_advisor(strategy, session)
                warmup_started = perf_counter()
                advisor.initialize()
                warmup_ms = (perf_counter() - warmup_started) * 1_000
                audit_method = getattr(advisor, "audit_info", None)
                if callable(audit_method):
                    strategy_audit.update(_json_safe(audit_method()))
                _validate_prepared_binding(strategy, strategy_audit)
            except Exception as exc:
                audit_method = getattr(advisor, "audit_info", None)
                if callable(audit_method):
                    try:
                        strategy_audit.update(_json_safe(audit_method()))
                    except Exception:
                        pass
                issues.append(_strategy_preflight_issue(strategy, strategy_audit, exc))

        return PreparedEvaluation(
            session_directory=session,
            strategy_id=strategy,
            frozen_truth=validated.frozen_truth,
            input_sha256=validated.input_sha256,
            truth_source=validated.truth_source,
            advisor=advisor,
            strategy_audit=strategy_audit,
            issues=tuple(_deduplicate_issues(issues)),
            eligible_self_decisions=validated.eligible_self_decisions,
            self_contexts=validated.self_contexts,
            finish_order=validated.finish_order,
            warmup_latency_ms=warmup_ms,
        )

    def validate_input(
        self,
        session_directory: Path | str,
    ) -> EvaluationInputValidation:
        """Freeze and strictly validate the session's saved TruthLog."""

        session = Path(session_directory).resolve()
        issues: list[EvaluationIssue] = []
        frozen: TruthLog | None = None
        frozen_sha: str | None = None
        truth_source = str(session / "truth_log.json")

        if not session.is_dir():
            issues.append(EvaluationIssue("INPUT_SESSION_MISSING", f"会话目录不存在：{session}"))
        else:
            try:
                frozen = load_truth_log(
                    session / "truth_log.json", session_id=session.name
                )
                frozen = truth_log_from_dict(_json_roundtrip(frozen.to_dict()))
                frozen_sha = evaluation_truth_sha256(frozen)
            except Exception as exc:
                issues.append(EvaluationIssue("INPUT_TRUTH_UNREADABLE", str(exc)))

        finish_order: tuple[str, ...] = ()
        contexts: tuple[str, ...] = ()
        if frozen is not None:
            strict = validate_evaluation_truth(session, frozen)
            issues.extend(strict.issues)
            contexts = strict.self_contexts
            finish_order = strict.finish_order

        return EvaluationInputValidation(
            session_directory=session,
            frozen_truth=frozen,
            input_sha256=frozen_sha,
            truth_source=truth_source,
            issues=tuple(_deduplicate_issues(issues)),
            eligible_self_decisions=len(contexts),
            self_contexts=contexts,
            finish_order=finish_order,
        )

    def run(
        self,
        prepared: PreparedEvaluation,
        *,
        progress: Callable[[EvaluationProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> EvaluationRunResult:
        run_id = _new_run_id()
        created_at = _utc_now()
        decisions: list[dict[str, object]] = []
        errors: list[EvaluationIssue] = list(prepared.issues)
        status: FinalStatus = "blocked" if not prepared.ready else "completed"
        cancelled = False
        fatal = False
        warmup_ms = prepared.warmup_latency_ms

        _emit(progress, "validating", 0, prepared.eligible_self_decisions)
        if prepared.ready:
            assert prepared.frozen_truth is not None
            assert prepared.advisor is not None
            reducer = LiveReducer(prepared.frozen_truth.source_session_id)
            _initialize_reducer(reducer, prepared.frozen_truth)
            self_index = 0
            _emit(progress, "running", 0, prepared.eligible_self_decisions)
            for turn in prepared.frozen_truth.turns:
                if turn.actor == "self":
                    if is_cancelled is not None and is_cancelled():
                        cancelled = True
                        break
                    context = prepared.self_contexts[self_index]
                    self_index += 1
                    snapshot = reducer.snapshot()
                    state_hash = _state_sha256(snapshot.semantic_dict())
                    request_id = f"{run_id}-T{turn.index:04d}"
                    trace = StrategyExecutionTrace(request_id)
                    started = perf_counter()
                    advice: AdviceResult | None = None
                    decision_error: EvaluationIssue | None = None
                    try:
                        # The strategy boundary receives only the current state and trace.
                        advice = prepared.advisor.recommend(
                            reducer.to_guandan_state(),
                            request_id=request_id,
                            trace=trace,
                        )
                        _validate_result_binding(prepared.strategy_id, advice)
                        _validate_predicted_action(snapshot, advice)
                    except BackendIdentityError as exc:
                        decision_error = EvaluationIssue(
                            "BACKEND_IDENTITY_DRIFT", str(exc), turn.index
                        )
                        fatal = True
                    except Exception as exc:
                        decision_error = EvaluationIssue(
                            "STRATEGY_DECISION_FAILED", str(exc), turn.index
                        )
                    latency_ms = (perf_counter() - started) * 1_000

                    # Construct the comparison only after prediction has returned.
                    actual = _action_dict(turn.is_pass, turn.cards)
                    if advice is not None and decision_error is None:
                        predicted = _action_dict(advice.is_pass, advice.cards)
                        exact = predicted == actual
                        pass_play = predicted["is_pass"] == actual["is_pass"]
                        decision = _decision_record(
                            prepared,
                            turn,
                            context=context,
                            snapshot=snapshot,
                            state_hash=state_hash,
                            request_id=request_id,
                            predicted=predicted,
                            actual=actual,
                            exact=exact,
                            pass_play=pass_play,
                            latency_ms=latency_ms,
                            status="evaluated",
                            error=None,
                            advice=advice,
                        )
                    else:
                        assert decision_error is not None
                        errors.append(decision_error)
                        decision = _decision_record(
                            prepared,
                            turn,
                            context=context,
                            snapshot=snapshot,
                            state_hash=state_hash,
                            request_id=request_id,
                            predicted=None,
                            actual=actual,
                            exact=None,
                            pass_play=None,
                            latency_ms=None,
                            status="error",
                            error=decision_error,
                            advice=None,
                        )
                    decisions.append(decision)
                    _apply_truth_turn(reducer, turn)
                    _emit(
                        progress,
                        "running",
                        len(decisions),
                        prepared.eligible_self_decisions,
                        f"turn {turn.index}",
                    )
                    if fatal:
                        break
                else:
                    _apply_truth_turn(reducer, turn)

            evaluated = sum(item["status"] == "evaluated" for item in decisions)
            decision_errors = sum(item["status"] == "error" for item in decisions)
            if cancelled:
                status = "cancelled"
            elif fatal:
                status = "failed"
            elif evaluated == prepared.eligible_self_decisions and decision_errors == 0:
                status = "completed"
            elif evaluated > 0:
                status = "completed_with_errors"
            else:
                status = "failed"

        _emit(progress, "finalizing", len(decisions), prepared.eligible_self_decisions)
        finished_at = _utc_now()
        metrics = _metrics(decisions, prepared.self_contexts)
        evaluated_count = sum(item["status"] == "evaluated" for item in decisions)
        error_count = sum(item["status"] == "error" for item in decisions)
        skipped_count = max(
            0, prepared.eligible_self_decisions - evaluated_count - error_count
        )
        summary: dict[str, object] = {
            "schema": self.schema,
            "schema_version": 1,
            "run_id": run_id,
            "created_at": created_at,
            "finished_at": finished_at,
            "status": status,
            "errors": [issue.to_dict() for issue in errors],
            "error_summary": {
                "count": len(errors),
                "codes": sorted({issue.code for issue in errors}),
            },
            "session_id": (
                prepared.frozen_truth.source_session_id
                if prepared.frozen_truth is not None
                else prepared.session_directory.name
            ),
            "truth_source": prepared.truth_source,
            "input_sha256": prepared.input_sha256,
            "strategy": dict(prepared.strategy_audit),
            "strategy_id": prepared.strategy_id,
            "actual_backend": prepared.strategy_audit.get("backend"),
            "total_turns": len(prepared.frozen_truth.turns) if prepared.frozen_truth else 0,
            "eligible_self_decisions": prepared.eligible_self_decisions,
            "evaluated_decisions": evaluated_count,
            "error_decisions": error_count,
            "skipped_decisions": skipped_count,
            "metrics": metrics,
            "warmup_latency_ms": warmup_ms,
            "cancelled": cancelled,
            "cancellation": {
                "requested": cancelled,
                "completed_decisions": len(decisions),
            },
            "report_path": "report.md",
            "decisions_path": "decisions.jsonl",
        }
        report = _render_report(summary, decisions)
        try:
            report_directory = _publish_artifacts(
                prepared.session_directory, run_id, summary, decisions, report
            )
        except Exception:
            # Artifact publication is itself a terminal failure and must never be
            # relabelled as a completed evaluation.
            summary["status"] = "failed"
            raise
        return EvaluationRunResult(
            run_id,
            summary["status"],  # type: ignore[arg-type]
            report_directory,
            summary,
            tuple(decisions),
        )

    def _build_advisor(self, strategy: str, session: Path) -> AdvicePort:
        if self._advisor_factory is not None:
            return self._advisor_factory(strategy)
        profiles_root = self.profiles_root
        profile_name = self.profile_name
        if profiles_root is None:
            # Expected layout: <profiles>/<profile>/sessions/<session>.
            profiles_root = session.parents[2]
            profile_name = session.parents[1].name
        return build_evaluation_advisor(
            strategy,
            profiles_root=profiles_root,
            profile_name=profile_name,
        )


class BackendIdentityError(RuntimeError):
    pass


def _strategy_preflight_issue(
    strategy_id: str,
    audit: dict[str, object],
    error: Exception,
) -> EvaluationIssue:
    status = str(audit.get("status", ""))
    if strategy_id == "fabledan_model" and status == "missing":
        code = "STRATEGY_FABLEDAN_WEIGHTS_MISSING"
    elif strategy_id == "fabledan_model" and status == "invalid":
        code = "STRATEGY_FABLEDAN_WEIGHTS_INVALID"
    elif isinstance(error, BackendIdentityError):
        code = "STRATEGY_BACKEND_MISMATCH"
    elif strategy_id == "danzero_model":
        code = "STRATEGY_DANZERO_UNAVAILABLE"
    elif strategy_id == "fabledan_rule":
        code = "STRATEGY_RULE_UNAVAILABLE"
    else:
        code = "STRATEGY_PREFLIGHT_FAILED"
    return EvaluationIssue(code, str(error))


def _validate_session_ownership(
    session: Path, truth: TruthLog
) -> list[EvaluationIssue]:
    if truth.source_session_id != session.name:
        return [
            EvaluationIssue(
                "INPUT_SESSION_MISMATCH",
                f"TruthLog belongs to {truth.source_session_id}, not {session.name}",
            )
        ]
    return []


def validate_evaluation_truth(
    session: Path | str,
    truth: TruthLog,
) -> EvaluationTruthValidation:
    """Apply strict game validation to one saved evaluation input."""

    session_path = Path(session).resolve()
    issues = _validate_session_ownership(session_path, truth)
    replay_issues, contexts, finish_order = _preflight_replay(truth)
    issues.extend(replay_issues)
    if not contexts:
        issues.append(
            EvaluationIssue("INPUT_NO_SELF_DECISIONS", "TruthLog 不包含我方决策点")
        )
    return EvaluationTruthValidation(
        tuple(_deduplicate_issues(issues)), contexts, finish_order
    )


def _preflight_replay(
    truth: TruthLog,
) -> tuple[list[EvaluationIssue], tuple[str, ...], tuple[str, ...]]:
    issues: list[EvaluationIssue] = []
    contexts: list[str] = []
    finish_order: list[str] = []
    if truth.initial_state.round_level not in RANKS:
        issues.append(EvaluationIssue("INPUT_LEVEL_INVALID", "初始级牌无效"))
        return issues, (), ()
    if len(truth.initial_state.my_hand) != 27:
        issues.append(EvaluationIssue("INPUT_HAND_COUNT", "我方初始手牌必须精确为 27 张"))
    if any(card.endswith("?") for card in truth.initial_state.my_hand):
        issues.append(EvaluationIssue("INPUT_UNKNOWN_SUIT", "初始手牌含未知花色"))
    physical = Counter(truth.initial_state.my_hand)
    overflow = sorted(card for card, count in physical.items() if count > 2)
    if overflow:
        issues.append(EvaluationIssue("INPUT_DECK_OVERFLOW", "双副牌数量超限：" + "、".join(overflow)))
    if issues:
        return issues, (), ()

    reducer = LiveReducer(truth.source_session_id)
    try:
        _initialize_reducer(reducer, truth)
    except Exception as exc:
        return [EvaluationIssue("INPUT_INITIAL_STATE_INVALID", str(exc))], (), ()
    previous_trick = 0
    for position, turn in enumerate(truth.turns, start=1):
        try:
            snapshot = reducer.snapshot()
            if turn.index != position:
                raise TruthValidationError(
                    "INPUT_TURN_SEQUENCE", "turn 编号必须从 1 连续递增"
                )
            if turn.trick_id <= 0 or turn.trick_id < previous_trick:
                raise TruthValidationError("INPUT_TRICK_SEQUENCE", "trick_id 必须为正且不递减")
            if turn.trick_id != snapshot.trick_id:
                raise TruthValidationError(
                    "INPUT_TRICK_MISMATCH",
                    f"trick_id={turn.trick_id}，重放状态为 {snapshot.trick_id}",
                )
            if snapshot.current_player != turn.actor:
                raise TruthValidationError(
                    "INPUT_TURN_ORDER",
                    f"当前应由 {snapshot.current_player} 行动，TruthLog 为 {turn.actor}",
                )
            _validate_action_rules(snapshot, turn)
            if turn.actor == "self":
                contexts.append("lead" if not snapshot.trick_plays else "follow")
            if turn.actor != "self" and not turn.is_pass:
                physical.update(turn.cards)
                overflow = sorted(card for card, count in physical.items() if count > 2)
                if overflow:
                    raise TruthValidationError(
                        "INPUT_DECK_OVERFLOW",
                        "我方初始手牌与对手历史超过双副牌数量：" + "、".join(overflow),
                    )
            before_finished = snapshot.finished_seats
            _apply_truth_turn(reducer, turn)
            after = reducer.snapshot()
            for seat in ("self", "right", "opposite", "left"):
                if seat in after.finished_seats and seat not in before_finished:
                    finish_order.append(seat)
            if len(after.finished_seats) >= 3 and len(finish_order) == 3:
                finish_order.extend(
                    seat
                    for seat in ("self", "right", "opposite", "left")
                    if seat not in after.finished_seats
                )
            previous_trick = turn.trick_id
        except TruthValidationError as exc:
            issues.append(EvaluationIssue(exc.code, str(exc), turn.index))
            break
        except Exception as exc:
            issues.append(EvaluationIssue("INPUT_REPLAY_FAILED", str(exc), turn.index))
            break
    return issues, tuple(contexts), tuple(finish_order)


class TruthValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _validate_action_rules(snapshot, turn: TruthTurn) -> None:
    if any(card.endswith("?") for card in turn.cards):
        raise TruthValidationError("INPUT_UNKNOWN_SUIT", "动作含未知花色")
    if turn.is_pass:
        if turn.cards:
            raise TruthValidationError("INPUT_PASS_WITH_CARDS", "不出动作不能带牌")
        return
    if not turn.cards:
        raise TruthValidationError("INPUT_PLAY_EMPTY", "出牌动作不能为空")
    actions = actions_for_cards(turn.cards, snapshot.round_level)
    if not actions:
        raise TruthValidationError("INPUT_PLAY_PATTERN_INVALID", "动作不是合法牌型")
    table = next(
        (event.cards for event in reversed(snapshot.trick_plays) if not event.is_pass),
        (),
    )
    if table and not play_beats_table(turn.cards, table, snapshot.round_level):
        raise TruthValidationError("INPUT_PLAY_DOES_NOT_BEAT", "动作不能压过当前桌面牌")
    if len(actions) > 1 and any(
        card == f"{snapshot.round_level}H" for card in turn.cards
    ):
        signatures = {(str(action[0]), str(action[1])) for action in actions}
        if len(signatures) > 1:
            raise TruthValidationError(
                "INPUT_WILDCARD_AMBIGUOUS",
                "百搭牌动作缺少唯一牌型元数据，不能猜测其声明牌型",
            )


def _initialize_reducer(reducer: LiveReducer, truth: TruthLog) -> None:
    reducer.confirm_initial_state(
        round_level=truth.initial_state.round_level,
        hand=truth.initial_state.my_hand,
        lead_player=truth.initial_state.lead_player,
        source="model_evaluation_truth",
    )


def _apply_truth_turn(reducer: LiveReducer, turn: TruthTurn) -> None:
    if turn.is_pass:
        reducer.record_pass(turn.actor, source="model_evaluation_truth")
    else:
        reducer.record_play(turn.actor, turn.cards, source="model_evaluation_truth")


def _validate_prepared_binding(strategy_id: str, audit: dict[str, object]) -> None:
    expected_backend = _BINDINGS[strategy_id][1]
    backend = str(audit.get("backend", expected_backend))
    if backend != expected_backend:
        raise BackendIdentityError(
            f"expected backend {expected_backend}, got {backend}"
        )
    if strategy_id == "fabledan_model" and str(audit.get("status", "loaded")) != "loaded":
        raise BackendIdentityError(
            f"FableDan model status is {audit.get('status')}"
        )


def _validate_result_binding(strategy_id: str, advice: AdviceResult) -> None:
    expected_strategy, expected_backend = _BINDINGS[strategy_id]
    backend = expected_backend
    if isinstance(advice.engine_input, dict):
        backend = str(advice.engine_input.get("backend", backend))
    if advice.strategy != expected_strategy or backend != expected_backend:
        raise BackendIdentityError(
            f"expected {expected_strategy}/{expected_backend}, got {advice.strategy}/{backend}"
        )


def _validate_predicted_action(snapshot, advice: AdviceResult) -> None:
    cards = tuple(str(card) for card in advice.cards)
    table = next(
        (event.cards for event in reversed(snapshot.trick_plays) if not event.is_pass),
        (),
    )
    if advice.is_pass:
        if cards:
            raise ValueError("strategy returned PASS with cards")
        if not table:
            raise ValueError("strategy returned PASS while leading")
        return
    if not cards:
        raise ValueError("strategy returned an empty play")
    if Counter(cards) - Counter(snapshot.my_hand):
        raise ValueError("strategy returned cards outside the current hand")
    if not actions_for_cards(cards, snapshot.round_level):
        raise ValueError("strategy returned an invalid play pattern")
    if table and not play_beats_table(cards, table, snapshot.round_level):
        raise ValueError("strategy returned a play that does not beat the table")


def _decision_record(
    prepared: PreparedEvaluation,
    turn: TruthTurn,
    *,
    context: str,
    snapshot,
    state_hash: str,
    request_id: str,
    predicted: dict[str, object] | None,
    actual: dict[str, object],
    exact: bool | None,
    pass_play: bool | None,
    latency_ms: float | None,
    status: str,
    error: EvaluationIssue | None,
    advice: AdviceResult | None,
) -> dict[str, object]:
    backend = prepared.strategy_audit.get("backend", _BINDINGS[prepared.strategy_id][1])
    if advice is not None and isinstance(advice.engine_input, dict):
        backend = advice.engine_input.get("backend", backend)
    return {
        "decision_id": f"T{turn.index:04d}",
        "turn_id": turn.index,
        "trick_id": turn.trick_id,
        "prefix_turn_count": turn.index - 1,
        "context": context,
        "state_revision": snapshot.revision,
        "state_before_sha256": state_hash,
        "request_id": request_id,
        "predicted_action": predicted,
        "actual_action": actual,
        "exact_match": exact,
        "pass_play_match": pass_play,
        "strategy_id": prepared.strategy_id,
        "actual_strategy": advice.strategy if advice is not None else None,
        "actual_backend": backend,
        "model_digest": prepared.strategy_audit.get("digest"),
        "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
        "status": status,
        "error_code": error.code if error else None,
        "error_message": error.message if error else None,
    }


def _action_dict(is_pass: bool, cards) -> dict[str, object]:
    return {
        "is_pass": bool(is_pass),
        "cards": [] if is_pass else sorted(str(card) for card in cards),
    }


def _metrics(
    decisions: list[dict[str, object]], contexts: tuple[str, ...]
) -> dict[str, object]:
    evaluated = [item for item in decisions if item["status"] == "evaluated"]
    eligible = len(contexts)
    exact = sum(item["exact_match"] is True for item in evaluated)
    binary = sum(item["pass_play_match"] is True for item in evaluated)
    actual_pass = [item for item in evaluated if item["actual_action"]["is_pass"]]
    actual_play = [item for item in evaluated if not item["actual_action"]["is_pass"]]
    predicted_pass = [item for item in evaluated if item["predicted_action"]["is_pass"]]
    confusion = {
        "actual_pass_predicted_pass": sum(item in predicted_pass for item in actual_pass),
        "actual_pass_predicted_play": sum(item not in predicted_pass for item in actual_pass),
        "actual_play_predicted_pass": sum(item in predicted_pass for item in actual_play),
        "actual_play_predicted_play": sum(item not in predicted_pass for item in actual_play),
    }
    grouped: dict[str, object] = {}
    for context in ("lead", "follow"):
        group = [item for item in evaluated if item["context"] == context]
        group_eligible = contexts.count(context)
        grouped[context] = {
            "eligible": group_eligible,
            "evaluated": len(group),
            "exact": _ratio(sum(item["exact_match"] is True for item in group), len(group)),
            "pass_play": _ratio(
                sum(item["pass_play_match"] is True for item in group), len(group)
            ),
            "coverage": _ratio(len(group), group_eligible),
        }
    latencies = sorted(float(item["latency_ms"]) for item in evaluated)
    return {
        "coverage": _ratio(len(evaluated), eligible),
        "exact_action_accuracy": _ratio(exact, len(evaluated)),
        "pass_play_accuracy": {
            **_ratio(binary, len(evaluated)),
            "actual_pass": _ratio(
                sum(item["pass_play_match"] is True for item in actual_pass),
                len(actual_pass),
            ),
            "actual_play": _ratio(
                sum(item["pass_play_match"] is True for item in actual_play),
                len(actual_play),
            ),
            "confusion_matrix": confusion,
        },
        "lead_follow": grouped,
        "latency_ms": {
            "count": len(latencies),
            "total": round(sum(latencies), 3) if latencies else 0.0,
            "mean": round(mean(latencies), 3) if latencies else None,
            "p50": round(_percentile(latencies, 0.50), 3) if latencies else None,
            "p95": round(_percentile(latencies, 0.95), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
    }


def _ratio(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _percentile(values: list[float], quantile: float) -> float:
    return values[max(0, math.ceil(len(values) * quantile) - 1)]


def _render_report(
    summary: dict[str, object], decisions: list[dict[str, object]]
) -> str:
    metrics = summary["metrics"]
    lines = [
        "# 模型整局评测报告",
        "",
        "> 本报告是 teacher-forced 历史动作比较，不是闭环胜率。",
        "> 当前 TruthLog 缺少四家完整初始牌；采用首个反事实动作后，真实后续动作即不再有效。",
        "",
        f"- 运行：`{summary['run_id']}`",
        f"- 状态：`{summary['status']}`",
        "- 输入：已保存的 `truth_log.json`",
        f"- 策略：`{summary['strategy_id']}` / `{summary['actual_backend']}`",
        f"- 输入 SHA-256：`{summary['input_sha256']}`",
        "",
        "## 指标",
        "",
        f"```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```",
        "",
        "## 错误",
        "",
    ]
    errors = summary.get("errors", [])
    if errors:
        lines.extend(
            f"- `{item.get('code')}` turn={item.get('turn_id', '-')}：{item.get('message')}"
            for item in errors
        )
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 逐决策",
            "",
            "| turn | 场景 | 真实动作 | 预测动作 | exact | 延迟(ms) | 状态 |",
            "| ---: | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for item in decisions:
        lines.append(
            "| {turn_id} | {context} | {actual} | {predicted} | {exact} | {latency} | {status} |".format(
                turn_id=item["turn_id"],
                context=item["context"],
                actual=_format_action(item["actual_action"]),
                predicted=_format_action(item["predicted_action"]),
                exact=item["exact_match"],
                latency=item["latency_ms"] if item["latency_ms"] is not None else "-",
                status=item["status"],
            )
        )
    return "\n".join(lines) + "\n"


def _format_action(action: object) -> str:
    if not isinstance(action, dict):
        return "-"
    if action.get("is_pass"):
        return "不出"
    return " ".join(str(card) for card in action.get("cards", ()))


def _publish_artifacts(
    session: Path,
    run_id: str,
    summary: dict[str, object],
    decisions: list[dict[str, object]],
    report: str,
) -> Path:
    root = session / "model_evaluation_runs"
    root.mkdir(parents=True, exist_ok=True)
    target = root / run_id
    temporary = root / f".{run_id}.tmp-{uuid4().hex}"
    if target.exists():
        raise FileExistsError(f"evaluation run already exists: {target}")
    temporary.mkdir()
    try:
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with (temporary / "decisions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for decision in decisions:
                handle.write(json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n")
        (temporary / "report.md").write_text(report, encoding="utf-8")
        loaded = json.loads((temporary / "summary.json").read_text(encoding="utf-8"))
        if loaded.get("run_id") != run_id or loaded.get("status") not in FINAL_STATUSES:
            raise ValueError("summary verification failed")
        for line in (temporary / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
            json.loads(line)
        if not (temporary / "report.md").read_text(encoding="utf-8").strip():
            raise ValueError("report verification failed")
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid4().hex[:8]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evaluation_truth_sha256(truth: TruthLog) -> str:
    """Return the canonical input identity used by evaluation reports."""

    return sha256(_canonical_json(truth.to_dict()).encode("utf-8")).hexdigest()


def evaluation_truth_semantic_sha256(truth: TruthLog) -> str:
    """Identity of state-affecting input, independent of labels and card order."""

    value = {
        "source_session_id": truth.source_session_id,
        "initial_state": {
            "round_level": truth.initial_state.round_level,
            "lead_player": truth.initial_state.lead_player,
            "my_hand": sorted(truth.initial_state.my_hand),
        },
        "turns": [
            {
                "turn_id": turn.index,
                "trick_id": turn.trick_id,
                "actor": turn.actor,
                "is_pass": turn.is_pass,
                "cards": [] if turn.is_pass else sorted(turn.cards),
            }
            for turn in truth.turns
        ],
    }
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _state_sha256(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_roundtrip(value: object):
    return json.loads(_canonical_json(value))


def _json_safe(value: object) -> dict[str, object]:
    safe = _json_roundtrip(value)
    return safe if isinstance(safe, dict) else {}


def _deduplicate_issues(issues: list[EvaluationIssue]) -> list[EvaluationIssue]:
    seen: set[tuple[str, str, int | None]] = set()
    result: list[EvaluationIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.turn_id)
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _emit(
    callback: Callable[[EvaluationProgress], None] | None,
    phase: str,
    completed: int,
    total: int,
    message: str = "",
) -> None:
    if callback is not None:
        callback(EvaluationProgress(phase, completed, total, message))
