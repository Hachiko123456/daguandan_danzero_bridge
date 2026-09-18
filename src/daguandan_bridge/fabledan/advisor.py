from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np

from ..action_semantics import canonical_fabledan_type
from ..config import LOGS_ROOT, PROFILES_ROOT
from ..danzero.state import (
    GameStateError,
    GuanDanState,
    LocalStrategySnapshot,
    PlayEvent,
    RANKS,
    Seat,
)
from ..domain.advice import AdviceResult, StrategyExecutionTrace
from ..live.turns import TURN_ORDER, WindCatchPolicy, project_trick_turn, round_is_decided
from ._vendor.fabledan.agents import NumpyAgent, RuleAgent
from ._vendor.fabledan.cards import RANK_NAMES, is_wildcard, order_of, rank_of
from ._vendor.fabledan.combos import (
    PASS,
    TYPE_NAMES,
    Move,
    beats,
    gen_moves,
)
from ._vendor.fabledan.encode import (
    BOS_TOK,
    FEAT_DIM,
    LEVEL_BASE,
    PLAYER_BASE,
    RANK_BASE,
    RETURN_TOK,
    TRIBUTE_TOK,
    TYPE_BASE,
    encode_decision,
)
from ._vendor.fabledan.model_np import NumpyModel


UPSTREAM_COMMIT = "7cc5e311b9860bc44f76c082d9c1b21fc8b2d3ec"
ADAPTER_SCHEMA = "fabledan-adapter/v1"
STANDARD_NO_TRIBUTE = True
DECISION_TRACE_SCHEMA = "fabledan-trace/1"
DEFAULT_WEIGHTS_FILENAME = "best.npz"
DiagnosticsMode = Literal["off", "basic", "full"]
_SEAT_TO_PLAYER: dict[Seat, int] = {
    seat: index for index, seat in enumerate(TURN_ORDER)
}
_SUIT_TO_INDEX = {"H": 0, "D": 1, "S": 2, "C": 3}
_INDEX_TO_SUIT = {value: key for key, value in _SUIT_TO_INDEX.items()}
_PLAYER_MAPPING: dict[str, int] = {
    "self": 0,
    "right": 1,
    "opposite": 2,
    "left": 3,
    "next": 1,
    "partner": 2,
    "previous": 3,
}
_LOGGER = logging.getLogger(__name__)


class FableDanStateError(GameStateError):
    """A confirmed state cannot be represented without guessing."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


@dataclass(frozen=True)
class _PolicyRuntime:
    agent: Any
    backend: str
    status: str
    path: Path
    digest: str | None
    error: str | None = None


@dataclass(frozen=True)
class _MappedState:
    observation: dict[str, object]
    legal: tuple[Move, ...]
    hand_ids: tuple[int, ...]
    lead_owner: int | None
    audit: dict[str, object]


@dataclass(frozen=True)
class DecisionCandidate:
    """按本次决策实际使用的 Q 值排序后的一项合法动作。"""

    rank: int
    action: Move
    action_text: str
    q_value: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "action": _move_audit(self.action),
            "action_text": self.action_text,
            "q": self.q_value,
        }


@dataclass(frozen=True)
class FableDanDecisionResult:
    """保留 ``recommend()`` 接口兼容性的详细策略结果。"""

    advice: AdviceResult
    best_action: Move
    best_action_text: str
    best_q: float | None
    second_q: float | None
    q_gap: float | None
    candidates: tuple[DecisionCandidate, ...]
    legal_action_count: int
    model_path: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "best_action": _move_audit(self.best_action),
            "best_action_text": self.best_action_text,
            "best_q": self.best_q,
            "second_q": self.second_q,
            "q_gap": self.q_gap,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "legal_action_count": self.legal_action_count,
            "model_path": self.model_path,
            "warnings": list(self.warnings),
        }


class _PhysicalCards:
    """Allocate duplicate-deck IDs only inside one adapter request."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = defaultdict(int)

    def allocate(self, code: str) -> int:
        base = _base_card_id(code)
        copy_index = self._seen[code]
        if copy_index >= 2:
            raise FableDanStateError(
                f"牌面 {code} 在已确认手牌与历史中超过双副牌上限"
            )
        self._seen[code] += 1
        return base + 54 * copy_index


class FableDanAdvisor:
    """Adapt confirmed ``GuanDanState`` snapshots to vendored FableDan."""

    strategy_id = "fabledan"
    display_name = "FableDan"
    requires_exact_history_suits = True

    def __init__(
        self,
        profiles_root: Path | str = PROFILES_ROOT,
        profile_name: str = "tencent_daguandan",
        *,
        runtime_policy: Literal["auto", "model_required", "rule_only"] = "auto",
        debug: bool = False,
        diagnostics: DiagnosticsMode | None = None,
        write_decision_log: bool = True,
        log_directory: Path | str | None = None,
        top_n: int = 3,
    ) -> None:
        if runtime_policy not in {"auto", "model_required", "rule_only"}:
            raise ValueError(f"不支持的 FableDan 运行策略：{runtime_policy}")
        if int(top_n) < 1:
            raise ValueError("top_n 至少必须为 1")
        self.profiles_root = Path(profiles_root)
        self.profile_name = str(profile_name)
        self.runtime_policy = runtime_policy
        self.diagnostics = _normalize_diagnostics_mode(
            diagnostics if diagnostics is not None else ("full" if debug else "off")
        )
        self.debug = self.diagnostics != "off"
        self.write_decision_log = bool(write_decision_log)
        self.top_n = int(top_n)
        self.log_directory = (
            Path(log_directory)
            if log_directory is not None
            else LOGS_ROOT / "fabledan_traces"
        )
        self.weights_path = (
            self.profiles_root
            / self.profile_name
            / "models"
            / DEFAULT_WEIGHTS_FILENAME
        )
        self._runtime: _PolicyRuntime | None = None
        self._runtime_lock = Lock()
        self._log_lock = Lock()

    def initialize(self) -> None:
        runtime = self._ensure_runtime()
        self._require_model_runtime(runtime)

    def decision_input_fingerprint(
        self,
        state: GuanDanState,
        *,
        request_id: str = "fabledan-input-fingerprint",
    ) -> tuple[str, str, int]:
        """Hash the exact encoded FableDan input without running the model."""

        mapped = self._map_snapshot(state.local_snapshot(), request_id=request_id)
        tokens, features = encode_decision(mapped.observation)
        token_array = np.asarray(tokens, dtype=np.int64)
        feature_array = np.ascontiguousarray(features)
        return (
            sha256(token_array.tobytes()).hexdigest(),
            sha256(feature_array.tobytes()).hexdigest(),
            len(mapped.legal),
        )

    def audit_info(self) -> dict[str, object]:
        runtime = self._ensure_runtime()
        return self._runtime_audit(runtime)

    def recommend(
        self,
        state: GuanDanState,
        *,
        request_id: str = "",
        trace: StrategyExecutionTrace | None = None,
    ) -> AdviceResult:
        return self.recommend_detailed(
            state,
            request_id=request_id,
            trace=trace,
        ).advice

    def recommend_detailed(
        self,
        state: GuanDanState,
        *,
        request_id: str = "",
        trace: StrategyExecutionTrace | None = None,
    ) -> FableDanDecisionResult:
        started = perf_counter()
        execution_trace = trace or StrategyExecutionTrace(request_id)
        execution_trace.begin("fabledan_validate")
        runtime = self._ensure_runtime()
        self._require_model_runtime(runtime)
        base_audit = self._runtime_audit(runtime)
        base_audit["request_id"] = request_id
        execution_trace.set_engine_input(base_audit)
        try:
            snapshot = state.local_snapshot()
            mapped = self._map_snapshot(snapshot, request_id=request_id)
        except Exception as exc:
            blocked = dict(base_audit)
            blocked.update(
                {
                    "request_id": request_id,
                    "validation_status": "blocked",
                    "validation_error": str(exc),
                    "project_snapshot": {
                        "round_level": state.round_level,
                        "wild_rank": state.wild_rank,
                        "phase": state.phase,
                        "current_player": state.current_player,
                        "lead_player": state.lead_player,
                        "my_hand": list(state.my_hand),
                        "play_history": [
                            event.to_dict() for event in state.play_history
                        ],
                        "remaining_cards": state.remaining_cards,
                        "revision": state.revision,
                    },
                }
            )
            if self.diagnostics != "off":
                blocked["fabledan_trace"] = _blocked_trace_payload(
                    request_id=request_id,
                    state=state,
                    error=exc,
                    diagnostics_mode=self.diagnostics,
                )
            execution_trace.set_engine_input(blocked)
            execution_trace.end()
            raise

        engine_input = dict(mapped.audit)
        engine_input.update(base_audit)
        engine_input["request_id"] = request_id
        engine_input["validation_status"] = "accepted"
        execution_trace.set_engine_input(engine_input)
        execution_trace.begin("fabledan_policy")
        policy_started = perf_counter()
        active_runtime = runtime
        q_values: tuple[float, ...] | None = None
        tokens: list[int] | None = None
        features: np.ndarray | None = None
        try:
            selected_index, q_values, tokens, features = self._evaluate_policy(
                runtime,
                mapped.observation,
            )
        except Exception as exc:
            if runtime.backend != "numpy" or self.runtime_policy == "model_required":
                execution_trace.end()
                backend = "model" if runtime.backend == "numpy" else "RuleAgent"
                raise RuntimeError(f"FableDan {backend} 执行失败：{exc}") from exc
            active_runtime = self._replace_invalid_numpy_runtime(exc)
            engine_input.update(self._runtime_audit(active_runtime))
            execution_trace.set_engine_input(engine_input)
            selected_index, q_values, tokens, features = self._evaluate_policy(
                active_runtime,
                mapped.observation,
            )
        policy_ms = (perf_counter() - policy_started) * 1_000
        if not 0 <= selected_index < len(mapped.legal):
            execution_trace.end()
            raise RuntimeError("FableDan 返回的合法动作索引越界")
        move = mapped.legal[selected_index]
        if move.type == PASS:
            cards: tuple[str, ...] = ()
        else:
            available = Counter(mapped.hand_ids)
            requested = Counter(int(card) for card in move.cards)
            if requested - available:
                execution_trace.end()
                raise RuntimeError("FableDan 动作包含当前我方手牌以外的物理牌")
            cards = tuple(sorted(_card_code(int(card)) for card in move.cards))
        legal_signature = _move_signature(move)
        if legal_signature not in {
            _move_signature(candidate) for candidate in mapped.legal
        }:
            execution_trace.end()
            raise RuntimeError("FableDan 返回动作不属于本次 legal 集合")

        candidates = _rank_candidates(mapped.legal, selected_index, q_values)
        best_q = candidates[0].q_value
        second_q = candidates[1].q_value if len(candidates) > 1 else None
        q_gap = (
            best_q - second_q
            if best_q is not None and second_q is not None
            else None
        )
        diagnostics = _decision_diagnostics(
            mapped,
            selected_index=selected_index,
            q_values=q_values,
            candidates=candidates,
            tokens=tokens,
            features=features,
        )
        warnings = list(diagnostics["warnings"])
        engine_input["selected_action_index"] = selected_index
        engine_input["selected_action"] = _move_audit(move)
        engine_input.update(self._runtime_audit(active_runtime))
        # Keep the exact model input needed for offline training even when the
        # optional, much larger diagnostic trace is disabled.  This snapshot is
        # intentionally independent from Q values: real-game labels must come
        # from the confirmed final result, never from the model's own output.
        engine_input["fabledan_training_input"] = _training_input_payload(
            mapped=mapped,
            runtime=active_runtime,
            tokens=tokens,
            features=features,
        )
        engine_input["debug"] = self.debug
        engine_input["diagnostics_mode"] = self.diagnostics
        decision_audit = {
            "best_action": _move_audit(move),
            "best_action_text": _move_text(move),
            "best_q": best_q,
            "second_q": second_q,
            "q_gap": q_gap,
            "legal_action_count": len(mapped.legal),
            # Keep the normal runtime payload compact.  The complete ranking is
            # still retained in the full diagnostic trace below when requested.
            "candidates": [
                candidate.to_dict() for candidate in candidates[: self.top_n]
            ],
            "warnings": list(warnings),
        }
        # The UI can show the model's best alternatives without enabling the
        # much larger token/feature/Q-value diagnostics payload.
        engine_input["top_n"] = self.top_n
        engine_input["decision"] = decision_audit
        if self.debug:
            trace_payload = _decision_trace_payload(
                request_id=request_id,
                mapped=mapped,
                runtime=active_runtime,
                diagnostics_mode=self.diagnostics,
                selected_index=selected_index,
                q_values=q_values,
                candidates=candidates,
                tokens=tokens,
                features=features,
                diagnostics=diagnostics,
            )
            engine_input.update(
                {
                    "encoding": trace_payload.get("encoding", {}),
                    "q_values": _q_values_audit(mapped.legal, q_values),
                    "validation_warnings": list(warnings),
                    "fabledan_trace": trace_payload,
                }
            )
            if self.write_decision_log:
                timestamp = datetime.now().astimezone()
                log_payload = dict(trace_payload)
                log_payload["timestamp"] = timestamp.isoformat(timespec="seconds")
                try:
                    log_path = self._append_decision_log(timestamp, log_payload)
                    engine_input["decision_log_path"] = str(log_path)
                except Exception as exc:
                    warning = f"FableDan 决策日志写入失败：{exc}"
                    warnings.append(warning)
                    decision_audit["warnings"] = list(warnings)
                    engine_input["validation_warnings"] = list(warnings)
                    _LOGGER.warning(warning, exc_info=True)

        elapsed_ms = (perf_counter() - started) * 1_000
        strategy = (
            "fabledan-numpy"
            if active_runtime.backend == "numpy"
            else "fabledan-rule"
        )
        advice = AdviceResult(
            strategy=strategy,
            cards=cards,
            play_type=TYPE_NAMES[move.type],
            is_pass=move.type == PASS,
            state_revision=snapshot.revision,
            elapsed_ms=elapsed_ms,
            request_id=request_id,
            engine_input=engine_input,
            timings={
                "fabledan_policy": policy_ms,
                "total": elapsed_ms,
            },
        )
        execution_trace.set_engine_input(engine_input)
        execution_trace.end()
        return FableDanDecisionResult(
            advice=advice,
            best_action=move,
            best_action_text=_move_text(move),
            best_q=best_q,
            second_q=second_q,
            q_gap=q_gap,
            candidates=candidates,
            legal_action_count=len(mapped.legal),
            model_path=str(active_runtime.path),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _evaluate_policy(
        runtime: _PolicyRuntime,
        observation: dict[str, object],
    ) -> tuple[int, tuple[float, ...] | None, list[int] | None, np.ndarray | None]:
        if runtime.backend != "numpy":
            return int(runtime.agent.act(observation)), None, None, None

        # 保持与上游 NumpyAgent 完全相同的推理路径，并让动作选择与调试信息
        # 共享同一次模型推理，避免调试模式引入第二次计算。
        tokens, features = encode_decision(observation)
        raw_q_values = runtime.agent.model.q_values(tokens, features)
        q_array = np.asarray(raw_q_values).reshape(-1)
        selected_index = int(np.argmax(q_array))
        return (
            selected_index,
            tuple(float(value) for value in q_array),
            tokens,
            features,
        )

    def _append_decision_log(
        self,
        timestamp: datetime,
        payload: dict[str, object],
    ) -> Path:
        path = self.log_directory / f"fabledan_trace-{timestamp.date().isoformat()}.jsonl"
        line = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        return path

    def _ensure_runtime(self) -> _PolicyRuntime:
        with self._runtime_lock:
            if self._runtime is not None:
                return self._runtime
            path = self.weights_path
            if self.runtime_policy == "rule_only":
                self._runtime = _PolicyRuntime(
                    RuleAgent(), "rule", "rule_only", path, None
                )
                return self._runtime
            if not path.is_file():
                self._runtime = _PolicyRuntime(
                    RuleAgent() if self.runtime_policy == "auto" else None,
                    "rule" if self.runtime_policy == "auto" else "numpy",
                    "missing",
                    path,
                    None,
                    "模型权重文件不存在",
                )
                return self._runtime
            try:
                digest = sha256(path.read_bytes()).hexdigest()
                model = NumpyModel(path)
                _validate_numpy_model(model)
                runtime = _PolicyRuntime(
                    NumpyAgent(model), "numpy", "loaded", path, digest
                )
            except Exception as exc:
                runtime = _PolicyRuntime(
                    RuleAgent() if self.runtime_policy == "auto" else None,
                    "rule" if self.runtime_policy == "auto" else "numpy",
                    "invalid",
                    path,
                    _safe_digest(path),
                    str(exc),
                )
            self._runtime = runtime
            return runtime

    def _replace_invalid_numpy_runtime(self, exc: Exception) -> _PolicyRuntime:
        with self._runtime_lock:
            current = self._runtime
            if current is not None and current.backend != "numpy":
                return current
            runtime = _PolicyRuntime(
                RuleAgent(),
                "rule",
                "invalid",
                self.weights_path,
                current.digest if current is not None else _safe_digest(self.weights_path),
                str(exc),
            )
            self._runtime = runtime
            return runtime

    def _runtime_audit(self, runtime: _PolicyRuntime) -> dict[str, object]:
        return {
            "backend": runtime.backend,
            "path": str(runtime.path),
            "model_path": str(runtime.path),
            "model_filename": runtime.path.name,
            "status": runtime.status,
            "digest": runtime.digest,
            "model_hash": runtime.digest,
            "upstream_commit": UPSTREAM_COMMIT,
            "schema": ADAPTER_SCHEMA,
            "standard_no_tribute": STANDARD_NO_TRIBUTE,
            "backend_error": runtime.error,
            "runtime_policy": self.runtime_policy,
            "debug": self.debug,
            "diagnostics_mode": self.diagnostics,
            "decision_log_mode": (
                "append"
                if self.debug and self.write_decision_log
                else "trace_embedded"
                if self.debug
                else "disabled"
            ),
        }

    def _require_model_runtime(self, runtime: _PolicyRuntime) -> None:
        if self.runtime_policy == "model_required" and (
            runtime.backend != "numpy" or runtime.status != "loaded"
        ):
            raise RuntimeError(
                f"FableDan model_required（模型模式）要求 {DEFAULT_WEIGHTS_FILENAME} 可用："
                f"{runtime.error or runtime.status}"
            )

    def _map_snapshot(
        self,
        snapshot: LocalStrategySnapshot,
        *,
        request_id: str,
    ) -> _MappedState:
        _validate_standard_snapshot(snapshot)
        allocation = _PhysicalCards()
        counts = {seat: 27 for seat in TURN_ORDER}
        done: set[Seat] = set()
        events: list[tuple[object, ...]] = []
        event_audit: list[dict[str, object]] = []
        expected: Seat | None = None
        trick_lead: Move | None = None
        trick_leader: Seat | None = None
        passed: set[Seat] = set()

        if snapshot.play_history:
            expected = snapshot.play_history[0].player
        for index, event in enumerate(snapshot.play_history, start=1):
            if expected is not None and event.player != expected:
                raise FableDanStateError(
                    f"第 {index} 条历史行动者应为 {expected}，实际为 {event.player}；"
                    "历史不是从第一手开始的完整标准轮转"
                )
            player = _SEAT_TO_PLAYER[event.player]
            if event.is_pass:
                if trick_lead is None:
                    raise FableDanStateError(f"第 {index} 条历史为无首出的不出")
                events.append(("pass", player))
                event_audit.append(
                    {
                        "index": index,
                        "kind": "pass",
                        "player": event.player,
                        "player_id": player,
                        "absolute_player": player,
                        "relative_player": (player - 0) % 4,
                        "relative_player_label": _relative_player_label(player),
                        "is_teammate": player == 2,
                        "source_turn_id": index,
                        "encoded_player_token": PLAYER_BASE + ((player - 0) % 4),
                        "encoded_player_token_name": _token_name(
                            PLAYER_BASE + ((player - 0) % 4)
                        ),
                        "is_pass": True,
                    }
                )
                passed.add(event.player)
            else:
                card_ids = tuple(allocation.allocate(card) for card in event.cards)
                move, semantic_resolution = _unique_observed_move(
                    card_ids,
                    snapshot.round_level,
                    index,
                    event.action_metadata,
                )
                if trick_lead is not None and not beats(
                    move, trick_lead, RANK_NAMES.index(snapshot.round_level)
                ):
                    move_audit = _move_audit(
                        move,
                        level=RANK_NAMES.index(snapshot.round_level),
                    )
                    lead_audit = _move_audit(
                        trick_lead,
                        level=RANK_NAMES.index(snapshot.round_level),
                    )
                    diagnostic = {
                        "code": "history_move_does_not_beat_lead",
                        "source_turn_id": index,
                        "player": event.player,
                        "physical_cards": list(event.cards),
                        "declared_move": move_audit,
                        "lead_owner": trick_leader,
                        "lead_move": lead_audit,
                        "reason": "该动作声明按掼蛋牌型比较不能压过当前桌面",
                    }
                    raise FableDanStateError(
                        f"第 {index} 条历史动作 {TYPE_NAMES[move.type]} "
                        f"{_move_text(move)}（实体牌 {' '.join(event.cards)}）"
                        f"不能压过 {trick_leader} 的 {TYPE_NAMES[trick_lead.type]} "
                        f"{_move_text(trick_lead)}",
                        diagnostic=diagnostic,
                    )
                counts[event.player] -= len(card_ids)
                if counts[event.player] < 0:
                    raise FableDanStateError(
                        f"{event.player} 的出牌历史超过标准起手 27 张"
                    )
                if counts[event.player] == 0:
                    done.add(event.player)
                trick_lead = move
                trick_leader = event.player
                passed.clear()
                events.append(("play", player, move))
                event_audit.append(
                    {
                        "index": index,
                        "kind": "play",
                        "player": event.player,
                        "player_id": player,
                        "absolute_player": player,
                        "relative_player": (player - 0) % 4,
                        "relative_player_label": _relative_player_label(player),
                        "is_teammate": player == 2,
                        "source_turn_id": index,
                        "encoded_player_token": PLAYER_BASE + ((player - 0) % 4),
                        "encoded_player_token_name": _token_name(
                            PLAYER_BASE + ((player - 0) % 4)
                        ),
                        "is_pass": False,
                        "move": _move_audit(move, level=RANK_NAMES.index(snapshot.round_level)),
                        "action_text": _move_text(move),
                        "semantic_resolution": semantic_resolution,
                    }
                )

            if round_is_decided(done):
                expected = None
                trick_lead = None
                trick_leader = None
                passed.clear()
                continue
            if trick_leader is None:
                raise FableDanStateError(f"第 {index} 条历史后缺少当前墩首出玩家")
            projection = project_trick_turn(
                trick_leader, done, passed,
                wind_catch_policy=WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
            )
            expected = projection.expected_after(event.player)
            if projection.is_complete:
                trick_lead = None
                trick_leader = None
                passed.clear()

        hand_ids = tuple(allocation.allocate(card) for card in snapshot.my_hand)
        counts["self"] = len(hand_ids)
        self_played = sum(
            len(event.cards)
            for event in snapshot.play_history
            if event.player == "self" and not event.is_pass
        )
        if self_played + len(hand_ids) != 27:
            raise FableDanStateError(
                "我方当前手牌与历史出牌不能还原标准 27 张起手；"
                "FableDan 需要从第一手开始的完整历史"
            )
        derived_counts = {
            seat: 27
            - sum(
                len(event.cards)
                for event in snapshot.play_history
                if event.player == seat and not event.is_pass
            )
            for seat in TURN_ORDER
        }
        derived_counts["self"] = len(hand_ids)
        if snapshot.remaining_cards is not None:
            if set(snapshot.remaining_cards) != set(TURN_ORDER):
                raise FableDanStateError("remaining_cards 必须包含四个标准座位")
            supplied = {
                seat: int(snapshot.remaining_cards[seat]) for seat in TURN_ORDER
            }
            if supplied != derived_counts:
                raise FableDanStateError(
                    "remaining_cards 与从第一手重建的完整出牌历史不一致"
                )
        counts = derived_counts
        done = {seat for seat, count in counts.items() if count == 0}

        if snapshot.play_history:
            if expected != snapshot.current_player:
                raise FableDanStateError(
                    "历史轮转不能唯一到达当前行动者；请补齐或修正历史"
                )
        elif snapshot.lead_player != "self":
            raise FableDanStateError(
                "空历史仅能表示我方首发；否则缺少从第一手开始的完整历史"
            )
        _validate_trick_suffix(snapshot)
        reconstructed_lead = trick_lead
        if snapshot.trick_plays:
            first = snapshot.trick_plays[0]
            if snapshot.lead_player != first.player:
                raise FableDanStateError("本轮首出者与 trick_plays 首条事件不一致")
        elif snapshot.lead_player != snapshot.current_player:
            raise FableDanStateError("新一轮首出者与当前行动者不一致")

        level = RANK_NAMES.index(snapshot.round_level)
        legal = tuple(gen_moves(hand_ids, level, reconstructed_lead))
        if not legal:
            raise FableDanStateError("FableDan 没有生成可用合法动作")
        lead_owner = (
            _SEAT_TO_PLAYER[trick_leader]
            if reconstructed_lead is not None and trick_leader is not None
            else None
        )
        left = [counts[seat] for seat in TURN_ORDER]
        done_flags = [seat in done for seat in TURN_ORDER]
        observation: dict[str, object] = {
            "level": level,
            "player": 0,
            "hand": list(hand_ids),
            "left": left,
            "done": done_flags,
            "events": events,
            "lead": reconstructed_lead,
            "legal": list(legal),
        }
        audit = {
            "request_id": request_id,
            "project_snapshot": {
                "round_level": snapshot.round_level,
                "wild_rank": snapshot.wild_rank,
                "current_player": snapshot.current_player,
                "lead_player": snapshot.lead_player,
                "my_hand": list(snapshot.my_hand),
                "remaining_cards": counts,
                "play_history": [event.to_dict() for event in snapshot.play_history],
                "trick_plays": [event.to_dict() for event in snapshot.trick_plays],
                "revision": snapshot.revision,
            },
            "player": 0,
            "viewer_absolute": 0,
            "current_player_absolute": _SEAT_TO_PLAYER[snapshot.current_player],
            "current_player_relative": (
                _SEAT_TO_PLAYER[snapshot.current_player] - 0
            ) % 4,
            "teammate_absolute": 2,
            "teammate_relative": 2,
            "seat_mapping": dict(_SEAT_TO_PLAYER),
            "player_mapping": dict(_PLAYER_MAPPING),
            "player_mapping_labels": {
                "0": "self/自己",
                "1": "right/下家",
                "2": "opposite/队友",
                "3": "left/上家",
            },
            "level": level,
            "level_text": snapshot.round_level,
            "hand": list(snapshot.my_hand),
            "hand_ids": [int(card_id) for card_id in hand_ids],
            "left": left,
            "left_by_player": {
                str(_SEAT_TO_PLAYER[seat]): {
                    "seat": seat,
                    "count": counts[seat],
                }
                for seat in TURN_ORDER
            },
            "done": done_flags,
            "lead": (
                _move_audit(reconstructed_lead, level=level)
                if reconstructed_lead is not None
                else None
            ),
            "lead_text": _move_text(reconstructed_lead),
            "lead_owner": lead_owner,
            "lead_owner_seat": trick_leader,
            "lead_owner_absolute": lead_owner,
            "lead_owner_relative": (
                (lead_owner - 0) % 4 if lead_owner is not None else None
            ),
            "lead_owner_is_teammate": lead_owner == 2 if lead_owner is not None else None,
            "history": event_audit,
            "events": event_audit,
            "legal_actions": [_move_audit(move, level=level) for move in legal],
            "feature_schema": "fabledan-token48-feat80/v1",
        }
        return _MappedState(observation, legal, hand_ids, lead_owner, audit)


def _validate_standard_snapshot(snapshot: LocalStrategySnapshot) -> None:
    if snapshot.phase != "playing":
        raise FableDanStateError("FableDan 仅支持 standard no-tribute playing 阶段")
    if snapshot.round_level not in RANKS or snapshot.wild_rank not in RANKS:
        raise FableDanStateError("FableDan 需要已确认的级牌与百搭牌")
    if snapshot.round_level != snapshot.wild_rank:
        raise FableDanStateError("FableDan standard 模式要求级牌与百搭级别一致")
    if snapshot.current_player != "self":
        raise FableDanStateError("FableDan 仅在我方当前行动时提供建议")
    for zone, cards in _snapshot_card_zones(snapshot):
        unknown = [card for card in cards if card.endswith("?")]
        if unknown:
            raise FableDanStateError(
                f"{zone} 含未知花色 {', '.join(unknown)}；FableDan 不会猜测花色"
            )
        invalid = [card for card in cards if not _is_known_card(card)]
        if invalid:
            raise FableDanStateError(
                f"{zone} 含无效牌码：{', '.join(invalid)}"
            )
    for index, event in enumerate(snapshot.play_history, start=1):
        if event.is_pass and event.cards:
            raise FableDanStateError(f"第 {index} 条不出历史含牌面")
        if not event.is_pass and not event.cards:
            raise FableDanStateError(f"第 {index} 条出牌历史没有牌面")
        # ``LiveReducer.record_play`` preserves the exact suit of a concrete
        # card as a one-item tuple, e.g. ``("D",)``.  Only multiple choices
        # are ambiguous; treating every non-empty tuple as ambiguous blocks
        # every replayed, fully-known history before the model can run.
        if any(len(options) > 1 for options in event.suit_options):
            raise FableDanStateError(
                f"第 {index} 条历史仍含花色候选，不能唯一解释"
            )


def _snapshot_card_zones(
    snapshot: LocalStrategySnapshot,
) -> Iterable[tuple[str, tuple[str, ...]]]:
    yield "当前手牌", snapshot.my_hand
    for index, event in enumerate(snapshot.play_history, start=1):
        yield f"第 {index} 条历史", event.cards


def _validate_trick_suffix(snapshot: LocalStrategySnapshot) -> None:
    trick = tuple(snapshot.trick_plays)
    if not trick:
        return
    history = tuple(snapshot.play_history)
    if len(trick) > len(history) or history[-len(trick) :] != trick:
        raise FableDanStateError("trick_plays 不是完整历史的当前轮后缀")


def _unique_observed_move(
    card_ids: tuple[int, ...],
    level_rank: str,
    history_index: int,
    action_metadata: dict[str, object] | None = None,
) -> tuple[Move, dict[str, object]]:
    level = RANK_NAMES.index(level_rank)
    candidates = [
        move
        for move in gen_moves(card_ids, level, None)
        if Counter(int(card) for card in move.cards) == Counter(card_ids)
    ]
    by_declaration: dict[tuple[object, ...], Move] = {}
    for move in candidates:
        by_declaration.setdefault(_move_declaration(move), move)
    if not by_declaration:
        diagnostic = {
            "code": "observed_move_invalid",
            "source_turn_id": history_index,
            "physical_cards": sorted(_card_code(card) for card in card_ids),
            "level": level_rank,
            "wild_rank": level_rank,
            "candidate_interpretations": [],
            "reason": "实体牌无法生成任何 FableDan 合法声明",
        }
        raise FableDanStateError(
            f"第 {history_index} 条历史不是合法牌型",
            diagnostic=diagnostic,
        )
    selected_semantics = _selected_move_semantics(action_metadata)
    if selected_semantics is not None:
        selection_source = str(
            (action_metadata or {}).get("selection_source", "exact_engine_state")
        )
        candidate_checks = [
            (
                move,
                _move_semantics_mismatches(
                    move,
                    selected_semantics,
                    level=level,
                    selection_source=selection_source,
                ),
            )
            for move in by_declaration.values()
        ]
        matches = [move for move, reasons in candidate_checks if not reasons]
        if len(matches) == 1:
            return matches[0], {
                "selection_source": selection_source,
                "ambiguity": len(by_declaration) > 1,
                "selected_interpretation": _move_audit(matches[0], level=level),
                "candidate_interpretations": [
                    _move_audit(move, level=level) for move in by_declaration.values()
                ],
            }
        mismatch_details = [
            {
                "candidate": _move_audit(move, level=level),
                "mismatch_reasons": reasons,
            }
            for move, reasons in candidate_checks
        ]
        physical_cards = sorted(_card_code(card) for card in card_ids)
        wildcard_count = sum(is_wildcard(card, level) for card in card_ids)
        if len(by_declaration) == 1 and wildcard_count == 0:
            move = next(iter(by_declaration.values()))
            reasons = mismatch_details[0]["mismatch_reasons"]
            reason_text = "；".join(str(reason) for reason in reasons)
            warning = {
                "code": "unique_physical_move_metadata_mismatch",
                "message": (
                    f"第 {history_index} 条历史不含逢人配，实体牌仅有一个合法解释；"
                    f"已采用 FableDan 唯一候选 {TYPE_NAMES[move.type]}，忽略不一致的动作语义元数据。"
                    f"差异：{reason_text}"
                ),
                "provided_interpretation": selected_semantics,
                "selected_interpretation": _move_audit(move, level=level),
                "mismatch_reasons": reasons,
            }
            return move, {
                "selection_source": "inferred_unique_after_metadata_warning",
                "provided_selection_source": selection_source,
                "ambiguity": False,
                "selected_interpretation": _move_audit(move, level=level),
                "candidate_interpretations": [_move_audit(move, level=level)],
                "metadata_warning": warning,
            }
        candidate_text = "；".join(
            f"{item['candidate']['type']} key={item['candidate']['key']}："
            + "、".join(str(reason) for reason in item["mismatch_reasons"])
            for item in mismatch_details
        )
        code = (
            "action_semantics_not_unique"
            if len(matches) > 1
            else "action_semantics_mismatch"
        )
        reason = (
            "已记录语义仍匹配多个 FableDan 候选，无法唯一确定动作"
            if len(matches) > 1
            else "已记录语义不能匹配任何 FableDan 候选"
        )
        diagnostic = {
            "code": code,
            "source_turn_id": history_index,
            "physical_cards": physical_cards,
            "level": level_rank,
            "wild_rank": level_rank,
            "wildcard_count": wildcard_count,
            "provided_interpretation": selected_semantics,
            "candidate_interpretations": [
                _move_audit(move, level=level) for move in by_declaration.values()
            ],
            "candidate_match_diagnostics": mismatch_details,
            "reason": reason,
        }
        raise FableDanStateError(
            f"第 {history_index} 条历史动作语义不一致：实体牌 "
            f"{' '.join(physical_cards)}；记录语义 "
            f"{json.dumps(selected_semantics, ensure_ascii=False, sort_keys=True)}；"
            f"{reason}。候选差异：{candidate_text}",
            diagnostic=diagnostic,
        )
    if len(by_declaration) != 1:
        declarations = ", ".join(
            str(value) for value in sorted(by_declaration, key=str)
        )
        diagnostic = {
            "code": "wildcard_ambiguity",
            "source_turn_id": history_index,
            "physical_cards": sorted(_card_code(card) for card in card_ids),
            "level": level_rank,
            "wild_rank": level_rank,
            "ambiguity": True,
            "candidate_declarations": [
                {
                    "type_id": int(move.type),
                    "key": int(move.key),
                    "claim_rank_ids": sorted(int(rank) for rank in move.claim_ranks),
                    "tuple": [
                        int(move.type),
                        int(move.key),
                        sorted(int(rank) for rank in move.claim_ranks),
                    ],
                }
                for move in by_declaration.values()
            ],
            "candidate_interpretations": [
                _move_audit(move, level=level) for move in by_declaration.values()
            ],
            "selected_interpretation": None,
            "selection_source": "unresolved",
            "reason": "只有实体牌信息，多个 wildcard 声明会消费同一组实体牌，不能静默选择",
        }
        raise FableDanStateError(
            f"第 {history_index} 条历史的 wildcard 声明不唯一：实体牌 "
            f"{' '.join(diagnostic['physical_cards'])} 在级牌 {level_rank} 下有 "
            f"{len(by_declaration)} 种解释（{declarations}）；"
            "缺少已确认的动作语义，不能替模型静默选择",
            diagnostic=diagnostic,
        )
    move = next(iter(by_declaration.values()))
    return move, {
        "selection_source": "inferred_unique",
        "ambiguity": False,
        "selected_interpretation": _move_audit(move, level=level),
        "candidate_interpretations": [_move_audit(move, level=level)],
    }


def _move_declaration(move: Move) -> tuple[object, ...]:
    return move.type, move.key, tuple(sorted(int(rank) for rank in move.claim_ranks))


def _selected_move_semantics(
    metadata: dict[str, object] | None,
) -> dict[str, object] | None:
    if not metadata or metadata.get("selection_source") == "unresolved":
        return None
    selected = metadata.get("selected_interpretation")
    if isinstance(selected, dict):
        return selected
    if any(key in metadata for key in ("move_type", "play_type", "type_id")):
        return metadata
    return None


def _move_matches_semantics(
    move: Move,
    semantics: dict[str, object],
    *,
    level: int,
    selection_source: str = "exact_engine_state",
) -> bool:
    return not _move_semantics_mismatches(
        move,
        semantics,
        level=level,
        selection_source=selection_source,
    )


def _move_semantics_mismatches(
    move: Move,
    semantics: dict[str, object],
    *,
    level: int,
    selection_source: str,
) -> list[str]:
    mismatches: list[str] = []
    actual_type = TYPE_NAMES[move.type]
    type_id = semantics.get("type_id")
    if type_id is not None:
        try:
            if int(type_id) != move.type:
                mismatches.append(
                    f"记录 type_id={type_id}，候选 type_id={move.type}（{actual_type}）"
                )
        except (TypeError, ValueError):
            mismatches.append(f"记录 type_id={type_id!r} 不是有效整数")
    move_type = semantics.get("move_type", semantics.get("play_type"))
    if move_type is not None:
        expected = canonical_fabledan_type(move_type)
        if expected is None:
            mismatches.append(f"记录牌型 {move_type!r} 无法识别")
        elif expected != actual_type:
            mismatches.append(
                f"记录牌型 {move_type!r} 归一化为 {expected}，候选牌型为 {actual_type}"
            )
    claim_ranks = semantics.get("claim_ranks")
    if isinstance(claim_ranks, (list, tuple)) and claim_ranks:
        normalized_claims = sorted(_normalize_semantic_rank(value) for value in claim_ranks)
        actual_claims = sorted(RANK_NAMES[int(rank)] for rank in move.claim_ranks)
        if normalized_claims != actual_claims:
            mismatches.append(
                f"记录声明点数={normalized_claims}，候选声明点数={actual_claims}"
            )
    assignments = semantics.get(
        "wildcard_assignments", semantics.get("wildcard_substitutions")
    )
    if isinstance(assignments, (list, tuple)) and assignments:
        expected_assignments = sorted(
            _normalize_semantic_rank(item.get("as_rank", ""))
            for item in assignments
            if isinstance(item, dict)
        )
        actual_assignments = sorted(
            _normalize_semantic_rank(item["as_rank"])
            for item in _move_audit(move, level=level).get(
                "wildcard_assignments", ()
            )
            if isinstance(item, dict)
        )
        if expected_assignments and actual_assignments != expected_assignments:
            mismatches.append(
                f"记录 wildcard 代替点数={expected_assignments}，"
                f"候选代替点数={actual_assignments}"
            )
    key = semantics.get("key")
    if key is not None:
        normalized_key, key_error = _normalized_semantic_key(
            key,
            move_type=actual_type,
            level=level,
            semantics=semantics,
            selection_source=selection_source,
        )
        if key_error:
            mismatches.append(key_error)
        elif normalized_key != int(move.key):
            mismatches.append(
                f"记录 key={key!r} 转换为 FableDan key={normalized_key}，"
                f"候选 key={int(move.key)}"
            )
    return mismatches


def _normalized_semantic_key(
    value: object,
    *,
    move_type: str,
    level: int,
    semantics: dict[str, object],
    selection_source: str,
) -> tuple[int | None, str | None]:
    source = selection_source.strip().casefold()
    project_sources = {
        "realtime_semantics",
        "ui_detection",
        "candidate_branch",
        "inferred_unique",
    }
    if semantics.get("type_id") is not None and source not in project_sources:
        try:
            return int(value), None
        except (TypeError, ValueError):
            return None, f"记录的 FableDan 内部 key={value!r} 不是有效整数"
    if move_type in {"PASS", "ROCKET"}:
        return 0, None
    rank_name = _normalize_semantic_rank(value)
    if rank_name not in RANK_NAMES:
        return None, (
            f"记录 key={value!r} 无法按牌型 {move_type} 转换为 FableDan 点数"
        )
    rank = RANK_NAMES.index(rank_name)
    if move_type in {"STRAIGHT", "PLATE", "TUBE", "SFLUSH"}:
        if rank >= 13:
            return None, f"序列牌型 {move_type} 的 key 不能是王：{value!r}"
        return 1 if rank == 0 else rank + 1, None
    return order_of(rank, level), None


def _normalize_semantic_rank(value: object) -> str:
    raw = str(value).strip()
    aliases = {
        "T": "10",
        "t": "10",
        "B": "sj",
        "b": "sj",
        "小王": "sj",
        "small_joker": "sj",
        "SJ": "sj",
        "R": "BJ",
        "r": "BJ",
        "大王": "BJ",
        "big_joker": "BJ",
        "bj": "BJ",
    }
    return aliases.get(raw, raw.upper() if raw.upper() in RANK_NAMES else raw)


def _move_signature(move: Move) -> tuple[object, ...]:
    return (
        *_move_declaration(move),
        tuple(sorted(int(card) for card in move.cards)),
    )


def _move_audit(move: Move, *, level: int | None = None) -> dict[str, object]:
    physical_cards = sorted(_card_code(int(card)) for card in move.cards)
    wildcard_assignments: list[dict[str, object]] = []
    if level is not None:
        occurrences: Counter[str] = Counter()
        for card, claim_rank in zip(move.cards, move.claim_ranks):
            card_id = int(card)
            if not is_wildcard(card_id, level):
                continue
            code = _card_code(card_id)
            occurrences[code] += 1
            wildcard_assignments.append(
                {
                    "physical_card": code,
                    "physical_card_id": card_id,
                    "occurrence": occurrences[code],
                    "as_rank": RANK_NAMES[int(claim_rank)],
                    "as_rank_id": int(claim_rank),
                }
            )
    return {
        "play_type": TYPE_NAMES[move.type],
        "type": TYPE_NAMES[move.type],
        "type_id": int(move.type),
        "cards": physical_cards,
        "physical_cards": physical_cards,
        "claim_ranks": [RANK_NAMES[int(rank)] for rank in move.claim_ranks],
        "claim_rank_ids": [int(rank) for rank in move.claim_ranks],
        "key": int(move.key),
        "size": int(move.size),
        "wildcard_count": len(wildcard_assignments),
        "wildcard_assignments": wildcard_assignments,
        "is_pass": move.type == PASS,
        "is_bomb": bool(move.is_bombish()),
    }


def _move_text(move: Move | None) -> str:
    if move is None:
        return "-"
    if move.type == PASS:
        return "PASS"
    return "".join(RANK_NAMES[int(rank)] for rank in move.claim_ranks)


def _rank_candidates(
    legal: tuple[Move, ...],
    selected_index: int,
    q_values: tuple[float, ...] | None,
) -> tuple[DecisionCandidate, ...]:
    remaining = [index for index in range(len(legal)) if index != selected_index]
    if q_values is not None:
        remaining.sort(
            key=lambda index: (
                0 if _safe_q_value(q_values, index) is not None else 1,
                -(_safe_q_value(q_values, index) or 0.0),
                index,
            )
        )
    order = [selected_index, *remaining]
    return tuple(
        DecisionCandidate(
            rank=rank,
            action=legal[index],
            action_text=_move_text(legal[index]),
            q_value=_safe_q_value(q_values, index),
        )
        for rank, index in enumerate(order, start=1)
    )


def _q_values_audit(
    legal: tuple[Move, ...],
    q_values: tuple[float, ...] | None,
) -> list[dict[str, object]]:
    if q_values is None:
        return []
    return [
        {
            "legal_index": index,
            "action": _move_audit(move),
            "action_text": _move_text(move),
            "q": _safe_q_value(q_values, index),
        }
        for index, move in enumerate(legal)
    ]


def _safe_q_value(
    q_values: tuple[float, ...] | None,
    index: int,
) -> float | None:
    if q_values is None or not 0 <= index < len(q_values):
        return None
    value = q_values[index]
    return float(value) if np.isfinite(value) else None


def _decision_diagnostics(
    mapped: _MappedState,
    *,
    selected_index: int,
    q_values: tuple[float, ...] | None,
    candidates: tuple[DecisionCandidate, ...],
    tokens: list[int] | None,
    features: np.ndarray | None,
) -> dict[str, object]:
    warnings: list[str] = []
    errors: list[str] = []
    invariants: dict[str, object] = {}
    observation = mapped.observation
    player = observation.get("player")
    invariants["player_id_valid"] = player in {0, 1, 2, 3}
    invariants["relative_two_is_teammate"] = (
        mapped.audit.get("teammate_relative") == 2
        and mapped.audit.get("teammate_absolute") == 2
    )
    invariants["seat_mapping_consistent"] = mapped.audit.get("seat_mapping") == {
        "self": 0,
        "right": 1,
        "opposite": 2,
        "left": 3,
    }
    hand = observation.get("hand")
    original_hand = mapped.audit.get("hand")
    invariants["hand_count_matches"] = (
        isinstance(hand, list)
        and isinstance(original_hand, list)
        and len(hand) == len(original_hand)
    )
    invariants["duplicate_cards_preserved"] = (
        isinstance(hand, list)
        and isinstance(original_hand, list)
        and Counter(_card_code(int(card)) for card in hand) == Counter(original_hand)
    )
    left = observation.get("left")
    invariants["left_count_shape_valid"] = isinstance(left, list) and len(left) == 4
    invariants["lead_owner_valid"] = (
        mapped.lead_owner is None or mapped.lead_owner in {0, 1, 2, 3}
    )
    invariants["legal_actions_non_empty"] = bool(mapped.legal)
    invariants["all_legal_actions_generated_by_rules_engine"] = True
    invariants["selected_index_in_range"] = 0 <= selected_index < len(mapped.legal)
    invariants["candidate_count_matches_legal"] = len(candidates) == len(mapped.legal)
    invariants["selected_action_matches_legal_index"] = (
        bool(candidates)
        and 0 <= selected_index < len(mapped.legal)
        and _move_signature(candidates[0].action)
        == _move_signature(mapped.legal[selected_index])
    )
    available = Counter(mapped.hand_ids)
    invariants["all_legal_actions_covered_by_hand"] = all(
        not (Counter(int(card) for card in move.cards) - available)
        for move in mapped.legal
    )
    follow = observation.get("lead") is not None
    pass_indices = [index for index, move in enumerate(mapped.legal) if move.type == PASS]
    invariants["follow_pass_present"] = (not follow) or bool(pass_indices)
    history = mapped.audit.get("history", [])
    source_history = mapped.audit.get("project_snapshot", {}).get("play_history", [])
    invariants["events_order_continuous"] = isinstance(history, list) and [
        item.get("source_turn_id") for item in history if isinstance(item, dict)
    ] == list(range(1, len(history) + 1))
    invariants["pass_events_preserved"] = (
        isinstance(history, list)
        and isinstance(source_history, list)
        and sum(bool(item.get("is_pass")) for item in history if isinstance(item, dict))
        == sum(
            bool(item.get("is_pass"))
            for item in source_history
            if isinstance(item, dict)
        )
    )
    if q_values is not None:
        invariants["q_count_matches_legal"] = len(q_values) == len(mapped.legal)
        invariants["q_values_finite"] = all(np.isfinite(value) for value in q_values)
        invariants["selected_index_matches_argmax"] = (
            bool(q_values) and selected_index == int(np.argmax(np.asarray(q_values)))
        )
    else:
        invariants["q_count_matches_legal"] = None
        invariants["q_values_finite"] = None
        invariants["selected_index_matches_argmax"] = None
    invariants["features_first_dim_matches_legal"] = (
        features is None
        or (features.ndim >= 1 and int(features.shape[0]) == len(mapped.legal))
    )
    invariants["features_width_matches_schema"] = (
        features is None
        or (features.ndim == 2 and int(features.shape[1]) == FEAT_DIM)
    )
    invariants["tokens_non_empty"] = tokens is None or bool(tokens)
    lead = observation.get("lead")
    if lead is not None:
        last_play = next(
            (
                event
                for event in reversed(mapped.audit.get("history", []))
                if isinstance(event, dict) and event.get("kind") == "play"
            ),
            None,
        )
        invariants["lead_owner_matches_latest_play"] = (
            isinstance(last_play, dict)
            and last_play.get("player_id") == mapped.lead_owner
        )
        invariants["lead_matches_latest_play"] = (
            isinstance(last_play, dict)
            and last_play.get("move")
            == _move_audit(lead, level=int(observation["level"]))
        )
    else:
        invariants["lead_owner_matches_latest_play"] = mapped.lead_owner is None
        invariants["lead_matches_latest_play"] = True
    for name, passed in invariants.items():
        if passed is False:
            errors.append(f"诊断不变量未通过：{name}")
    for item in history if isinstance(history, list) else []:
        if not isinstance(item, dict):
            continue
        resolution = item.get("semantic_resolution")
        if not isinstance(resolution, dict):
            continue
        metadata_warning = resolution.get("metadata_warning")
        if isinstance(metadata_warning, dict) and metadata_warning.get("message"):
            warnings.append(str(metadata_warning["message"]))
        if resolution.get("ambiguity"):
            warnings.append(
                f"第 {item.get('source_turn_id')} 条历史使用显式动作语义消解了 wildcard 多候选"
            )
    return {"warnings": warnings, "errors": errors, "invariants": invariants}


def _decision_trace_payload(
    *,
    request_id: str,
    mapped: _MappedState,
    runtime: _PolicyRuntime,
    diagnostics_mode: DiagnosticsMode,
    selected_index: int,
    q_values: tuple[float, ...] | None,
    candidates: tuple[DecisionCandidate, ...],
    tokens: list[int] | None,
    features: np.ndarray | None,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    observation = mapped.observation
    level = int(observation["level"])
    audit = mapped.audit
    project_snapshot = audit["project_snapshot"]
    original_hand = list(audit.get("hand", ()))
    rank_counts = Counter(RANK_NAMES[rank_of(int(card))] for card in mapped.hand_ids)
    event_trace = list(audit.get("events", ()))
    legal_actions = [
        {
            "legal_index": index,
            **_move_audit(move, level=level),
            "readable_action": _move_text(move),
        }
        for index, move in enumerate(mapped.legal)
    ]
    encoding = _encoding_trace(
        observation,
        mapped.legal,
        tokens=tokens,
        features=features,
        diagnostics_mode=diagnostics_mode,
    )
    q_rows = []
    if q_values is not None:
        q_rows = [
            {
                "legal_index": index,
                "action": legal_actions[index],
                "readable_action": _move_text(move),
                "q": float(q_values[index]),
            }
            for index, move in enumerate(mapped.legal)
        ]
        ranking_indices = sorted(
            range(len(q_values)), key=lambda index: (-q_values[index], index)
        )
    else:
        ranking_indices = [selected_index] + [
            index for index in range(len(mapped.legal)) if index != selected_index
        ]
    q_ranking = [
        {
            "rank": rank,
            "legal_index": index,
            "readable_action": _move_text(mapped.legal[index]),
            "q": _safe_q_value(q_values, index),
        }
        for rank, index in enumerate(ranking_indices, start=1)
        if 0 <= index < len(mapped.legal)
    ]
    second_index = ranking_indices[1] if len(ranking_indices) > 1 else None
    selected_q = _safe_q_value(q_values, selected_index)
    second_q = _safe_q_value(q_values, second_index) if second_index is not None else None
    source_state = {
        "state_revision": project_snapshot.get("revision"),
        "round_level": project_snapshot.get("round_level"),
        "wild_rank": project_snapshot.get("wild_rank"),
        "viewer_absolute": audit.get("viewer_absolute"),
        "seat_mapping": audit.get("seat_mapping"),
        "current_player_absolute": audit.get("current_player_absolute"),
        "current_player_relative": audit.get("current_player_relative"),
        "teammate_absolute": audit.get("teammate_absolute"),
        "teammate_relative": audit.get("teammate_relative"),
        "project_snapshot": project_snapshot,
    }
    adapter_observation: dict[str, object] = {
        "player": {
            "absolute_id": observation["player"],
            "viewer_absolute": audit.get("viewer_absolute"),
            "relative_id": 0,
            "seat": "self",
            "seat_mapping": audit.get("seat_mapping"),
        },
        "level": {
            "level_id": level,
            "level": RANK_NAMES[level],
            "wild_rank": RANK_NAMES[level],
            "wildcard_card_definition": {
                "physical_card": f"{RANK_NAMES[level]}H",
                "rule": "heart card of the current level",
            },
        },
        "hand": {
            "original_cards": original_hand,
            "internal_representation": [int(card) for card in mapped.hand_ids],
            "rank_counts": dict(sorted(rank_counts.items())),
            "total_card_count": len(mapped.hand_ids),
            "wildcard_count": sum(
                1 for card in mapped.hand_ids if is_wildcard(int(card), level)
            ),
        },
        "left": list(observation["left"]),
        "done": list(observation["done"]),
        "lead": audit.get("lead"),
        "lead_owner_absolute": audit.get("lead_owner_absolute"),
        "lead_owner_relative": audit.get("lead_owner_relative"),
        "lead_owner_is_teammate": audit.get("lead_owner_is_teammate"),
        "events": event_trace,
    }
    if diagnostics_mode == "full":
        adapter_observation["actual_observation"] = {
            "level": level,
            "player": int(observation["player"]),
            "hand": [int(card) for card in observation["hand"]],
            "left": list(observation["left"]),
            "done": list(observation["done"]),
            "events": event_trace,
            "lead": audit.get("lead"),
            "legal": legal_actions,
        }
    return {
        "schema": DECISION_TRACE_SCHEMA,
        "schema_version": 1,
        "status": "completed",
        "diagnostics_mode": diagnostics_mode,
        "run_id": None,
        "request_id": request_id,
        "decision_id": None,
        "turn_id": None,
        "trick_id": None,
        "state_revision": project_snapshot.get("revision"),
        "state_before_sha256": None,
        "source_state": source_state,
        "adapter_observation": adapter_observation,
        "legal_actions": {
            "legal_count_before_any_cap": len(mapped.legal),
            "legal_count_after_cap": len(mapped.legal),
            "cap_applied": False,
            "cap_limit": None,
            "ordering_strategy": "fabledan.gen_moves native order",
            "actions": legal_actions,
        },
        "encoding": encoding,
        "model_output": {
            "backend": runtime.backend,
            "model_path": str(runtime.path),
            "model_hash": runtime.digest,
            "q_values": q_rows,
            "q_ranking": q_ranking,
            "selected_index": selected_index,
            "selected_action": legal_actions[selected_index],
            "selected_q": selected_q,
            "second_best_index": second_index,
            "second_best_q": second_q,
            "q_margin": (
                selected_q - second_q
                if selected_q is not None and second_q is not None
                else None
            ),
            "selection_reason": (
                f"np.argmax(q_values) returned legal_index {selected_index}"
                if q_values is not None
                else "RuleAgent returned the legal action index"
            ),
        },
        "diagnostics": diagnostics,
    }


def _encoding_trace(
    observation: dict[str, object],
    legal: tuple[Move, ...],
    *,
    tokens: list[int] | None,
    features: np.ndarray | None,
    diagnostics_mode: DiagnosticsMode,
) -> dict[str, object]:
    token_values = [int(value) for value in tokens] if tokens is not None else []
    token_array = np.asarray(token_values, dtype="<i8")
    token_trace: dict[str, object] = {
        "token_length": len(token_values),
        "tokens_sha256": sha256(token_array.tobytes()).hexdigest(),
    }
    if diagnostics_mode == "full":
        token_trace["tokens"] = token_values
        token_trace["decoded_tokens"] = [
            {"position": index, "token_id": token, "name": _token_name(token)}
            for index, token in enumerate(token_values)
        ]
    feature_trace: dict[str, object] = {
        "feats_shape": list(features.shape) if features is not None else None,
        "feats_dtype": str(features.dtype) if features is not None else None,
        "feats_sha256": (
            sha256(np.ascontiguousarray(features).tobytes()).hexdigest()
            if features is not None
            else None
        ),
        "feature_summary": [
            _feature_summary(observation, move, index)
            for index, move in enumerate(legal)
        ],
    }
    if diagnostics_mode == "full" and features is not None:
        feature_trace["feats"] = features.tolist()
    return {"tokens": token_trace, "features": feature_trace}


def _training_input_payload(
    *,
    mapped: _MappedState,
    runtime: _PolicyRuntime,
    tokens: list[int] | None,
    features: np.ndarray | None,
) -> dict[str, object]:
    """Return the compact, self-contained input required for offline labels.

    The values are serialized directly rather than reconstructed from a later
    game state.  That makes training samples reproducible if card-recognition
    or action-semantic logic changes after a real game was recorded.
    """

    token_values = [int(value) for value in tokens or ()]
    if features is None:
        feature_values: list[list[float]] = []
    else:
        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(mapped.legal), FEAT_DIM):
            raise RuntimeError("FableDan 编码特征维度与合法动作集合不一致")
        feature_values = matrix.tolist()
    token_array = np.asarray(token_values, dtype="<i8")
    feature_array = np.ascontiguousarray(
        np.asarray(feature_values, dtype="<f4")
    )
    return {
        "schema": "fabledan-decision-input/1",
        "feature_schema": "fabledan-token48-feat80/v1",
        "tokens": token_values,
        "tokens_sha256": sha256(token_array.tobytes()).hexdigest(),
        "features": feature_values,
        "features_sha256": sha256(feature_array.tobytes()).hexdigest(),
        "legal_action_count": len(mapped.legal),
        "model_hash": runtime.digest,
        "adapter_schema": ADAPTER_SCHEMA,
        "upstream_commit": UPSTREAM_COMMIT,
        "standard_no_tribute": STANDARD_NO_TRIBUTE,
    }


def _feature_summary(
    observation: dict[str, object], move: Move, legal_index: int
) -> dict[str, object]:
    level = int(observation["level"])
    lead = observation.get("lead")
    return {
        "legal_index": legal_index,
        "hand_count": len(observation["hand"]),
        "hand_wildcard_count": sum(
            1 for card in observation["hand"] if is_wildcard(int(card), level)
        ),
        "left_counts": list(observation["left"]),
        "done": list(observation["done"]),
        "level": RANK_NAMES[level],
        "action_type": TYPE_NAMES[move.type],
        "action_key": int(move.key),
        "action_size": int(move.size),
        "action_wildcard_count": sum(
            1 for card in move.cards if is_wildcard(int(card), level)
        ),
        "lead_type": TYPE_NAMES[lead.type] if isinstance(lead, Move) else None,
        "lead_key": int(lead.key) if isinstance(lead, Move) else None,
    }


def _blocked_trace_payload(
    *,
    request_id: str,
    state: GuanDanState,
    error: Exception,
    diagnostics_mode: DiagnosticsMode,
) -> dict[str, object]:
    diagnostic = (
        error.diagnostic
        if isinstance(error, FableDanStateError) and error.diagnostic is not None
        else {"code": "adapter_observation_failed", "reason": str(error)}
    )
    source_state = {
        "state_revision": state.revision,
        "round_level": state.round_level,
        "wild_rank": state.wild_rank,
        "viewer_absolute": 0,
        "seat_mapping": dict(_SEAT_TO_PLAYER),
        "current_player_absolute": (
            _SEAT_TO_PLAYER.get(state.current_player) if state.current_player else None
        ),
        "current_player_relative": (
            _SEAT_TO_PLAYER.get(state.current_player) if state.current_player else None
        ),
        "teammate_absolute": 2,
        "teammate_relative": 2,
        "project_snapshot": {
            "round_level": state.round_level,
            "wild_rank": state.wild_rank,
            "current_player": state.current_player,
            "lead_player": state.lead_player,
            "my_hand": list(state.my_hand),
            "play_history": [event.to_dict() for event in state.play_history],
            "remaining_cards": state.remaining_cards,
            "revision": state.revision,
        },
    }
    return {
        "schema": DECISION_TRACE_SCHEMA,
        "schema_version": 1,
        "status": "adapter_error",
        "diagnostics_mode": diagnostics_mode,
        "run_id": None,
        "request_id": request_id,
        "decision_id": None,
        "turn_id": None,
        "trick_id": None,
        "state_revision": state.revision,
        "state_before_sha256": None,
        "source_state": source_state,
        "adapter_observation": None,
        "legal_actions": None,
        "encoding": None,
        "model_output": None,
        "diagnostics": {
            "warnings": [],
            "errors": [str(error)],
            "invariants": {
                "relative_two_is_teammate": True,
                "adapter_observation_constructed": False,
            },
            "root_cause": diagnostic,
        },
    }


def _normalize_diagnostics_mode(value: object) -> DiagnosticsMode:
    normalized = str(value or "off").strip().lower()
    if normalized not in {"off", "basic", "full"}:
        raise ValueError(f"不支持的 FableDan 诊断模式：{value}")
    return normalized  # type: ignore[return-value]


def _relative_player_label(relative: int) -> str:
    return ("self", "next", "partner", "prev")[int(relative) % 4]


def _token_name(token: int) -> str:
    value = int(token)
    if value == BOS_TOK:
        return "BOS"
    if LEVEL_BASE <= value < LEVEL_BASE + 13:
        return f"LEVEL_{RANK_NAMES[value - LEVEL_BASE]}"
    if PLAYER_BASE <= value < PLAYER_BASE + 4:
        return f"PLAYER_{_relative_player_label(value - PLAYER_BASE).upper()}"
    if TYPE_BASE <= value < TYPE_BASE + len(TYPE_NAMES):
        return TYPE_NAMES[value - TYPE_BASE]
    if value == TRIBUTE_TOK:
        return "TRIBUTE"
    if value == RETURN_TOK:
        return "RETURN"
    if RANK_BASE <= value < RANK_BASE + len(RANK_NAMES):
        return f"RANK_{RANK_NAMES[value - RANK_BASE]}"
    if value == 0:
        return "PAD"
    return f"UNKNOWN_{value}"


def _base_card_id(code: str) -> int:
    if code == "small_joker":
        return 52
    if code == "big_joker":
        return 53
    if not _is_known_card(code):
        raise FableDanStateError(f"无效牌码：{code}")
    rank = code[:-1]
    suit = code[-1]
    return RANK_NAMES.index(rank) * 4 + _SUIT_TO_INDEX[suit]


def _card_code(card_id: int) -> str:
    base = int(card_id) % 54
    if base == 52:
        return "small_joker"
    if base == 53:
        return "big_joker"
    return f"{RANK_NAMES[base // 4]}{_INDEX_TO_SUIT[base % 4]}"


def _is_known_card(code: str) -> bool:
    if code in {"small_joker", "big_joker"}:
        return True
    return len(code) >= 2 and code[:-1] in RANKS and code[-1] in _SUIT_TO_INDEX


def _validate_numpy_model(model: NumpyModel) -> None:
    required = {
        "token_emb.weight",
        "rope_cos",
        "rope_sin",
        "final_norm.weight",
    }
    missing = sorted(required - set(model.w))
    if missing:
        raise ValueError("模型缺少数组：" + ", ".join(missing))
    if model.n_blocks < 1 or model.n_heads < 1 or model.qk < 2 or model.v < 1:
        raise ValueError("FableDan 模型配置无效")
    for index in range(model.n_blocks):
        prefix = f"blocks.{index}."
        block_required = {
            prefix + "attn_norm.weight",
            prefix + "ffn_norm.weight",
            prefix + "attn.q_proj.weight",
            prefix + "attn.k_proj.weight",
            prefix + "attn.v_proj.weight",
            prefix + "attn.out_proj.weight",
            prefix + "attn.q_norm.weight",
            prefix + "attn.k_norm.weight",
            prefix + "ffn.gate_proj.weight",
            prefix + "ffn.up_proj.weight",
            prefix + "ffn.down_proj.weight",
        }
        missing = sorted(block_required - set(model.w))
        if missing:
            raise ValueError("模型缺少数组：" + ", ".join(missing))
    if not any(key.startswith("hand_mlp.") for key in model.w):
        raise ValueError("模型缺少 hand_mlp 数组")
    if not any(key.startswith("q_head.") for key in model.w):
        raise ValueError("模型缺少 q_head 数组")


def _safe_digest(path: Path) -> str | None:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
