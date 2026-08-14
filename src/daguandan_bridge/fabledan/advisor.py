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

from ..config import PROFILES_ROOT
from ..danzero.state import (
    GameStateError,
    GuanDanState,
    LocalStrategySnapshot,
    PlayEvent,
    RANKS,
    Seat,
)
from ..domain.advice import AdviceResult, StrategyExecutionTrace
from ._vendor.fabledan.agents import NumpyAgent, RuleAgent
from ._vendor.fabledan.cards import RANK_NAMES, rank_of
from ._vendor.fabledan.combos import (
    PASS,
    TYPE_NAMES,
    Move,
    beats,
    gen_moves,
)
from ._vendor.fabledan.encode import encode_decision
from ._vendor.fabledan.model_np import NumpyModel


UPSTREAM_COMMIT = "7cc5e311b9860bc44f76c082d9c1b21fc8b2d3ec"
ADAPTER_SCHEMA = "fabledan-adapter/v1"
STANDARD_NO_TRIBUTE = True
DECISION_LOG_SCHEMA = "fabledan-decision/1"
_TURN_ORDER: tuple[Seat, ...] = ("self", "right", "opposite", "left")
_SEAT_TO_PLAYER: dict[Seat, int] = {
    seat: index for index, seat in enumerate(_TURN_ORDER)
}
_PARTNER: dict[Seat, Seat] = {
    "self": "opposite",
    "opposite": "self",
    "right": "left",
    "left": "right",
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

    requires_exact_history_suits = True

    def __init__(
        self,
        profiles_root: Path | str = PROFILES_ROOT,
        profile_name: str = "tencent_daguandan",
        *,
        runtime_policy: Literal["auto", "model_required", "rule_only"] = "auto",
        debug: bool = False,
        write_decision_log: bool = True,
        log_directory: Path | str | None = None,
        top_n: int = 5,
    ) -> None:
        if runtime_policy not in {"auto", "model_required", "rule_only"}:
            raise ValueError(f"unsupported FableDan runtime policy: {runtime_policy}")
        if int(top_n) < 1:
            raise ValueError("top_n must be at least 1")
        self.profiles_root = Path(profiles_root)
        self.profile_name = str(profile_name)
        self.runtime_policy = runtime_policy
        self.debug = bool(debug)
        self.write_decision_log = bool(write_decision_log)
        self.top_n = int(top_n)
        self.log_directory = (
            Path(log_directory)
            if log_directory is not None
            else Path(__file__).resolve().parents[3]
            / "logs"
            / "fabledan_decisions"
        )
        self.weights_path = (
            self.profiles_root
            / self.profile_name
            / "models"
            / "fabledan_weights.npz"
        )
        self._runtime: _PolicyRuntime | None = None
        self._runtime_lock = Lock()
        self._log_lock = Lock()

    def initialize(self) -> None:
        runtime = self._ensure_runtime()
        self._require_model_runtime(runtime)

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
        encoding_audit: dict[str, object] = {}
        try:
            selected_index, q_values, encoding_audit = self._evaluate_policy(
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
            selected_index, q_values, encoding_audit = self._evaluate_policy(
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
        warnings = (
            _validate_decision_state(
                mapped,
                selected_index=selected_index,
                q_values=q_values,
                candidates=candidates,
            )
            if self.debug
            else []
        )
        engine_input["selected_action_index"] = selected_index
        engine_input["selected_action"] = _move_audit(move)
        engine_input.update(self._runtime_audit(active_runtime))
        engine_input["debug"] = self.debug
        decision_audit = {
            "best_action": _move_audit(move),
            "best_action_text": _move_text(move),
            "best_q": best_q,
            "second_q": second_q,
            "q_gap": q_gap,
            "legal_action_count": len(mapped.legal),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "warnings": list(warnings),
        }
        if self.debug:
            engine_input.update(
                {
                    "top_n": self.top_n,
                    "encoding": encoding_audit,
                    "q_values": _q_values_audit(mapped.legal, q_values),
                    "decision": decision_audit,
                    "validation_warnings": list(warnings),
                }
            )
            timestamp = datetime.now().astimezone()
            log_payload = _decision_log_payload(
                timestamp=timestamp,
                engine_input=engine_input,
                decision=decision_audit,
            )
            if self.write_decision_log:
                try:
                    log_path = self._append_decision_log(timestamp, log_payload)
                    engine_input["decision_log_path"] = str(log_path)
                except Exception as exc:
                    warning = f"FableDan 决策日志写入失败：{exc}"
                    warnings.append(warning)
                    decision_audit["warnings"] = list(warnings)
                    engine_input["validation_warnings"] = list(warnings)
                    _LOGGER.warning(warning, exc_info=True)
            else:
                # 整局评测把同一记录嵌入自己的原子产物，无需再写实时日志。
                engine_input["decision_log"] = log_payload

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
    ) -> tuple[int, tuple[float, ...] | None, dict[str, object]]:
        if runtime.backend != "numpy":
            return int(runtime.agent.act(observation)), None, {
                "q_values_available": False,
                "reason": "RuleAgent does not produce model Q-values",
            }

        # 保持与上游 NumpyAgent 完全相同的推理路径，并让动作选择与调试信息
        # 共享同一次模型推理，避免调试模式引入第二次计算。
        tokens, features = encode_decision(observation)
        raw_q_values = runtime.agent.model.q_values(tokens, features)
        q_array = np.asarray(raw_q_values).reshape(-1)
        selected_index = int(np.argmax(q_array))
        return selected_index, tuple(float(value) for value in q_array), {
            "q_values_available": True,
            "token_count": len(tokens),
            "tokens": [int(token) for token in tokens],
            "feature_shape": [int(value) for value in features.shape],
        }

    def _append_decision_log(
        self,
        timestamp: datetime,
        payload: dict[str, object],
    ) -> Path:
        path = self.log_directory / f"{timestamp.date().isoformat()}.jsonl"
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
                    "weights file does not exist",
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
            "decision_log_mode": (
                "append"
                if self.debug and self.write_decision_log
                else "embedded"
                if self.debug
                else "disabled"
            ),
        }

    def _require_model_runtime(self, runtime: _PolicyRuntime) -> None:
        if self.runtime_policy == "model_required" and (
            runtime.backend != "numpy" or runtime.status != "loaded"
        ):
            raise RuntimeError(
                "FableDan model_required requires a valid fabledan_weights.npz: "
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
        counts = {seat: 27 for seat in _TURN_ORDER}
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
                        "is_pass": True,
                    }
                )
                passed.add(event.player)
            else:
                card_ids = tuple(allocation.allocate(card) for card in event.cards)
                move = _unique_observed_move(card_ids, snapshot.round_level, index)
                if trick_lead is not None and not beats(
                    move, trick_lead, RANK_NAMES.index(snapshot.round_level)
                ):
                    raise FableDanStateError(
                        f"第 {index} 条历史声明不能压过当前桌面"
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
                        "is_pass": False,
                        "move": _move_audit(move),
                        "action_text": _move_text(move),
                    }
                )

            active = set(_TURN_ORDER) - done
            if len(done) >= 3:
                expected = None
                trick_lead = None
                trick_leader = None
                passed.clear()
                continue
            if (
                trick_lead is not None
                and trick_leader is not None
                and (active - {trick_leader}).issubset(passed)
            ):
                next_leader = trick_leader
                if next_leader in done:
                    next_leader = _PARTNER[next_leader]
                if next_leader in done:
                    next_leader = _next_active(next_leader, done)
                expected = next_leader
                trick_lead = None
                trick_leader = None
                passed.clear()
            else:
                expected = _next_active(event.player, done)

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
            for seat in _TURN_ORDER
        }
        derived_counts["self"] = len(hand_ids)
        if snapshot.remaining_cards is not None:
            if set(snapshot.remaining_cards) != set(_TURN_ORDER):
                raise FableDanStateError("remaining_cards 必须包含四个标准座位")
            supplied = {
                seat: int(snapshot.remaining_cards[seat]) for seat in _TURN_ORDER
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
        left = [counts[seat] for seat in _TURN_ORDER]
        done_flags = [seat in done for seat in _TURN_ORDER]
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
                for seat in _TURN_ORDER
            },
            "done": done_flags,
            "lead": (
                _move_audit(reconstructed_lead)
                if reconstructed_lead is not None
                else None
            ),
            "lead_text": _move_text(reconstructed_lead),
            "lead_owner": lead_owner,
            "lead_owner_seat": trick_leader,
            "history": event_audit,
            "events": event_audit,
            "legal_actions": [_move_audit(move) for move in legal],
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
) -> Move:
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
        raise FableDanStateError(f"第 {history_index} 条历史不是合法牌型")
    if len(by_declaration) != 1:
        declarations = ", ".join(
            str(value) for value in sorted(by_declaration, key=str)
        )
        raise FableDanStateError(
            f"第 {history_index} 条历史的 wildcard 声明不唯一：{declarations}"
        )
    return next(iter(by_declaration.values()))


def _move_declaration(move: Move) -> tuple[object, ...]:
    return move.type, move.key, tuple(sorted(int(rank) for rank in move.claim_ranks))


def _move_signature(move: Move) -> tuple[object, ...]:
    return (
        *_move_declaration(move),
        tuple(sorted(int(card) for card in move.cards)),
    )


def _move_audit(move: Move) -> dict[str, object]:
    return {
        "play_type": TYPE_NAMES[move.type],
        "cards": sorted(_card_code(int(card)) for card in move.cards),
        "claim_ranks": [RANK_NAMES[int(rank)] for rank in move.claim_ranks],
        "key": int(move.key),
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


def _validate_decision_state(
    mapped: _MappedState,
    *,
    selected_index: int,
    q_values: tuple[float, ...] | None,
    candidates: tuple[DecisionCandidate, ...],
) -> list[str]:
    warnings: list[str] = []
    observation = mapped.observation
    player = observation.get("player")
    if player not in {0, 1, 2, 3}:
        warnings.append(f"player 无效：{player}")
    hand = observation.get("hand")
    if not isinstance(hand, list) or not 0 <= len(hand) <= 27:
        warnings.append("hand 数量不在 0..27 范围内")
    left = observation.get("left")
    if not isinstance(left, list) or len(left) != 4:
        warnings.append("left 必须包含四个玩家")
    if mapped.lead_owner is not None and mapped.lead_owner not in {0, 1, 2, 3}:
        warnings.append(f"lead_owner 无效：{mapped.lead_owner}")
    if not mapped.legal:
        warnings.append("legal actions 为空")
    if not 0 <= selected_index < len(mapped.legal):
        warnings.append("best action 索引不属于 legal actions")
    if len(candidates) != len(mapped.legal):
        warnings.append("candidates 数量与 legal actions 不一致")
    elif _move_signature(candidates[0].action) != _move_signature(
        mapped.legal[selected_index]
    ):
        warnings.append("candidates[0] 与模型最终动作不一致")
    if q_values is not None:
        if len(q_values) != len(mapped.legal):
            warnings.append("Q-value 数量与 legal actions 不一致")
        if any(not np.isfinite(value) for value in q_values):
            warnings.append("Q-value 中包含 NaN 或 Inf")
        finite_q = [
            candidate.q_value
            for candidate in candidates
            if candidate.q_value is not None
        ]
        if any(first < second for first, second in zip(finite_q, finite_q[1:])):
            warnings.append("candidates 未按 Q-value 降序排列")
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
        if not isinstance(last_play, dict) or last_play.get("player_id") != mapped.lead_owner:
            warnings.append("lead_owner 与最新非 PASS 事件不一致")
        if isinstance(last_play, dict) and last_play.get("move") != _move_audit(lead):
            warnings.append("lead 与最新非 PASS 事件不一致")
    return warnings


def _decision_log_payload(
    *,
    timestamp: datetime,
    engine_input: dict[str, object],
    decision: dict[str, object],
) -> dict[str, object]:
    model_path = str(engine_input.get("model_path") or engine_input.get("path", ""))
    return {
        "schema": DECISION_LOG_SCHEMA,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "request_id": engine_input.get("request_id", ""),
        "model_path": model_path,
        "model_filename": Path(model_path).name,
        "model_hash": engine_input.get("model_hash") or engine_input.get("digest"),
        "backend": engine_input.get("backend"),
        "model_status": engine_input.get("status"),
        "player": engine_input.get("player"),
        "player_mapping": engine_input.get("player_mapping"),
        "player_mapping_labels": engine_input.get("player_mapping_labels"),
        "level": engine_input.get("level"),
        "level_text": engine_input.get("level_text"),
        "hand": engine_input.get("hand"),
        "hand_ids": engine_input.get("hand_ids"),
        "left": engine_input.get("left"),
        "left_by_player": engine_input.get("left_by_player"),
        "lead": engine_input.get("lead"),
        "lead_text": engine_input.get("lead_text"),
        "lead_owner": engine_input.get("lead_owner"),
        "lead_owner_seat": engine_input.get("lead_owner_seat"),
        "done": engine_input.get("done"),
        "history": engine_input.get("history"),
        "events": engine_input.get("events"),
        "legal_actions": engine_input.get("legal_actions"),
        "legal_action_count": decision.get("legal_action_count"),
        "q_values": engine_input.get("q_values"),
        "candidates": decision.get("candidates"),
        "best_action": decision.get("best_action"),
        "best_action_text": decision.get("best_action_text"),
        "best_q": decision.get("best_q"),
        "second_q": decision.get("second_q"),
        "q_gap": decision.get("q_gap"),
        "encoding": engine_input.get("encoding"),
        "validation_warnings": engine_input.get("validation_warnings"),
        "project_snapshot": engine_input.get("project_snapshot"),
    }


def _next_active(current: Seat, done: set[Seat]) -> Seat:
    start = _TURN_ORDER.index(current)
    for offset in range(1, len(_TURN_ORDER) + 1):
        candidate = _TURN_ORDER[(start + offset) % len(_TURN_ORDER)]
        if candidate not in done:
            return candidate
    raise FableDanStateError("标准对局中没有仍在行动的玩家")


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
        raise ValueError("missing arrays: " + ", ".join(missing))
    if model.n_blocks < 1 or model.n_heads < 1 or model.qk < 2 or model.v < 1:
        raise ValueError("invalid FableDan model configuration")
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
            raise ValueError("missing arrays: " + ", ".join(missing))
    if not any(key.startswith("hand_mlp.") for key in model.w):
        raise ValueError("missing hand_mlp arrays")
    if not any(key.startswith("q_head.") for key in model.w):
        raise ValueError("missing q_head arrays")


def _safe_digest(path: Path) -> str | None:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
