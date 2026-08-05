from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from importlib.resources import files
import os
from pathlib import Path
import sys
from time import perf_counter
from threading import Lock
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np

if TYPE_CHECKING:
    from .state import LocalStrategySnapshot, PlayEvent
from .rules import action_for_cards, from_engine_card, to_engine_card


DEFAULT_STRATEGY = "danzero"
_STRATEGIES = (
    "danzero",
    "base1",
    "base2",
    "base3",
    "base4",
    "base5",
    "base6",
    "base7",
    "base8",
)
_SEAT_TO_PLAYER = {"self": 0, "left": 1, "opposite": 2, "right": 3}
_DANZERO_CHECKPOINT_SHA256 = "a6f132e50e205709c0efdb1b550d935c5bfc313b21726b0e70e300b6cd7ea1a3"


class LocalStrategyError(RuntimeError):
    """The current observed state cannot be evaluated by a local agent."""


class StrategyExecutionTrace:
    """Thread-safe progress data retained when a request fails or times out."""

    def __init__(self, request_id: str = "") -> None:
        self.request_id = str(request_id)
        self._lock = Lock()
        self._phase = ""
        self._phase_started = perf_counter()
        self._timings: dict[str, float] = {}
        self._engine_input: dict[str, object] | None = None

    def begin(self, phase: str) -> None:
        with self._lock:
            self._close_phase()
            self._phase = str(phase)
            self._phase_started = perf_counter()

    def end(self) -> None:
        with self._lock:
            self._close_phase()
            self._phase = ""

    def set_engine_input(self, value: dict[str, object]) -> None:
        with self._lock:
            self._engine_input = LocalGuandanAdvisor._json_safe(value)  # type: ignore[assignment]

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            current_phase = self._phase
            phase_elapsed = (
                (perf_counter() - self._phase_started) * 1000
                if current_phase
                else 0.0
            )
            timings = dict(self._timings)
            if current_phase:
                timings[current_phase] = (
                    timings.get(current_phase, 0.0) + phase_elapsed
                )
            return {
                "request_id": self.request_id,
                "current_phase": current_phase or None,
                "phase_elapsed_ms": round(phase_elapsed, 3),
                "timings": {
                    key: round(value, 3) for key, value in timings.items()
                },
                "engine_input": self._engine_input,
            }

    def _close_phase(self) -> None:
        if self._phase:
            self._timings[self._phase] = (
                self._timings.get(self._phase, 0.0)
                + (perf_counter() - self._phase_started) * 1000
            )


def _missing_dependency_error(
    strategy: str,
    exc: ModuleNotFoundError,
) -> LocalStrategyError:
    """Turn an import failure into a repairable, user-facing message."""
    module_name = str(getattr(exc, "name", "") or "")
    if not module_name:
        message = str(exc)
        for candidate in ("rlcard", "torch", "numpy"):
            if f"'{candidate}'" in message or f'"{candidate}"' in message:
                module_name = candidate
                break
    if module_name == "torch" or module_name.startswith("torch."):
        dependency = "PyTorch"
    elif module_name == "rlcard" or module_name.startswith("rlcard."):
        dependency = "rlcard"
    elif module_name.startswith("daguandan_bridge.danzero._vendor.guandan_rlcard"):
        dependency = "项目内置 guandan_rlcard"
    else:
        dependency = module_name or "本地策略运行依赖"
    return LocalStrategyError(
        f"本地策略 {strategy} 依赖缺失：{dependency}；"
        "请在当前运行 Python 解释器中执行："
        f'"{sys.executable}" -m pip install -r requirements.txt'
    )


@dataclass(frozen=True)
class LocalAdvice:
    strategy: str
    cards: tuple[str, ...]
    play_type: str
    is_pass: bool
    state_revision: int
    elapsed_ms: float
    request_id: str = ""
    engine_input: dict[str, object] | None = None
    timings: dict[str, float] = field(default_factory=dict)


def available_strategies() -> tuple[str, ...]:
    return _STRATEGIES


_SEAT_LABELS = {"self": "我方", "left": "左家", "opposite": "对家", "right": "右家"}


def _display_cards(cards: object) -> str:
    values = []
    for raw in cards or ():
        code = str(raw)
        if code == "small_joker":
            values.append("小王")
            continue
        if code == "big_joker":
            values.append("大王")
            continue
        values.append(code)
    return " ".join(values) or "—"


def format_engine_input_summary(engine_input: dict[str, object] | None) -> str:
    """Build a human-readable Chinese summary of the state sent to a strategy.

    The full JSON is retained in ``engine_input`` and logs; this formatter is
    for quick reading during live testing and audit review.
    """
    if not isinstance(engine_input, dict):
        return "（无策略输入）"
    project = engine_input.get("project_snapshot")
    if not isinstance(project, dict):
        return "（策略输入缺少 project_snapshot）"
    strategy = str(engine_input.get("strategy", "") or "")

    round_level = str(project.get("round_level", "") or "未设置")
    wild_rank = str(project.get("wild_rank", "") or "未设置")
    current = str(project.get("current_player", "") or "未设置")
    lead = str(project.get("lead_player", "") or "未设置")
    hand = tuple(str(card) for card in project.get("my_hand", ()))

    lines = [f"策略：{strategy or '未知'}"]
    lines.append(
        f"级牌：{round_level}（逢人配=红桃{round_level}）"
        if round_level == wild_rank and round_level != "未设置"
        else f"级牌：{round_level} | 百搭：{wild_rank}"
    )
    lines.append(
        f"当前行动：{_SEAT_LABELS.get(current, current)} | "
        f"首出：{_SEAT_LABELS.get(lead, lead)}"
    )
    lines.append(f"手牌（{len(hand)}张）：{_display_cards(hand)}")

    trick = project.get("trick")
    if isinstance(trick, list):
        events = [
            (str(item.get("player", "")), bool(item.get("is_pass", False)),
             item.get("cards"))
            for item in trick
            if isinstance(item, dict)
        ]
        if events:
            parts = []
            for player, is_pass, cards in events:
                label = _SEAT_LABELS.get(player, player)
                parts.append(
                    f"{label} 不出" if is_pass else f"{label} 出 {_display_cards(cards)}"
                )
            lines.append("本轮：" + "；".join(parts))

    history = project.get("history")
    if isinstance(history, list):
        events = [
            (str(item.get("player", "")), bool(item.get("is_pass", False)),
             item.get("cards"))
            for item in history
            if isinstance(item, dict)
        ]
        if events:
            parts = []
            for player, is_pass, cards in events:
                label = _SEAT_LABELS.get(player, player)
                parts.append(
                    f"{label} 不出" if is_pass else f"{label} 出 {_display_cards(cards)}"
                )
            lines.append("历史：" + "；".join(parts))

    legal_count = int(engine_input.get("legal_action_count", 0))
    lines.append(f"合法动作：{legal_count} 个")
    return "\n".join(lines)


def _missing_card_counts(
    cards: tuple[str, ...],
    hand: tuple[str, ...],
) -> Counter[str]:
    available = Counter(hand)
    missing: Counter[str] = Counter()
    for card, count in Counter(cards).items():
        deficit = count - available.get(card, 0)
        if deficit > 0:
            missing[card] = deficit
    return missing


def _cards_are_available(
    cards: tuple[str, ...],
    hand: tuple[str, ...],
) -> bool:
    return not _missing_card_counts(cards, hand)


def _is_legal_engine_action(
    candidate: list[object],
    legal_actions: list[list[object]],
) -> bool:
    if len(candidate) < 3:
        return False
    if str(candidate[0]) == "PASS":
        return any(str(action[0]) == "PASS" for action in legal_actions)
    candidate_cards = Counter(str(card) for card in candidate[2])
    return any(
        len(action) >= 3
        and str(action[0]) == str(candidate[0])
        and str(action[1]) == str(candidate[1])
        and Counter(str(card) for card in action[2]) == candidate_cards
        for action in legal_actions
    )


class LocalGuandanAdvisor:
    """Run a selected rlcard-guandan agent entirely inside this process."""

    def __init__(self, strategy: str = DEFAULT_STRATEGY) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"未知本地策略：{strategy}；可选：{', '.join(_STRATEGIES)}"
            )
        self.strategy = strategy
        self._agent: Any | None = None
        self._agent_lock = Lock()

    def initialize(self, *, trace: StrategyExecutionTrace | None = None) -> None:
        """Create the reusable local agent outside a user-facing recommendation."""
        if self._agent is not None:
            return
        with self._agent_lock:
            if self._agent is not None:
                return
            if trace is not None:
                trace.begin("create_agent")
            self._agent = self._create_agent()

    def recommend(
        self,
        snapshot: LocalStrategySnapshot,
        *,
        request_id: str = "",
        trace: StrategyExecutionTrace | None = None,
    ) -> LocalAdvice:
        self._validate_snapshot(snapshot)
        started = perf_counter()
        execution = trace or StrategyExecutionTrace(request_id)
        try:
            execution.begin("build_engine_state")
            level_index, actions, state, history = self._build_engine_state(snapshot)
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error(self.strategy, exc) from exc
        source_trace = self._json_safe(state["trace"])
        self.initialize(trace=execution)
        agent = self._agent
        assert agent is not None
        if self.strategy == "danzero":
            execution.begin("prime_danzero")
            self._prime_danzero(agent, snapshot, history)
            # The fresh agent must not replay the final events a second time.
            state["trace"] = []
        engine_input = self._engine_input_audit(
            snapshot,
            request_id=request_id,
            level_index=level_index,
            state=state,
            actions=actions,
            source_trace=source_trace,
        )
        execution.set_engine_input(engine_input)
        try:
            execution.begin("agent_step")
            action = agent.step(state)
        except Exception as exc:
            raise LocalStrategyError(
                f"本地策略 {self.strategy} 推理失败：{exc}"
            ) from exc
        execution.begin("validate_action")
        if not action:
            raise LocalStrategyError(f"本地策略 {self.strategy} 没有返回动作")
        cards = () if action[0] == "PASS" else tuple(
            from_engine_card(card) for card in action[2]
        )
        if not _cards_are_available(cards, snapshot.my_hand):
            missing = _missing_card_counts(cards, snapshot.my_hand)
            details = ", ".join(
                f"{card}\u00d7{count}" for card, count in sorted(missing.items())
            )
            raise LocalStrategyError(
                f"strategy {self.strategy} returned cards not in the current hand: {details}"
            )
        if not _is_legal_engine_action(action, actions):
            raise LocalStrategyError(
                f"strategy {self.strategy} returned an action that is not legal "
                "for the current trick"
            )
        execution.end()
        timing_data = execution.snapshot()["timings"]
        return LocalAdvice(
            strategy=self.strategy,
            cards=cards,
            play_type=str(action[0]),
            is_pass=action[0] == "PASS",
            state_revision=snapshot.revision,
            elapsed_ms=(perf_counter() - started) * 1000,
            request_id=request_id,
            engine_input=engine_input,
            timings=dict(timing_data),
        )

    @staticmethod
    def _json_safe(value: Any) -> object:
        if isinstance(value, dict):
            return {
                str(key): LocalGuandanAdvisor._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [LocalGuandanAdvisor._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _engine_input_audit(
        self,
        snapshot: LocalStrategySnapshot,
        *,
        request_id: str,
        level_index: int,
        state: dict[str, object],
        actions: list[list[object]],
        source_trace: object,
    ) -> dict[str, object]:
        """Return JSON-safe input snapshots for the local strategy call."""
        return {
            "request_id": request_id,
            "strategy": self.strategy,
            "project_snapshot": {
                "round_level": snapshot.round_level,
                "wild_rank": snapshot.wild_rank,
                "phase": snapshot.phase,
                "current_player": snapshot.current_player,
                "lead_player": snapshot.lead_player,
                "my_hand": list(snapshot.my_hand),
                "trick": [event.to_dict() for event in snapshot.trick_plays],
                "history": [event.to_dict() for event in snapshot.play_history],
                "revision": snapshot.revision,
            },
            "seat_to_player": dict(_SEAT_TO_PLAYER),
            "level_index": level_index,
            "source_trace_for_danzero_prime": source_trace,
            "agent_step_state": self._json_safe(state),
            "legal_action_count": len(actions),
            "legal_actions": self._json_safe(actions),
        }

    @staticmethod
    def _validate_snapshot(snapshot: LocalStrategySnapshot) -> None:
        errors = tuple(snapshot.readiness_errors)
        if errors:
            raise LocalStrategyError("；".join(errors))
        if snapshot.round_level != snapshot.wild_rank:
            raise LocalStrategyError("级牌与百搭牌级别必须相同")

    def _create_agent(self) -> Any:
        try:
            from daguandan_bridge.danzero._vendor.guandan_rlcard.baselines import get_agent_class

            agent_class = get_agent_class(self.strategy)
            previous_checkpoint = os.environ.get("GUANDAN_DANZERO_CKPT")
            if self.strategy == "danzero":
                os.environ["GUANDAN_DANZERO_CKPT"] = str(
                    self._danzero_checkpoint_path()
                )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The use of `x.T`.*",
                    category=UserWarning,
                    module=r"daguandan_bridge\.danzero\._vendor\.guandan_rlcard\.baselines\.danzero\.model",
                )
                try:
                    return agent_class(0, np.random.RandomState(0))
                finally:
                    if self.strategy == "danzero":
                        if previous_checkpoint is None:
                            os.environ.pop("GUANDAN_DANZERO_CKPT", None)
                        else:
                            os.environ["GUANDAN_DANZERO_CKPT"] = previous_checkpoint
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error(self.strategy, exc) from exc
        except Exception as exc:
            raise LocalStrategyError(
                f"无法初始化本地策略 {self.strategy}：{exc}"
            ) from exc

    @staticmethod
    def _danzero_checkpoint_path() -> Path:
        configured = os.environ.get("DAGUANDAN_DANZERO_CKPT")
        if configured:
            checkpoint = Path(configured)
        else:
            checkpoint = Path(
                files("daguandan_bridge.danzero._vendor.guandan_rlcard").joinpath(
                    "baselines",
                    "danzero",
                    "q_network.ckpt",
                )
            )
        if not checkpoint.is_file():
            raise LocalStrategyError(f"未找到本地 DanZero 权重：{checkpoint}")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != _DANZERO_CHECKPOINT_SHA256:
            raise LocalStrategyError("本地 DanZero 权重校验失败，已拒绝加载")
        return checkpoint

    def _build_engine_state(
        self,
        snapshot: LocalStrategySnapshot,
    ) -> tuple[int, list[list[object]], dict[str, object], list[list[object]]]:
        from daguandan_bridge.danzero._vendor.guandan_rlcard.constants import CARD_RANK, REMAIN_RANK_INDEX
        from daguandan_bridge.danzero._vendor.guandan_rlcard.game.action_compare import get_gt_actions
        from daguandan_bridge.danzero._vendor.guandan_rlcard.game.card_utils import card_from_str
        from daguandan_bridge.danzero._vendor.guandan_rlcard.game.judger import GuandanJudger
        from daguandan_bridge.danzero._vendor.guandan_rlcard.game.player import GuandanPlayer

        level_rank = "T" if snapshot.round_level == "10" else snapshot.round_level
        level_index = CARD_RANK.index(level_rank)
        engine_hand = [card_from_str(to_engine_card(card)) for card in snapshot.my_hand]
        all_actions = GuandanJudger.playable_actions_from_hand(engine_hand, level_index)
        history: list[list[object]] = [
            [_SEAT_TO_PLAYER[event.player], self._event_to_action(event, level_index)]
            for event in snapshot.play_history
        ]
        current_trick = [
            [_SEAT_TO_PLAYER[event.player], self._event_to_action(event, level_index)]
            for event in snapshot.trick_plays
        ]
        last_non_pass = next(
            ((player, action) for player, action in reversed(current_trick) if action[0] != "PASS"),
            None,
        )
        if last_non_pass is None:
            actions = all_actions
            greater_position = -1
            greater_action: list[object] = []
        else:
            greater_position, greater_action = last_non_pass
            greater_player = GuandanPlayer(greater_position, np.random.RandomState(0))
            greater_player.played_action = greater_action
            actions = get_gt_actions(level_index, greater_player, all_actions)
        if not actions:
            raise LocalStrategyError("当前局面没有可用的合法动作")
        remaining = self._remaining_counts(snapshot)
        played_cards: dict[int, list[str]] = {player: [] for player in range(4)}
        reactions: list[dict[str, list[str]]] = [{} for _ in range(4)]
        remain_cards = {
            "S": [2] * 14,
            "H": [2] * 14,
            "C": [2] * 13 + [0],
            "D": [2] * 13 + [0],
        }
        pass_num = [0, 0, 0, 0]
        my_pass_num = [0, 0, 0, 0]
        previous_combo = "None"
        for player, action in history:
            player = int(player)
            if action[0] == "PASS":
                combo_type, key_rank = previous_combo, "PASS"
                pass_num[player] += 1
                pass_num[(player + 2) % 4] += 1
                my_pass_num[player] += 1
            else:
                combo_type, key_rank = str(action[0]), str(action[1])
                previous_combo = combo_type
                pass_num[player] = 0
                pass_num[(player + 2) % 4] = 0
                my_pass_num[player] = 0
                for card in action[2]:
                    remain_cards[card[0]][REMAIN_RANK_INDEX[card[1]]] -= 1
                    played_cards[player].append(card)
            reactions[player].setdefault(combo_type, []).append(key_rank)
        state: dict[str, object] = {
            "rank_list": [level_index, level_index],
            "play_team": 0,
            "greaterAction": greater_action,
            "greaterPos": greater_position,
            "remain_cards": remain_cards,
            "pass_num": pass_num,
            "my_pass_num": my_pass_num,
            "played_cards": played_cards,
            "num_cards_left": remaining,
            "actions": actions,
            "current_hand": [to_engine_card(card) for card in snapshot.my_hand],
            "trace": history,
            "reactions": reactions,
            "self": 0,
            "teamid": 0,
        }
        return level_index, actions, state, history

    @staticmethod
    def _remaining_counts(snapshot: LocalStrategySnapshot) -> list[int]:
        counts = [27, 27, 27, 27]
        for event in snapshot.play_history:
            if not event.is_pass:
                counts[_SEAT_TO_PLAYER[event.player]] -= len(event.cards)
        counts[0] = len(snapshot.my_hand)
        if any(count < 0 or count > 27 for count in counts):
            raise LocalStrategyError("出牌历史与各座位剩余牌数不一致")
        return counts

    @staticmethod
    def _event_to_action(event: PlayEvent, level_index: int) -> list[object]:
        if event.is_pass:
            return ["PASS", "PASS", "PASS"]
        from daguandan_bridge.danzero._vendor.guandan_rlcard.constants import CARD_RANK

        level_rank = CARD_RANK[level_index]
        action = action_for_cards(
            event.cards,
            "10" if level_rank == "T" else level_rank,
        )
        if action is None:
            cards = "、".join(event.cards)
            raise LocalStrategyError(f"历史出牌不是当前级牌下的合法牌型：{cards}")
        return action

    @staticmethod
    def _prime_danzero(
        agent: Any,
        snapshot: LocalStrategySnapshot,
        history: list[list[object]],
    ) -> None:
        """Populate DanZero's local, non-public feature cache once."""
        from daguandan_bridge.danzero._vendor.guandan_rlcard.baselines.danzero.danutil import card2num

        known_cards = [to_engine_card(card) for card in snapshot.my_hand]
        for _player, action in history:
            if action[0] != "PASS":
                known_cards.extend(action[2])
        unseen = [2] * 54
        for card in card2num(known_cards):
            if card < 0 or unseen[card] == 0:
                raise LocalStrategyError("已确认手牌与出牌历史超过双副牌数量")
            unseen[card] -= 1
        agent.begin = False
        agent.other_left_hands = unseen
        agent.history_action = {player: [] for player in range(4)}
        agent.action_order = []
        agent.action_seq = []
        for player, action in history:
            numeric = [-1] if action[0] == "PASS" else card2num(action[2])
            agent.action_order.append(player)
            agent.action_seq.append(numeric)
            agent.history_action[player].append(numeric)
