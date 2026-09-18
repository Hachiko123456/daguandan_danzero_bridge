"""Pure semantic validation for canonical TruthLogs.

The validator deliberately performs no image recognition.  It replays the
recorded actions through the same card classifier, table comparison and turn
projection used by the production LiveV2 rule gateway.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..danzero.rules import actions_for_cards, play_beats_table
from ..live.truth_log import TruthLog, TruthLogCardInventoryError, validate_truth_log_card_inventory
from ..live.turns import TURN_ORDER, WindCatchPolicy, next_active_seat, project_trick_turn, round_is_decided

ValidationMode = Literal["logic", "publish"]


@dataclass(frozen=True)
class TruthLogFinding:
    code: str
    severity: Literal["error", "warning"]
    message: str
    turn_id: int | None = None
    field: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "turn_id": self.turn_id,
            "field": self.field,
        }


@dataclass(frozen=True)
class TruthLogValidationReport:
    session_id: str
    mode: ValidationMode
    findings: tuple[TruthLogFinding, ...]
    next_actor: str | None
    finish_order: tuple[str, ...]
    wind_catches: tuple[tuple[int, str, str], ...]
    derived_trick_ids: tuple[int, ...]

    @property
    def errors(self) -> tuple[TruthLogFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def warnings(self) -> tuple[TruthLogFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "warning")

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "guandan.truth-validation/1",
            "session_id": self.session_id,
            "mode": self.mode,
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "next_actor": self.next_actor,
            "finish_order": list(self.finish_order),
            "derived_trick_ids": list(self.derived_trick_ids),
            "wind_catches": [
                {"turn_id": turn_id, "from": source, "receiver": receiver}
                for turn_id, source, receiver in self.wind_catches
            ],
            "findings": [item.to_dict() for item in self.findings],
        }

    def format_errors(self) -> str:
        return "\n".join(
            f"[{item.code}]"
            f"{' 第 ' + str(item.turn_id) + ' 手' if item.turn_id is not None else ''}: "
            f"{item.message}"
            for item in self.errors
        )


def _finding(
    findings: list[TruthLogFinding], code: str, message: str, *,
    turn_id: int | None = None, field: str = "",
    severity: Literal["error", "warning"] = "error",
) -> None:
    findings.append(TruthLogFinding(code, severity, message, turn_id, field))


def validate_truth_log_semantics(
    log: TruthLog,
    *,
    mode: ValidationMode = "logic",
    wind_catch_policy: WindCatchPolicy = WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    standard_playing: bool = False,
) -> TruthLogValidationReport:
    """Validate one TruthLog without reading screenshots or running a model."""

    if not isinstance(log, TruthLog):
        raise TypeError("log must be a TruthLog")
    if mode not in {"logic", "publish"}:
        raise ValueError("mode must be 'logic' or 'publish'")

    findings: list[TruthLogFinding] = []
    try:
        validate_truth_log_card_inventory(log)
    except TruthLogCardInventoryError as exc:
        _finding(findings, "TL-CARD-INVENTORY", str(exc), field="turns")

    if not log.initial_state.my_hand:
        _finding(findings, "TL-INITIAL-HAND-EMPTY", "初始手牌不能为空", field="initial_state.my_hand")
    if log.initial_state.lead_player not in TURN_ORDER:
        _finding(findings, "TL-LEAD-SEAT", "首出玩家无效", field="initial_state.lead_player")

    previous_trick = 0
    seen_turn_ids: set[int] = set()
    previous_anchor: int | None = None
    for position, turn in enumerate(log.turns, start=1):
        if turn.index in seen_turn_ids:
            _finding(findings, "TL-TURN-DUPLICATE", f"turn_id {turn.index} 重复", turn_id=turn.index, field="turn_id")
        seen_turn_ids.add(turn.index)
        if turn.index != position:
            _finding(findings, "TL-TURN-SEQUENCE", f"turn_id 应为 {position}，实际为 {turn.index}", turn_id=turn.index, field="turn_id")
        if turn.trick_id <= 0:
            _finding(
                findings, "TL-TRICK-ID", "trick_id 必须为正数",
                turn_id=turn.index, field="trick_id", severity="warning",
            )
        if turn.trick_id < previous_trick or turn.trick_id > previous_trick + 1 and previous_trick:
            _finding(
                findings, "TL-TRICK-SEQUENCE",
                f"trick_id 从 {previous_trick} 跳到 {turn.trick_id}",
                turn_id=turn.index, field="trick_id", severity="warning",
            )
        previous_trick = max(previous_trick, turn.trick_id)
        if turn.is_pass and turn.cards:
            _finding(findings, "TL-PASS-HAS-CARDS", "PASS 动作不能包含牌", turn_id=turn.index, field="cards")
        if not turn.is_pass and not turn.cards:
            _finding(findings, "TL-PLAY-EMPTY", "出牌动作不能为空", turn_id=turn.index, field="cards")
        if mode == "publish":
            if turn.label_status != "verified":
                _finding(findings, "TL-TURN-NOT-VERIFIED", "已发布 TruthLog 的每一手都必须 verified", turn_id=turn.index, field="label_status")
            if turn.uncertainty:
                _finding(findings, "TL-UNCERTAINTY", f"存在未解决不确定项：{', '.join(turn.uncertainty)}", turn_id=turn.index, field="uncertainty")
            if not turn.evidence.frame_indices:
                _finding(findings, "TL-EVIDENCE-MISSING", "缺少证据帧", turn_id=turn.index, field="evidence.frame_indices")
            anchor = turn.evidence.monotonic_ms
            if anchor is None:
                _finding(findings, "TL-EVIDENCE-TIME-MISSING", "缺少动作单调时间", turn_id=turn.index, field="evidence.monotonic_ms")
            elif previous_anchor is not None and anchor < previous_anchor:
                _finding(findings, "TL-EVIDENCE-TIME-REGRESSION", f"动作时间 {anchor} 早于前一手 {previous_anchor}", turn_id=turn.index, field="evidence.monotonic_ms")
            if anchor is not None:
                previous_anchor = anchor

    if mode == "publish" and log.label_status != "verified":
        _finding(findings, "TL-LOG-NOT-VERIFIED", "发布校验要求顶层 label_status=verified", field="label_status")

    size_map = dict(log.initial_state.seat_hand_sizes)
    if standard_playing:
        for seat, count in size_map.items():
            if seat != "self" and int(count) != 27:
                _finding(
                    findings, "TL-STARTING-HAND-SIZE",
                    f"正常出牌阶段 {seat} 起手必须为 27 张，实际为 {count} 张",
                    field=f"initial_state.seat_hand_sizes.{seat}",
                )
    remaining = {seat: int(size_map.get(seat, 27)) for seat in TURN_ORDER}
    remaining["self"] = len(log.initial_state.my_hand)
    finished: set[str] = set()
    finish_order: list[str] = []
    wind_catches: list[tuple[int, str, str]] = []
    derived_trick_ids: list[int] = []
    expected: str | None = str(log.initial_state.lead_player)
    trick_index = 1
    trick_leader: str | None = None
    table_cards: tuple[str, ...] = ()
    passed: set[str] = set()

    for turn in log.turns:
        derived_trick_ids.append(trick_index)
        actor = str(turn.actor)
        if expected is None:
            _finding(findings, "TL-POST-TERMINAL-ACTION", "对局已经结束，不能再有动作", turn_id=turn.index, field="actor")
            break
        if actor not in TURN_ORDER:
            _finding(findings, "TL-ACTOR", f"无效玩家 {actor}", turn_id=turn.index, field="actor")
            break
        if actor != expected:
            _finding(findings, "TL-ACTOR-ORDER", f"应由 {expected} 行动，实际为 {actor}", turn_id=turn.index, field="actor")
            break
        if turn.trick_id != trick_index:
            _finding(
                findings, "TL-TRICK-MISMATCH",
                f"规则推导应为第 {trick_index} 墩，TruthLog 写为第 {turn.trick_id} 墩",
                turn_id=turn.index, field="trick_id", severity="warning",
            )
        if actor in finished:
            _finding(findings, "TL-FINISHED-ACTOR", f"{actor} 已经出完牌，不能继续行动", turn_id=turn.index, field="actor")
            break

        if turn.is_pass:
            if trick_leader is None or not table_cards:
                _finding(findings, "TL-LEAD-PASS", "新墩首出玩家不能 PASS", turn_id=turn.index, field="is_pass")
                break
            passed.add(actor)
        else:
            cards = tuple(str(card) for card in turn.cards)
            try:
                legal_shapes = actions_for_cards(cards, log.initial_state.round_level)
            except Exception as exc:
                legal_shapes = []
                _finding(findings, "TL-CARD-CODE", str(exc), turn_id=turn.index, field="cards")
            if not legal_shapes:
                _finding(findings, "TL-ILLEGAL-PLAY", f"牌组 {' '.join(cards)} 不能组成合法掼蛋牌型", turn_id=turn.index, field="cards")
            if table_cards and legal_shapes and not play_beats_table(cards, table_cards, log.initial_state.round_level):
                _finding(findings, "TL-DOES-NOT-BEAT", f"{' '.join(cards)} 不能压过 {' '.join(table_cards)}", turn_id=turn.index, field="cards")
            if len(cards) > remaining[actor]:
                _finding(findings, "TL-REMAINING-NEGATIVE", f"出牌 {len(cards)} 张，但 {actor} 只剩 {remaining[actor]} 张", turn_id=turn.index, field="cards")
                remaining[actor] = 0
            else:
                remaining[actor] -= len(cards)
            if remaining[actor] == 0 and actor not in finished:
                finished.add(actor)
                finish_order.append(actor)
            trick_leader = actor
            table_cards = cards
            passed.clear()

        if round_is_decided(finished):
            expected = None
            trick_leader = None
            table_cards = ()
            passed.clear()
            continue
        if trick_leader is None:
            expected = next_active_seat(actor, frozenset(finished))
            continue
        projection = project_trick_turn(
            trick_leader,
            finished,
            passed,
            wind_catch_policy=wind_catch_policy,
        )
        expected = projection.expected_after(actor)
        if projection.is_complete:
            if projection.wind_receiver is not None:
                wind_catches.append((turn.index, trick_leader, projection.wind_receiver))
            trick_leader = None
            table_cards = ()
            passed.clear()
            trick_index += 1

    declared = tuple(log.outcome.finish_order)
    derived = tuple(finish_order)
    shared_finish_count = min(len(declared), len(derived))
    if declared and derived[:shared_finish_count] != declared[:shared_finish_count]:
        _finding(findings, "TL-OUTCOME-ORDER", f"声明名次 {declared} 与动作推导名次 {derived} 不一致", field="outcome.finish_order")
    if log.outcome.complete and set(declared) != set(TURN_ORDER):
        _finding(findings, "TL-OUTCOME-INCOMPLETE", "complete outcome 必须包含四个座位", field="outcome.finish_order")

    return TruthLogValidationReport(
        log.source_session_id,
        mode,
        tuple(findings),
        expected,
        derived,
        tuple(wind_catches),
        tuple(derived_trick_ids),
    )


def require_valid_truth_log(
    log: TruthLog, *, mode: ValidationMode = "logic",
    wind_catch_policy: WindCatchPolicy = WindCatchPolicy.AUTO_HANDOFF_TO_PARTNER,
    standard_playing: bool = False,
) -> TruthLogValidationReport:
    report = validate_truth_log_semantics(
        log, mode=mode, wind_catch_policy=wind_catch_policy,
        standard_playing=standard_playing,
    )
    if not report.valid:
        raise ValueError("TruthLog 语义校验未通过：\n" + report.format_errors())
    return report


def normalize_truth_log_trick_ids(
    log: TruthLog, *, standard_playing: bool = False,
) -> TruthLog:
    """Return a validated copy whose trick IDs come only from the action chain."""

    from dataclasses import replace

    report = require_valid_truth_log(
        log, mode="logic", standard_playing=standard_playing,
    )
    if len(report.derived_trick_ids) != len(log.turns):
        raise ValueError("TruthLog 动作链未能完整推导 trick_id")
    turns = tuple(
        turn if turn.trick_id == trick_id else replace(turn, trick_id=trick_id)
        for turn, trick_id in zip(
            log.turns, report.derived_trick_ids, strict=True
        )
    )
    return log if turns == log.turns else replace(log, turns=turns)


__all__ = [
    "TruthLogFinding",
    "TruthLogValidationReport",
    "ValidationMode",
    "normalize_truth_log_trick_ids",
    "require_valid_truth_log",
    "validate_truth_log_semantics",
]
