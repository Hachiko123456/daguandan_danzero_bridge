from __future__ import annotations

from dataclasses import dataclass, replace

from ..live.truth_log import TruthInitialState, TruthLog, TruthTurn
from ..live.turns import (
    TURN_ORDER,
    next_active_seat,
    project_trick_turn,
    round_is_decided,
)


def next_actor_after_prefix(
    initial_state: TruthInitialState,
    turns: tuple[TruthTurn, ...] | list[TruthTurn],
) -> str | None:
    """Validate a prefix and return the next expected actor.

    This intentionally derives turns from their ordered actions and card
    counts, not from the informational ``trick_id`` field.  A stale trick id
    is editable only indirectly in the GUI and must never make a valid actor
    chain unsaveable (or make an invalid chain look valid).
    """

    lead = getattr(initial_state, "lead_player", None)
    if lead not in TURN_ORDER:
        raise ValueError("首出玩家无效")
    hand = tuple(getattr(initial_state, "my_hand", ()))
    remaining = {seat: 27 for seat in TURN_ORDER}
    remaining["self"] = len(hand)
    finished: set[str] = set()
    trick_leader: str | None = None
    passed: set[str] = set()
    expected: str | None = str(lead)

    for position, turn in enumerate(turns, start=1):
        actor = str(turn.actor)
        if actor not in TURN_ORDER:
            raise ValueError(f"第 {position} 条动作的玩家无效")
        if expected is None:
            raise ValueError(f"第 {position} 条动作发生在对局已经结束之后")
        if actor != expected:
            raise ValueError(
                f"第 {position} 条动作玩家顺序错误："
                f"应为 {expected}，实际为 {actor}"
            )
        if not turn.is_pass:
            played = len(turn.cards)
            if played > remaining[actor]:
                raise ValueError(f"第 {position} 条动作的出牌数量超过 {actor} 的剩余手牌")
            remaining[actor] -= played
            if remaining[actor] == 0:
                finished.add(actor)
            trick_leader = actor
            passed.clear()
        elif trick_leader is not None:
            passed.add(actor)
        if round_is_decided(finished):
            expected = None
            continue

        if trick_leader is None:
            expected = next_active_seat(actor, frozenset(finished))
            continue
        projection = project_trick_turn(trick_leader, finished, passed)
        expected = projection.expected_after(actor)
        if projection.is_complete:
            trick_leader = None
            passed.clear()
    return expected


# Kept for callers that imported the former private helper while the editor
# migrates to the supported prefix-derivation API.
def _actor_chain_after(
    initial_state: TruthInitialState,
    turns: tuple[TruthTurn, ...] | list[TruthTurn],
) -> str | None:
    return next_actor_after_prefix(initial_state, turns)


def validate_turn_actor_chain(log: TruthLog) -> None:
    """Raise when a truth-log action sequence does not follow turn ownership."""

    next_actor_after_prefix(log.initial_state, log.turns)


@dataclass(frozen=True)
class ReplayTurnDraftAppend:
    accepted: bool
    reason: str
    turn: TruthTurn | None
    truth_log: TruthLog
    status: str


class ReplayTurnDraftAssembler:
    """Assemble confirmed replay actions in memory without UI dependencies."""

    def __init__(self, baseline: TruthLog) -> None:
        self._baseline = baseline
        self._turns = list(baseline.turns)
        self._source_turn_ids = {turn.index for turn in baseline.turns}
        self._row_by_source_turn_id = {
            turn.index: row for row, turn in enumerate(baseline.turns)
        }
        # A pre-existing baseline is rare for scans, but validate it once so
        # an unsafe scan cannot silently extend a broken actor sequence.
        self._next_actor = next_actor_after_prefix(
            baseline.initial_state,
            self._turns,
        )

    @property
    def truth_log(self) -> TruthLog:
        return TruthLog(
            source_session_id=self._baseline.source_session_id,
            initial_state=self._baseline.initial_state,
            turns=tuple(self._turns),
            source_video=self._baseline.source_video,
            frame_index_path=self._baseline.frame_index_path,
            label_status=self._baseline.label_status,
            provenance=self._baseline.provenance,
            outcome=self._baseline.outcome,
        )

    def append(self, raw: dict[str, object]) -> ReplayTurnDraftAppend:
        if str(raw.get("kind", "action")) == "suit_corrected":
            return self.apply_suit_correction(raw)
        try:
            source_turn_id = int(raw.get("turn_id", 0) or 0)
            frame_index = (
                int(raw["frame_index"])
                if raw.get("frame_index") is not None
                else None
            )
            trick_id = (
                int(raw["trick_id"])
                if raw.get("trick_id") is not None
                else None
            )
        except (TypeError, ValueError):
            return self._rejected("回合、牌墩或帧编号无效")
        actor = str(raw.get("actor", ""))
        is_pass = bool(raw.get("recognized_pass", raw.get("is_pass", False)))
        cards = tuple(
            str(card)
            for card in (
                raw.get("recognized_cards", raw.get("cards", ())) or ()
            )
        )
        if source_turn_id <= 0:
            return self._rejected("缺少有效 turn_id")
        if source_turn_id in self._source_turn_ids:
            return self._rejected("该回合已确认")
        if actor not in TURN_ORDER:
            return self._rejected(f"回合 {source_turn_id} 的 actor 无效")
        if self._next_actor is None:
            return self._rejected("对局已结束，不能追加新的动作")
        if actor != self._next_actor:
            return self._rejected(
                f"回合 {source_turn_id} 的玩家顺序错误："
                f"应为 {self._next_actor}，实际为 {actor}"
            )
        if is_pass and cards:
            return self._rejected(f"回合 {source_turn_id} 的不出动作不能带牌")
        if not is_pass and not cards:
            return self._rejected(f"回合 {source_turn_id} 的出牌动作缺少牌面")
        turn = TruthTurn(
            len(self._turns) + 1,
            actor,
            is_pass,
            () if is_pass else cards,
            frame_index=frame_index,
            trick_id=trick_id,
        )
        self._turns.append(turn)
        self._source_turn_ids.add(source_turn_id)
        self._row_by_source_turn_id[source_turn_id] = len(self._turns) - 1
        try:
            self._next_actor = next_actor_after_prefix(
                self._baseline.initial_state,
                self._turns,
            )
        except ValueError as exc:
            # Keep the in-memory draft transaction-like: rejected input cannot
            # leave a half-appended row that shifts every later recognition.
            self._turns.pop()
            self._source_turn_ids.remove(source_turn_id)
            self._row_by_source_turn_id.pop(source_turn_id, None)
            self._next_actor = next_actor_after_prefix(
                self._baseline.initial_state,
                self._turns,
            )
            return self._rejected(str(exc))
        return ReplayTurnDraftAppend(
            True,
            "",
            turn,
            self.truth_log,
            "扫描确认",
        )

    def apply_suit_correction(self, raw: dict[str, object]) -> ReplayTurnDraftAppend:
        """Replace one already-drafted action after a confirmed suit reread.

        A correction is not another turn.  It must preserve both the row count
        and the original actor; only suit information on a rank-equivalent
        non-pass action is allowed to change.
        """

        try:
            source_turn_id = int(raw.get("target_turn_id", 0) or 0)
        except (TypeError, ValueError):
            return self._rejected("花色修正缺少有效目标回合")
        row = self._row_by_source_turn_id.get(source_turn_id)
        if row is None:
            return self._rejected(f"花色修正找不到第 {source_turn_id} 手原动作")
        original = self._turns[row]
        raw_cards = raw.get("recognized_cards", raw.get("cards", ())) or ()
        cards = tuple(str(card) for card in raw_cards)
        actor = str(raw.get("actor", original.actor))
        if original.is_pass or not cards:
            return self._rejected("花色修正只能回填已有的出牌动作")
        if actor != original.actor:
            return self._rejected("花色修正玩家与原动作不一致")
        if _rank_signature(cards) != _rank_signature(original.cards):
            return self._rejected("花色修正不能改变原动作的点数或张数")
        corrected = replace(original, cards=cards)
        self._turns[row] = corrected
        return ReplayTurnDraftAppend(
            True,
            "",
            corrected,
            self.truth_log,
            "花色修正已回填",
        )

    def _rejected(self, reason: str) -> ReplayTurnDraftAppend:
        return ReplayTurnDraftAppend(False, reason, None, self.truth_log, "已忽略")


def _rank_signature(cards: tuple[str, ...]) -> tuple[str, ...]:
    def rank(card: str) -> str:
        if card in {"small_joker", "big_joker"}:
            return card
        return card[:-1] if len(card) >= 2 else card

    return tuple(sorted(rank(card) for card in cards))
