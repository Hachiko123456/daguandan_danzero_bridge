from __future__ import annotations

from dataclasses import dataclass

from ..live.truth_log import TruthLog, TruthTurn


_SEATS = {"self", "right", "opposite", "left"}


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
        if actor not in _SEATS:
            return self._rejected(f"回合 {source_turn_id} 的 actor 无效")
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
        return ReplayTurnDraftAppend(
            True,
            "",
            turn,
            self.truth_log,
            "扫描确认",
        )

    def _rejected(self, reason: str) -> ReplayTurnDraftAppend:
        return ReplayTurnDraftAppend(False, reason, None, self.truth_log, "已忽略")
