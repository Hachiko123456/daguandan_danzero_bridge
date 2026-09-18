"""Pure projection of live state into the compact companion presentation."""
from __future__ import annotations

from dataclasses import dataclass

from ..live.local_rule_hint import LocalRuleHint


PLAY_TYPE_LABELS = {
    "SINGLE": "单张", "PAIR": "对子", "TRIPS": "三张", "TRIPLE": "三张",
    "THREEPAIR": "三连对", "THREEWITHTWO": "三带二", "TWOTRIPS": "钢板",
    "STRAIGHT": "顺子", "STRAIGHTFLUSH": "同花顺", "BOMB": "炸弹", "PASS": "不出",
}


@dataclass(frozen=True)
class CompactViewState:
    kind: str
    title: str
    detail: str = ""
    cards: tuple[str, ...] = ()
    request_id: str = ""


def advice_matches_snapshot(advice: object, snapshot: object) -> bool:
    key = getattr(advice, "key", None)
    if key is None:
        return False
    for field, key_field in (("session_id", "session_id"), ("turn_id", "turn_id"), ("revision", "state_revision")):
        value = getattr(snapshot, field, None)
        if value is not None and value != getattr(key, key_field, None):
            return False
    return True


def is_terminal_update(update: object) -> bool:
    event = getattr(update, "event", None)
    events = tuple(getattr(update, "events", ()) or ())
    return bool(
        getattr(update, "status", None) in {"finalizing", "sealed"}
        or (getattr(update, "status", None) == "running" and getattr(getattr(update, "snapshot", None), "current_player", None) is None)
        or getattr(event, "event_type", None) == "game_end_detected"
        or any(getattr(item, "event_type", None) == "game_end_detected" for item in events)
    )


def _blocked_detail(reason: str, missing_player: str = "", missing_action_kind: str = "") -> str:
    # Do not turn arbitrary exception strings or uncertain chronology into a
    # supposed exact missing-card explanation. Technical details stay in logs.
    if missing_player in {"self", "left", "opposite", "right"}:
        seat = {"self": "己方", "left": "左家", "opposite": "对家", "right": "右家"}[missing_player]
        if missing_action_kind == "lead":
            return f"缺少{seat}首出，请先手动出牌"
        if missing_action_kind == "action":
            return f"{seat}上一手未确认，请先手动出牌"
        return f"{seat}动作未确认，请先手动出牌"
    if reason in {"turn_recovery_pending", "turn_recovery_expired", "turn_recovery_budget_exceeded", "turn_desynchronized"}:
        return "回合信息未对齐，请先手动出牌"
    if reason == "wind_catch_pass_recovery_pending":
        return "接风前的动作未确认，请先手动出牌"
    if reason == "previous_action_reread_pending":
        return "上一手尚未确认，请先手动出牌"
    return "牌局记录未完整跟上，请先手动出牌"


def project_compact_view(update: object, *, now_ms: int) -> CompactViewState:
    snapshot = getattr(update, "snapshot", None)
    status = getattr(update, "status", "")
    raw = getattr(update, "advice", None)
    player = getattr(snapshot, "current_player", None)
    if is_terminal_update(update):
        return CompactViewState("terminal", "本局已结束")
    if status == "paused":
        return CompactViewState("paused", "识别已暂停", "请排除遮挡后点击继续")
    if status not in {"running", "initializing", "waiting_lead"}:
        return CompactViewState("blocked", "暂无法推荐", "请打开完整助手处理识别问题")

    hint = getattr(update, "local_rule_hint", None)
    hint_pending = bool(getattr(update, "local_rule_hint_pending", False))
    generation = getattr(update, "capture_generation", None)
    session_id = str(getattr(snapshot, "session_id", "") or "")
    if hint_pending and status == "running" and player == "self":
        return CompactViewState("confirming", "确认中…")
    if isinstance(hint, LocalRuleHint) and status == "running" and hint.is_current(
        session_id=session_id, capture_generation=generation, now_ms=now_ms,
    ):
        detail = "牌局记录待同步" if getattr(raw, "status", None) == "withheld" or player != "self" else ""
        return CompactViewState("local_rule_hint", "不出", detail)

    matches = raw is not None and advice_matches_snapshot(raw, snapshot)
    fast = getattr(update, "fast_signals", None)
    visual_self = bool(
        getattr(fast, "active_player", None) == "self"
        and getattr(fast, "self_action_buttons_visible", False)
    )
    # The canonical actor may be behind the visual actor; never call that a
    # harmless foreign turn. A stale withheld object cannot outrank new state.
    withheld_reason = str(
        getattr(update, "block_reason", "")
        or getattr(raw, "withhold_reason", "")
        or ""
    )
    # NOT_LOCAL_TURN is a normal waiting state, not a recognition failure.
    # Actual recovery gaps retain priority even when the missing actor is a
    # foreign seat; otherwise a real desynchronization would look harmless.
    if matches and getattr(raw, "status", None) == "withheld":
        if withheld_reason == "not_local_turn" and player != "self":
            return CompactViewState("waiting", "等待自己回合")
        if withheld_reason in {"previous_action_reread_pending", "turn_recovery_pending", "wind_catch_pass_recovery_pending"}:
            return CompactViewState("confirming", "确认中…")
        return CompactViewState(
            "blocked", "暂无法推荐",
            _blocked_detail(
                withheld_reason, str(getattr(update, "missing_player", "") or ""),
                str(getattr(update, "missing_action_kind", "") or ""),
            ),
        )
    if status == "running" and player != "self":
        if visual_self:
            return CompactViewState("blocked", "暂无法推荐", "回合信息未对齐，请先手动出牌")
        return CompactViewState("waiting", "等待自己回合")
    if not matches:
        return CompactViewState("waiting", "等待建议")
    raw_status = getattr(raw, "status", None)
    if raw_status == "failed":
        return CompactViewState("failed", "暂无法推荐", "建议计算失败，请先手动出牌")
    if raw_status == "requested":
        return CompactViewState("calculating", "计算中…")
    advice = getattr(raw, "advice", None)
    if raw_status != "ready" or advice is None:
        return CompactViewState("waiting", "等待建议")
    if not getattr(raw, "visible", False):
        return CompactViewState("confirming", "确认中…")
    # A one-frame cannot-beat candidate must hide a preceding model card, but
    # cannot itself become a final PASS without the independent tracker.
    if (
        visual_self and getattr(fast, "cannot_beat_visible", False)
        and getattr(advice, "strategy", None) != "button_cannot_beat"
        and not getattr(fast, "effect_visible", False)
    ):
        return CompactViewState("confirming", "确认中…")
    is_pass = bool(getattr(advice, "is_pass", False))
    play_type = str(getattr(advice, "play_type", "")).replace("_", "").upper()
    return CompactViewState(
        "ready", "不出" if is_pass else "出牌 · " + PLAY_TYPE_LABELS.get(play_type, "出牌"),
        cards=() if is_pass else tuple(str(card) for card in getattr(advice, "cards", ())),
        request_id=str(getattr(getattr(raw, "key", None), "request_id", "") or ""),
    )


class CompactUpdateGate:
    """Reject old lifecycle/state updates before any widget is touched."""

    def __init__(self) -> None:
        self.session_id = ""
        self.generation = -1
        self.turn_id = -1
        self.revision = -1
        self.terminal = False
        self.update_sequence = -1
        self._retired: tuple[str, ...] = ()

    def begin_listening(self) -> None:
        if self.session_id:
            self._retired = (*self._retired[-31:], self.session_id)
        self.session_id = ""
        self.turn_id = self.revision = -1
        self.terminal = False
        self.update_sequence = -1

    def accept(self, update: object, *, expected_session_id: str = "") -> bool:
        snapshot = getattr(update, "snapshot", None)
        key = getattr(getattr(update, "advice", None), "key", None)
        session_id = str(getattr(snapshot, "session_id", None) or getattr(key, "session_id", None) or "")
        generation = getattr(update, "capture_generation", None)
        sequence = getattr(update, "update_sequence", None)
        if isinstance(generation, int) and generation < self.generation:
            return False
        if session_id in self._retired or (expected_session_id and session_id and session_id != expected_session_id):
            return False
        if session_id and session_id != self.session_id:
            if self.session_id:
                self._retired = (*self._retired[-31:], self.session_id)
            self.session_id = session_id
            self.turn_id = self.revision = -1
            self.terminal = False
            self.update_sequence = -1
        if isinstance(generation, int) and generation > self.generation:
            self.update_sequence = -1
        if isinstance(sequence, int) and sequence > 0 and sequence <= self.update_sequence:
            return False
        turn_id = getattr(snapshot, "turn_id", None)
        revision = getattr(snapshot, "revision", None)
        if isinstance(turn_id, int) and turn_id < self.turn_id:
            return False
        if isinstance(revision, int) and revision < self.revision:
            return False
        if self.terminal and not is_terminal_update(update):
            return False
        if isinstance(generation, int):
            self.generation = generation
        if isinstance(turn_id, int):
            self.turn_id = turn_id
        if isinstance(revision, int):
            self.revision = revision
        if isinstance(sequence, int) and sequence > 0:
            self.update_sequence = sequence
        self.terminal = self.terminal or is_terminal_update(update)
        return True
