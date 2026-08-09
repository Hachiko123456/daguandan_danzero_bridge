from __future__ import annotations

from collections.abc import Iterable

from .models import LiveEvent
from .truth_log import card_code_to_text


_SEAT_LABELS = {
    "self": "自己",
    "right": "右家",
    "opposite": "对家",
    "left": "左家",
}
_STATUS_LABELS = {
    "initializing": "初始化中",
    "waiting_lead": "等待首出",
    "running": "运行中",
    "review_required": "需要人工确认",
    "paused": "已暂停",
    "finalizing": "正在结束",
    "sealed": "已封存",
}
_REASON_LABELS = {
    "action_timeout": "等待动作超时",
    "candidate_conflict": "候选动作相互冲突",
    "cards_not_in_known_hand": "出牌不在已确认手牌中",
    "does_not_beat_table": "出牌未压过当前牌型",
    "empty_play": "未识别到出牌",
    "first_action_not_captured": "首出玩家首手未捕获，尚未开始识别后续动作",
    "self_action_waiting": "等待自己选择出牌",
    "self_hand_changed_play_unrecognized": "已检测到自己手牌变化，但未识别到落牌区牌面",
    "exceeds_double_deck_limit": "单张牌数量超过双副牌限制",
    "exceeds_remaining_cards": "出牌数量超过剩余手牌",
    "illegal_pattern": "牌型不符合规则",
    "insufficient_consensus": "多帧结果不一致",
    "lead_player_timeout": "等待首出标志超时",
    "pass_not_allowed": "首出不能不出",
}


def seat_text(value: object | None, *, unknown: str = "未知座位") -> str:
    if value is None or value == "":
        return unknown
    return _SEAT_LABELS.get(str(value), f"未知座位（内部码：{value}）")


def live_status_text(value: str) -> str:
    return _STATUS_LABELS.get(value, f"其他状态（内部码：{value}）")


def reason_text(value: object | None) -> str:
    if value is None or value == "":
        return "未知原因"
    raw = str(value)
    if raw.startswith("recognition_failed:"):
        detail = raw.partition(":")[2]
        return f"识别服务异常（{detail}）"
    return _REASON_LABELS.get(raw, f"其他原因（内部码：{raw}）")


def reasons_text(value: object | None) -> str:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Iterable):
        values = [str(item) for item in value]
    else:
        values = [value]
    return "；".join(reason_text(item) for item in values if item is not None) or "未知原因"


def event_action_text(event: LiveEvent) -> str:
    seat = seat_text(event.actor, unknown="系统")
    cards = "、".join(
        card_code_to_text(str(card)) for card in event.payload.get("cards", ())
    )
    if event.event_type == "waiting_for_lead":
        return "等待加倍结束与首出标志"
    if event.event_type == "deal_complete":
        return "加倍阶段结束，开始等待首出标志"
    if event.event_type == "initial_state_confirmed":
        lead = seat_text(event.payload.get("lead_player"), unknown="等待自动识别")
        return f"初始状态确认，首出：{lead}"
    if event.event_type == "lead_player_confirmed":
        lead = seat_text(event.payload.get("lead_player", event.actor))
        return f"首出确认：{lead}"
    if event.event_type == "turn_started":
        return f"轮到{seat_text(event.payload.get('player', event.actor))}"
    if event.event_type == "player_played":
        return f"{seat}出牌：{cards or '未识别到牌面'}"
    if event.event_type == "player_passed":
        return f"{seat}不出"
    if event.event_type == "review_required":
        return f"识别暂停，请确认{seat}动作：{reasons_text(event.payload.get('reason'))}"
    if event.event_type == "recognition_retry":
        stage = "等待首出" if event.payload.get("stage") == "waiting_lead" else "当前行动"
        return f"{stage}识别未完成，自动重试：{reasons_text(event.payload.get('reason'))}"
    if event.event_type == "advice_requested":
        return "DanZero 开始计算建议"
    if event.event_type == "advice_ready":
        return "DanZero 建议已就绪" if event.payload.get("visible") else "DanZero 建议已就绪，等待自己回合旁证"
    if event.event_type == "advice_visible":
        return "DanZero 建议已显示"
    if event.event_type == "advice_failed":
        return "DanZero 建议计算失败"
    if event.event_type == "event_correction":
        return "已更正最近一条动作记录"
    return f"其他事件（内部码：{event.event_type}）"


def event_prefix(event: LiveEvent) -> str:
    return f"[第{event.turn_id}手]"
