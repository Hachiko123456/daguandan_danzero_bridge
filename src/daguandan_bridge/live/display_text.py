from __future__ import annotations

from collections.abc import Iterable

from .models import LiveEvent


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
    "conflicting_valid_candidates": "多帧牌面候选不一致",
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
_PLACEMENT_LABELS = {
    "head": "头游",
    "second": "二游",
    "third": "三游",
    "last": "末游",
}
_SUIT_GLYPHS = {
    "S": "♠",
    "H": "♥",
    "C": "♣",
    "D": "♦",
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
    raw_cards = tuple(str(card) for card in event.payload.get("cards", ()))
    cards = compact_cards_text(
        raw_cards,
        event.payload.get("suit_options", ()),
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
        detail = ""
        logical_label = str(event.payload.get("logical_label", "") or "")
        substitutions = event.payload.get("wildcard_substitutions", ())
        if logical_label and isinstance(substitutions, Iterable) and tuple(substitutions):
            mappings = []
            for item in substitutions:
                if not isinstance(item, dict):
                    continue
                physical = str(item.get("card", ""))
                rank = str(item.get("as_rank", ""))
                if physical and rank:
                    mappings.append(f"{physical}→{rank}")
            suffix = f"，万能牌：{'/'.join(mappings)}" if mappings else ""
            detail = f"（逻辑：{logical_label}{suffix}）"
        return f"{seat}出牌：{cards or '未识别到牌面'}{detail}"
    if event.event_type == "player_passed":
        return f"{seat}不出"
    if event.event_type == "player_finished":
        placement = _PLACEMENT_LABELS.get(
            str(event.payload.get("placement", "")),
            "出完牌",
        )
        return f"{seat}出完牌：{placement}"
    if event.event_type == "wind_caught":
        from_player = seat_text(event.payload.get("from_player"))
        to_player = seat_text(event.payload.get("to_player"))
        return f"接风：{from_player} → {to_player}"
    if event.event_type == "game_end_detected":
        control = {
            "continue_game": "再来一局",
            "change_table": "换桌",
        }.get(str(event.payload.get("control", "")), "结算界面")
        return f"检测到{control}，正在自动结束并封存本局"
    if event.event_type == "review_required":
        return f"识别暂停，请确认{seat}动作：{reasons_text(event.payload.get('reason'))}"
    if event.event_type == "recognition_retry":
        stage = "等待首出" if event.payload.get("stage") == "waiting_lead" else "当前行动"
        message = str(event.payload.get("message", "") or "").strip()
        if message:
            return f"{stage}识别未完成，自动重试：{message}"
        reason = str(event.payload.get("reason", ""))
        if reason == "conflicting_valid_candidates":
            return f"{stage}画面短暂不一致，已清空候选并继续识别"
        if reason == "action_timeout":
            return f"尚未捕获{seat}的新动作，继续等待"
        return f"{stage}识别未完成，自动重试：{reasons_text(event.payload.get('reason'))}"
    if event.event_type == "advice_requested":
        return f"{_advice_model_name(event)} 开始计算建议"
    if event.event_type == "advice_ready":
        name = _advice_model_name(event)
        return (
            f"{name} 建议已就绪"
            if event.payload.get("visible")
            else f"{name} 建议已就绪，等待自己回合旁证"
        )
    if event.event_type == "advice_visible":
        return f"{_advice_model_name(event)} 建议已显示"
    if event.event_type == "advice_failed":
        error = str(event.payload.get("error", "") or "").strip()
        suffix = f"：{error}" if error else ""
        return f"{_advice_model_name(event)} 建议计算失败{suffix}"
    if event.event_type == "event_correction":
        return "已更正最近一条动作记录"
    if event.event_type == "suit_corrected":
        return f"{seat}花色修正：{cards or '未识别到牌面'}（不影响已确认对局流程）"
    return f"其他事件（内部码：{event.event_type}）"


def _advice_model_name(event: LiveEvent) -> str:
    configured = str(event.payload.get("advisor_name", "") or "").strip()
    if configured:
        return configured
    return {
        "danzero": "DanZero",
        "fabledan": "FableDan",
    }.get(str(event.payload.get("advisor_strategy", "")).lower(), "建议模型")


def compact_cards_text(
    cards: object,
    suit_options: object = (),
) -> str:
    """Format visible card faces as one readable, single-line token.

    The timeline must not throw away physical suit information just to be
    compact.  An occluded suit remains an explicit ``?〔候选〕`` on that one
    card instead of making an entire action look uncertain.
    """

    values: list[str] = []
    raw_cards = cards if isinstance(cards, Iterable) and not isinstance(cards, str) else ()
    card_values = tuple(str(raw) for raw in raw_cards)
    raw_options = (
        tuple(tuple(str(suit) for suit in choices) for choices in suit_options)
        if isinstance(suit_options, Iterable) and not isinstance(suit_options, str)
        else ()
    )
    for index, card in enumerate(card_values):
        if card == "small_joker":
            values.append("小王")
        elif card == "big_joker":
            values.append("大王")
        elif card.endswith("?"):
            candidates = tuple(
                dict.fromkeys(
                    _SUIT_GLYPHS[suit]
                    for suit in raw_options[index] if index < len(raw_options)
                    if suit in _SUIT_GLYPHS
                )
            )
            suffix = f"〔{'/'.join(candidates)}〕" if candidates else ""
            values.append(f"{card}{suffix}")
        elif len(card) >= 2 and card[-1] in _SUIT_GLYPHS:
            values.append(f"{card[:-1]}{_SUIT_GLYPHS[card[-1]]}")
        else:
            values.append(card)
    return " ".join(values)


def event_prefix(event: LiveEvent) -> str:
    return f"[第{event.turn_id}手]"
