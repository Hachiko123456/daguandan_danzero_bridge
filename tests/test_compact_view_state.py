from dataclasses import replace
from types import SimpleNamespace as NS
import pytest

from daguandan_bridge.domain.advice import LocalAdvice
from daguandan_bridge.live.local_rule_hint import LocalRuleHint
from daguandan_bridge.live.orchestrator import AdviceRequestKey, LiveAdvice, LiveUpdate
from daguandan_bridge.gui.compact_view_state import CompactUpdateGate, project_compact_view


def update(*, session="s1", turn=2, revision=3, player="self", **kwargs):
    key = AdviceRequestKey(session, turn, revision)
    advice = LiveAdvice(key=key, status="ready", visible=True, advice=LocalAdvice(
        strategy="fabledan", cards=("4S",), play_type="SINGLE", is_pass=False,
        state_revision=revision, request_id=key.request_id, elapsed_ms=999,
        engine_input={"debug": True, "decision": {"best_q": 3, "q_gap": 1}},
    ))
    return NS(status="running", snapshot=NS(session_id=session, turn_id=turn, revision=revision, current_player=player),
              advice=kwargs.pop("advice", advice), capture_generation=kwargs.pop("capture_generation", 1), **kwargs)


def test_ready_contains_only_play_type_and_cards_without_debug_log():
    state = project_compact_view(update(), now_ms=100)
    assert state.title == "出牌 · 单张" and state.cards == ("4S",)
    assert state.detail == ""


def test_cross_session_old_advice_cannot_show_on_same_numeric_turn():
    current = update(session="next")
    current.advice = update().advice
    assert not project_compact_view(current, now_ms=100).cards


def test_canonical_history_hold_can_show_independent_local_hint_without_mutation():
    current = update(player="right")
    original = replace(current.advice, status="withheld", visible=False, withhold_reason="turn_recovery_pending")
    current.advice = original
    current.local_rule_hint = LocalRuleHint("s1", 1, 1, 100, 110, 600, .99, (400, 600, 80, 30))
    state = project_compact_view(current, now_ms=200)
    assert state.kind == "local_rule_hint" and state.title == "不出"
    assert state.detail == "牌局记录待同步"
    assert current.advice is original and current.snapshot.current_player == "right"
    assert project_compact_view(current, now_ms=601).kind == "confirming"
    current.capture_generation = 2
    assert project_compact_view(current, now_ms=200).kind == "confirming"


def test_pause_terminal_and_expired_hints_never_keep_cards():
    for status in ("paused", "sealed", "finalizing"):
        current = update()
        current.status = status
        current.local_rule_hint = LocalRuleHint("s1", 1, 1, 100, 110, 600, .99, (400, 600, 80, 30))
        assert project_compact_view(current, now_ms=200).kind != "local_rule_hint"
        assert not project_compact_view(current, now_ms=200).cards


def test_foreign_wait_and_visual_conflict_are_not_confused():
    current = update(player="left")
    assert project_compact_view(current, now_ms=100).title == "等待自己回合"
    current.fast_signals = NS(active_player="self", self_action_buttons_visible=True)
    assert project_compact_view(current, now_ms=100).title == "暂无法推荐"
    assert "未对齐" in project_compact_view(current, now_ms=100).detail


def test_generation_turn_revision_and_retired_session_updates_rejected():
    gate = CompactUpdateGate()
    assert gate.accept(update())
    assert not gate.accept(update(turn=1))
    assert not gate.accept(update(revision=2))
    assert not gate.accept(update(capture_generation=0))
    assert gate.accept(update(session="s2"))
    assert not gate.accept(update())
    assert not gate.accept(update(session="s3"), expected_session_id="s2")


def test_terminal_cannot_be_overwritten_by_late_same_session_running_update():
    gate = CompactUpdateGate()
    terminal = update()
    terminal.status = "sealed"
    assert gate.accept(terminal)
    assert not gate.accept(update())
    gate.begin_listening()
    assert not gate.accept(update())
    assert gate.accept(update(session="s2"))


def test_same_turn_revision_late_hold_rejected_by_update_sequence():
    gate = CompactUpdateGate()
    assert gate.accept(update(update_sequence=10))
    old_hold = update(update_sequence=9)
    old_hold.advice = replace(old_hold.advice, status="withheld", withhold_reason="turn_recovery_pending")
    assert not gate.accept(old_hold)
    assert not gate.accept(update(update_sequence=10))
    assert gate.accept(update(update_sequence=11))
    assert gate.accept(update(capture_generation=2, update_sequence=1))


def test_recovery_budget_has_structured_cause_without_rendering_internal_text():
    current = update()
    current.advice = replace(current.advice, status="withheld", withhold_reason="turn_recovery_pending", error="traceback INTERNAL-REREAD path")
    assert project_compact_view(current, now_ms=100).title == "确认中…"
    current.block_reason = "turn_recovery_budget_exceeded"
    current.missing_player = "right"
    current.missing_action_kind = "action"
    state = project_compact_view(current, now_ms=100)
    assert state.title == "暂无法推荐"
    assert state.detail == "右家上一手未确认，请先手动出牌"
    assert "INTERNAL" not in state.detail


@pytest.mark.parametrize("action_kind,expected", [
    ("lead", "缺少右家首出，请先手动出牌"),
    ("action", "右家上一手未确认，请先手动出牌"),
    ("", "右家动作未确认，请先手动出牌"),
    ("unrecognized-kind", "右家动作未确认，请先手动出牌"),
])
def test_production_live_update_projects_structured_missing_action_kind(action_kind, expected):
    original = update(player="right")
    withheld = replace(
        original.advice, status="withheld", visible=False,
        withhold_reason="turn_recovery_budget_exceeded",
        error="错误中文上下文：误认结算，左家首出；不能解析这段文字",
    )
    production_update = LiveUpdate(
        status="running", snapshot=original.snapshot, advice=withheld,
        capture_generation=3, update_sequence=20,
        block_reason="turn_recovery_budget_exceeded", missing_player="right",
        missing_action_kind=action_kind,
    )
    state = project_compact_view(production_update, now_ms=100)
    assert state.kind == "blocked" and state.title == "暂无法推荐"
    assert state.detail == expected and not state.cards
    assert "误认" not in state.detail and "左家" not in state.detail


def test_unknown_seat_or_terminal_history_conflict_stays_neutral():
    original = update()
    withheld = replace(original.advice, status="withheld", visible=False, withhold_reason="visual_finish_without_complete_history")
    production_update = LiveUpdate(
        status="running", snapshot=original.snapshot, advice=withheld,
        capture_generation=3, update_sequence=20,
        block_reason="visual_finish_without_complete_history", missing_action_kind="lead",
    )
    state = project_compact_view(production_update, now_ms=100)
    assert state.detail == "牌局记录未完整跟上，请先手动出牌"
    assert "误检" not in state.detail and "首出" not in state.detail
