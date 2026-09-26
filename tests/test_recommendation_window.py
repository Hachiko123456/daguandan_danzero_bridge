from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from time import monotonic_ns

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
from daguandan_bridge.gui import main_window as main_window_module
from daguandan_bridge.gui.main_window import DaguandanBridgeWindow
from daguandan_bridge.live.orchestrator import AdviceRequestKey, LiveAdvice, LiveUpdate
from daguandan_bridge.recognition_service import FastSignalResult
from daguandan_bridge.live.local_rule_hint import LocalRuleHint


def _app():
    return QApplication.instance() or QApplication([])


class FakeRuntime(QObject):
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    listening_status = Signal(object)
    log_delivery_status = Signal(object)
    live_fault = Signal(object)


def _cannot_beat_fast(**overrides) -> FastSignalResult:
    values = {
        "expected_player": "self",
        "active_player": "self",
        "pass_visible": False,
        "self_action_buttons_visible": True,
        "effect_visible": False,
        "cannot_beat_visible": True,
        "cannot_beat_confidence": 0.99,
    }
    values.update(overrides)
    return FastSignalResult(**values)


def test_float_window_distinguishes_fast_recovery_from_confirmed_history_gap():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(current_player="self", finished_seats=frozenset())

    runtime.update_ready.emit(
        SimpleNamespace(
            status="running",
            snapshot=snapshot,
            advice=LiveAdvice(
                key=AdviceRequestKey("session", 1, 1),
                status="withheld",
                withhold_reason="turn_recovery_pending",
                error="正在补齐刚才的快速出牌，暂缓推荐",
            ),
            event=None,
            events=(),
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()
    assert window.suggestion_label.text() == "确认中…"
    assert window.detail_label.text() == "正在确认上一手"

    runtime.update_ready.emit(
        SimpleNamespace(
            status="running",
            snapshot=snapshot,
            advice=LiveAdvice(
                key=AdviceRequestKey("session", 2, 2),
                status="withheld",
                withhold_reason="visual_finish_without_complete_history",
                error="history gap",
            ),
            event=None,
            events=(),
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()
    assert window.suggestion_label.text() == "暂不推荐"
    assert "记录未完整跟上" in window.detail_label.text()


def test_float_window_reports_generated_automatic_log():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    runtime.log_delivery_status.emit(
        {"status": "PASS", "diagnostic_zip_path": "C:/logs/game.zip"}
    )
    app.processEvents()

    assert window.suggestion_label.text() == "等待建议"
    assert window.capture_label.text() == "日志已保存"
    assert window.detail_label.text() == ""
    assert window.detail_label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse


def test_float_window_shows_provisional_cannot_beat_status_without_committing_pass():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", finished_seats=frozenset()),
            advice=None,
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "等待建议"
    assert "不直接" not in window.suggestion_label.text()
    assert window.detail_label.text() == ""
    assert window._card_badges == []  # A one-frame button is not a final PASS.
    window.hide()


def test_float_window_does_not_show_cannot_beat_as_local_when_turn_is_foreign():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="right", finished_seats=frozenset()),
            advice=None,
            fast_signals=FastSignalResult(
                expected_player="right",
                active_player="right",
                pass_visible=False,
                self_action_buttons_visible=True,
                effect_visible=False,
                cannot_beat_visible=True,
                cannot_beat_confidence=0.99,
            ),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "等待自己回合"
    window.hide()


def test_float_window_rejects_unsafe_cannot_beat_candidates_and_clears_status():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(current_player="self", finished_seats=frozenset())

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=None,
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()
    assert window.suggestion_label.text() == "等待建议"

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=None,
            fast_signals=_cannot_beat_fast(active_player=None),
        )
    )
    app.processEvents()
    assert window.suggestion_label.text() == "等待建议"

    unsafe_signals = (
        _cannot_beat_fast(cannot_beat_confidence=0.79),
        _cannot_beat_fast(active_player="right"),
        _cannot_beat_fast(effect_visible=True),
        _cannot_beat_fast(self_action_buttons_visible=False),
        _cannot_beat_fast(cannot_beat_visible=False),
    )
    for signal in unsafe_signals:
        runtime.update_ready.emit(
            LiveUpdate(
                status="running",
                snapshot=snapshot,
                advice=None,
                fast_signals=signal,
            )
        )
        app.processEvents()
        assert window.suggestion_label.text() == "等待建议"
        assert "要不起" not in window.detail_label.text()
    window.hide()


def test_float_window_labels_wind_catch_recovery_separately_from_history_gap():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", finished_seats=frozenset()),
            advice=LiveAdvice(
                key=AdviceRequestKey("session", 3, 4),
                status="withheld",
                withhold_reason="wind_catch_pass_recovery_pending",
                error="等待接风前最后一个不出",
            ),
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "确认中…"
    assert "历史" not in window.suggestion_label.text()
    window.hide()


def test_float_window_reports_log_loading_failure_and_disabled_states():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    runtime.log_delivery_status.emit(
        {"status": "RUNNING", "message": "正在后台整理诊断"}
    )
    app.processEvents()
    assert window.suggestion_label.text() == "等待建议"
    assert window.capture_label.text() == "日志整理中"

    runtime.log_delivery_status.emit({"status": "FAIL", "error": "磁盘空间不足"})
    app.processEvents()
    assert window.suggestion_label.text() == "等待建议"
    assert "日志保存失败" in window.capture_label.text()
    assert window.detail_label.text() == ""

    runtime.log_delivery_status.emit({"status": "DISABLED"})
    app.processEvents()
    assert window.suggestion_label.text() == "等待建议"
    assert window.capture_label.text() == "未保存对局日志"


def _ready_advice(*, visible: bool, debug: bool = False) -> LiveAdvice:
    return LiveAdvice(
        key=AdviceRequestKey("session", 7, 8),
        status="ready",
        visible=visible,
        advice=LocalAdvice(
            strategy="test",
            cards=("3S", "4H", "5D", "6C", "7S"),
            play_type="Straight",
            is_pass=False,
            state_revision=8,
            elapsed_ms=12.0,
            request_id="ADV-0007-0008",
            engine_input=(
                {
                    "debug": True,
                    "decision": {"best_q": 1.3274, "q_gap": 0.4153},
                }
                if debug
                else {}
            ),
            timings={},
        ),
    )


def test_confirmed_button_pass_is_final_advice_not_provisional_status():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    base = _ready_advice(visible=True)
    advice = replace(base, advice=replace(base.advice, strategy="button_cannot_beat", is_pass=True, cards=(), play_type="PASS"))
    runtime.update_ready.emit(LiveUpdate(
        status="running", snapshot=SimpleNamespace(current_player="self", turn_id=7, revision=8),
        advice=advice, fast_signals=_cannot_beat_fast(),
    ))
    app.processEvents()
    assert window.suggestion_label.text() == "不出"
    assert window.detail_label.text() == ""
    assert "正在确认" not in window.suggestion_label.text()
    window.hide()


def test_opening_status_replaces_reconnected_but_cannot_overwrite_new_session():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    window.apply_listening_status({"state": "recovered", "generation": 2})
    window.apply_listening_status({"state": "opening", "phase": "opening_seed_invalid", "generation": 2, "message": "已识别27张，正在确认首出"})
    assert window.suggestion_label.text() == "确认开局中…"
    window.apply_listening_status({"state": "opening", "generation": 1, "message": "旧状态"})
    assert "旧状态" not in window.suggestion_label.text()
    runtime.orchestrator = object()
    runtime.update_ready.emit(LiveUpdate(status="running", snapshot=SimpleNamespace(current_player="self"), advice=_ready_advice(visible=True)))
    app.processEvents()
    previous = window.suggestion_label.text()
    window.apply_listening_status({"state": "opening", "generation": 2, "message": "晚到开局状态"})
    assert window.suggestion_label.text() == previous
    window.hide()


def test_structured_opening_readiness_shows_reason_and_suggested_action():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    window.apply_listening_status({
        "state": "opening",
        "generation": 4,
        "status": "WAIT",
        "primary_reason": "HAND_UNSTABLE",
        "message": "起手牌识别仍在变化",
        "recoverable": True,
        "suggested_action": "保持牌桌清晰，等待连续一致的起手牌识别",
        "compact_allowed": False,
        "diagnostic_compact_allowed": True,
        "session_allowed": False,
    })

    assert window.suggestion_label.text() == "起手牌识别不稳定"
    assert "起手牌识别仍在变化" in window.detail_label.text()
    assert "建议：保持牌桌清晰" in window.detail_label.text()
    assert "确认开局中" not in window.suggestion_label.text()
    window.hide()


def test_structured_opening_hard_error_clears_compact_cards():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    window._render_cards(("3S",))

    window.apply_listening_status({
        "state": "failed",
        "generation": 5,
        "status": "FAIL",
        "primary_reason": "ROI_FATAL",
        "message": "识别区域配置存在致命错误",
        "recoverable": False,
        "suggested_action": "打开完整助手修复 ROI 配置后重新连接",
        "compact_allowed": False,
    })

    assert window.suggestion_label.text() == "识别区域配置错误"
    assert "修复 ROI 配置" in window.detail_label.text()
    assert window._card_badges == []
    window.hide()


def test_recording_capacity_warning_preserves_active_recommendation():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    runtime.update_ready.emit(LiveUpdate(status="running", snapshot=SimpleNamespace(current_player="self"), advice=_ready_advice(visible=True)))
    app.processEvents()
    previous = window.suggestion_label.text()
    window.apply_recording_status({"reason": "recording_capacity_reached", "message": "录像已满"})
    assert window.suggestion_label.text() == previous
    assert "识别和推荐继续" in window.capture_label.text()
    window.hide()


def test_float_window_cannot_beat_candidate_hides_stale_ready_advice():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    old_advice = _ready_advice(visible=True)

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(
                current_player="self",
                finished_seats=frozenset(),
                turn_id=7,
                revision=8,
            ),
            advice=old_advice,
        )
    )
    app.processEvents()
    assert window._card_badges

    current_snapshot = SimpleNamespace(
        current_player="self",
        finished_seats=frozenset(),
        turn_id=8,
        revision=9,
    )
    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=current_snapshot,
            advice=old_advice,
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "等待建议"
    assert window._card_badges == []

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=current_snapshot,
            advice=old_advice,
            fast_signals=_cannot_beat_fast(cannot_beat_visible=False),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "等待建议"
    assert window._card_badges == []
    window.hide()


def test_float_window_shows_one_prominent_suggestion_for_one_request():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(current_player="self")

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=_ready_advice(visible=False),
        )
    )
    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=_ready_advice(visible=True),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "出牌 · 顺子"
    assert window.detail_label.text() == ""
    assert len(window._card_badges) == 5
    window.hide()


def test_float_window_hides_unconfirmed_recommendation_cards():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(current_player="self")

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=_ready_advice(visible=False),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "确认中…"
    assert window._card_badges == []
    window.hide()


def test_float_window_clears_recommendation_when_history_is_withheld():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(current_player="self")

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=_ready_advice(visible=True),
        )
    )
    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=LiveAdvice(
                key=AdviceRequestKey("session", 8, 9),
                status="withheld",
                error="牌局历史不完整，暂停推荐",
            ),
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "暂不推荐"
    assert window._card_badges == []
    window.hide()


def test_float_window_prioritizes_terminal_state_over_a_stale_withhold():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    withheld = LiveAdvice(
        key=AdviceRequestKey("session", 8, 9),
        status="withheld",
        error="牌局历史不完整，暂停推荐",
        withhold_reason="visual_finish_without_complete_history",
    )

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player=None),
            advice=withheld,
        )
    )
    app.processEvents()

    assert window.suggestion_label.text() == "本局已结束"
    assert "历史不完整" not in window.suggestion_label.text()
    window.hide()


def test_float_window_treats_short_adjacent_reread_as_updating_not_missing_history():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(current_player="self")
    withheld = LiveAdvice(
        key=AdviceRequestKey("session", 8, 9),
        status="withheld",
        error="上一手牌面待复核，暂停推荐",
        withhold_reason="previous_action_reread_pending",
    )

    runtime.update_ready.emit(
        LiveUpdate(status="running", snapshot=snapshot, advice=withheld)
    )
    app.processEvents()
    assert window.suggestion_label.text() == "确认中…"
    assert window._card_badges == []

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=_ready_advice(visible=True),
        )
    )
    QTest.qWait(600)
    app.processEvents()
    assert window.suggestion_label.text() == "出牌 · 顺子"
    window.hide()


def test_float_window_labels_long_adjacent_reread_as_verification():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    withheld = LiveAdvice(
        key=AdviceRequestKey("session", 8, 9),
        status="withheld",
        error="上一手牌面待复核，暂停推荐",
        withhold_reason="previous_action_reread_pending",
    )

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self"),
            advice=withheld,
        )
    )
    QTest.qWait(600)
    app.processEvents()
    assert window.suggestion_label.text() == "确认中…"
    assert "历史不完整" not in window.suggestion_label.text()
    window.hide()


def test_float_window_marks_capture_backend_and_occlusion_pause():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    runtime.frame_ready.emit(
        SimpleNamespace(frame=SimpleNamespace(backend="printwindow"))
    )
    app.processEvents()
    assert window.capture_label.text() == ""

    runtime.error.emit("屏幕采集已暂停：目标牌桌被其他窗口遮挡")
    app.processEvents()
    assert window.suggestion_label.text() == "窗口遮挡，已暂停"
    window.hide()


def test_float_window_clears_terminal_failure_across_new_listening_recovery():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    window._render_cards(("3S",))
    runtime.listening_status.emit(
        {
            "state": "failed",
            "message": "监听已停止，请打开完整助手",
            "reason": "窗口持续抖动",
        }
    )
    app.processEvents()

    assert window.suggestion_label.text() == "监听已停止"
    assert window.detail_label.text() == "请打开完整助手重新连接牌桌"
    assert window.capture_label.text() == ""

    runtime.listening_status.emit(
        {
            "state": "listening",
            "message": "持续监听页面中",
        }
    )
    app.processEvents()
    assert window.suggestion_label.text() == "等待开局"
    assert window.detail_label.text() == ""
    assert window.capture_label.text() == ""
    assert window.cards_host.isHidden()

    runtime.listening_status.emit(
        {
            "state": "recovering",
            "message": "牌桌窗口发生变化，正在重新连接",
        }
    )
    app.processEvents()
    assert window.suggestion_label.text() == "重新连接中…"
    assert window.capture_label.text() == ""

    runtime.listening_status.emit(
        {
            "state": "recovered",
            "message": "牌桌窗口已重新连接，继续监听",
        }
    )
    app.processEvents()
    assert window.suggestion_label.text() == "等待开局"
    assert "监听已停止" not in window.detail_label.text()
    window.hide()


def test_float_window_keeps_fabledan_q_summary_out_of_compact_view():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(current_player="self")

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            advice=_ready_advice(visible=True, debug=True),
        )
    )
    app.processEvents()

    assert window.detail_label.text() == ""
    window.hide()


def test_float_window_uses_selected_fabledan_name_while_requesting():
    app = _app()
    runtime = FakeRuntime()
    runtime.advisor_strategy = "fabledan"
    window = RecommendationFloatWindow(runtime)
    request = LiveAdvice(
        key=AdviceRequestKey("session", 7, 8),
        status="requested",
    )

    runtime.update_ready.emit(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self"),
            advice=request,
        )
    )
    app.processEvents()

    assert window.windowTitle() == "FableDan 极简推荐"
    assert window.suggestion_label.text() == "计算中…"
    assert window.detail_label.text() == ""
    window.hide()


def test_float_window_old_listening_preselection_and_timer_never_replace_ready():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    ready = _ready_advice(visible=True)
    current = SimpleNamespace(
        status="running", snapshot=SimpleNamespace(session_id="session", current_player="self", turn_id=7, revision=8),
        advice=ready, capture_generation=3, update_sequence=8,
    )
    window.apply_update(current)
    original = window.suggestion_label.text()
    for state in ("opening", "listening", "recovering", "recovered", "failed"):
        window.apply_listening_status({"state": state, "generation": 2, "message": "old worker"})
    for _ in range(10):
        window.apply_preselection_result(SimpleNamespace(request_id=ready.key.request_id, status="preselected", detail="重复成功信息"))
    window._show_delayed_transient_withhold()
    window.show_error("旧线程：屏幕采集已暂停，遮挡")
    app.processEvents()
    assert window.suggestion_label.text() == original
    assert window._card_badges and window.detail_label.text() == ""
    window.hide()


def test_float_window_same_key_late_hold_cannot_erase_newer_ready():
    app = _app()
    window = RecommendationFloatWindow(FakeRuntime())
    ready = _ready_advice(visible=True)
    snapshot = SimpleNamespace(session_id="session", current_player="self", turn_id=7, revision=8)
    window.apply_update(SimpleNamespace(status="running", snapshot=snapshot, advice=ready, capture_generation=1, update_sequence=10))
    window.apply_update(SimpleNamespace(status="running", snapshot=snapshot, advice=replace(ready, status="withheld", withhold_reason="turn_recovery_pending"), capture_generation=1, update_sequence=9))
    assert window.suggestion_label.text() == "出牌 · 顺子"
    assert window._card_badges
    window.hide()


def test_float_window_independent_hint_expires_without_resurrecting_model_cards():
    app = _app()
    window = RecommendationFloatWindow(FakeRuntime())
    now_ms = monotonic_ns() // 1_000_000
    raw = replace(_ready_advice(visible=True), status="withheld", withhold_reason="turn_recovery_pending")
    current = SimpleNamespace(
        status="running", snapshot=SimpleNamespace(session_id="session", current_player="right", turn_id=7, revision=8),
        advice=raw, capture_generation=1, update_sequence=1,
        local_rule_hint=LocalRuleHint("session", 1, 1, now_ms, now_ms, now_ms + 80, .99, (400, 600, 80, 30)),
    )
    window.apply_update(current)
    assert window.suggestion_label.text() == "确认中…"
    assert window.detail_label.text() == "正在确认上一手"
    assert not window._card_badges
    QTest.qWait(120)
    app.processEvents()
    assert window.suggestion_label.text() == "确认中…"
    assert window.detail_label.text() == "正在确认上一手"
    assert not window._card_badges
    window.hide()


def test_float_window_hint_expiry_cannot_erase_later_current_advice():
    app = _app()
    window = RecommendationFloatWindow(FakeRuntime())
    now_ms = monotonic_ns() // 1_000_000
    snapshot = SimpleNamespace(session_id="session", current_player="self", turn_id=7, revision=8)
    window.apply_update(SimpleNamespace(
        status="running", snapshot=snapshot, advice=None, capture_generation=1, update_sequence=1,
        local_rule_hint=LocalRuleHint("session", 1, 1, now_ms, now_ms, now_ms + 80, .99, (400, 600, 80, 30)),
    ))
    window.apply_update(SimpleNamespace(status="running", snapshot=snapshot, advice=_ready_advice(visible=True), capture_generation=1, update_sequence=2))
    QTest.qWait(120)
    app.processEvents()
    assert window.suggestion_label.text() == "出牌 · 顺子"
    assert window._card_badges
    window.hide()


def test_float_window_current_fatal_worker_fault_clears_cards_and_old_faults_do_not():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    snapshot = SimpleNamespace(session_id="session", current_player="self", turn_id=7, revision=8)
    ready = _ready_advice(visible=True)
    window.apply_update(SimpleNamespace(status="running", snapshot=snapshot, advice=ready, capture_generation=3, update_sequence=10))
    runtime.live_fault.emit({"session_id": "old", "capture_generation": 3, "kind": "analysis"})
    runtime.live_fault.emit({"session_id": "session", "capture_generation": 2, "kind": "capture"})
    app.processEvents()
    assert window._card_badges
    runtime.live_fault.emit({"session_id": "session", "capture_generation": 3, "kind": "analysis"})
    app.processEvents()
    assert not window._card_badges
    assert window.suggestion_label.text() == "识别已暂停"
    assert window.detail_label.text() == "识别失败，请重新连接牌桌"
    window.apply_update(SimpleNamespace(status="running", snapshot=snapshot, advice=ready, capture_generation=3, update_sequence=11))
    assert not window._card_badges
    window.apply_update(SimpleNamespace(status="running", snapshot=snapshot, advice=ready, capture_generation=4, update_sequence=12))
    assert window._card_badges
    window.hide()


def test_float_window_keeps_its_size_and_shows_the_current_trick():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    window.resize(500, 245)
    snapshot = SimpleNamespace(
        current_player="left",
        turn_id=8,
        trick_plays=(
            SimpleNamespace(
                player="self",
                cards=("JC", "JD", "9H", "2D", "2H"),
                is_pass=False,
                suit_options=(),
            ),
            SimpleNamespace(
                player="right",
                cards=("3C", "3D", "3H", "3S"),
                is_pass=False,
                suit_options=(),
            ),
            SimpleNamespace(
                player="opposite",
                cards=(),
                is_pass=True,
                suit_options=(),
            ),
        ),
    )

    runtime.update_ready.emit(
        LiveUpdate(status="running", snapshot=snapshot, advice=None)
    )
    app.processEvents()

    assert window.size().width() == 500
    assert window.size().height() == 245
    assert window.trick_strip.turn_label.text() == "第8手"
    assert window.trick_strip.cells["self"].action_label.text() == "JJ922"
    assert window.trick_strip.cells["right"].action_label.text() == "3333"
    assert window.trick_strip.cells["opposite"].action_label.text() == "不出"
    assert window.trick_strip.cells["left"].action_label.text() == "等待"
    window.hide()


def test_float_window_prefers_a_non_overlapping_side_position():
    _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    target = QRect(0, 0, 800, 500)

    safe = window.place_beside(target)

    if safe:
        assert not window.geometry().intersects(target)
    window.hide()


def test_float_window_no_safe_slot_never_accepts_overlap(monkeypatch):
    _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)
    target = QRect(0, 0, 800, 500)

    class Screen:
        def availableGeometry(self):
            return QRect(0, 0, 800, 500)
        def name(self):
            return "test-screen"
        def devicePixelRatio(self):
            return 1.0

    screen = Screen()
    # place_beside resolves QGuiApplication from its own module.
    import daguandan_bridge.gui.recommendation_window as recommendation_module
    monkeypatch.setattr(recommendation_module.QGuiApplication, "screens", staticmethod(lambda: (screen,)))
    monkeypatch.setattr(recommendation_module.QGuiApplication, "screenAt", staticmethod(lambda _point: screen))

    assert window.place_beside(target) is False
    assert window.last_placement_diagnostic["reason"] == "no_safe_slot"
    window.hide()


def test_main_window_no_safe_slot_keeps_full_assistant_visible(monkeypatch):
    class Runtime:
        def target_client_rect(self):
            return SimpleNamespace(left=0, top=0, width=800, height=500)

    class Float:
        def __init__(self):
            self.place_calls = 0
            self.show_calls = 0
            self.raise_calls = 0
        def place_beside(self, _target):
            self.place_calls += 1
            return False
        def show(self): self.show_calls += 1
        def raise_(self): self.raise_calls += 1

    fake = DaguandanBridgeWindow.__new__(DaguandanBridgeWindow)
    fake.live_runtime = Runtime()
    fake.recommendation_window = Float()
    calls = {"full": 0, "minimized": 0, "message": 0}
    fake.show_full_assistant = lambda: calls.__setitem__("full", calls["full"] + 1)
    fake.showMinimized = lambda: calls.__setitem__("minimized", calls["minimized"] + 1)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", staticmethod(lambda *args: calls.__setitem__("message", calls["message"] + 1)))

    DaguandanBridgeWindow.show_compact_recommendation(fake)

    assert fake.recommendation_window.place_calls == 1
    assert fake.recommendation_window.show_calls == 0
    assert fake.recommendation_window.raise_calls == 0
    assert calls == {"full": 1, "minimized": 0, "message": 1}


def test_compact_action_bar_uses_icon_buttons_with_chinese_tooltips():
    app = _app()
    window = RecommendationFloatWindow(FakeRuntime())
    buttons = [
        window.capture_button, window.screenshot_folder_button,
        window.debug_button, window.copy_issue_button,
        window.copy_summary_button, window.open_button, window.stop_button,
    ]
    assert all(button.toolTip() for button in buttons)
    assert all(button.accessibleName() for button in buttons)
    assert all(not button.text() for button in buttons)
    assert "截取当前画面" in window.capture_button.toolTip()
    assert window.screenshot_folder_button.toolTip() == "打开截图目录"
    assert all(hasattr(button, "_compact_tooltip_filter") for button in buttons)
    actions = window.layout().itemAt(1).layout()
    assert actions.indexOf(window.screenshot_folder_button) == actions.indexOf(window.capture_button) + 1
    assert "打开窗口与牌局诊断" in window.debug_button.toolTip()
    window.close()


def test_float_window_renders_ready_waiting_first_action_for_structured_and_legacy_payloads():
    app = _app()
    window = RecommendationFloatWindow(FakeRuntime())

    window.apply_listening_status({
        "state": "opening",
        "generation": 8,
        "status": "PASS",
        "primary_reason": "READY_WAITING_FIRST_ACTION",
        "message": "已进入牌桌，等待自己首出",
        "suggested_action": "等待自己首出；首出后继续识别出牌",
    })
    assert window.suggestion_label.text() == "已进入牌桌，等待自己首出"

    window.apply_listening_status({
        "state": "opening",
        "phase": "ready_waiting_first_action",
        "generation": 9,
    })
    assert window.suggestion_label.text() == "已进入牌桌，等待自己首出"
    window.hide()
