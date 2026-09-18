from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.live_assistant_page import LiveAssistantPage
from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.live.models import LiveEvent
from daguandan_bridge.gui.main_window import DaguandanBridgeWindow
from daguandan_bridge.live.orchestrator import (
    AdviceRequestKey,
    LiveAdvice,
    LiveUpdate,
    ReviewCandidate,
    ReviewRequest,
)
from daguandan_bridge.recognition_service import FastSignalResult
from qfluentwidgets import FluentWindow


class FakeRuntime(QObject):
    initial_recognized = Signal(object, object)
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)
    session_finished = Signal(object)
    listening_status = Signal(object)
    log_delivery_status = Signal(object)

    def __init__(self):
        super().__init__()
        self.recording_mode = "game"
        self.recording_mode_updates = []
        self.recording_max_total_bytes = 20 * 1024 ** 3
        self.automatic_log_include_media = False
        self.recording_capacity_updates = []
        self.automatic_log_media_updates = []
        self.session_data_recording_enabled = True
        self.session_data_recording_updates = []
        self.started = None
        self.listening_started = 0
        self.start_listening_result = None
        self.confirmed_candidate_id = None
        self.manual_action = None
        self.correction = None
        self.full_diagnostic_requests = 0
        self.opened_log_directory = None
        self.open_log_requests = 0

    def recognize_initial(self):
        pass

    def start_listening(self):
        self.listening_started += 1
        return self.start_listening_result

    def start_session(
        self,
        *,
        round_level,
        hand,
        lead_player,
        recognition_strategy="missing",
    ):
        self.started = (round_level, hand, lead_player, recognition_strategy)

    def confirm_lead_player(self, seat):
        self.confirmed_lead = seat

    def confirm_candidate(self, candidate_id):
        self.confirmed_candidate_id = candidate_id

    def confirm_manual_action(self, *, cards=(), is_pass):
        self.manual_action = (cards, is_pass)

    def correct_latest(self, *, cards=(), is_pass):
        self.correction = (cards, is_pass)

    def pause(self):
        pass

    def resume(self):
        pass

    def finish(self):
        pass

    def request_full_diagnostic_export(self):
        self.full_diagnostic_requests += 1

    def open_automatic_log_directory(self):
        self.opened_log_directory = "C:/logs"
        return self.opened_log_directory

    def request_open_automatic_log_directory(self):
        self.open_log_requests += 1

    def set_session_data_recording_enabled(self, enabled):
        self.session_data_recording_enabled = bool(enabled)
        self.session_data_recording_updates.append(bool(enabled))

    def set_recording_mode(self, mode):
        self.recording_mode = str(mode)
        self.recording_mode_updates.append(self.recording_mode)
        self.session_data_recording_enabled = self.recording_mode != "none"

    def set_recording_max_total_gb(self, value):
        self.recording_max_total_bytes = int(float(value) * 1024 ** 3)
        self.recording_capacity_updates.append(int(value))
        return self.recording_max_total_bytes

    def set_automatic_log_include_media(self, enabled):
        self.automatic_log_include_media = bool(enabled)
        self.automatic_log_media_updates.append(bool(enabled))
        return self.automatic_log_include_media

    def recording_storage_summary(self):
        return {
            "limit_bytes": self.recording_max_total_bytes,
            "used_bytes": 4 * 1024 ** 3,
            "remaining_bytes": 16 * 1024 ** 3,
            "capacity_exhausted": False,
        }

    def shutdown(self):
        pass


def _app():
    return QApplication.instance() or QApplication([])


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


def _recognition(hand):
    return SimpleNamespace(
        round_level="2",
        wild_rank="2",
        lead_player="right",
        current_player="right",
        my_hand=hand,
        diagnostics=(),
    )


def test_full_assistant_intro_describes_staged_hand_and_lead_confirmation():
    from PySide6.QtWidgets import QLabel
    _app()
    page = LiveAssistantPage(FakeRuntime())
    texts = [label.text() for label in page.findChildren(QLabel)]
    assert any("确认起手牌和首出信息后自动开始" in text for text in texts)
    assert not any("稳定识别两次相同" in text for text in texts)


def test_full_assistant_log_controls_show_loading_success_and_failure():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    page.export_full_diagnostic_button.click()
    assert runtime.full_diagnostic_requests == 1
    assert page.export_full_diagnostic_button.isEnabled() is False

    runtime.log_delivery_status.emit(
        {
            "status": "PASS",
            "include_media": True,
            "diagnostic_zip_path": "C:/logs/full.zip",
        }
    )
    app.processEvents()
    assert page.export_full_diagnostic_button.isEnabled() is True
    assert "C:/logs/full.zip" in page.log_delivery_status.text()

    runtime.log_delivery_status.emit(
        {"status": "FAIL", "include_media": True, "error": "disk full"}
    )
    app.processEvents()
    assert "disk full" in page.log_delivery_status.text()

    page.open_log_directory_button.click()
    assert runtime.open_log_requests == 1
    assert page.open_log_directory_button.isEnabled() is False
    runtime.log_delivery_status.emit(
        {
            "status": "PASS",
            "action": "open_directory",
            "output_directory": "C:/logs",
        }
    )
    app.processEvents()
    assert page.open_log_directory_button.isEnabled() is True
    assert "C:/logs" in page.log_delivery_status.text()


def test_full_assistant_disables_log_actions_when_recording_is_disabled():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    index = page.recording_mode_combo.findData("none")
    assert index >= 0
    page.recording_mode_combo.setCurrentIndex(index)
    app.processEvents()

    assert page.open_log_directory_button.isEnabled() is False
    assert page.export_full_diagnostic_button.isEnabled() is False
    assert "日志功能不可用" in page.log_delivery_status.text()

    runtime.log_delivery_status.emit({"status": "DISABLED"})
    app.processEvents()
    assert page.export_full_diagnostic_button.isEnabled() is False


def test_full_assistant_clears_only_geometry_terminal_error_on_restart():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    runtime.listening_status.emit(
        {
            "state": "failed",
            "message": "监听已停止，请打开完整助手",
            "reason": "窗口持续抖动",
        }
    )
    app.processEvents()
    assert page.error_status.text() == "错误：窗口持续抖动"

    runtime.listening_status.emit(
        {
            "state": "listening",
            "message": "持续监听页面中",
        }
    )
    app.processEvents()
    assert page.initialization_status.text() == "持续监听页面中"
    assert page.error_status.text() == ""

    runtime.listening_status.emit(
        {
            "state": "recovering",
            "message": "牌桌窗口发生变化，正在重新连接",
        }
    )
    app.processEvents()
    assert page.initialization_status.text() == "牌桌窗口发生变化，正在重新连接"
    assert page.error_status.text() == ""

    runtime.listening_status.emit(
        {
            "state": "recovered",
            "message": "牌桌窗口已重新连接，继续监听",
        }
    )
    app.processEvents()
    assert page.initialization_status.text() == "牌桌窗口已重新连接，继续监听"
    assert page.error_status.text() == ""

    page.show_error("模板资源仍然不可用")
    runtime.listening_status.emit(
        {"state": "listening", "message": "持续监听页面中"}
    )
    app.processEvents()
    assert page.error_status.text() == "错误：模板资源仍然不可用"
    page.shutdown()


def test_live_page_requires_exactly_27_cards_before_start():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    page.apply_initial_recognition(_recognition(("3S",)), None)
    app.processEvents()

    assert not hasattr(page, "start_session_button")
    assert "27" in page.initialization_status.text()
    page.close()


def test_live_page_shows_recognized_initial_state_and_block_reason():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())

    page.apply_initial_recognition(
        SimpleNamespace(
            round_level=None,
            wild_rank=None,
            lead_player="left",
            current_player="right",
            my_hand=("3S",) * 27,
            diagnostics=("未识别到当前级牌",),
        ),
        None,
    )
    app.processEvents()

    status = page.initialization_status.text()
    assert "识别级牌：未识别" in status
    assert "百搭级牌：未识别" in status
    assert "起手牌：27/27 张" in status
    assert "当前行动：右家" in status
    assert "首发候选：左家" in status
    assert "建局状态：等待级牌识别" in status
    page.close()


def test_live_page_starts_persistent_listener_without_manual_start_button():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    page.recognize_initial_button.click()
    app.processEvents()

    assert runtime.listening_started == 1
    assert not hasattr(page, "start_session_button")
    page.close()


def test_live_page_keeps_full_assistant_visible_when_window_lock_fails():
    app = _app()
    runtime = FakeRuntime()
    runtime.start_listening_result = False
    page = LiveAssistantPage(runtime)
    requested = []
    page.compact_mode_requested.connect(lambda: requested.append(True))

    page.recognize_initial_button.click()
    app.processEvents()

    assert runtime.listening_started == 1
    assert requested == []
    page.close()


def test_live_page_persists_selected_recording_mode_for_the_next_listener():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    assert {
        page.recording_mode_combo.itemData(index)
        for index in range(page.recording_mode_combo.count())
    } == {"none", "game", "all"}
    page.recording_mode_combo.setCurrentIndex(
        page.recording_mode_combo.findData("all")
    )
    app.processEvents()

    assert runtime.recording_mode_updates == ["all"]
    assert runtime.recording_mode == "all"
    assert runtime.session_data_recording_enabled is True
    page.close()


def test_live_page_persists_recording_capacity_and_auto_media_preferences():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    page.recording_capacity_spin.setValue(30)
    page.recording_capacity_spin.editingFinished.emit()
    assert runtime.recording_capacity_updates == [30]
    assert runtime.recording_max_total_bytes == 30 * 1024 ** 3

    page.automatic_log_media_check.setChecked(True)
    assert runtime.automatic_log_media_updates == [True]
    assert runtime.automatic_log_include_media is True
    assert "已用" in page.recording_storage_status.text()
    page.close()


def test_live_page_does_not_reuse_a_stale_single_image_lead_candidate():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    page.apply_initial_recognition(_recognition(("3S",)), None)
    assert page.lead_player_combo.currentData() == "right"

    page.apply_initial_recognition(
        SimpleNamespace(
            round_level="2",
            wild_rank="2",
            lead_player="right",
            current_player="right",
            my_hand=("3S",),
            diagnostics=(),
            buttons=("super_double",),
        ),
        None,
    )
    assert page.lead_player_combo.currentData() == ""

    page.apply_initial_recognition(
        SimpleNamespace(
            round_level="2",
            wild_rank="2",
            lead_player=None,
            current_player=None,
            my_hand=("3S",),
            diagnostics=(),
            buttons=(),
        ),
        None,
    )
    app.processEvents()

    assert page.lead_player_combo.currentData() == ""
    page.close()


def test_live_page_does_not_show_a_lead_candidate_during_normal_double():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())

    page.apply_initial_recognition(
        SimpleNamespace(
            round_level="2",
            wild_rank="2",
            lead_player="opposite",
            current_player="opposite",
            my_hand=("3S",),
            diagnostics=(),
            buttons=("double",),
        ),
        None,
    )
    app.processEvents()

    assert page.lead_player_combo.currentData() == ""
    page.close()


def test_live_page_exposes_four_recognition_strategies():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())

    options = {
        page.recognition_strategy_combo.itemData(index)
        for index in range(page.recognition_strategy_combo.count())
    }

    assert options == {
        "reference_single_shot",
        "two_valid_streak",
        "stable_single_shot",
        "valid_candidate_vote",
    }
    page.close()


def test_live_page_shows_super_double_decision_state():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    update = LiveUpdate(
        status="running",
        snapshot=SimpleNamespace(current_player="left", trick_id=1, turn_id=1),
        fast_signals=FastSignalResult(
            expected_player="left",
            active_player=None,
            pass_visible=True,
            self_action_buttons_visible=False,
            effect_visible=True,
            super_double_visible=True,
        ),
    )

    page.apply_update(update)
    app.processEvents()

    assert page.live_status.text() == "状态：正在决定是否加倍"
    assert "加倍按钮显示期间不进行首出或出牌识别" in page.turn_status.text()
    page.close()


def test_live_page_shows_provisional_cannot_beat_status_for_one_frame():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    update = LiveUpdate(
        status="running",
        snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=2),
        fast_signals=_cannot_beat_fast(),
    )

    page.apply_update(update)
    app.processEvents()

    assert page.live_status.text() == "状态：检测到要不起，正在确认不出"
    assert "单帧信号不会直接提交动作" in page.turn_status.text()
    page.close()


def test_live_page_cannot_beat_candidate_hides_stale_ready_advice():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    old_advice = LiveAdvice(
        key=AdviceRequestKey("session", 7, 8),
        status="ready",
        visible=True,
        advice=LocalAdvice(
            strategy="test",
            cards=("3S",),
            play_type="Single",
            is_pass=False,
            state_revision=8,
            elapsed_ms=12.0,
            request_id="ADV-0007-0008",
            engine_input={
                "debug": True,
                "decision": {
                    "best_action_text": "3S",
                    "best_q": 1.0,
                    "q_gap": 0.2,
                    "candidates": [],
                },
            },
            timings={},
        ),
    )

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(
                current_player="self",
                trick_id=1,
                turn_id=7,
                revision=8,
            ),
            advice=old_advice,
        )
    )
    app.processEvents()
    assert not page.fabledan_debug_card.isHidden()

    current_snapshot = SimpleNamespace(
        current_player="self",
        trick_id=1,
        turn_id=8,
        revision=9,
    )
    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=current_snapshot,
            advice=old_advice,
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()

    assert page.live_status.text() == "状态：检测到要不起，正在确认不出"
    assert page.fabledan_debug_card.isHidden()

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=current_snapshot,
            advice=old_advice,
            fast_signals=_cannot_beat_fast(cannot_beat_visible=False),
        )
    )
    app.processEvents()

    assert page.live_status.text() == "状态：运行中"
    assert page.fabledan_debug_card.isHidden()
    page.close()


def test_live_page_rejects_unsafe_cannot_beat_candidates_and_clears_status():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    snapshot = SimpleNamespace(current_player="self", trick_id=1, turn_id=2)

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()
    assert page.live_status.text() == "状态：检测到要不起，正在确认不出"

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=snapshot,
            fast_signals=_cannot_beat_fast(active_player=None),
        )
    )
    app.processEvents()
    assert page.live_status.text() == "状态：检测到要不起，正在确认不出"

    unsafe_signals = (
        _cannot_beat_fast(cannot_beat_confidence=0.79),
        _cannot_beat_fast(active_player="right"),
        _cannot_beat_fast(effect_visible=True),
        _cannot_beat_fast(self_action_buttons_visible=False),
        _cannot_beat_fast(cannot_beat_visible=False),
    )
    for signal in unsafe_signals:
        page.apply_update(
            LiveUpdate(
                status="running",
                snapshot=snapshot,
                fast_signals=signal,
            )
        )
        app.processEvents()
        assert page.live_status.text() == "状态：运行中"
        assert "要不起" not in page.turn_status.text()
    page.close()


def test_live_page_recovery_status_wins_over_cannot_beat_candidate():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=2),
            advice=LiveAdvice(
                key=AdviceRequestKey("session", 2, 2),
                status="withheld",
                withhold_reason="turn_recovery_pending",
                error="正在补齐刚才的快速出牌，暂缓推荐",
            ),
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()

    assert page.live_status.text() == "状态：正在补齐刚才的快速出牌"
    assert "要不起" not in page.live_status.text()
    page.close()


def test_live_page_labels_wind_catch_recovery_as_temporary_state():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=2),
            advice=LiveAdvice(
                key=AdviceRequestKey("session", 2, 3),
                status="withheld",
                withhold_reason="wind_catch_pass_recovery_pending",
                error="等待接风前最后一个不出",
            ),
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()

    assert page.live_status.text() == "状态：正在确认接风前的不出"
    assert "历史不完整" not in page.live_status.text()
    page.close()


def test_live_page_reserves_confirmed_history_gap_text_for_terminal_withhold():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=2),
            advice=LiveAdvice(
                key=AdviceRequestKey("session", 2, 4),
                status="withheld",
                withhold_reason="visual_finish_without_complete_history",
                error="牌局历史不完整，暂停推荐",
            ),
            fast_signals=_cannot_beat_fast(),
        )
    )
    app.processEvents()

    assert page.live_status.text() == "状态：已确认牌局历史存在缺口"
    page.close()



def _event(event_type, *, actor=None, payload=None):
    return LiveEvent(
        event_id="EVT-1",
        event_type=event_type,
        session_id="session",
        seq=1,
        monotonic_ms=0,
        wall_time="2026-08-06T00:00:00+00:00",
        trick_id=1,
        turn_id=1,
        actor=actor,
        payload=payload or {},
        confidence=1.0,
        source="test",
        state_revision_before=0,
        state_revision_after=1,
    )


def test_live_page_formats_lead_turn_and_advice_events():
    assert "[第1手] 首出确认：对家" == LiveAssistantPage._event_text(
        _event("lead_player_confirmed", actor="opposite", payload={"lead_player": "opposite"})
    )
    assert "轮到自己" in LiveAssistantPage._event_text(
        _event("turn_started", actor="self", payload={"player": "self"})
    )
    assert "等待加倍结束与首出标志" in LiveAssistantPage._event_text(
        _event("waiting_for_lead")
    )
    assert "识别暂停，请确认对家动作：首出不能不出；多帧结果不一致" in LiveAssistantPage._event_text(
        _event(
            "review_required",
            actor="opposite",
            payload={"reason": "pass_not_allowed,insufficient_consensus"},
        )
    )
    assert "FableDan 开始计算建议" in LiveAssistantPage._event_text(
        _event("advice_requested", payload={"advisor_name": "FableDan"})
    )
    assert "等待自己回合旁证" in LiveAssistantPage._event_text(
        _event("advice_ready", payload={"visible": False})
    )


def test_live_page_timeline_is_selectable_and_distinguishes_visible_advice():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    advice = LiveAdvice(
        key=AdviceRequestKey("session", 1, 1),
        status="ready",
        visible=True,
        advice=LocalAdvice(
            strategy="test",
            cards=("2S", "2H"),
            play_type="Pair",
            is_pass=False,
            state_revision=1,
            elapsed_ms=12.0,
            request_id="ADV-0001-0001",
            engine_input={},
            timings={},
        ),
    )
    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=1),
            event=_event(
                "player_played",
                actor="left",
                payload={"cards": ["QC", "QH", "QS", "QS"]},
            ),
            advice=advice,
        )
    )
    app.processEvents()

    timeline_html = page.timeline.toHtml().lower()
    timeline_text = page.timeline.toPlainText()

    assert page.timeline.isReadOnly()
    assert page.timeline.textInteractionFlags() & Qt.TextSelectableByMouse
    assert "QC QH QS QS" not in timeline_text
    assert "2S 2H" not in timeline_text
    assert "左家出牌：" in timeline_text
    assert "♥" in timeline_text
    cards_html = page._cards_html(("QC", "QH", "QS", "QS"))
    assert "<br>" not in cards_html
    assert "min-width:36px" in cards_html
    assert "font-size:21px" in cards_html
    assert "#0f766e" in timeline_html
    assert page.timeline.verticalScrollBar().value() == page.timeline.verticalScrollBar().maximum()
    page.close()


def test_live_page_exposes_compact_recommendation_mode():
    _app()
    page = LiveAssistantPage(FakeRuntime())
    requested = []
    page.compact_mode_requested.connect(lambda: requested.append(True))

    page.compact_button.click()

    assert requested == [True]
    page.close()


def test_live_page_shows_unconfirmed_pass_advice_instead_of_hiding_it():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    advice = LiveAdvice(
        key=AdviceRequestKey("session", 5, 6),
        status="ready",
        visible=False,
        advice=LocalAdvice(
            strategy="test",
            cards=(),
            play_type="PASS",
            is_pass=True,
            state_revision=6,
            elapsed_ms=11.0,
            request_id="ADV-0005-0006",
            engine_input={},
            timings={},
        ),
    )

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=5),
            advice=advice,
        )
    )
    app.processEvents()

    assert "建议：不出" in page.timeline.toPlainText()
    assert "待确认" not in page.timeline.toPlainText()
    assert "DanZero 建议" in page.timeline.toPlainText()
    assert "#0f766e" in page.timeline.toHtml().lower()
    page.close()


def test_live_page_renders_one_compact_entry_when_advice_becomes_visible():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    local = LocalAdvice(
        strategy="test",
        cards=(),
        play_type="PASS",
        is_pass=True,
        state_revision=8,
        elapsed_ms=12.0,
        request_id="ADV-0007-0008",
        engine_input={},
        timings={},
    )
    hidden = LiveAdvice(
        key=AdviceRequestKey("session", 7, 8),
        status="ready",
        visible=False,
        advice=local,
    )
    visible = replace(hidden, visible=True)
    snapshot = SimpleNamespace(current_player="self", trick_id=1, turn_id=7)

    page.apply_update(LiveUpdate(status="running", snapshot=snapshot, advice=hidden))
    page.apply_update(LiveUpdate(status="running", snapshot=snapshot, advice=visible))
    app.processEvents()

    text = page.timeline.toPlainText()
    assert text.count("DanZero 建议") == 1
    assert text.count("建议：不出") == 1
    assert "请求 ADV" not in text
    page.close()


def test_live_page_labels_fabledan_failure_without_danzero_wording():
    app = _app()
    runtime = FakeRuntime()
    runtime.advisor_strategy = "fabledan"
    page = LiveAssistantPage(runtime)
    failed = LiveAdvice(
        key=AdviceRequestKey("session", 2, 3),
        status="failed",
        error="FableDan model 执行失败",
    )

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=2),
            advice=failed,
        )
    )
    app.processEvents()

    text = page.timeline.toPlainText()
    assert "FableDan 计算失败" in text
    assert "DanZero" not in text
    assert page.danzero_warmup_status.text() == "FableDan 模型准备中"
    page.close()


def test_live_page_renders_all_action_and_outcome_events_from_one_update():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    played = _event("player_played", actor="right", payload={"cards": ["7S"]})
    finished = _event("player_finished", actor="right", payload={"placement": "head"})
    wind = _event(
        "wind_caught",
        actor="left",
        payload={"from_player": "right", "to_player": "left"},
    )
    # The helper uses deliberately repeated ids; make the update mirror real
    # state-machine output where every auxiliary event has its own id.
    finished = replace(finished, event_id="AUX-000001")
    wind = replace(wind, event_id="AUX-000002")

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="opposite", trick_id=1, turn_id=2),
            event=played,
            events=(played, finished, wind),
        )
    )
    app.processEvents()

    text = page.timeline.toPlainText()
    assert "头游" in text
    assert "接风：右家 → 左家" in text
    assert "7S" not in text
    page.close()


def test_live_page_puts_played_cards_on_the_same_timeline_line():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    played = _event(
        "player_played",
        actor="left",
        payload={
            "cards": ["3H", "3S", "4D", "4S", "5?", "7H"],
            "suit_options": [
                ["H"], ["S"], ["D"], ["S"], ["S", "C"], ["H"],
            ],
        },
    )

    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="opposite", trick_id=1, turn_id=2),
            event=played,
        )
    )
    app.processEvents()

    timeline_html = page.timeline.toHtml().lower()
    timeline_text = page.timeline.toPlainText()
    assert "左家出牌：" in timeline_text
    assert "♥ 3" in timeline_text
    assert "vertical-align:middle" in timeline_html
    assert "候选：♠/♣" in page._cards_html(
        ("3H", "3S", "4D", "4S", "5?", "7H"),
        (("H",), ("S",), ("D",), ("S",), ("S", "C"), ("H",)),
    )
    page.close()


def test_live_page_renders_initial_hand_as_cards_without_internal_codes():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())

    page.apply_initial_recognition(
        _recognition(("small_joker", "2D", "QC")),
        None,
    )
    app.processEvents()

    assert [badge.card_code for badge in page.initial_hand_badges] == [
        "small_joker",
        "2D",
        "QC",
    ]
    assert page.hand_edit.isHidden()
    assert "2D" not in page.initial_hand_cards.toolTip()
    assert "方块2" in page.initial_hand_cards.toolTip()
    assert all(badge.width() == 36 and badge.height() == 50 for badge in page.initial_hand_badges)
    page.close()


def test_live_page_shows_automatic_retry_without_review_panel():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(
                current_player="opposite",
                trick_id=1,
                turn_id=2,
            ),
            event=_event(
                "recognition_retry",
                actor="opposite",
                payload={"reason": "action_timeout"},
            ),
        )
    )
    app.processEvents()

    assert not page.review_bar.isVisible()
    assert "尚未捕获对家的新动作，继续等待" in page.timeline.toPlainText()
    assert "⌛ 等待动作" in page.timeline.toPlainText()
    page.close()


def test_live_page_clears_only_transient_timeline_when_a_new_session_starts():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("2", "3", "4", "5", "6", "7")
        for suit in "SHCD"
    ) + ("8S", "8H", "8C")
    page.apply_initial_recognition(_recognition(hand), None)
    page.apply_update(
        LiveUpdate(
            status="running",
            snapshot=SimpleNamespace(current_player="self", trick_id=1, turn_id=1),
            event=_event("player_played", actor="right", payload={"cards": ["9S"]}),
        )
    )
    page.error_status.setText("old error")
    app.processEvents()
    assert page.timeline.toPlainText()

    page._reset_transient_session_ui()

    assert page.timeline.toPlainText() == ""
    assert page.error_status.text() == "old error"
    assert page.lead_player_combo.currentData() == ""
    page.close()


def test_live_page_uses_readable_hand_strip_and_has_no_quick_correction_controls():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())

    assert page.initial_hand_scroll.height() >= 64
    assert not hasattr(page, "correct_cards_button")
    assert not hasattr(page, "correct_pass_button")
    assert page.timeline_title.text() == "对局动态"
    page.close()


def test_live_page_shows_fabledan_top_three_and_reuses_detail_payload():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    candidates = [
        {
            "rank": index,
            "action_text": action,
            "q": q_value,
            "action": {"play_type": "PAIR", "cards": [], "claim_ranks": []},
        }
        for index, (action, q_value) in enumerate(
            (
                ("9999", 1.3274),
                ("PASS", 0.9121),
                ("66", 0.8473),
                ("77", 0.8036),
                ("88", 0.7625),
                ("1010", 0.7000),
            ),
            start=1,
        )
    ]
    advice = LocalAdvice(
        strategy="fabledan-numpy",
        cards=("9S", "9H", "9D", "9C"),
        play_type="BOMB",
        is_pass=False,
        state_revision=2,
        elapsed_ms=12.0,
        request_id="fabledan-debug",
        engine_input={
            "debug": True,
            "top_n": 3,
            "model_path": "models/fabledan/best.npz",
            "player": 0,
            "level": 3,
            "level_text": "6",
            "hand": ["9S", "9H", "9D", "9C"],
            "left": [27, 27, 27, 25],
            "lead_text": "55",
            "lead_owner": 3,
            "history": [],
            "legal_actions": [candidate["action"] for candidate in candidates],
            "q_values": [],
            "decision": {
                "best_action_text": "9999",
                "best_q": 1.3274,
                "second_q": 0.9121,
                "q_gap": 0.4153,
                "candidates": candidates,
            },
        },
        timings={},
    )

    page._show_fabledan_decision(advice)
    app.processEvents()

    assert not page.fabledan_debug_card.isHidden()
    assert page.fabledan_recommendation.text() == "9999"
    assert page.fabledan_q_value.text() == "Q值：1.3274"
    assert page.fabledan_q_gap.text() == "Top1 - Top2：0.4153"
    assert "当前需要压：55" in page.fabledan_context.text()
    assert "不是可直接解读为百分比" in page.fabledan_top_three_hint.text()
    assert page.fabledan_candidates.text().splitlines() == [
        "1. 9999    Q=1.3274",
        "2. PASS    Q=0.9121",
        "3. 66    Q=0.8473",
    ]
    assert "77" not in page.fabledan_candidates.text()
    assert "1010" not in page.fabledan_candidates.text()
    assert '"model_path": "models/fabledan/best.npz"' in page.fabledan_detail.toPlainText()
    page.fabledan_detail_button.click()
    assert not page.fabledan_detail.isHidden()
    assert page.fabledan_detail_button.text() == "收起模型诊断"
    page.close()


def test_live_page_shows_compact_fabledan_top_three_without_diagnostics():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())
    advice = LocalAdvice(
        strategy="fabledan-numpy",
        cards=("9S", "9H", "9D", "9C"),
        play_type="BOMB",
        is_pass=False,
        state_revision=2,
        elapsed_ms=12.0,
        request_id="fabledan-compact",
        engine_input={
            "debug": False,
            "top_n": 3,
            "level_text": "6",
            "lead_text": "55",
            "decision": {
                "best_action_text": "9999",
                "best_q": 1.3274,
                "second_q": 0.9121,
                "q_gap": 0.4153,
                "candidates": [
                    {"rank": 1, "action_text": "9999", "q": 1.3274},
                    {"rank": 2, "action_text": "PASS", "q": 0.9121},
                    {"rank": 3, "action_text": "66", "q": 0.8473},
                ],
            },
        },
        timings={},
    )

    page._show_fabledan_decision(advice)
    app.processEvents()

    assert not page.fabledan_debug_card.isHidden()
    assert not page.fabledan_detail_button.isVisible()
    assert page.fabledan_detail.isHidden()
    assert page.fabledan_candidates.text().splitlines() == [
        "1. 9999    Q=1.3274",
        "2. PASS    Q=0.9121",
        "3. 66    Q=0.8473",
    ]
    page.close()


def test_review_bar_confirms_existing_candidate_with_one_click():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)
    review = ReviewRequest(
        reason="candidate_conflict",
        player="right",
        candidates=(
            ReviewCandidate("CAND-1", ("7H", "7S"), False, 3, 0.94, True),
            ReviewCandidate("CAND-2", ("7D", "7S"), False, 2, 0.82, True),
        ),
    )

    page.show_review(review)
    app.processEvents()
    page.review_candidate_buttons[0].click()

    assert runtime.confirmed_candidate_id == "CAND-1"
    assert not page.review_bar.isVisible()
    page.close()


def test_review_bar_disables_candidate_that_failed_rule_validation():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)
    review = ReviewRequest(
        reason="does_not_beat_table",
        player="right",
        candidates=(
            ReviewCandidate(
                "CAND-invalid",
                ("6S",),
                False,
                3,
                0.95,
                False,
                "does_not_beat_table",
            ),
        ),
    )

    page.show_review(review)
    app.processEvents()

    assert not page.review_candidate_buttons[0].isEnabled()
    page.close()


def test_main_window_registers_live_page_in_fluent_navigation(monkeypatch):
    app = _app()
    registrations = []
    add_sub_interface = DaguandanBridgeWindow.addSubInterface

    def record_sub_interface(window, interface, icon, text, *args, **kwargs):
        registrations.append((interface, icon, text))
        return add_sub_interface(window, interface, icon, text, *args, **kwargs)

    monkeypatch.setattr(
        DaguandanBridgeWindow, "addSubInterface", record_sub_interface
    )
    window = DaguandanBridgeWindow(live_runtime=FakeRuntime())

    assert isinstance(window, FluentWindow)
    assert not hasattr(window, "capture_page")
    assert window.annotation_page.objectName() == "annotationPage"
    assert window.live_assistant_page.objectName() == "liveAssistantPage"
    assert not hasattr(window, "model_evaluation_page")
    assert all(text != "模型整局评测" for _interface, _icon, text in registrations)

    window.close()
    app.processEvents()
