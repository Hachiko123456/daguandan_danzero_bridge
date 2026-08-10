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

    def __init__(self):
        super().__init__()
        self.started = None
        self.listening_started = 0
        self.confirmed_candidate_id = None
        self.manual_action = None
        self.correction = None

    def recognize_initial(self):
        pass

    def start_listening(self):
        self.listening_started += 1

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

    def shutdown(self):
        pass


def _app():
    return QApplication.instance() or QApplication([])


def _recognition(hand):
    return SimpleNamespace(
        round_level="2",
        wild_rank="2",
        lead_player="right",
        current_player="right",
        my_hand=hand,
        diagnostics=(),
    )


def test_live_page_requires_exactly_27_cards_before_start():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)

    page.apply_initial_recognition(_recognition(("3S",)), None)
    app.processEvents()

    assert not hasattr(page, "start_session_button")
    assert "27" in page.initialization_status.text()
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
    assert "DanZero 开始计算建议" in LiveAssistantPage._event_text(
        _event("advice_requested")
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
    assert "min-width:26px" in page._cards_html(("QC", "QH", "QS", "QS"))
    assert "#0f766e" in timeline_html
    assert page.timeline.verticalScrollBar().value() == page.timeline.verticalScrollBar().maximum()
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
    assert "待画面确认" in page.timeline.toPlainText()
    assert "#b45309" in page.timeline.toHtml().lower()
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
    assert "自动重试" in page.timeline.toPlainText()
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


def test_live_page_uses_compact_hand_strip_and_has_no_quick_correction_controls():
    app = _app()
    page = LiveAssistantPage(FakeRuntime())

    assert page.initial_hand_scroll.height() <= 52
    assert not hasattr(page, "correct_cards_button")
    assert not hasattr(page, "correct_pass_button")
    assert page.timeline_title.text() == "对局动态"
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


def test_main_window_registers_live_page_in_fluent_navigation():
    app = _app()
    window = DaguandanBridgeWindow(live_runtime=FakeRuntime())

    assert isinstance(window, FluentWindow)
    assert not hasattr(window, "capture_page")
    assert window.annotation_page.objectName() == "annotationPage"
    assert window.live_assistant_page.objectName() == "liveAssistantPage"

    window.close()
    app.processEvents()
