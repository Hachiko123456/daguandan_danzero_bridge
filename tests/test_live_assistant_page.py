from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.live_assistant_page import LiveAssistantPage
from daguandan_bridge.gui.main_window import DaguandanBridgeWindow
from daguandan_bridge.live.orchestrator import ReviewCandidate, ReviewRequest
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
        self.confirmed_candidate_id = None
        self.manual_action = None
        self.correction = None

    def recognize_initial(self):
        pass

    def start_session(self, *, round_level, hand, lead_player):
        self.started = (round_level, hand, lead_player)

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

    assert not page.start_session_button.isEnabled()
    assert "27" in page.initialization_status.text()
    page.close()


def test_live_page_starts_with_manually_confirmed_single_image_values():
    app = _app()
    runtime = FakeRuntime()
    page = LiveAssistantPage(runtime)
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("2", "3", "4", "5", "6", "7")
        for suit in "SHCD"
    ) + ("8S", "8H", "8C")
    page.apply_initial_recognition(_recognition(hand), None)
    app.processEvents()

    assert page.start_session_button.isEnabled()
    page.start_session_button.click()

    assert runtime.started == ("2", hand, "right")
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


def test_main_window_registers_live_page_in_fluent_navigation():
    app = _app()
    window = DaguandanBridgeWindow(live_runtime=FakeRuntime())

    assert isinstance(window, FluentWindow)
    assert window.capture_page.objectName() == "capturePage"
    assert window.annotation_page.objectName() == "annotationPage"
    assert window.live_assistant_page.objectName() == "liveAssistantPage"

    window.close()
    app.processEvents()
