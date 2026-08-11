from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QRect, Signal
from PySide6.QtWidgets import QApplication

from daguandan_bridge.danzero.advisor import LocalAdvice
from daguandan_bridge.gui.recommendation_window import RecommendationFloatWindow
from daguandan_bridge.live.orchestrator import AdviceRequestKey, LiveAdvice, LiveUpdate


def _app():
    return QApplication.instance() or QApplication([])


class FakeRuntime(QObject):
    update_ready = Signal(object)
    frame_ready = Signal(object)
    error = Signal(str)


def _ready_advice(*, visible: bool) -> LiveAdvice:
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
            engine_input={},
            timings={},
        ),
    )


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
    assert window.detail_label.text() == "耗时 12 ms"
    assert len(window._card_badges) == 5
    window.hide()


def test_float_window_marks_capture_backend_and_occlusion_pause():
    app = _app()
    runtime = FakeRuntime()
    window = RecommendationFloatWindow(runtime)

    runtime.frame_ready.emit(
        SimpleNamespace(frame=SimpleNamespace(backend="printwindow"))
    )
    app.processEvents()
    assert "后台窗口采集" in window.capture_label.text()

    runtime.error.emit("屏幕采集已暂停：目标牌桌被其他窗口遮挡")
    app.processEvents()
    assert window.suggestion_label.text() == "窗口遮挡，已暂停"
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
