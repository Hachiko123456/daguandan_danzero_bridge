from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

import daguandan_bridge.gui.replay_page as replay_page_module
from daguandan_bridge.application.replay_turn_draft import ReplayTurnDraftAssembler
from daguandan_bridge.gui.main_window import DaguandanBridgeWindow
from daguandan_bridge.gui.replay_page import FrameInspectDialog, ReplayPage
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    save_truth_log,
)
from daguandan_bridge.recognition_service import RecognitionResult, RecognizedEvent


def _app():
    return QApplication.instance() or QApplication([])


def _contrast(first: QColor, second: QColor) -> float:
    def luminance(color: QColor) -> float:
        channels = (color.redF(), color.greenF(), color.blueF())
        linear = tuple(
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        )
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _recorded_session(tmp_path, *, with_initial=False):
    session = tmp_path / "sessions" / "game-test"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "game-test",
                "status": "sealed",
                "frame_count": 3,
                "dropped_frames": 0,
            }
        ),
        encoding="utf-8",
    )
    timeline = session / "timeline.jsonl"
    if with_initial:
        hand = [
            f"{rank}{suit}"
            for rank in ("2", "3", "4", "5", "6", "7")
            for suit in "SHCD"
        ] + ["8S", "8H", "8C"]
        timeline.write_text(
            json.dumps(
                {
                    "event_id": "EVT-000001",
                    "event_type": "initial_state_confirmed",
                    "session_id": "game-test",
                    "seq": 1,
                    "monotonic_ms": 0,
                    "wall_time": "t0",
                    "trick_id": 1,
                    "turn_id": 1,
                    "actor": "self",
                    "payload": {"round_level": "2", "hand": hand, "lead_player": "self"},
                    "confidence": 1.0,
                    "source": "test",
                    "state_revision_before": 0,
                    "state_revision_after": 1,
                    "evidence_refs": [],
                    "schema_version": 1,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        timeline.touch()
    recorder = SessionRecorder(session, size=(64, 32), fps=10)
    for index in range(3):
        recorder.write_frame(
            np.full((32, 64, 3), index * 50, np.uint8),
            captured_monotonic_ms=index * 100,
            wall_time=f"t{index}",
        )
    recorder.close()
    return session


def _fake_result(**overrides) -> RecognitionResult:
    fields = dict(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="self",
        my_hand=("2S", "3H"),
        events=(),
        field_confidences={},
        sources={},
        unresolved_fields=(),
        diagnostics=(),
    )
    fields.update(overrides)
    return RecognitionResult(**fields)


class _FakeRecognition:
    def __init__(self, result):
        self.result = result

    def recognize(self, image):
        return self.result


def test_replay_page_can_load_recorded_session(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)

    page.select_session(session)
    app.processEvents()

    assert page.play_button.isEnabled()
    assert page.rewind_button.isEnabled()
    assert page.forward_button.isEnabled()
    assert page.frame_spin.maximum() == 2
    assert page.playback_toolbar.play_button is page.play_button
    assert page.playback_toolbar.rewind_button is page.rewind_button
    assert page.step_button.text() == "单图标注"
    assert not hasattr(page, "inspect_frame_button")
    assert not hasattr(page, "incident_combo")
    assert "game-test" in page.session_summary.text()
    assert "3" in page.session_summary.text()
    page.shutdown()
    page.close()


def test_replay_page_single_image_annotation_button_does_not_step_video(
    tmp_path,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        ReplayPage,
        "open_frame_inspect",
        lambda page: calls.append(page.current_session),
    )
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)
    page.select_session(session)
    app.processEvents()

    page.step_button.click()

    assert calls == [session.resolve()]
    assert page._decode_thread is None
    page.shutdown()
    page.close()


def test_replay_page_uses_overlay_seek_and_hysteretic_responsive_layout(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)
    page.show()

    assert page.playback_toolbar.rewind_button.isHidden()
    assert page.playback_toolbar.forward_button.isHidden()
    assert page.playback_toolbar.frame_jump_button.isHidden()
    assert page.content_scroll.horizontalScrollBarPolicy() == (
        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )
    page.resize(1024, 768)
    app.processEvents()
    assert page._content_vertical is True
    page.resize(1180, 768)
    app.processEvents()
    assert page._content_vertical is True
    page.resize(1366, 768)
    app.processEvents()
    assert page._content_vertical is False
    page.resize(1180, 768)
    app.processEvents()
    assert page._content_vertical is False

    requested = []
    page.playback_toolbar.seek_seconds_requested.connect(requested.append)
    page.rewind_overlay_button.click()
    page.forward_overlay_button.click()
    assert requested[-2:] == [-5.0, 5.0]
    requested_frames = []
    page.playback_toolbar.seek_requested.connect(requested_frames.append)
    page.frame_spin.setValue(1)
    page.frame_spin.lineEdit().returnPressed.emit()
    assert requested_frames[-1] == 1
    page.shutdown()
    page.close()


def test_replay_toolbar_uses_one_compact_row_when_video_pane_is_wide(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)
    page.show()
    page.playback_toolbar.resize(900, 100)
    app.processEvents()

    layout = page.playback_toolbar._layout
    for widget in (
        page.play_button,
        page.step_button,
        page.speed_combo,
        page.frame_spin,
        page.frame_status,
    ):
        index = layout.indexOf(widget)
        row, _column, _row_span, _column_span = layout.getItemPosition(index)
        assert row == 0
    page.shutdown()
    page.close()


def test_replay_page_light_and_dark_palettes_cover_native_scroll_and_editor_views(
    tmp_path,
    monkeypatch,
):
    app = _app()
    session = _recorded_session(tmp_path, with_initial=True)
    monkeypatch.setattr(replay_page_module, "isDarkTheme", lambda: False)
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._show_truth_log_editor()
    app.processEvents()

    light_viewport = page.content_scroll.viewport().palette().color(
        QPalette.ColorRole.Window
    )
    light_base = page.diagnostics.palette().color(QPalette.ColorRole.Base)
    light_text = page.diagnostics.palette().color(QPalette.ColorRole.Text)
    editor = page._truth_editor
    assert editor is not None
    light_table_base = editor.table.palette().color(QPalette.ColorRole.Base)
    assert light_viewport.lightnessF() > 0.8
    assert light_table_base.lightnessF() > 0.85
    assert _contrast(light_base, light_text) >= 7.0

    monkeypatch.setattr(replay_page_module, "isDarkTheme", lambda: True)
    page._apply_theme()
    app.processEvents()
    dark_viewport = page.content_scroll.viewport().palette().color(
        QPalette.ColorRole.Window
    )
    dark_card = page.diagnostics_card.palette().color(QPalette.ColorRole.Window)
    dark_base = page.diagnostics.palette().color(QPalette.ColorRole.Base)
    dark_text = page.diagnostics.palette().color(QPalette.ColorRole.Text)
    dark_table_base = editor.table.palette().color(QPalette.ColorRole.Base)
    dark_table_text = editor.table.palette().color(QPalette.ColorRole.Text)
    assert dark_viewport.lightnessF() < 0.2
    assert dark_card.lightnessF() < 0.25
    assert dark_base.lightnessF() < light_base.lightnessF()
    assert dark_table_base.lightnessF() < light_table_base.lightnessF()
    assert _contrast(dark_base, dark_text) >= 7.0
    assert _contrast(dark_table_base, dark_table_text) >= 7.0
    page.shutdown()
    page.close()


def test_streamed_truth_rows_update_unsaved_editor_draft(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    baseline = page._truth_log_for_video_scan()
    page._truth_scan_base = baseline
    page._truth_draft_assembler = ReplayTurnDraftAssembler(baseline)
    page.truth_log = baseline
    page._show_truth_log_editor()

    page._collect_truth_scan_turn(
        {
            "turn_id": 1,
            "trick_id": 1,
            "frame_index": 12,
            "actor": "self",
            "recognized_cards": ["2S"],
            "recognized_pass": False,
        }
    )
    editor = page._truth_editor
    assert editor is not None
    assert editor.table.rowCount() == 1
    assert editor.table.columnCount() == 5
    assert editor._frame_scan_provider == page._frames_after_current

    page._collect_truth_scan_turn(
        {
            "turn_id": 2,
            "trick_id": 1,
            "frame_index": 25,
            "actor": "right",
            "recognized_cards": [],
            "recognized_pass": True,
        }
    )
    assert editor.table.rowCount() == 2
    assert [turn.actor for turn in page.truth_log.turns] == ["self", "right"]
    assert not (session / "truth_log.json").exists()
    page.shutdown()
    page.close()


def test_streamed_suit_correction_replaces_draft_row_without_overwriting_saved_log(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=True)
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("2", "3", "4", "5", "6", "7")
        for suit in "SHCD"
    ) + ("8S", "8H", "8C")
    source = TruthLog("game-test", TruthInitialState("2", "self", hand), ())
    truth_path = session / "truth_log.json"
    save_truth_log(truth_path, source)
    source_bytes = truth_path.read_bytes()
    page = ReplayPage(session.parent)
    page.select_session(session)
    baseline = page._truth_log_for_video_scan()
    page._truth_scan_base = baseline
    page._truth_draft_assembler = ReplayTurnDraftAssembler(baseline)
    page.truth_log = baseline
    page._show_truth_log_editor()

    page._collect_truth_scan_turn(
        {
            "kind": "action",
            "turn_id": 1,
            "actor": "self",
            "recognized_pass": False,
            "recognized_cards": ["J?"],
        }
    )
    page._collect_truth_scan_turn(
        {
            "kind": "suit_corrected",
            "target_turn_id": 1,
            "actor": "self",
            "recognized_cards": ["JD"],
        }
    )

    assert page._truth_editor is not None
    assert page._truth_editor.table.rowCount() == 1
    assert page.truth_log.turns[0].cards == ("JD",)
    assert truth_path.read_bytes() == source_bytes
    page.shutdown()
    page.close()


def test_replay_page_reuses_existing_button_for_trusted_advisor_mode(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)

    trusted_index = page.replay_mode_combo.findData("trusted_advisor")
    assert trusted_index >= 0
    page.replay_mode_combo.setCurrentIndex(trusted_index)
    app.processEvents()

    assert page.truth_replay_button.text() == "开始实时助手测试"
    page.shutdown()
    page.close()


def test_replay_page_exposes_the_same_recognition_strategies(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)

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
    page.shutdown()
    page.close()


def test_replay_page_only_exposes_live_replay_modes(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)

    modes = {
        page.replay_mode_combo.itemData(index): page.replay_mode_combo.itemText(index)
        for index in range(page.replay_mode_combo.count())
    }

    assert modes == {
        "pipeline": "状态机管线（实时同核心）",
        "trusted_advisor": "可信日志驱动（测试实时策略）",
    }
    page.shutdown()
    page.close()


def test_replay_page_only_enables_recognition_strategy_for_pipeline_mode(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)

    assert page.replay_mode_combo.currentData() == "pipeline"
    assert page.recognition_strategy_combo.isEnabled()

    trusted_index = page.replay_mode_combo.findData("trusted_advisor")
    page.replay_mode_combo.setCurrentIndex(trusted_index)
    app.processEvents()

    assert page.recognition_strategy_combo.isEnabled() is False
    page.shutdown()
    page.close()


def test_replay_page_prefills_truth_log_for_editor(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)

    assert page.truth_log is not None
    assert page.truth_edit_button.isEnabled()
    assert not page.truth_replay_button.isEnabled()
    assert "0" in page.truth_status.text()
    page.shutdown()
    page.close()


def test_replay_page_turns_confirmed_frame_scan_into_editable_truth_log(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    assert page.truth_log is not None

    generated = page._truth_log_from_scan_turns(
        page.truth_log,
        (
            {
                "turn_id": 1,
                "frame_index": 12,
                "actor": "self",
                "recognized_cards": ["7S", "7H"],
                "recognized_pass": False,
            },
            {
                "turn_id": 2,
                "frame_index": 26,
                "actor": "right",
                "recognized_cards": [],
                "recognized_pass": True,
            },
        ),
    )
    app.processEvents()

    assert "逐帧分析" in page.visual_replay_button.text()
    assert page.visual_replay_button.parentWidget() is page.diagnostics_stack.widget(0)
    assert [(turn.actor, turn.is_pass, turn.cards, turn.frame_index) for turn in generated.turns] == [
        ("self", False, ("7S", "7H"), 12),
        ("right", True, (), 26),
    ]
    assert generated.initial_state == page.truth_log.initial_state
    page.shutdown()
    page.close()


def test_replay_page_opens_editor_with_unsaved_frame_scan_draft(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    assert page.truth_log is not None
    page._truth_scan_base = page.truth_log

    page._collect_truth_scan_turn(
        {
            "turn_id": 1,
            "frame_index": 12,
            "actor": "self",
            "recognized_cards": ["7S"],
            "recognized_pass": False,
        }
    )
    page._truth_scan_completed(None)
    app.processEvents()

    editor = page.truth_editor_host.itemAt(0).widget()
    assert page.truth_log is not None
    assert page.truth_log.turns[0].frame_index == 12
    assert page.diagnostics_stack.currentIndex() == 1
    assert editor.table.rowCount() == 1
    assert "待校验" in page.truth_status.text()
    page.shutdown()
    page.close()


def test_replay_page_builds_truth_log_from_recognition(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path, with_initial=False)
    page = ReplayPage(session.parent)
    page.select_session(session)

    assert page.truth_log is None

    log = page._truth_log_from_recognition(
        np.zeros((32, 64, 3), np.uint8),
        recognition=_FakeRecognition(_fake_result()),
    )

    assert log.initial_state.round_level == "2"
    assert log.initial_state.lead_player == "self"
    assert log.initial_state.my_hand == ("2S", "3H")
    assert log.turns == ()
    page.shutdown()
    page.close()


def test_replay_page_recognition_requires_hand(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path, with_initial=False)
    page = ReplayPage(session.parent)
    page.select_session(session)

    with pytest.raises(ValueError):
        page._truth_log_from_recognition(
            np.zeros((32, 64, 3), np.uint8),
            recognition=_FakeRecognition(_fake_result(my_hand=())),
        )
    page.shutdown()
    page.close()


def test_frame_inspect_dialog_shows_frame_and_recognizes():
    app = _app()
    frame = np.zeros((64, 128, 3), np.uint8)
    dialog = FrameInspectDialog(frame, _FakeRecognition(_fake_result()), "测试帧", None)
    app.processEvents()

    assert dialog.page._source_image is frame
    assert dialog.windowTitle() == "单图标注"
    assert dialog.save_frame_button.text() == "保存当前画面为截图"
    assert not dialog.page.isWindow(), "embedded page must not be a separate window"
    assert dialog.page.image_preview.pixmap() is not None
    assert dialog.page.recognition_elapsed_ms is not None
    dialog.shutdown()
    dialog.close()


def test_seek_to_frame_starts_playing_from_target(tmp_path):
    import time

    app = _app()
    session = _recorded_session(tmp_path)
    page = ReplayPage(session.parent)
    page.select_session(session)
    app.processEvents()

    page.frame_spin.setValue(2)
    page.seek_to_frame()
    assert page._playing is True

    deadline = time.time() + 5
    while (
        page._current_record is None or page._current_record.frame_index < 2
    ) and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)

    assert page._current_record is not None
    assert page._current_record.frame_index >= 2
    page.shutdown()
    page.close()


    app = _app()
    window = DaguandanBridgeWindow()

    assert window.replay_page.objectName() == "replayPage"

    window.close()
    app.processEvents()
