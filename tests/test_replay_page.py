from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QDialog

import daguandan_bridge.gui.replay_page as replay_page_module
from daguandan_bridge.application.replay_turn_draft import ReplayTurnDraftAssembler
from daguandan_bridge.gui.main_window import DaguandanBridgeWindow
from daguandan_bridge.gui.replay_page import (
    FrameInspectDialog,
    PureVideoScanThread,
    ReplayPage,
    ScanInitialStateDialog,
)
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


def test_replay_page_uses_explicit_profile_for_arbitrary_session_path(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path / "copied", with_initial=False)
    profiles_root = tmp_path / "resources" / "profiles"
    (profiles_root / "custom").mkdir(parents=True)
    (profiles_root / "custom" / "profile.json").write_text("{}", encoding="utf-8")
    page = ReplayPage(
        session.parent,
        profiles_root=profiles_root,
        profile_name="custom",
    )
    page.select_session(session)
    assert page.profiles_root == profiles_root.resolve()
    assert page.profile_name == "custom"
    page.configure_context(
        session.parent,
        profiles_root=profiles_root,
        profile_name="custom",
        session=session,
    )
    assert page.current_session == session.resolve()
    page.shutdown()
    page.close()


def test_replay_page_scan_requires_only_avi_not_timeline_truth_or_frame_index(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=False)
    (session / "video" / "frame_index.jsonl").unlink()
    (session / "timeline.jsonl").unlink()
    page = ReplayPage(session.parent)
    page.select_session(session)

    assert page.visual_replay_button.isEnabled()
    assert "缺少已确认的首出玩家" not in page.truth_scan_status.text()
    page.shutdown()
    page.close()


def test_scan_entry_uses_pure_avi_thread_without_truth_log_baseline(tmp_path, monkeypatch):
    _app()
    session = _recorded_session(tmp_path, with_initial=False)
    (session / "timeline.jsonl").unlink()
    page = ReplayPage(session.parent)
    page.select_session(session)
    started = []
    monkeypatch.setattr(
        ReplayPage,
        "_truth_log_for_video_scan",
        lambda *_args: pytest.fail("pure AVI scan must not ask for a TruthLog baseline"),
    )
    monkeypatch.setattr(page, "_recognition", lambda: object())
    monkeypatch.setattr(PureVideoScanThread, "start", lambda thread: started.append(thread))

    page.analyze_video_to_truth_log()

    assert len(started) == 1
    assert started[0].video_path == session / "video" / "game.avi"
    assert not hasattr(started[0], "truth_log")
    assert started[0].frame_index_path == session / "video" / "frame_index.jsonl"
    page.shutdown()
    page.close()


def test_pure_scan_result_builds_draft_only_from_external_scan_artifacts(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=False)
    output = tmp_path / "derived" / "scan"
    output.mkdir(parents=True)
    opening = output / "opening_candidates.json"
    opening.write_text(
        json.dumps({"status": "needs_review", "lead_player": "left"}),
        encoding="utf-8",
    )
    observations = output / "frame_observations.jsonl.gz"
    observation = {
        "opening": {
            "round_level": "2",
            "my_hand": ["2S", "3H"],
            "lead_player_signal": "left",
        }
    }
    with gzip.open(observations, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(observation) + "\n")
    actions = output / "action_trace.jsonl"
    actions.write_text(
        json.dumps(
            {
                "actor": "left",
                "is_pass": False,
                "cards": ["4C"],
                "frame_start": 12,
                "evidence_frames": [12, 13],
                "uncertainty": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    page = ReplayPage(session.parent)
    page.select_session(session)
    result = {
        "opening_path": opening,
        "observations_path": observations,
        "action_trace_path": actions,
    }
    draft = page._truth_log_from_pure_scan_result(result)

    assert draft is not None
    assert draft.provenance.source == "video_scan+canonical_reconciliation"
    assert draft.initial_state.lead_player == "left"
    assert draft.initial_state.round_level == "2"
    assert draft.turns[0].cards == ("4C",)
    page.shutdown()
    page.close()


def test_pure_scan_result_preserves_shared_converter_review_items(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=False)
    output = tmp_path / "derived" / "scan-review"
    output.mkdir(parents=True)
    actions = output / "action_trace.jsonl"
    actions.write_text(
        json.dumps(
            {
                "action_id": "pass-with-cards",
                "actor": "left",
                "is_pass": True,
                "cards": ["4C"],
                "frame_start": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    page = ReplayPage(session.parent)
    page.select_session(session)

    draft = page._truth_log_from_pure_scan_result(
        {"action_trace_path": actions},
        initial_override={
            "lead_player": "left",
            "round_level": "2",
            "my_hand": ["2S"],
        },
    )

    assert draft is not None
    assert draft.turns[0].is_pass is True
    assert draft.turns[0].cards == ()
    assert len(page._pure_scan_review_items) == 1
    assert (
        "pass_cards_present_normalized_to_empty"
        in page._pure_scan_review_items[0]["reasons"]
    )
    page.shutdown()
    page.close()


def test_pure_scan_result_does_not_default_missing_lead_to_self(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=False)
    page = ReplayPage(session.parent)
    page.select_session(session)

    assert page._truth_log_from_pure_scan_result({}) is None
    page.shutdown()
    page.close()


def test_scan_initial_state_dialog_requires_explicit_initial_facts(tmp_path):
    _app()
    dialog = ScanInitialStateDialog({})
    assert dialog.lead_combo.currentData() == ""
    assert dialog.level_combo.currentData() == ""
    dialog.confirm_button.click()
    assert dialog.result() == 0

    dialog.lead_combo.setCurrentIndex(dialog.lead_combo.findData("left"))
    dialog.level_combo.setCurrentIndex(dialog.level_combo.findData("2"))
    dialog._hand = ("2S", "3H")
    dialog.confirm_button.click()
    assert dialog.result() == 1
    assert dialog.initial_state() == {
        "lead_player": "left",
        "round_level": "2",
        "my_hand": ["2S", "3H"],
    }
    dialog.close()


def test_scan_initial_state_review_turns_scan_package_into_editable_draft(
    tmp_path,
    monkeypatch,
):
    _app()
    session = _recorded_session(tmp_path, with_initial=False)
    output = tmp_path / "derived" / "scan"
    output.mkdir(parents=True)
    actions = output / "action_trace.jsonl"
    actions.write_text(
        json.dumps(
            {
                "actor": "left",
                "is_pass": False,
                "cards": ["4C"],
                "frame_start": 12,
                "evidence_frames": [12],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._pure_scan_result = {"action_trace_path": actions}
    rendered = []
    monkeypatch.setattr(page, "_show_truth_log_editor", rendered.append)

    class _ConfirmedInitialState:
        def __init__(self, *_args):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        @staticmethod
        def initial_state():
            return {
                "lead_player": "left",
                "round_level": "2",
                "my_hand": ["2S", "3H"],
            }

    monkeypatch.setattr(replay_page_module, "ScanInitialStateDialog", _ConfirmedInitialState)
    page._open_pure_scan_initial_state_review()

    assert page._proposed_scan_log is not None
    assert page._proposed_scan_log.initial_state.lead_player == "left"
    assert page._proposed_scan_log.turns[0].cards == ("4C",)
    assert rendered == [page._proposed_scan_log]
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
    page._begin_truth_scan(baseline)

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
    assert editor.isEnabled() is False
    assert editor.table.rowCount() == 1
    assert editor.table.columnCount() == 4
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
    assert page.truth_log is not None
    assert page.truth_log.turns == ()
    assert [turn.actor for turn in page._truth_scan_log.turns] == ["self", "right"]
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
    page._begin_truth_scan(baseline)

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
    assert page.truth_log is not None
    assert page.truth_log.turns == ()
    assert page._truth_scan_log.turns[0].cards == ("JD",)
    assert truth_path.read_bytes() == source_bytes
    page.shutdown()
    page.close()


def test_streamed_event_correction_replaces_draft_row_without_overwriting_saved_log(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=True)
    truth_path = session / "truth_log.json"
    save_truth_log(
        truth_path,
        TruthLog(
            "game-test",
            TruthInitialState(
                "2",
                "self",
                tuple(
                    f"{rank}{suit}"
                    for rank in ("2", "3", "4", "5", "6", "7")
                    for suit in "SHCD"
                )
                + ("8S", "8H", "8C"),
            ),
            (),
        ),
    )
    source_bytes = truth_path.read_bytes()
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._begin_truth_scan(page._truth_log_for_video_scan())

    page._collect_truth_scan_turn(
        {
            "turn_id": 1,
            "actor": "self",
            "recognized_pass": False,
            "recognized_cards": ["A?", "K?"],
        }
    )
    page._collect_truth_scan_turn(
        {
            "kind": "event_correction",
            "target_turn_id": 1,
            "actor": "self",
            "recognized_pass": False,
            "recognized_cards": ["AC", "AS", "KC", "KS", "QD", "QS"],
        }
    )

    assert page.truth_log is not None
    assert page.truth_log.turns == ()
    assert page._truth_scan_log is not None
    assert page._truth_scan_log.turns[0].cards == (
        "AC", "AS", "KC", "KS", "QD", "QS"
    )
    assert truth_path.read_bytes() == source_bytes
    page.shutdown()
    page.close()


def test_scan_stays_in_memory_until_explicit_save_writes_canonical_truth(
    tmp_path,
):
    _app()
    session = _recorded_session(tmp_path, with_initial=True)
    hand = tuple(
        f"{rank}{suit}"
        for rank in ("2", "3", "4", "5", "6", "7")
        for suit in "SHCD"
    ) + ("8S", "8H", "8C")
    canonical = TruthLog(
        "game-test",
        TruthInitialState("2", "self", hand),
        (),
    )
    canonical_path = session / "truth_log.json"
    save_truth_log(canonical_path, canonical)
    canonical_before = canonical_path.read_bytes()
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._begin_truth_scan(page._truth_log_for_video_scan())

    editor = page._truth_editor
    assert editor is not None
    assert editor.save_button.text() == "保存 TruthLog"
    assert editor.isEnabled() is False
    page._collect_truth_scan_turn(
        {
            "turn_id": 1,
            "actor": "self",
            "recognized_pass": False,
            "recognized_cards": ["2S"],
        }
    )
    page._collect_truth_scan_turn(
        {
            "turn_id": 2,
            "actor": "right",
            "recognized_pass": True,
            "recognized_cards": [],
        }
    )

    assert canonical_path.read_bytes() == canonical_before
    page._truth_scan_completed(None)

    assert canonical_path.read_bytes() == canonical_before
    assert page._truth_scan_log is not None
    assert [turn.actor for turn in page._truth_scan_log.turns] == ["self", "right"]
    assert editor.isEnabled() is True
    assert "可编辑并点击『保存 TruthLog』" in page.truth_scan_status.text()

    editor._save()

    assert canonical_path.read_bytes() != canonical_before
    saved = replay_page_module.load_truth_log(canonical_path, session_id="game-test")
    assert [turn.actor for turn in saved.turns] == ["self", "right"]
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

    assert page.truth_replay_button.text() == "助手复测"
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


def test_session_lead_confirmation_builds_opposite_scan_baseline_and_keeps_turn_one(
    tmp_path,
):
    _app()
    session = _recorded_session(tmp_path, with_initial=True)
    timeline_path = session / "timeline.jsonl"
    initial = json.loads(timeline_path.read_text("utf-8"))
    initial["actor"] = None
    initial["payload"]["lead_player"] = None
    confirmed = {
        **initial,
        "event_id": "EVT-000002",
        "event_type": "lead_player_confirmed",
        "seq": 2,
        "actor": "opposite",
        "payload": {"lead_player": "opposite"},
        "state_revision_before": 1,
        "state_revision_after": 2,
    }
    timeline_path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in (initial, confirmed)
        )
        + "\n",
        encoding="utf-8",
    )
    page = ReplayPage(session.parent)
    page.select_session(session)

    assert page.truth_log is not None
    assert page.truth_log.initial_state.lead_player == "opposite"
    baseline = page._truth_log_for_video_scan()
    assert baseline.initial_state.lead_player == "opposite"
    page._begin_truth_scan(baseline)
    page._collect_truth_scan_turn(
        {
            "turn_id": 1,
            "actor": "opposite",
            "recognized_pass": False,
            "recognized_cards": ["2S"],
        }
    )

    assert page._truth_scan_failure is None
    assert page._truth_scan_log is not None
    assert [(turn.index, turn.actor, turn.cards) for turn in page._truth_scan_log.turns] == [
        (1, "opposite", ("2S",)),
    ]
    page.shutdown()
    page.close()


def test_rejected_or_out_of_sequence_streamed_action_marks_scan_failed_without_saving(
    tmp_path,
):
    _app()
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._begin_truth_scan(page._truth_log_for_video_scan())
    editor = page._truth_editor
    assert editor is not None and editor.isEnabled() is False

    page._collect_truth_scan_turn(
        {
            "turn_id": 2,
            "actor": "right",
            "recognized_pass": False,
            "recognized_cards": ["2S"],
        }
    )

    assert page._truth_scan_failure is not None
    assert "turn_id 不连续" in page._truth_scan_failure
    assert "扫描失败" in page.truth_scan_status.text()
    assert editor.isEnabled() is True
    assert not (session / "truth_log.json").exists()
    page.shutdown()
    page.close()


def test_scan_completion_and_failure_emit_page_level_infobars(tmp_path, monkeypatch):
    _app()
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        replay_page_module.InfoBar,
        "success",
        lambda **kwargs: calls.append(("success", kwargs)),
    )
    monkeypatch.setattr(
        replay_page_module.InfoBar,
        "error",
        lambda **kwargs: calls.append(("error", kwargs)),
    )
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._begin_truth_scan(page._truth_log_for_video_scan())
    page._collect_truth_scan_turn(
        {
            "turn_id": 1,
            "actor": "self",
            "recognized_pass": False,
            "recognized_cards": ["2S"],
        }
    )
    page._truth_scan_completed(None)

    assert calls[0][0] == "success"
    assert calls[0][1]["title"] == "扫描完成"
    assert "可编辑并点击『保存 TruthLog』" in str(calls[0][1]["content"])
    assert calls[0][1]["parent"] is page

    page._fail_truth_scan("第 2 手被拒绝")

    assert calls[1][0] == "error"
    assert calls[1][1]["title"] == "扫描失败"
    assert "未保存" in str(calls[1][1]["content"])
    assert calls[1][1]["parent"] is page
    page.shutdown()
    page.close()


def test_truth_scan_progress_bar_tracks_worker_progress_and_completion(
    tmp_path, monkeypatch
):
    _app()
    monkeypatch.setattr(replay_page_module.InfoBar, "success", lambda **_kwargs: None)
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._begin_truth_scan(page._truth_log_for_video_scan())

    assert page.truth_scan_progress.isHidden() is False
    assert page.truth_scan_progress.isEnabled()
    assert page.truth_scan_progress.value() == 0

    page._truth_scan_progress_changed(720, 1568, 719)

    assert page.truth_scan_progress.maximum() == 1568
    assert page.truth_scan_progress.value() == 720
    assert page.truth_scan_status.text() == "扫描中：720/1568 帧（46%）"

    page._truth_scan_completed(type("ReplayResult", (), {"frame_count": 1568})())

    assert page.truth_scan_progress.isHidden() is False
    assert page.truth_scan_progress.isEnabled() is False
    assert page.truth_scan_progress.value() == 1568
    assert "扫描完成" in page.truth_scan_status.text()
    page.shutdown()
    page.close()


def test_truth_scan_progress_resets_after_partial_completion_and_ignores_stale_session(
    tmp_path, monkeypatch
):
    _app()
    monkeypatch.setattr(replay_page_module.InfoBar, "error", lambda **_kwargs: None)
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    page._begin_truth_scan(page._truth_log_for_video_scan())
    page._truth_scan_progress_changed(720, 1568, 719)

    page._truth_scan_completed(type("ReplayResult", (), {"frame_count": 720})())

    assert page.truth_scan_progress.isHidden()
    assert page.truth_scan_progress.isEnabled() is False
    assert page.truth_scan_progress.value() == 0
    assert "扫描失败" in page.truth_scan_status.text()

    page.select_session(session)
    page._truth_scan_progress_changed(1568, 1568, 1567)

    assert page.truth_scan_progress.isHidden()
    assert page.truth_scan_status.text() == "扫描：未开始"
    page.shutdown()
    page.close()


def test_replay_page_uses_short_scan_and_replay_action_labels(tmp_path):
    _app()
    page = ReplayPage(tmp_path / "sessions")

    assert page.visual_replay_button.text() == "扫描出牌"
    assert page.truth_edit_button.text() == "编辑日志"
    assert page.state_replay_button.text() == "状态重放"
    assert page.truth_replay_button.text() == "复测"
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

    assert page.visual_replay_button.text() == "扫描出牌"
    assert page.visual_replay_button.parentWidget() is page.diagnostics_stack.widget(0)
    assert [(turn.actor, turn.is_pass, turn.cards, turn.frame_index) for turn in generated.turns] == [
        ("self", False, ("7S", "7H"), 12),
        ("right", True, (), 26),
    ]
    assert generated.initial_state == page.truth_log.initial_state
    page.shutdown()
    page.close()


def test_replay_page_loads_latest_batch_scan_actions_without_published_truth_log(tmp_path):
    _app()
    session = _recorded_session(tmp_path, with_initial=True)
    reports = tmp_path / "reports"
    batch = reports / "unverified_20260912_000000_test"
    output = batch / session.name
    output.mkdir(parents=True)
    (batch / "summary.json").write_text(
        json.dumps({
            "selected_count": 1,
            "completed_count": 1,
            "failed_count": 0,
            "cancelled": False,
            "sessions": [{"session_id": session.name, "status": "complete"}],
        }),
        encoding="utf-8",
    )
    (output / "scan_summary.json").write_text(
        json.dumps({"status": "complete", "session_id": session.name}),
        encoding="utf-8",
    )
    (output / "action_trace.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "action_id": 1,
                    "actor": "self",
                    "is_pass": False,
                    "cards": ["2S"],
                    "frame_start": 1,
                    "frame_end": 1,
                    "best_frame": 1,
                    "evidence_frames": [1],
                },
                {
                    "action_id": 2,
                    "actor": "right",
                    "is_pass": True,
                    "cards": [],
                    "frame_start": 2,
                    "frame_end": 2,
                    "best_frame": 2,
                    "evidence_frames": [2],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    page = ReplayPage(session.parent, batch_output_root=reports)
    page.select_session(session)
    app = QApplication.instance()
    assert app is not None
    app.processEvents()

    assert page.truth_log is not None
    assert len(page.truth_log.turns) == 2
    assert page._proposed_scan_log is not None
    assert page._batch_scan_draft_source == output.resolve()
    assert "批量扫描草稿" in page.truth_status.text()
    assert page.truth_edit_button.isEnabled()

    page.shutdown()
    page.close()


def test_replay_page_opens_editor_with_unsaved_frame_scan(tmp_path):
    app = _app()
    session = _recorded_session(tmp_path, with_initial=True)
    page = ReplayPage(session.parent)
    page.select_session(session)
    assert page.truth_log is not None
    baseline = page._truth_log_for_video_scan()
    page._begin_truth_scan(baseline)

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
    assert page._truth_scan_log is not None
    assert page._truth_scan_log.turns[0].frame_index == 12
    assert page.diagnostics_stack.currentIndex() == 1
    assert editor.table.rowCount() == 1
    assert editor.isEnabled() is True
    assert "未保存" in page.truth_status.text()
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


def test_replay_page_owns_unverified_batch_scan_with_progress(tmp_path):
    _app()
    sessions = tmp_path / "sessions"
    verified = _recorded_session(tmp_path / "verified", with_initial=True)
    draft = _recorded_session(tmp_path / "draft", with_initial=True)
    # Put the helpers under one selector root while keeping the fixture paths
    # independent, matching how the page discovers real session directories.
    sessions.mkdir(parents=True)
    verified_target = sessions / "verified"
    draft_target = sessions / "draft"
    shutil.copytree(verified, verified_target)
    shutil.copytree(draft, draft_target)
    for target, session_id in ((verified_target, "verified"), (draft_target, "draft")):
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        manifest["session_id"] = session_id
        (target / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    save_truth_log(
        verified_target / "truth_log.json",
        TruthLog(
            "verified",
            TruthInitialState("2", "self", tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")),
            (),
            label_status="verified",
        ),
    )
    calls = []

    class FakeBatchService:
        @staticmethod
        def recommended_workers():
            return 2

        def scan(self, descriptors, **kwargs):
            calls.append((tuple(descriptors), kwargs))
            kwargs["on_progress"]("draft", 4, 4, 3, 0)
            return type(
                "Result",
                (),
                {
                    "selected_count": 1,
                    "completed_count": 1,
                    "failed_count": 0,
                    "cancelled": False,
                    "summary_path": tmp_path / "reports" / "summary.json",
                    "output_directory": tmp_path / "reports" / "unverified",
                },
            )()

    page = ReplayPage(
        sessions,
        profiles_root=tmp_path / "profiles",
        profile_name="test-profile",
        unverified_batch_service=FakeBatchService(),
        batch_output_root=tmp_path / "reports",
    )
    assert page.scan_unverified_button.text() == "扫描未验证对局"
    assert page.scan_unverified_button.isEnabled()
    page.scan_unverified_button.click()
    deadline = time.time() + 3
    while page._unverified_batch_thread is not None and time.time() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)

    assert calls
    assert {item.session_id for item in calls[0][0]} == {"draft"}
    assert calls[0][1]["profile_root"] == tmp_path / "profiles" / "test-profile"
    assert page.batch_progress.value() == 100
    assert "1/1 局成功" in page.batch_progress_label.text()
    assert not page.cancel_unverified_button.isEnabled()
    page.shutdown()
    page.close()


def test_session_selector_labels_verified_and_draft_truth_logs(tmp_path):
    _app()
    verified = _recorded_session(tmp_path / "verified", with_initial=True)
    draft = _recorded_session(tmp_path / "draft", with_initial=True)
    save_truth_log(
        verified / "truth_log.json",
        TruthLog("game-test", TruthInitialState("2", "self", tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")), (), label_status="verified"),
    )
    save_truth_log(
        draft / "truth_log.json",
        TruthLog("game-test", TruthInitialState("2", "self", tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")), (), label_status="draft"),
    )
    selector_root = tmp_path / "selector"
    selector_root.mkdir()
    verified_target = selector_root / "verified-game"
    draft_target = selector_root / "draft-game"
    shutil.copytree(verified, verified_target)
    shutil.copytree(draft, draft_target)
    page = ReplayPage(selector_root)
    verified_index = page.session_combo.findData(str(verified_target))
    draft_index = page.session_combo.findData(str(draft_target))
    assert page.session_combo.itemText(verified_index).startswith("✓ 已验证")
    assert page.session_combo.itemText(draft_index).startswith("△ 草稿")
    assert not page.session_combo.itemIcon(verified_index).isNull()
    assert not page.session_combo.itemIcon(draft_index).isNull()
    page.shutdown()
    page.close()
