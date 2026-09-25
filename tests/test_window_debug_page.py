from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from daguandan_bridge.gui.window_debug_page import WindowDebugPage, _key_blockers


class FakeWindowDebugService:
    def __init__(self) -> None:
        self.build_calls: list[dict[str, object]] = []
        self.windows = [
            {
                "hwnd": 101,
                "pid": 7,
                "process_name": "WeChat.exe",
                "title": "牌桌",
                "class_name": "WeChatAppEx",
                "visible": True,
                "iconic": False,
                "client_rect": {"left": 0, "top": 0, "width": 320, "height": 180},
            }
        ]

    def list_windows(self):
        return list(self.windows)

    def probe(self, hwnd: int):
        return {**self.windows[0], "hwnd": hwnd}

    def build(self, *, hwnd: int, capture: bool, recognize: bool):
        self.build_calls.append(
            {"hwnd": hwnd, "capture": capture, "recognize": recognize}
        )
        return {
            "schema": "test-report/v1",
            "read_only": True,
            "control_policy": {
                "move": False,
                "resize": False,
                "focus": False,
                "click": False,
                "restore": False,
            },
            "window": self.probe(hwnd),
            "capture": {
                "backend": "printwindow",
                "image_persisted": False,
            },
            "errors": [],
            "opening_readiness_inputs": {
                "readiness": {
                    "status": "WAIT",
                    "primary_reason": "OPENING_UNRESOLVED",
                    "message": "首出信息尚未确认",
                }
            },
        }


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_window_debug_page_refreshes_selects_and_runs_one_frame_diagnosis():
    _app()
    service = FakeWindowDebugService()
    page = WindowDebugPage(report_service=service)

    page.refresh_windows()

    assert page.window_selector.count() == 1
    assert page.selected_hwnd() == 101
    assert "牌桌" in page.status_view.toPlainText()

    page.diagnose_current_window()

    assert service.build_calls == [
        {"hwnd": 101, "capture": True, "recognize": True}
    ]
    assert "image_persisted" in page.json_view.toPlainText()
    assert "OPENING_UNRESOLVED" in page.blockers_view.item(0).text()
    assert page.export_button.isEnabled()
    page.close()


def test_window_debug_page_exports_only_after_explicit_path_selection(tmp_path, monkeypatch):
    _app()
    service = FakeWindowDebugService()
    page = WindowDebugPage(service)
    page.refresh_windows()
    page.diagnose_current_window()

    target = tmp_path / "window-debug.json"
    monkeypatch.setattr(
        "daguandan_bridge.gui.window_debug_page.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(target), "JSON 文件 (*.json)"),
    )

    assert page.export_report() == target
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["capture"]["image_persisted"] is False
    assert not (tmp_path / "window-debug.png").exists()
    page.close()

class _Readiness:
    def __init__(self, *, status: str, hard_error: bool, reason: str, message: str, action: str):
        self.status = status
        self.hard_error = hard_error
        self.primary_reason = reason
        self.message = message
        self.suggested_action = action

    def to_dict(self):
        return {
            "status": self.status,
            "hard_error": self.hard_error,
            "primary_reason": self.primary_reason,
            "message": self.message,
            "suggested_action": self.suggested_action,
        }


class _CompactWindow:
    def __init__(self) -> None:
        self.placed = []
        self.applied_status = []
        self.show_calls = 0
        self.raise_calls = 0

    def apply_listening_status(self, payload):
        self.applied_status.append(payload)

    def place_beside(self, rect):
        self.placed.append(rect)
        return True

    def show(self):
        self.show_calls += 1

    def raise_(self):
        self.raise_calls += 1


class _CompactHarness:
    def __init__(self, readiness) -> None:
        self.live_runtime = type("Runtime", (), {})()
        self.live_runtime.opening_readiness = readiness
        self.target_calls = 0
        self.live_runtime.target_client_rect = self._target_client_rect
        self.recommendation_window = _CompactWindow()
        self.full_assistant_calls = 0
        self.minimized_calls = 0

    def _target_client_rect(self):
        self.target_calls += 1
        return type("Rect", (), {"left": 10, "top": 20, "width": 320, "height": 180})()

    def show_full_assistant(self):
        self.full_assistant_calls += 1

    def showMinimized(self):
        self.minimized_calls += 1


def test_main_window_keeps_full_assistant_visible_on_readiness_hard_error(monkeypatch):
    from daguandan_bridge.gui import main_window

    harness = _CompactHarness(
        _Readiness(
            status="FAIL",
            hard_error=True,
            reason="ROI_FATAL",
            message="识别区域配置存在致命错误",
            action="打开完整助手修复 ROI 配置",
        )
    )
    shown = {}
    monkeypatch.setattr(
        main_window.QMessageBox,
        "warning",
        lambda *_args: shown.update(title=_args[1], text=_args[2]),
    )

    main_window.DaguandanBridgeWindow.show_compact_recommendation(harness)

    assert harness.full_assistant_calls == 1
    assert harness.target_calls == 0
    assert harness.recommendation_window.placed == []
    assert harness.minimized_calls == 0
    assert "识别区域配置存在致命错误" in shown["text"]
    assert "打开完整助手修复 ROI 配置" in shown["text"]
    assert "ROI_FATAL" in shown["text"]


def test_main_window_keeps_structured_wait_reason_in_diagnostic_compact(monkeypatch):
    from daguandan_bridge.gui import main_window

    harness = _CompactHarness(
        _Readiness(
            status="WAIT",
            hard_error=False,
            reason="OPENING_UNRESOLVED",
            message="首出信息尚未确认",
            action="等待牌局开局证据",
        )
    )
    monkeypatch.setattr(main_window.QMessageBox, "information", lambda *_args: None)

    main_window.DaguandanBridgeWindow.show_compact_recommendation(harness)

    assert harness.full_assistant_calls == 0
    assert harness.target_calls == 1
    assert harness.recommendation_window.show_calls == 1
    assert harness.minimized_calls == 1
    assert harness.recommendation_window.applied_status[0]["primary_reason"] == "OPENING_UNRESOLVED"
    assert harness._last_compact_readiness is harness.live_runtime.opening_readiness


def test_waiting_state_uses_diagnostic_geometry_without_authorizing_recommendations(monkeypatch):
    from daguandan_bridge.gui import main_window

    readiness = _Readiness(
        status="WAIT",
        hard_error=False,
        reason="LOBBY",
        message="本局已结束，等待下一局",
        action="点击继续游戏",
    )
    harness = _CompactHarness(readiness)
    harness.live_runtime.target_client_rect = lambda: None
    harness.live_runtime.diagnostic_target_client_rect = harness._target_client_rect
    monkeypatch.setattr(main_window.QMessageBox, "information", lambda *_args: None)

    main_window.DaguandanBridgeWindow.show_compact_recommendation(harness)

    assert harness.target_calls == 1
    assert harness.recommendation_window.show_calls == 1
    assert harness.minimized_calls == 1


def test_detailed_report_renders_structured_recognition_fields_and_threshold_reason():
    _app()
    page = WindowDebugPage(FakeWindowDebugService())
    report = {
        "diagnosis_mode": "saved_live_listener_frame",
        "recognition_summary": {
            "game_phase": "进行中",
            "opening_status": "WAIT",
            "round_level": "7",
            "wild_rank": "7",
            "lead_player": "opposite",
            "current_player": "self",
            "lead_confidence": 0.86,
        },
        "recognition": {
            "result": {
                "my_hand": ["3S", "7H", "10D"],
                "events": [
                    {
                        "player": "opposite",
                        "cards": ["10C", "JC"],
                        "is_pass": False,
                        "confidence": 0.91,
                        "source": "roi:opposite_play",
                    },
                    {
                        "player": "left",
                        "cards": [],
                        "is_pass": True,
                        "confidence": 0.88,
                        "source": "template:passed_left",
                    },
                ],
                "field_confidences": {"my_hand": 0.72},
                "sources": {"my_hand": "template:cards"},
                "unresolved_fields": ["my_hand"],
            },
        },
        "field_assessments": [
            {
                "field": "my_hand",
                "label": "我的手牌",
                "value": "3S 7H 10D",
                "status": "below_threshold",
                "confidence": 0.72,
                "threshold": 0.80,
                "source": "template:cards",
                "reason": "最佳候选低于阈值",
            }
        ],
        "threshold_analysis": [
            {"field": "my_hand", "score": 0.72, "threshold": 0.80, "reason": "低于阈值"}
        ],
        "user_view": {"可复制诊断摘要": "牌局进行中；手牌识别不完整"},
    }

    page._apply_report(report, source_label="实时监听帧（listener）")
    text = page.detailed_view.toPlainText()

    assert "来源：实时监听帧（listener）" in text
    assert "级牌：7" in text
    assert "百变牌：7" in text
    assert "手牌数量：3 张" in text
    assert "3S 7H 10D" in text
    assert "对家（opposite）" in text
    assert "动作：不出" in text
    assert "roi:opposite_play" in text
    assert "低于阈值" in text
    assert "0.800" in text
    assert "未解决字段" in text
    page.close()


def test_switching_session_frame_clears_old_detailed_report(tmp_path):
    _app()
    first = tmp_path / "000001.png"
    second = tmp_path / "000002.png"
    image = QImage(16, 10, QImage.Format.Format_RGB32)
    image.fill(0xFF202020)
    assert image.save(str(first))
    assert image.save(str(second))

    page = WindowDebugPage(FakeWindowDebugService())
    page._session_frames = [
        {"image_path": first, "metadata_path": first.with_suffix(".json"), "sequence": 1},
        {"image_path": second, "metadata_path": second.with_suffix(".json"), "sequence": 2},
    ]
    page.frame_selector.addItem("第1张", userData=0)
    page.frame_selector.addItem("第2张", userData=1)
    page._apply_report({"recognition": {"result": {"round_level": "7"}}}, source_label="listener")
    assert page._last_report is not None

    page._session_frame_changed(1)

    assert page._last_report is None
    assert "尚未识别" in page.detailed_view.toPlainText()
    assert "7" not in page.detailed_view.toPlainText()
    page.close()


def test_old_report_is_rendered_without_guessing_missing_fields():
    _app()
    page = WindowDebugPage(FakeWindowDebugService())
    page._apply_report(
        {"recognition": {"result": {"my_hand": ["3S"], "round_level": "2"}}},
        source_label="保存报告截图（manual capture）",
    )
    text = page.detailed_view.toPlainText()

    assert "来源：保存报告截图（manual capture）" in text
    assert "级牌：2" in text
    assert "手牌数量：1 张" in text
    assert "目标数量：尚未提供" in text
    assert "首出玩家：未识别/尚未提供" in text
    assert "各家出牌/不出事件" in text
    assert "未识别/尚未提供" in text
    page.close()


def test_key_blockers_ignore_action_only_roi_overlap_warning():
    blockers = _key_blockers({
        "roi_validation": {
            "status": "fail",
            "issues": [{
                "code": "roi.critical_play_overlap",
                "severity": "warning",
                "region": "right_play",
                "related_region": "my_play",
            }],
        },
    })

    assert blockers == []
