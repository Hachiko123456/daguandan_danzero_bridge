from __future__ import annotations

import base64
import json

import cv2
from types import SimpleNamespace

import numpy as np

from daguandan_bridge.application.session_diagnostic_frames import SessionDiagnosticFrameStore
from daguandan_bridge.application.window_debug_report import WindowDebugReportService
from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.image_io import StandardizationResult
from daguandan_bridge.window_capture import CapturedStandardizedFrame
from daguandan_bridge.application.window_debug_storage import WindowDebugStorage
from daguandan_bridge.profiles import ProfileConfig


class FakeWindowApi:
    def EnumWindows(self, callback, extra): callback(77, extra)
    def IsWindow(self, hwnd): return hwnd == 77
    def IsWindowVisible(self, hwnd): return True
    def IsIconic(self, hwnd): return False
    def GetWindowText(self, hwnd): return "腾讯欢乐掼蛋"
    def GetClassName(self, hwnd): return "WeChatAppEx"
    def GetWindowRect(self, hwnd): return (0, 0, 320, 180)
    def GetClientRect(self, hwnd): return (0, 0, 320, 180)
    def ClientToScreen(self, hwnd, point): return point
    def GetWindowThreadProcessId(self, hwnd): return (1, 123)
    def GetDpiForWindow(self, hwnd): return 120


class FakeRecognizer:
    def validate_configuration(self, image):
        return {"status": "pass", "image_size": [image.shape[1], image.shape[0]], "issues": []}

    def recognize(self, _image, *, allow_unknown_suit=False):
        return SimpleNamespace(
            round_level="2", my_hand=("3S", "4H"), lead_player=None,
            current_player=None, buttons=(), events=(), unresolved_fields=(),
        )

    def recognize_page_anchor_scores(self, _image):
        return {"table_anchor_1_score": 0.91, "table_anchor_2_score": 0.2, "game_logo_anchor_score": 0.1}

    def recognize_opening_signal(self, _image):
        return SimpleNamespace(super_double_visible=False, marker_player=None, active_player=None, self_action_buttons_visible=False)

    def get_last_diagnostic_trace(self):
        return {"schema": "test-trace/1"}


def test_report_includes_capture_roi_recognition_and_readiness_inputs(tmp_path):
    service = WindowDebugReportService(
        window_api=FakeWindowApi(),
        process_name_resolver={123: "WeChat.exe"},
        capture_function=lambda _target, rect: np.full((rect.height, rect.width, 3), 20, dtype=np.uint8),
        profile_config=ProfileConfig(
            name="test", display_name="test", window_title_keywords=("test",),
            base_size=(320, 180), detect_black_bars=False, allow_resize=False,
        ),
        recognizer=FakeRecognizer(),
    )

    report = service.build(hwnd=77, capture=True, recognize=True)
    assert report["detailed_diagnostic"]["summary"]["round_level"] == "2"
    assert report["detailed_diagnostic"]["summary"]["hand_count"] == 2
    assert report["detailed_diagnostic"]["evidence_only"] is True
    output = service.write(report, tmp_path / "window-debug.json")
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted["read_only"] is True
    assert all(value is False for value in persisted["control_policy"].values())
    assert persisted["window"]["hwnd"] == 77
    assert persisted["window"]["portable_identity"]["hwnd_is_identity"] is False
    assert persisted["capture"]["backend"] == "printwindow"
    assert persisted["capture"]["standardization"]["standardized_size"] == [320, 180]
    assert persisted["roi_validation"]["status"] == "pass"
    assert persisted["recognition"]["trace"]["schema"] == "test-trace/1"
    assert persisted["opening_readiness_inputs"]["readiness"]["status"] == "WAIT"


def test_list_mode_returns_visible_windows_only():
    report = WindowDebugReportService(
        window_api=FakeWindowApi(), process_name_resolver={123: "WeChat.exe"}
    ).build(list_windows=True)
    assert len(report["windows"]) == 1
    assert report["windows"][0]["pid"] == 123
    assert report["user_view"]["状态"] == "等待处理"
    assert report["user_view"]["可复制诊断摘要"]


def test_default_write_uses_managed_storage_without_absolute_path_in_report(tmp_path):
    storage = WindowDebugStorage(tmp_path / "diagnostics")
    service = WindowDebugReportService(
        window_api=FakeWindowApi(),
        process_name_resolver={123: "WeChat.exe"},
        storage=storage,
    )
    report = service.build(list_windows=True)
    output = service.write(report)

    assert output.is_relative_to(storage.root)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["storage_run_id"] == output.parent.name
    assert persisted["storage_ref"] == f"window_debug/{output.parent.name}/report.json"


def test_explicit_export_does_not_create_managed_run(tmp_path):
    storage = WindowDebugStorage(tmp_path / "diagnostics")
    service = WindowDebugReportService(
        window_api=FakeWindowApi(),
        process_name_resolver={123: "WeChat.exe"},
        storage=storage,
    )
    output = tmp_path / "export" / "report.json"
    service.write(service.build(list_windows=True), output)

    assert output.is_file()
    assert not storage.root.exists()



def _write_saved_frame_report(tmp_path, *, media_item=True, content=None, recognition=None):
    image = np.full((180, 320, 3), 20, dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    source = {
        "schema": "guandan.window-debug-report/v1",
        "profile_name": "test",
        "window": {"title": "现场窗口", "hwnd": 77},
        "recognition": {"result": recognition or {"round_level": "old", "my_hand": ["AS"]}},
        "opening_readiness_inputs": {
            "opening_signal": {"marker_player": "left"},
            "readiness": {"status": "WAIT", "primary_reason": "OPENING_UNRESOLVED"},
        },
        "media": {"media_saved": bool(media_item), "items": {}},
    }
    if media_item:
        source["media"]["items"]["current_frame"] = {
            "encoding": "base64",
            "content": base64.b64encode(encoded if content is None else content).decode("ascii"),
        }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


def _replay_service():
    return WindowDebugReportService(
        profile_config=ProfileConfig(
            name="test", display_name="test", window_title_keywords=("test",),
            base_size=(320, 180), detect_black_bars=False, allow_resize=False,
        ),
        recognizer=FakeRecognizer(),
    )


def test_build_from_report_file_restores_current_frame_and_reruns_recognition(tmp_path):
    report = _replay_service().build_from_report_file(_write_saved_frame_report(tmp_path))

    assert report["diagnosis_mode"] == "saved_snapshot_re_diagnosis"
    assert report["live_window_diagnosis"] is False
    assert report["captured_snapshot"]["status"] == "PRESENT"
    assert report["re_diagnosis"]["status"] == "PASS"
    assert report["re_diagnosis"]["recognition"]["result"]["round_level"] == "2"
    assert report["source_report"]["window"]["title"] == "现场窗口"
    assert report["source_report"]["first_play_evidence"]["opening_signal"]["marker_player"] == "left"
    json.dumps(report, ensure_ascii=False)


def test_build_from_report_file_returns_structured_media_not_present(tmp_path):
    report = _replay_service().build_from_report_file(
        _write_saved_frame_report(tmp_path, media_item=False)
    )

    assert report["captured_snapshot"]["status"] == "MEDIA_NOT_PRESENT"
    assert report["re_diagnosis"]["status"] == "MEDIA_NOT_PRESENT"
    assert report["re_diagnosis"]["recognition"] is None
    assert report["user_view"]["实时窗口诊断"] is False


def test_build_from_report_file_returns_structured_error_for_corrupt_base64(tmp_path):
    path = _write_saved_frame_report(tmp_path, content=b"not-an-image")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["media"]["items"]["current_frame"]["content"] = "%%%broken-base64%%%"
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = _replay_service().build_from_report_file(path)

    assert report["captured_snapshot"]["status"] == "MEDIA_INVALID"
    assert report["captured_snapshot"]["error_code"] == "MEDIA_BASE64_INVALID"
    assert report["re_diagnosis"]["recognition"] is None
    json.dumps(report)


def test_build_from_report_file_reports_recognition_difference(tmp_path):
    report = _replay_service().build_from_report_file(
        _write_saved_frame_report(
            tmp_path,
            recognition={"round_level": "1", "my_hand": ["AS"], "lead_player": "right"},
        )
    )

    assert report["comparison"]["status"] == "CHANGED"
    assert report["comparison"]["recognition_changed"] is True
    fields = {item["field"] for item in report["comparison"]["recognition"]["differences"]}
    assert "round_level" in fields
    assert "lead_player" in fields


def test_build_from_report_file_re_diagnoses_embedded_frame(tmp_path):
    import cv2
    import numpy as np

    storage_report = {
        "schema": "guandan.window-debug-report/v1",
        "storage_run_id": "source-1",
        "recognition": {"result": {"my_hand": ["3S"]}},
        "opening_readiness_inputs": {"readiness": {"status": "WAIT", "primary_reason": "OPENING_UNRESOLVED"}},
        "media": {"media_saved": True, "items": {}},
    }
    ok, encoded = cv2.imencode(".png", np.full((180, 320, 3), 20, dtype=np.uint8))
    assert ok
    storage_report["media"]["items"]["current_frame"] = {
        "encoding": "base64", "content": base64.b64encode(encoded.tobytes()).decode("ascii")
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(storage_report), encoding="utf-8")

    service = WindowDebugReportService(
        profile_config=ProfileConfig(name="test", display_name="test", window_title_keywords=("test",), base_size=(320, 180), detect_black_bars=False, allow_resize=False),
        recognizer=FakeRecognizer(),
    )
    report = service.build_from_report_file(source)

    assert report["diagnostic_mode"] == "saved_screenshot"
    assert report["capture"]["backend"] == "saved_report"
    assert report["source_report"]["storage_run_id"] == "source-1"
    assert "comparison" in report


def test_build_from_report_file_reports_missing_media(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"schema": "guandan.window-debug-report/v1", "media": {"media_saved": False, "items": {}}}), encoding="utf-8")
    service = WindowDebugReportService(profile_config=ProfileConfig(name="test", display_name="test", window_title_keywords=("test",), base_size=(320, 180), detect_black_bars=False, allow_resize=False), recognizer=FakeRecognizer())
    report = service.build_from_report_file(source)
    assert report["errors"][0]["code"] == "MEDIA_NOT_PRESENT"



def _saved_listener_frame(tmp_path):
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    standardization = StandardizationResult(
        image=image,
        source_size=(8, 6),
        source_viewport=SimpleNamespace(to_list=lambda: [0, 0, 8, 6]),
        content_box=SimpleNamespace(to_list=lambda: [0, 0, 8, 6]),
        scale=1.0,
        padding=(0, 0, 0, 0),
        aspect_error=0.0,
        aspect_compatible=True,
    )
    snapshot = FrameSnapshot(
        frame=CapturedStandardizedFrame(
            standardization=standardization,
            rect=SimpleNamespace(left=1, top=2, width=8, height=6),
            backend="printwindow",
            dpi=120,
            window_title="牌桌",
            raw_image=image.copy(),
        ),
        captured_monotonic_ms=987,
        evidence_frame_id="frame-1",
    )
    return SessionDiagnosticFrameStore().save_snapshot(
        tmp_path / "session",
        snapshot,
        session_id="session-1",
        capture_generation=4,
        capture_seq=55,
    )


def _replay_service_with_capture(capture_function):
    return WindowDebugReportService(
        profile_config=ProfileConfig(
            name="test", display_name="test", window_title_keywords=("test",),
            base_size=(320, 180), detect_black_bars=False, allow_resize=False,
        ),
        recognizer=FakeRecognizer(),
        capture_function=capture_function,
    )


def test_build_from_listener_frame_uses_saved_pixels_without_capture(tmp_path):
    record = _saved_listener_frame(tmp_path)
    called = False

    def capture(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("saved listener frame diagnosis must not capture a window")

    report = _replay_service_with_capture(capture).build_from_listener_frame(record.image_path)

    assert called is False
    assert report["diagnosis_mode"] == "saved_live_listener_frame"
    assert report["live_window_diagnosis"] is False
    assert report["source_metadata"]["session_id"] == "session-1"
    assert report["source_frame"]["sequence"] == 1
    assert report["capture"]["standardization"]["applied"] is False
    assert report["recognition"]["result"]["round_level"] == "2"
    assert report["detailed_diagnostic"]["summary"]["round_level"] == "2"
    assert report["detailed_diagnostic"]["summary"]["hand_count"] == 2
    assert report["user_view"]["实时窗口诊断"] is False


def test_build_from_listener_frame_recognize_false_still_validates_roi(tmp_path):
    record = _saved_listener_frame(tmp_path)
    report = _replay_service_with_capture(lambda *_args, **_kwargs: None).build_from_listener_frame(
        record.image_path,
        metadata_path=record.metadata_path,
        recognize=False,
    )

    assert report["recognition_status"] == "NOT_REQUESTED"
    assert report["roi_validation"]["status"] == "pass"
    assert "recognition" not in report
