import numpy as np
import pytest

from daguandan_bridge.capture_service import (
    CaptureService,
    FrameSnapshot,
    LiveCaptureInterrupted,
)
from daguandan_bridge.image_io import standardize_to_base
from daguandan_bridge.models import ClientRect
from daguandan_bridge.models import TargetWindow
from daguandan_bridge.profiles import ProfileConfig, create_profile
from daguandan_bridge.window_capture import CapturedStandardizedFrame


def _frame_snapshot() -> FrameSnapshot:
    image = np.full((720, 1280, 3), 127, dtype=np.uint8)
    standardization = standardize_to_base(image, (1280, 720))
    frame = CapturedStandardizedFrame(
        standardization=standardization,
        rect=ClientRect(10, 20, 1280, 720),
        backend="test",
        dpi=96,
        window_title="Test Window",
    )
    return FrameSnapshot(frame)


def test_live_source_reuses_window_lookup_and_capture_backend(tmp_path, monkeypatch):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",)),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    calls = {"find": 0, "capture": 0, "close": 0}

    def find(_keywords):
        calls["find"] += 1
        return target

    class FakeCapture:
        def close(self):
            calls["close"] += 1

    monkeypatch.setattr("daguandan_bridge.capture_service.find_target_window", find)
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.get_client_rect_on_screen",
        lambda _target: ClientRect(10, 20, 1280, 720),
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.LazyMssCapture", FakeCapture
    )

    def capture(_target, screen_capture, _config):
        assert isinstance(screen_capture, FakeCapture)
        calls["capture"] += 1
        return _frame_snapshot().frame

    monkeypatch.setattr(
        "daguandan_bridge.capture_service.capture_standardized_client_frame",
        capture,
    )

    source = service.open_live_source("test_game")
    source.capture()
    source.capture()
    source.close()

    assert source.window_lookup_count == 1
    assert calls == {"find": 1, "capture": 2, "close": 1}


def test_live_source_reports_geometry_change_as_interruption(tmp_path, monkeypatch):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",)),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    rectangles = iter(
        (
            ClientRect(10, 20, 1280, 720),
            ClientRect(10, 20, 1200, 700),
        )
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_target_window", lambda _keywords: target
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.get_client_rect_on_screen",
        lambda _target: next(rectangles),
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.capture_standardized_client_frame",
        lambda *_args: _frame_snapshot().frame,
    )

    source = service.open_live_source("test_game")
    with pytest.raises(LiveCaptureInterrupted, match="geometry"):
        source.capture()
    source.close()


def test_visible_screen_capture_rejects_occluded_target_before_frame_is_returned(
    tmp_path,
    monkeypatch,
):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",)),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    frame = _frame_snapshot().frame
    frame = CapturedStandardizedFrame(
        standardization=frame.standardization,
        rect=frame.rect,
        backend="screen",
        dpi=frame.dpi,
        window_title=frame.window_title,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_target_window",
        lambda _keywords: target,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.get_client_rect_on_screen",
        lambda _target: ClientRect(10, 20, 1280, 720),
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.capture_standardized_client_frame",
        lambda *_args: frame,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_screen_occluders",
        lambda *_args: (TargetWindow(hwnd=456, title="DanZero 推荐"),),
    )

    source = service.open_live_source("test_game")
    with pytest.raises(LiveCaptureInterrupted, match="DanZero 推荐"):
        source.capture()
    source.close()


def test_background_window_capture_ignores_screen_occluders(tmp_path, monkeypatch):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",)),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    frame = _frame_snapshot().frame
    frame = CapturedStandardizedFrame(
        standardization=frame.standardization,
        rect=frame.rect,
        backend="printwindow",
        dpi=frame.dpi,
        window_title=frame.window_title,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_target_window",
        lambda _keywords: target,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.get_client_rect_on_screen",
        lambda _target: ClientRect(10, 20, 1280, 720),
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.capture_standardized_client_frame",
        lambda *_args: frame,
    )
    checked = []
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_screen_occluders",
        lambda *_args: checked.append(True),
    )

    source = service.open_live_source("test_game")
    assert source.capture().frame.backend == "printwindow"
    assert checked == []
    source.close()
