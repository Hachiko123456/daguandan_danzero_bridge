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
from daguandan_bridge.profiles import (
    ProfileConfig,
    create_profile,
    get_profile_paths,
    load_profile_config,
)
from daguandan_bridge.window_capture import (
    CapturedStandardizedFrame,
    TargetWindowError,
    resize_target_client,
)


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


def test_resize_target_client_preserves_client_origin_and_accounts_for_frame():
    target = TargetWindow(hwnd=123, title="Test Window")
    calls = []

    class WindowApi:
        @staticmethod
        def IsWindow(_hwnd):
            return True

        @staticmethod
        def IsIconic(_hwnd):
            return False

        @staticmethod
        def IsZoomed(_hwnd):
            return False

        @staticmethod
        def GetWindowRect(_hwnd):
            return (90, 160, 1010, 930)

        @staticmethod
        def SetWindowPos(*args):
            calls.append(args)

    rectangles = iter(
        (
            ClientRect(100, 200, 900, 700),
            ClientRect(100, 200, 1280, 720),
        )
    )

    resized = resize_target_client(
        target,
        (1280, 720),
        window_api=WindowApi(),
        client_rect_getter=lambda _target: next(rectangles),
    )

    assert resized == ClientRect(100, 200, 1280, 720)
    assert calls == [(123, 0, 90, 160, 1300, 790, 532)]


def test_resize_target_client_rejects_a_window_that_refuses_requested_size():
    target = TargetWindow(hwnd=123, title="Test Window")

    class WindowApi:
        @staticmethod
        def IsWindow(_hwnd):
            return True

        @staticmethod
        def IsIconic(_hwnd):
            return False

        @staticmethod
        def IsZoomed(_hwnd):
            return False

        @staticmethod
        def GetWindowRect(_hwnd):
            return (90, 160, 1010, 930)

        @staticmethod
        def SetWindowPos(*_args):
            pass

    with pytest.raises(TargetWindowError, match="未接受请求尺寸"):
        resize_target_client(
            target,
            (1280, 720),
            window_api=WindowApi(),
            client_rect_getter=lambda _target: ClientRect(100, 200, 900, 700),
        )


def test_capture_service_locks_the_client_to_its_profile_base_size(tmp_path, monkeypatch):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",), base_size=(1280, 720)),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    locked = []
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_target_window",
        lambda _keywords: target,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.resize_target_client",
        lambda passed_target, size: locked.append((passed_target, size))
        or ClientRect(10, 20, 1280, 720),
    )

    result = service.lock_target_client_size("test_game")

    assert result == ClientRect(10, 20, 1280, 720)
    assert locked == [(target, (1280, 720))]


def test_capture_service_uses_a_profile_specific_client_size_when_configured(
    tmp_path,
    monkeypatch,
):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig(
            "test_game",
            "Test",
            ("Test",),
            base_size=(1280, 720),
            target_client_size=(1280, 764),
        ),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    locked = []
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_target_window",
        lambda _keywords: target,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.resize_target_client",
        lambda passed_target, size: locked.append((passed_target, size))
        or ClientRect(10, 20, 1280, 764),
    )

    result = service.lock_target_client_size("test_game")

    assert result == ClientRect(10, 20, 1280, 764)
    assert locked == [(target, (1280, 764))]


def test_capture_service_does_not_resize_when_profile_disallows_resize(
    tmp_path,
    monkeypatch,
):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig(
            "test_game",
            "Test",
            ("Test",),
            base_size=(1280, 720),
            target_client_size=(1280, 764),
            allow_resize=False,
        ),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    actual_rect = ClientRect(10, 20, 1000, 600)
    resize_calls = []
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_target_window",
        lambda _keywords: target,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.get_client_rect_on_screen",
        lambda passed_target: actual_rect if passed_target == target else None,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.resize_target_client",
        lambda *_args: resize_calls.append(_args),
    )

    result = service.lock_target_client_size("test_game")

    assert result == actual_rect
    assert resize_calls == []


def test_profile_allow_resize_is_appended_after_existing_positional_fields():
    config = ProfileConfig(
        "test_game",
        "Test",
        ("Test",),
        (1280, 720),
        0.5,
        7,
        0.04,
        "screen",
        False,
        False,
        "bottom_aspect",
        4 / 3,
        (1280, 764),
        ProfileConfig.__dataclass_fields__["match_settings"].default_factory(),
        ProfileConfig.__dataclass_fields__["counter_settings"].default_factory(),
        "danzero",
    )

    assert config.allow_resize is True
    assert config.advisor_strategy == "danzero"


def test_profile_allow_resize_defaults_true_for_legacy_json(tmp_path):
    service = CaptureService(tmp_path / "profiles")
    paths = create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",), allow_resize=False),
    )
    data = paths.profile_config_path.read_text(encoding="utf-8")
    assert '"allow_resize"' in data
    assert load_profile_config(paths).allow_resize is False

    legacy_paths = get_profile_paths(service.profiles_root, "legacy_game")
    legacy_paths.ensure_dirs()
    legacy_paths.profile_config_path.write_text(
        '''{
  "name": "legacy_game",
  "display_name": "Legacy",
  "window_title_keywords": ["Legacy"],
  "schema_version": 2
}
''',
        encoding="utf-8",
    )

    assert load_profile_config(legacy_paths).allow_resize is True


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
    with pytest.raises(LiveCaptureInterrupted, match="geometry") as captured:
        source.capture()
    assert captured.value.details["old_rect"] == [10, 20, 1280, 720]
    assert captured.value.details["new_rect"] == [10, 20, 1200, 700]
    assert captured.value.details["change_types"] == ["resize"]
    source.close()


def test_live_source_reports_dpi_change_even_when_rect_is_unchanged(
    tmp_path,
    monkeypatch,
):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",)),
    )
    target = TargetWindow(hwnd=123, title="Test Window")
    dpi_values = iter((96, 144))
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.find_target_window",
        lambda _keywords: target,
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.get_client_rect_on_screen",
        lambda _target: ClientRect(10, 20, 1280, 720),
    )
    monkeypatch.setattr(
        "daguandan_bridge.capture_service.get_window_dpi",
        lambda _target: next(dpi_values),
    )

    source = service.open_live_source("test_game")
    with pytest.raises(LiveCaptureInterrupted) as captured:
        source.capture()

    assert captured.value.code == "GEOMETRY-CHANGED"
    assert captured.value.details["change_types"] == ["dpi"]
    assert captured.value.details["old_dpi"] == 96
    assert captured.value.details["new_dpi"] == 144
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
