from __future__ import annotations

import numpy as np

from daguandan_bridge.window_debug import (
    capture_printwindow_frame,
    enumerate_visible_windows,
    inspect_window,
    portable_window_identity,
)


class FakeWindowApi:
    def __init__(self) -> None:
        self.windows = {
            101: {"visible": True, "iconic": False, "title": "牌桌", "class": "WeChatAppEx", "outer": (10, 20, 410, 320), "client": (0, 0, 380, 270), "origin": (20, 40), "pid": 7, "dpi": 144},
            202: {"visible": False, "iconic": False, "title": "隐藏", "class": "Hidden", "outer": (0, 0, 10, 10), "client": (0, 0, 10, 10), "origin": (0, 0), "pid": 8, "dpi": 96},
        }

    def EnumWindows(self, callback, extra):
        for hwnd in self.windows:
            callback(hwnd, extra)

    def IsWindow(self, hwnd): return hwnd in self.windows
    def IsWindowVisible(self, hwnd): return self.windows[hwnd]["visible"]
    def IsIconic(self, hwnd): return self.windows[hwnd]["iconic"]
    def GetWindowText(self, hwnd): return self.windows[hwnd]["title"]
    def GetClassName(self, hwnd): return self.windows[hwnd]["class"]
    def GetWindowRect(self, hwnd): return self.windows[hwnd]["outer"]
    def GetClientRect(self, hwnd): return self.windows[hwnd]["client"]
    def ClientToScreen(self, hwnd, _point): return self.windows[hwnd]["origin"]
    def GetWindowThreadProcessId(self, hwnd): return (1, self.windows[hwnd]["pid"])
    def GetDpiForWindow(self, hwnd): return self.windows[hwnd]["dpi"]


def test_enumerates_visible_top_level_windows_with_complete_probe_fields():
    api = FakeWindowApi()
    windows = enumerate_visible_windows(window_api=api, process_name_resolver={7: "WeChat.exe"})

    assert [item.hwnd for item in windows] == [101]
    item = windows[0]
    assert item.pid == 7
    assert item.process_name == "WeChat.exe"
    assert item.title == "牌桌"
    assert item.class_name == "WeChatAppEx"
    assert item.visible is True and item.iconic is False
    assert item.outer_rect.to_dict()["width"] == 400
    assert item.client_rect.to_dict()["width"] == 380
    assert item.dpi == 144


def test_probe_and_capture_are_read_only_and_identity_does_not_use_hwnd():
    api = FakeWindowApi()
    first = inspect_window(101, window_api=api, process_name_resolver={7: "WeChat.exe"})
    second = first.__class__(999, *first.__dict__.values().__iter__().__next__()) if False else first
    identity = portable_window_identity(first)

    frame = capture_printwindow_frame(
        first,
        window_api=api,
        capture_function=lambda _target, rect: np.full((rect.height, rect.width, 3), 9, dtype=np.uint8),
    )

    assert frame.backend == "printwindow"
    assert frame.image.shape == (270, 380, 3)
    assert "101" not in identity["key"]
    assert "hwnd" not in identity["basis"]
    assert identity["hwnd_is_identity"] is False
