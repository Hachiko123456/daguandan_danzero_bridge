from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DpiAwarenessStatus:
    success: bool
    awareness: str
    method: str


def get_windows_dpi_awareness() -> str:
    """查询当前进程 DPI 感知级别。"""
    if sys.platform != "win32":
        return "not_windows"
    awareness = ctypes.c_int(-1)
    try:
        result = ctypes.windll.shcore.GetProcessDpiAwareness(  # type: ignore[attr-defined]
            None,
            ctypes.byref(awareness),
        )
        if int(result) == 0:
            return {
                0: "unaware",
                1: "system",
                2: "per_monitor",
            }.get(awareness.value, "unknown")
    except Exception:
        pass
    return "unknown"


def enable_windows_dpi_awareness() -> DpiAwarenessStatus:
    """尽早启用 Per-Monitor V2，并验证实际进程状态。"""
    if sys.platform != "win32":
        return DpiAwarenessStatus(True, "not_windows", "not_windows")

    method = "existing"
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        function = user32.SetProcessDpiAwarenessContext
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_bool
        pointer_bits = ctypes.sizeof(ctypes.c_void_p) * 8
        per_monitor_v2 = ctypes.c_void_p((1 << pointer_bits) - 4)
        if bool(function(per_monitor_v2)):
            method = "SetProcessDpiAwarenessContext"
    except Exception:
        pass

    awareness = get_windows_dpi_awareness()
    if awareness == "per_monitor":
        return DpiAwarenessStatus(True, awareness, method)

    try:
        result = ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        if int(result) == 0:
            method = "SetProcessDpiAwareness"
    except Exception:
        pass

    awareness = get_windows_dpi_awareness()
    if awareness == "per_monitor":
        return DpiAwarenessStatus(True, awareness, method)

    try:
        if bool(ctypes.windll.user32.SetProcessDPIAware()):  # type: ignore[attr-defined]
            method = "SetProcessDPIAware"
    except Exception:
        pass
    awareness = get_windows_dpi_awareness()
    return DpiAwarenessStatus(
        success=awareness == "per_monitor",
        awareness=awareness,
        method=method,
    )
