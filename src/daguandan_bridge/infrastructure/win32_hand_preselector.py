"""Win32-only adapter for selecting cards in the target game's hand.

The public surface intentionally has one operation.  It accepts only a plan
whose points were derived from hand annotations; there is no generic click or
button-click method in this adapter.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from time import sleep
from typing import Any

from ..gui.hand_preselection import PreselectionPlan, PreselectionResult
from ..window_capture import find_target_window, get_client_rect_on_screen


_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_VIRTUALDESK = 0x4000
_INTER_CARD_DELAY_SECONDS = 0.04


def _move_flags() -> int:
    """Move in absolute coordinates relative to the full virtual desktop."""

    return _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK


class Win32HandPreselector:
    """Inject left-clicks only after revalidating the captured client window."""

    def __init__(self, window_title_keywords: tuple[str, ...]) -> None:
        self._window_title_keywords = tuple(str(item) for item in window_title_keywords)

    def preselect_hand_cards(self, plan: PreselectionPlan) -> PreselectionResult:
        if sys.platform != "win32":
            return self._reject(plan, "自动预选仅支持 Windows")
        if not plan.points:
            return self._reject(plan, "预选计划没有手牌坐标")
        try:
            from ..dependencies import import_required

            win32gui = import_required("win32gui", "pywin32")
            target = find_target_window(self._window_title_keywords)
            if not win32gui.IsWindow(target.hwnd):
                return self._reject(plan, "目标窗口句柄已失效")
            if win32gui.IsIconic(target.hwnd):
                return self._reject(plan, "目标窗口已最小化")
            if int(win32gui.GetForegroundWindow()) != int(target.hwnd):
                return self._reject(plan, "目标游戏窗口不在前台")
            current_rect = get_client_rect_on_screen(target)
        except Exception as exc:
            return self._reject(plan, f"无法验证目标游戏窗口：{exc}")

        if current_rect != plan.expected_client_rect:
            return self._reject(plan, "游戏窗口位置或尺寸已变化")
        if any(not _is_inside_client(point, current_rect) for point in plan.points):
            return self._reject(plan, "预选坐标超出游戏客户区")
        try:
            _send_left_clicks(plan.points)
        except Exception as exc:
            return PreselectionResult(
                request_id=plan.request_id,
                status="failed",
                detail=f"自动预选失败：{exc}",
            )
        return PreselectionResult(
            request_id=plan.request_id,
            status="preselected",
            detail="推荐手牌已预选，请手动点击出牌",
        )

    @staticmethod
    def _reject(plan: PreselectionPlan, detail: str) -> PreselectionResult:
        return PreselectionResult(
            request_id=plan.request_id,
            status="rejected",
            detail=str(detail),
        )


def _is_inside_client(point: tuple[int, int], rect: Any) -> bool:
    x, y = (int(value) for value in point)
    return rect.left <= x < rect.left + rect.width and rect.top <= y < rect.top + rect.height


def _send_left_clicks(
    points: tuple[tuple[int, int], ...],
    *,
    user32: Any | None = None,
    wait: Callable[[float], None] = sleep,
) -> None:
    """Send each validated hand click as a separate ``SendInput`` batch.

    WebView/miniprogram game clients update the raised-card state after a
    message turn.  Sending all clicks in one bulk batch can make each later
    click land against the original DOM state, so a successful click is given
    one short processing interval before the next card.  A failed send stops
    immediately; this adapter never contains a submit-button path.
    """

    user32 = user32 or ctypes.windll.user32  # type: ignore[attr-defined]
    virtual_left = int(user32.GetSystemMetrics(76))
    virtual_top = int(user32.GetSystemMetrics(77))
    virtual_width = int(user32.GetSystemMetrics(78))
    virtual_height = int(user32.GetSystemMetrics(79))
    if virtual_width <= 0 or virtual_height <= 0:
        raise RuntimeError("虚拟桌面尺寸无效")

    class _MouseInput(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.c_void_p),
        ]

    class _InputUnion(ctypes.Union):
        _fields_ = [("mi", _MouseInput)]

    class _Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("union", _InputUnion)]

    for index, point in enumerate(points):
        x, y = (int(value) for value in point)
        absolute_x = round((x - virtual_left) * 65535 / max(1, virtual_width - 1))
        absolute_y = round((y - virtual_top) * 65535 / max(1, virtual_height - 1))
        inputs = (
            _Input(0, _InputUnion(_MouseInput(absolute_x, absolute_y, 0, _move_flags(), 0, None))),
            _Input(0, _InputUnion(_MouseInput(0, 0, 0, 0x0002, 0, None))),
            _Input(0, _InputUnion(_MouseInput(0, 0, 0, 0x0004, 0, None))),
        )
        buffer = (_Input * len(inputs))(*inputs)
        sent = int(user32.SendInput(len(buffer), ctypes.byref(buffer), ctypes.sizeof(_Input)))
        if sent != len(buffer):
            raise RuntimeError(f"SendInput 仅发送了 {sent}/{len(buffer)} 个鼠标事件")
        if index + 1 < len(points):
            wait(_INTER_CARD_DELAY_SECONDS)
