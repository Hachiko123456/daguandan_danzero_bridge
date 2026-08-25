from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass
from typing import Any

from .dependencies import import_required
from .image_io import StandardizationResult, standardize_to_base
from .models import ClientRect, TargetWindow
from .profiles import ProfileConfig


class TargetWindowError(RuntimeError):
    """无法定位或截取目标窗口时抛出的错误。"""


_SW_RESTORE = 9
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_NOOWNERZORDER = 0x0200


@dataclass(frozen=True)
class CapturedClientImage:
    image: Any
    rect: ClientRect
    backend: str
    dpi: int


@dataclass(frozen=True)
class CapturedStandardizedFrame:
    standardization: StandardizationResult
    rect: ClientRect
    backend: str
    dpi: int
    window_title: str

    @property
    def image(self) -> Any:
        return self.standardization.image


class LazyMssCapture:
    """仅在 PrintWindow 回退真正发生时创建 MSS 会话。"""

    def __init__(self) -> None:
        self._session: Any = None

    def __enter__(self) -> "LazyMssCapture":
        return self

    def grab(self, monitor: dict[str, int]) -> Any:
        if self._session is None:
            mss_module = import_required("mss", "mss")
            self._session = mss_module.MSS()
        return self._session.grab(monitor)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


def backend_uses_visible_screen(backend: str) -> bool:
    """Return whether a backend reads pixels currently visible on screen."""
    return str(backend).strip().lower() in {"screen", "gdi_screen"}


def find_target_window(title_keywords: tuple[str, ...]) -> TargetWindow:
    """按标题关键字查找目标窗口，优先精确匹配，再做包含匹配。"""
    if sys.platform != "win32":
        raise TargetWindowError("窗口截图功能只能在 Windows 上运行")

    win32gui = import_required("win32gui", "pywin32")
    candidates: list[TargetWindow] = []

    def enum_callback(hwnd: int, _: Any) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd).strip()
        if title and any(keyword in title for keyword in title_keywords):
            candidates.append(TargetWindow(hwnd=hwnd, title=title))
        return True

    win32gui.EnumWindows(enum_callback, None)
    exact_candidates = [
        candidate
        for candidate in candidates
        if candidate.title in title_keywords
    ]
    preferred = exact_candidates or candidates
    unique_candidates = {candidate.hwnd: candidate for candidate in preferred}
    if len(unique_candidates) == 1:
        return next(iter(unique_candidates.values()))
    if len(unique_candidates) > 1:
        titles = "；".join(
            candidate.title for candidate in unique_candidates.values()
        )
        raise TargetWindowError(
            f"找到多个标题匹配的窗口：{titles}。请收窄 profile 的窗口关键字"
        )

    keywords = "、".join(title_keywords)
    raise TargetWindowError(f"没有找到标题包含 {keywords} 的可见窗口")


def get_client_rect_on_screen(target: TargetWindow) -> ClientRect:
    """获取目标窗口客户区在屏幕上的真实像素坐标。"""
    win32gui = import_required("win32gui", "pywin32")

    if not win32gui.IsWindow(target.hwnd):
        raise TargetWindowError("目标窗口句柄已经失效，请重新进入采集模式")
    if win32gui.IsIconic(target.hwnd):
        raise TargetWindowError("目标窗口处于最小化状态，无法可靠截图")

    left, top, right, bottom = win32gui.GetClientRect(target.hwnd)
    width = int(right - left)
    height = int(bottom - top)
    if width <= 0 or height <= 0:
        raise TargetWindowError("目标窗口客户区尺寸无效")

    screen_left, screen_top = win32gui.ClientToScreen(target.hwnd, (left, top))
    return ClientRect(
        left=int(screen_left),
        top=int(screen_top),
        width=width,
        height=height,
    )


def resize_target_client(
    target: TargetWindow,
    client_size: tuple[int, int],
    *,
    window_api: Any | None = None,
    client_rect_getter: Any | None = None,
) -> ClientRect:
    """Resize a target's client area and verify the resulting pixel size.

    The outer window frame varies with DPI and window style, so derive its
    current insets instead of assuming fixed caption or border dimensions.
    """

    width, height = (int(client_size[0]), int(client_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("目标客户区尺寸必须为正整数")
    if sys.platform != "win32":
        raise TargetWindowError("窗口尺寸锁定只能在 Windows 上运行")

    win32gui = window_api or import_required("win32gui", "pywin32")
    if not win32gui.IsWindow(target.hwnd):
        raise TargetWindowError("目标窗口句柄已经失效，无法锁定尺寸")

    show_window = getattr(win32gui, "ShowWindow", None)
    is_iconic = bool(win32gui.IsIconic(target.hwnd))
    is_zoomed = bool(getattr(win32gui, "IsZoomed", lambda _hwnd: False)(target.hwnd))
    if (is_iconic or is_zoomed) and callable(show_window):
        show_window(target.hwnd, _SW_RESTORE)

    get_client = client_rect_getter or get_client_rect_on_screen
    current = get_client(target)
    outer_left, outer_top, outer_right, outer_bottom = (
        int(value) for value in win32gui.GetWindowRect(target.hwnd)
    )
    left_inset = current.left - outer_left
    top_inset = current.top - outer_top
    right_inset = outer_right - (current.left + current.width)
    bottom_inset = outer_bottom - (current.top + current.height)
    if min(left_inset, top_inset, right_inset, bottom_inset) < 0:
        raise TargetWindowError("无法计算目标窗口边框，拒绝调整尺寸")

    outer_width = width + left_inset + right_inset
    outer_height = height + top_inset + bottom_inset
    try:
        win32gui.SetWindowPos(
            target.hwnd,
            0,
            outer_left,
            outer_top,
            outer_width,
            outer_height,
            _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_NOOWNERZORDER,
        )
    except Exception as exc:
        raise TargetWindowError(f"调整目标窗口尺寸失败：{exc}") from exc

    resized = get_client(target)
    if (resized.width, resized.height) != (width, height):
        raise TargetWindowError(
            "目标窗口未接受请求尺寸："
            f"期望 {width}x{height}，实际 {resized.width}x{resized.height}"
        )
    return resized


def get_window_dpi(target: TargetWindow) -> int:
    """读取窗口当前所在显示器的 DPI；旧 API 不可用时返回 96。"""
    try:
        dpi = int(ctypes.windll.user32.GetDpiForWindow(target.hwnd))  # type: ignore[attr-defined]
        return dpi if dpi > 0 else 96
    except Exception:
        return 96


def _validate_captured_image(image: Any, rect: ClientRect, backend: str) -> Any:
    if image is None or not hasattr(image, "shape"):
        raise TargetWindowError(f"{backend} 没有返回图像")
    if len(image.shape) < 2 or tuple(image.shape[:2]) != (rect.height, rect.width):
        raise TargetWindowError(
            f"{backend} 返回尺寸无效：期望 {rect.width}x{rect.height}"
        )
    if image.size == 0:
        raise TargetWindowError(f"{backend} 返回空图像")
    if backend == "printwindow" and float(image.max()) <= 0:
        raise TargetWindowError("PrintWindow 返回全黑图像")
    return image


def _rectangles_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return (
        max(first[0], second[0]) < min(first[2], second[2])
        and max(first[1], second[1]) < min(first[3], second[3])
    )


def find_screen_occluders(
    target: TargetWindow,
    rect: ClientRect,
    *,
    window_api: Any | None = None,
) -> tuple[TargetWindow, ...]:
    """List visible top-level windows above the target that overlap its client."""
    win32gui = window_api or import_required("win32gui", "pywin32")
    target_rect = (
        rect.left,
        rect.top,
        rect.left + rect.width,
        rect.top + rect.height,
    )
    blockers: list[TargetWindow] = []
    inspection_errors: list[str] = []
    target_seen = False

    def enum_callback(hwnd: int, _: Any) -> bool:
        nonlocal target_seen
        if int(hwnd) == target.hwnd:
            target_seen = True
            return False
        try:
            if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
                return True
            window_rect = tuple(
                int(value) for value in win32gui.GetWindowRect(hwnd)
            )
            if not _rectangles_intersect(window_rect, target_rect):
                return True
            title = str(win32gui.GetWindowText(hwnd)).strip()
            if not title:
                title = f"[{win32gui.GetClassName(hwnd)}]"
            blockers.append(TargetWindow(hwnd=int(hwnd), title=title))
        except Exception as exc:
            inspection_errors.append(f"hwnd={int(hwnd)}: {exc}")
            return True
        return True

    try:
        win32gui.EnumWindows(enum_callback, None)
    except Exception as exc:
        raise TargetWindowError(f"无法检查窗口遮挡状态：{exc}") from exc
    if inspection_errors:
        raise TargetWindowError(
            "无法可靠检查窗口遮挡状态："
            + "; ".join(inspection_errors[:3])
        )
    if not target_seen:
        raise TargetWindowError(
            "无法在顶层窗口 Z 序中确认目标牌桌，已拒绝截图"
        )
    return tuple(blockers)


def capture_client_image_printwindow(
    target: TargetWindow,
    rect: ClientRect,
) -> Any:
    """通过 PrintWindow 捕获客户区，避免屏幕遮挡进入结果。"""
    if sys.platform != "win32":
        raise TargetWindowError("PrintWindow 只能在 Windows 上运行")
    win32gui = import_required("win32gui", "pywin32")
    win32ui = import_required("win32ui", "pywin32")
    np = import_required("numpy", "numpy")

    window_dc_handle = 0
    source_dc = None
    memory_dc = None
    bitmap = None
    try:
        window_dc_handle = int(win32gui.GetWindowDC(target.hwnd))
        if not window_dc_handle:
            raise TargetWindowError("无法获取目标窗口 DC")
        source_dc = win32ui.CreateDCFromHandle(window_dc_handle)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, rect.width, rect.height)
        memory_dc.SelectObject(bitmap)
        flags = 0x00000001 | 0x00000002  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
        ok = bool(
            ctypes.windll.user32.PrintWindow(  # type: ignore[attr-defined]
                target.hwnd,
                memory_dc.GetSafeHdc(),
                flags,
            )
        )
        if not ok:
            raise TargetWindowError("PrintWindow 调用失败")
        raw = bitmap.GetBitmapBits(True)
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
            (rect.height, rect.width, 4)
        )
        image = bgra[:, :, :3].copy()
        return _validate_captured_image(image, rect, "printwindow")
    except TargetWindowError:
        raise
    except Exception as exc:
        raise TargetWindowError(f"PrintWindow 捕获失败：{exc}") from exc
    finally:
        if bitmap is not None:
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
        if memory_dc is not None:
            try:
                memory_dc.DeleteDC()
            except Exception:
                pass
        if source_dc is not None:
            try:
                source_dc.DeleteDC()
            except Exception:
                pass
        if window_dc_handle:
            try:
                win32gui.ReleaseDC(target.hwnd, window_dc_handle)
            except Exception:
                pass


def capture_client_image_screen(
    target: TargetWindow,
    screen_capture: Any,
    rect: ClientRect,
) -> Any:
    """使用 MSS 捕获屏幕客户区；该后端可能受窗口遮挡影响。"""
    if screen_capture is None:
        raise TargetWindowError("screen 捕获后端缺少 MSS 会话")
    cv2 = import_required("cv2", "opencv-python")
    np = import_required("numpy", "numpy")
    try:
        raw = np.array(screen_capture.grab(rect.to_mss_monitor()))
        image = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    except Exception as exc:
        raise TargetWindowError(f"屏幕捕获失败：{exc}") from exc
    return _validate_captured_image(image, rect, "screen")


def capture_client_image_gdi(
    target: TargetWindow,
    rect: ClientRect,
) -> Any:
    """Capture the target client area's currently visible pixels with Win32 GDI."""
    if sys.platform != "win32":
        raise TargetWindowError("gdi_screen is only available on Windows")
    win32gui = import_required("win32gui", "pywin32")
    win32ui = import_required("win32ui", "pywin32")
    np = import_required("numpy", "numpy")

    desktop_dc_handle = 0
    source_dc = None
    memory_dc = None
    bitmap = None
    try:
        desktop_dc_handle = int(win32gui.GetDC(0))
        if not desktop_dc_handle:
            raise TargetWindowError("无法获取桌面 DC")
        source_dc = win32ui.CreateDCFromHandle(desktop_dc_handle)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, rect.width, rect.height)
        memory_dc.SelectObject(bitmap)
        # SRCCOPY | CAPTUREBLT; CAPTUREBLT is not exported by every pywin32 build.
        memory_dc.BitBlt(
            (0, 0),
            (rect.width, rect.height),
            source_dc,
            (rect.left, rect.top),
            0x00CC0020 | 0x40000000,
        )
        raw = bitmap.GetBitmapBits(True)
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
            (rect.height, rect.width, 4)
        )
        image = bgra[:, :, :3].copy()
        return _validate_captured_image(image, rect, "gdi_screen")
    except TargetWindowError:
        raise
    except Exception as exc:
        raise TargetWindowError(f"GDI 屏幕采集失败：{exc}") from exc
    finally:
        if bitmap is not None:
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
        if memory_dc is not None:
            try:
                memory_dc.DeleteDC()
            except Exception:
                pass
        if source_dc is not None:
            try:
                source_dc.DeleteDC()
            except Exception:
                pass
        if desktop_dc_handle:
            try:
                win32gui.ReleaseDC(0, desktop_dc_handle)
            except Exception:
                pass


def capture_client_image(
    target: TargetWindow,
    screen_capture: Any,
    backend: str = "auto",
    allow_screen_fallback: bool = True,
) -> CapturedClientImage:
    """按 profile 选择客户区捕获后端，并明确记录实际后端。"""
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {
        "auto",
        "printwindow",
        "screen",
        "gdi_screen",
    }:
        raise TargetWindowError(
            "捕获后端只能是 auto、printwindow、screen 或 gdi_screen"
        )
    rect = get_client_rect_on_screen(target)
    dpi = get_window_dpi(target)

    if normalized_backend in {"auto", "printwindow"}:
        try:
            image = capture_client_image_printwindow(target, rect)
            return CapturedClientImage(image, rect, "printwindow", dpi)
        except TargetWindowError:
            if normalized_backend == "printwindow" or not allow_screen_fallback:
                raise

    if normalized_backend == "gdi_screen":
        image = capture_client_image_gdi(target, rect)
        return CapturedClientImage(image, rect, "gdi_screen", dpi)

    image = capture_client_image_screen(target, screen_capture, rect)
    return CapturedClientImage(image, rect, "screen", dpi)


def capture_standardized_client_image(
    target: TargetWindow,
    sct: Any,
    base_size: tuple[int, int],
) -> Any:
    """兼容旧调用：使用屏幕后端并等比标准化客户区。"""
    captured = capture_client_image(
        target,
        sct,
        backend="screen",
        allow_screen_fallback=True,
    )
    return standardize_to_base(
        captured.image,
        base_size,
        aspect_tolerance=1.0,
        detect_black_bars=False,
    ).image


def capture_standardized_client_frame(
    target: TargetWindow,
    screen_capture: Any,
    config: ProfileConfig,
) -> CapturedStandardizedFrame:
    """捕获目标客户区，并保留后端、DPI 和几何变换元数据。"""
    captured = capture_client_image(
        target,
        screen_capture,
        backend=config.capture_backend,
        allow_screen_fallback=config.allow_screen_fallback,
    )
    standardization = standardize_to_base(
        captured.image,
        config.base_size,
        aspect_tolerance=config.aspect_ratio_tolerance,
        detect_black_bars=config.detect_black_bars,
        viewport_mode=config.viewport_mode,
        viewport_aspect_ratio=config.viewport_aspect_ratio,
    )
    return CapturedStandardizedFrame(
        standardization=standardization,
        rect=captured.rect,
        backend=captured.backend,
        dpi=captured.dpi,
        window_title=target.title,
    )
