"""Read-only Win32 window diagnostics used by support tooling.

The public helpers in this module deliberately stop at observation.  They do
not activate, restore, move, resize, click, or otherwise control a window.
An HWND is a process/session-local observation key; portable report identity is
computed from semantic window attributes and never from the HWND.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .image_io import standardize_to_base
from .models import ClientRect, TargetWindow
from .window_capture import (
    TargetWindowError,
    capture_client_image_printwindow,
)

SCHEMA = "guandan.window-debug/v1"
WINDOW_IDENTITY_SCHEMA = "guandan.window-identity/v1"


class WindowDebugError(RuntimeError):
    """A read-only window inspection or PrintWindow failure."""

    def __init__(self, message: str, *, code: str = "WINDOW_DEBUG_FAILED") -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class DebugRect:
    """A JSON-friendly rectangle in screen pixels."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def to_dict(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
            "right": int(self.right),
            "bottom": int(self.bottom),
        }


@dataclass(frozen=True)
class WindowInfo:
    """The complete observation returned for one top-level window."""

    hwnd: int
    pid: int | None
    process_name: str | None
    title: str
    class_name: str
    visible: bool
    iconic: bool
    outer_rect: DebugRect | None
    client_rect: DebugRect | None
    dpi: int | None

    def to_dict(self, *, include_hwnd: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "pid": self.pid,
            "process_name": self.process_name,
            "title": self.title,
            "class_name": self.class_name,
            "visible": bool(self.visible),
            "iconic": bool(self.iconic),
            "outer_rect": self.outer_rect.to_dict() if self.outer_rect else None,
            "client_rect": self.client_rect.to_dict() if self.client_rect else None,
            "dpi": self.dpi,
        }
        # HWND is useful for the live probe, but it is intentionally omitted
        # from portable identity documents.
        if include_hwnd:
            value["hwnd"] = int(self.hwnd)
        return value


@dataclass(frozen=True)
class CapturedWindowFrame:
    """One in-memory PrintWindow result and the geometry used to get it."""

    image: Any
    window: WindowInfo
    backend: str = "printwindow"

    def to_dict(self) -> dict[str, object]:
        shape = getattr(self.image, "shape", None)
        return {
            "backend": self.backend,
            "size": [int(shape[1]), int(shape[0])] if shape is not None and len(shape) >= 2 else None,
            "channels": int(shape[2]) if shape is not None and len(shape) >= 3 else 1 if shape is not None else None,
            "window_client_rect": (
                self.window.client_rect.to_dict() if self.window.client_rect else None
            ),
        }


class WindowApi(Protocol):
    """Minimal pywin32-like API, also convenient for unit-test doubles."""

    def EnumWindows(self, callback: Callable[[int, object], bool], extra: object) -> object: ...


ProcessNameResolver = Callable[[int], str | None]
CaptureFunction = Callable[[TargetWindow, ClientRect], Any]


def _load_win32gui() -> Any:
    if sys.platform != "win32":
        raise WindowDebugError(
            "窗口调试后端只能在 Windows 上运行；测试请注入 mock window_api",
            code="WINDOW_DEBUG_UNSUPPORTED",
        )
    try:
        import win32gui  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - platform/dependency boundary
        raise WindowDebugError("缺少 pywin32 的 win32gui 模块", code="WINDOW_DEBUG_DEPENDENCY") from exc
    return win32gui


def _load_win32process() -> Any | None:
    if sys.platform != "win32":
        return None
    try:
        import win32process  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - platform/dependency boundary
        return None
    return win32process


def _native_process_name(pid: int) -> str | None:
    """Resolve only the executable basename, never persist a machine path."""

    if sys.platform != "win32" or pid <= 0:
        return None
    try:  # pragma: no cover - exercised only on a real Windows desktop
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        process = kernel32.OpenProcess(0x1000 | 0x0010, False, int(pid))
        if not process:
            return None
        try:
            buffer = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_uint32(len(buffer))
            query = getattr(kernel32, "QueryFullProcessImageNameW", None)
            if query is not None and query(process, 0, buffer, ctypes.byref(size)):
                return os.path.basename(buffer.value) or None
            return None
        finally:
            kernel32.CloseHandle(process)
    except Exception:
        return None


def _mapping_or_call(value: object, key: int) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    if callable(value):
        return value(key)
    return None


def _rect_from_tuple(value: object, *, coordinate_mode: str = "xyxy") -> DebugRect | None:
    if value is None:
        return None
    try:
        values = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(values) != 4:
        return None
    if coordinate_mode == "xywh":
        left, top, width, height = values
    else:
        left, top, right, bottom = values
        width, height = right - left, bottom - top
    if width < 0 or height < 0:
        return None
    return DebugRect(left, top, width, height)


def _get_window_rect(api: Any, hwnd: int) -> DebugRect | None:
    try:
        return _rect_from_tuple(api.GetWindowRect(hwnd))
    except Exception:
        return None


def _get_client_rect(api: Any, hwnd: int, outer: DebugRect | None = None) -> DebugRect | None:
    try:
        raw = api.GetClientRect(hwnd)
        local = _rect_from_tuple(raw)
        if local is None:
            return None
        if hasattr(api, "ClientToScreen"):
            left, top = api.ClientToScreen(hwnd, (local.left, local.top))
        elif outer is not None:
            left, top = outer.left + local.left, outer.top + local.top
        else:
            left, top = local.left, local.top
        return DebugRect(int(left), int(top), local.width, local.height)
    except Exception:
        return None


def _get_pid(api: Any, hwnd: int, process_api: Any | None = None) -> int | None:
    source = process_api or api
    getter = getattr(source, "GetWindowThreadProcessId", None)
    if not callable(getter):
        return None
    try:
        raw = getter(hwnd)
        if isinstance(raw, (tuple, list)):
            return int(raw[-1])
        return int(raw)
    except (TypeError, ValueError, IndexError, OSError):
        return None


def _get_dpi(api: Any, hwnd: int, dpi_getter: Callable[[int], int] | None = None) -> int | None:
    try:
        if dpi_getter is not None:
            value = int(dpi_getter(hwnd))
        elif callable(getattr(api, "GetDpiForWindow", None)):
            value = int(api.GetDpiForWindow(hwnd))
        elif sys.platform == "win32":  # pragma: no cover - real Windows boundary
            value = int(ctypes.windll.user32.GetDpiForWindow(hwnd))  # type: ignore[attr-defined]
        else:
            value = 96
        return value if value > 0 else 96
    except Exception:
        return 96 if sys.platform == "win32" else None


def _get_process_name(
    pid: int | None,
    resolver: ProcessNameResolver | Mapping[int, str | None] | None,
    process_api: Any | None,
) -> str | None:
    if pid is None:
        return None
    supplied = _mapping_or_call(resolver, pid) if resolver is not None else None
    if supplied is not None:
        return str(supplied) if str(supplied).strip() else None
    if process_api is not None:
        getter = getattr(process_api, "GetModuleFileNameEx", None)
        opener = getattr(process_api, "OpenProcess", None)
        if callable(getter) and callable(opener):
            try:
                handle = opener(0x0410, False, pid)
                value = str(getter(handle, 0))
                close = getattr(process_api, "CloseHandle", None)
                if callable(close):
                    close(handle)
                return os.path.basename(value) or None
            except Exception:
                pass
    return _native_process_name(pid)


def inspect_window(
    hwnd: int,
    *,
    window_api: Any | None = None,
    process_api: Any | None = None,
    process_name_resolver: ProcessNameResolver | Mapping[int, str | None] | None = None,
    dpi_getter: Callable[[int], int] | None = None,
) -> WindowInfo:
    """Read one HWND without changing any window state."""

    try:
        handle = int(hwnd)
    except (TypeError, ValueError) as exc:
        raise WindowDebugError("HWND 必须是整数", code="WINDOW_INVALID_HWND") from exc
    if handle <= 0:
        raise WindowDebugError("HWND 必须是正整数", code="WINDOW_INVALID_HWND")
    api = window_api or _load_win32gui()
    resolved_process_api = process_api
    if resolved_process_api is None and not callable(getattr(api, "GetWindowThreadProcessId", None)):
        resolved_process_api = _load_win32process()
    is_window = getattr(api, "IsWindow", None)
    if callable(is_window):
        try:
            if not bool(is_window(handle)):
                raise WindowDebugError("HWND 不是有效窗口", code="WINDOW_NOT_FOUND")
        except WindowDebugError:
            raise
        except Exception as exc:
            raise WindowDebugError(f"无法验证 HWND：{exc}", code="WINDOW_PROBE_FAILED") from exc

    def safe_bool(name: str, default: bool = False) -> bool:
        method = getattr(api, name, None)
        if not callable(method):
            return default
        try:
            return bool(method(handle))
        except Exception:
            return default

    def safe_text(name: str) -> str:
        method = getattr(api, name, None)
        if not callable(method):
            return ""
        try:
            return str(method(handle) or "").strip()
        except Exception:
            return ""

    outer = _get_window_rect(api, handle)
    return WindowInfo(
        hwnd=handle,
        pid=_get_pid(api, handle, resolved_process_api),
        process_name=_get_process_name(
            _get_pid(api, handle, resolved_process_api),
            process_name_resolver,
            resolved_process_api,
        ),
        title=safe_text("GetWindowText"),
        class_name=safe_text("GetClassName"),
        visible=safe_bool("IsWindowVisible"),
        iconic=safe_bool("IsIconic"),
        outer_rect=outer,
        client_rect=_get_client_rect(api, handle, outer),
        dpi=_get_dpi(api, handle, dpi_getter),
    )


def enumerate_visible_windows(
    *,
    window_api: Any | None = None,
    process_api: Any | None = None,
    process_name_resolver: ProcessNameResolver | Mapping[int, str | None] | None = None,
    dpi_getter: Callable[[int], int] | None = None,
) -> tuple[WindowInfo, ...]:
    """Enumerate visible top-level windows in the OS-provided Z-order."""

    api = window_api or _load_win32gui()
    resolved_process_api = process_api
    if resolved_process_api is None and not callable(getattr(api, "GetWindowThreadProcessId", None)):
        resolved_process_api = _load_win32process()
    enum = getattr(api, "EnumWindows", None)
    if not callable(enum):
        raise WindowDebugError("window_api 缺少 EnumWindows", code="WINDOW_DEBUG_DEPENDENCY")
    result: list[WindowInfo] = []

    def callback(raw_hwnd: int, _extra: object) -> bool:
        try:
            hwnd = int(raw_hwnd)
            visible = bool(api.IsWindowVisible(hwnd)) if callable(getattr(api, "IsWindowVisible", None)) else True
            if not visible:
                return True
            result.append(
                inspect_window(
                    hwnd,
                    window_api=api,
                    process_api=resolved_process_api,
                    process_name_resolver=process_name_resolver,
                    dpi_getter=dpi_getter,
                )
            )
        except WindowDebugError:
            # A window can disappear between EnumWindows and inspection.  It
            # is safer to omit that transient item than to invent metadata.
            return True
        return True

    try:
        enum(callback, None)
    except Exception as exc:
        raise WindowDebugError(f"枚举顶层窗口失败：{exc}", code="WINDOW_ENUM_FAILED") from exc
    return tuple(result)


def portable_window_identity(
    window: WindowInfo,
    *,
    application_id: str | None = None,
    title_role: str | None = None,
) -> dict[str, object]:
    """Build an identity safe to persist across machines.

    The basis intentionally excludes HWND, PID, screen coordinates and full
    process paths.  Those values describe a current observation only.
    """

    client_size = (
        [window.client_rect.width, window.client_rect.height]
        if window.client_rect is not None
        else None
    )
    basis = {
        "application_id": str(application_id or window.process_name or "").strip().lower(),
        "process_name": str(window.process_name or "").strip().lower(),
        "window_class": str(window.class_name or "").strip(),
        "title_role": str(title_role or window.title or "").strip().lower(),
        "client_size": client_size,
    }
    encoded = json.dumps(basis, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "schema": WINDOW_IDENTITY_SCHEMA,
        "key": f"window:{digest}",
        "basis": basis,
        "hwnd_is_identity": False,
    }


def capture_printwindow_frame(
    window: WindowInfo | int,
    *,
    window_api: Any | None = None,
    process_api: Any | None = None,
    process_name_resolver: ProcessNameResolver | Mapping[int, str | None] | None = None,
    dpi_getter: Callable[[int], int] | None = None,
    capture_function: CaptureFunction | None = None,
) -> CapturedWindowFrame:
    """Capture exactly one frame through PrintWindow, without fallback/control."""

    info = (
        window
        if isinstance(window, WindowInfo)
        else inspect_window(
            window,
            window_api=window_api,
            process_api=process_api,
            process_name_resolver=process_name_resolver,
            dpi_getter=dpi_getter,
        )
    )
    if not info.visible:
        raise WindowDebugError("目标窗口不可见，拒绝 PrintWindow 单帧测试", code="WINDOW_NOT_VISIBLE")
    if info.iconic:
        raise WindowDebugError("目标窗口已最小化，拒绝自动还原后截图", code="WINDOW_MINIMIZED")
    if info.client_rect is None or info.client_rect.width <= 0 or info.client_rect.height <= 0:
        raise WindowDebugError("目标窗口客户区尺寸无效", code="WINDOW_GEOMETRY_INVALID")
    target = TargetWindow(hwnd=info.hwnd, title=info.title)
    rect = ClientRect(
        left=info.client_rect.left,
        top=info.client_rect.top,
        width=info.client_rect.width,
        height=info.client_rect.height,
    )
    capture = capture_function or capture_client_image_printwindow
    try:
        image = capture(target, rect)
    except TargetWindowError as exc:
        raise WindowDebugError(str(exc), code=getattr(exc, "code", "PRINTWINDOW_FAILED")) from exc
    except Exception as exc:
        raise WindowDebugError(f"PrintWindow 单帧测试失败：{exc}", code="PRINTWINDOW_FAILED") from exc
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2 or tuple(shape[:2]) != (rect.height, rect.width):
        raise WindowDebugError("PrintWindow 返回尺寸与客户区不一致", code="CAPTURE_GEOMETRY_INVALID")
    if getattr(image, "size", 0) == 0:
        raise WindowDebugError("PrintWindow 返回空图像", code="CAPTURE_EMPTY")
    return CapturedWindowFrame(image=image, window=info)


def standardize_frame(
    image: Any,
    *,
    base_size: Sequence[int] = (1280, 720),
    aspect_ratio_tolerance: float = 0.03,
    detect_black_bars: bool = True,
    viewport_mode: str = "full",
    viewport_aspect_ratio: float = 16 / 9,
) -> tuple[Any, dict[str, object]]:
    """Standardize one captured image and expose all coordinate metadata."""

    base = (int(base_size[0]), int(base_size[1]))
    result = standardize_to_base(
        image,
        base,
        aspect_tolerance=float(aspect_ratio_tolerance),
        detect_black_bars=bool(detect_black_bars),
        viewport_mode=str(viewport_mode),
        viewport_aspect_ratio=float(viewport_aspect_ratio),
    )
    shape = getattr(result.image, "shape", ())
    metadata: dict[str, object] = {
        "source_size": list(result.source_size),
        "source_viewport": result.source_viewport.to_list(),
        "content_box": result.content_box.to_list(),
        "scale": result.scale,
        "padding": list(result.padding),
        "aspect_error": result.aspect_error,
        "aspect_compatible": result.aspect_compatible,
        "standardized_size": [int(shape[1]), int(shape[0])] if len(shape) >= 2 else None,
        "base_size": list(base),
    }
    return result.image, metadata


def json_safe(value: object) -> object:
    """Convert diagnostic values, dataclasses and NumPy scalars to JSON data."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int, float)):
        return getattr(value, "value")
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {str(key): json_safe(item) for key, item in vars(value).items() if not str(key).startswith("_")}
    return str(value)


__all__ = [
    "SCHEMA",
    "WINDOW_IDENTITY_SCHEMA",
    "WindowDebugError",
    "DebugRect",
    "WindowInfo",
    "CapturedWindowFrame",
    "WindowApi",
    "enumerate_visible_windows",
    "inspect_window",
    "portable_window_identity",
    "capture_printwindow_frame",
    "standardize_frame",
    "json_safe",
]
