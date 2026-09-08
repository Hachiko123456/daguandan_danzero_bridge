from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic_ns
from typing import Any, Mapping
from uuid import uuid4

from .config import PROFILES_ROOT
from .profiles import ProfileConfig, ProfilePaths, get_profile_paths, load_profile_config
from .window_capture import (
    CapturedStandardizedFrame,
    LazyMssCapture,
    TargetWindowError,
    backend_uses_visible_screen,
    capture_standardized_client_frame,
    find_screen_occluders,
    find_target_window,
    get_client_rect_on_screen,
    get_window_dpi,
    resize_target_client,
)


@dataclass(frozen=True)
class FrameSnapshot:
    frame: CapturedStandardizedFrame
    captured_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    captured_monotonic_ms: int = field(
        default_factory=lambda: monotonic_ns() // 1_000_000
    )
    evidence_frame_id: str = field(default_factory=lambda: uuid4().hex)

    @property
    def image(self) -> Any:
        return self.frame.image


@dataclass(frozen=True)
class LoadedProfile:
    paths: ProfilePaths
    config: ProfileConfig


class LiveCaptureInterrupted(RuntimeError):
    """A persistent source can no longer guarantee aligned window frames."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CAPTURE-BACKEND-FAILED",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def _rect_payload(rect: object | None) -> list[int] | None:
    if rect is None:
        return None
    try:
        return [
            int(getattr(rect, "left")),
            int(getattr(rect, "top")),
            int(getattr(rect, "width")),
            int(getattr(rect, "height")),
        ]
    except (TypeError, ValueError):
        return None


def _geometry_change_types(
    old_rect: object,
    new_rect: object,
    *,
    old_dpi: int,
    new_dpi: int,
) -> list[str]:
    changes: list[str] = []
    if (
        getattr(old_rect, "left", None),
        getattr(old_rect, "top", None),
    ) != (
        getattr(new_rect, "left", None),
        getattr(new_rect, "top", None),
    ):
        changes.append("move")
    if (
        getattr(old_rect, "width", None),
        getattr(old_rect, "height", None),
    ) != (
        getattr(new_rect, "width", None),
        getattr(new_rect, "height", None),
    ):
        changes.append("resize")
    if int(old_dpi) != int(new_dpi):
        changes.append("dpi")
    return changes or ["geometry"]


class LiveCaptureSource:
    def __init__(self, loaded: LoadedProfile) -> None:
        self.loaded = loaded
        self.target = find_target_window(loaded.config.window_title_keywords)
        self.window_lookup_count = 1
        self._initial_rect = get_client_rect_on_screen(self.target)
        self._initial_dpi = get_window_dpi(self.target)
        self._screen_capture = LazyMssCapture()
        self._last_backend: str | None = None
        self._closed = False

    def diagnostic_state(self) -> dict[str, object]:
        return {
            "rect": _rect_payload(self._initial_rect),
            "dpi": int(self._initial_dpi),
            "backend": self._last_backend,
            "window_title": self.target.title,
            "hwnd": int(self.target.hwnd),
        }

    def capture(self) -> FrameSnapshot:
        if self._closed:
            raise RuntimeError("持续采集源已经关闭")
        try:
            current_rect = get_client_rect_on_screen(self.target)
            current_dpi = get_window_dpi(self.target)
            if current_rect != self._initial_rect or current_dpi != self._initial_dpi:
                change_types = _geometry_change_types(
                    self._initial_rect,
                    current_rect,
                    old_dpi=self._initial_dpi,
                    new_dpi=current_dpi,
                )
                raise LiveCaptureInterrupted(
                    "target window geometry changed; reopen live capture source",
                    code="GEOMETRY-CHANGED",
                    details={
                        "old_rect": _rect_payload(self._initial_rect),
                        "new_rect": _rect_payload(current_rect),
                        "old_dpi": int(self._initial_dpi),
                        "new_dpi": int(current_dpi),
                        "change_types": change_types,
                        "capture_backend": self._last_backend,
                        "window_title": self.target.title,
                        "hwnd": int(self.target.hwnd),
                    },
                )
            frame = capture_standardized_client_frame(
                self.target,
                self._screen_capture,
                self.loaded.config,
            )
            self._last_backend = str(frame.backend)
            if backend_uses_visible_screen(frame.backend):
                blockers = find_screen_occluders(self.target, current_rect)
                if blockers:
                    names = "、".join(
                        dict.fromkeys(blocker.title for blocker in blockers)
                    )
                    raise LiveCaptureInterrupted(
                        "屏幕采集已暂停：目标牌桌被其他窗口遮挡"
                        f"（{names}）。请把推荐浮窗和完整助手移到牌桌客户区外，"
                        "再点击继续；被遮挡帧不会进入识别或策略计算。",
                        code="CAPTURE-OCCLUDED",
                    )
        except LiveCaptureInterrupted:
            raise
        except TargetWindowError as exc:
            source_code = str(getattr(exc, "code", "CAPTURE-BACKEND-FAILED"))
            change_types = (
                ["minimized"]
                if source_code == "WINDOW-MINIMIZED"
                else ["geometry"]
                if source_code == "GEOMETRY-CHANGED"
                else []
            )
            raise LiveCaptureInterrupted(
                str(exc),
                code=source_code,
                details={
                    "old_rect": _rect_payload(self._initial_rect),
                    "new_rect": None,
                    "old_dpi": int(self._initial_dpi),
                    "new_dpi": None,
                    "change_types": change_types,
                    "capture_backend": self._last_backend,
                    "window_title": self.target.title,
                    "hwnd": int(self.target.hwnd),
                },
            ) from exc
        return FrameSnapshot(frame)

    def close(self) -> None:
        if not self._closed:
            self._screen_capture.close()
            self._closed = True

    def __enter__(self) -> "LiveCaptureSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class CaptureService:
    """Qt-free source of standardized frames for the real-time assistant."""

    def __init__(self, profiles_root: Path = PROFILES_ROOT):
        self.profiles_root = Path(profiles_root)

    def load_profile(self, name: str) -> LoadedProfile:
        paths = get_profile_paths(self.profiles_root, name)
        return LoadedProfile(paths, load_profile_config(paths))

    def open_live_source(self, profile_name: str) -> LiveCaptureSource:
        return LiveCaptureSource(self.load_profile(profile_name))

    def capture_frame(self, profile_name: str) -> FrameSnapshot:
        """Capture one frame through the same validated persistent-source adapter."""

        with self.open_live_source(profile_name) as source:
            return source.capture()

    def target_client_rect(self, profile_name: str):
        """Locate the target for companion-window placement without capturing."""

        loaded = self.load_profile(profile_name)
        target = find_target_window(loaded.config.window_title_keywords)
        return get_client_rect_on_screen(target)

    def lock_target_client_size(self, profile_name: str):
        """Make the target client match this profile's canonical base size."""

        loaded = self.load_profile(profile_name)
        target = find_target_window(loaded.config.window_title_keywords)
        target_size = loaded.config.target_client_size or loaded.config.base_size
        return resize_target_client(target, target_size)
