from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
)


@dataclass(frozen=True)
class FrameSnapshot:
    frame: CapturedStandardizedFrame
    captured_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    @property
    def image(self) -> Any:
        return self.frame.image


@dataclass(frozen=True)
class LoadedProfile:
    paths: ProfilePaths
    config: ProfileConfig


class LiveCaptureInterrupted(RuntimeError):
    """A persistent source can no longer guarantee aligned window frames."""


class LiveCaptureSource:
    def __init__(self, loaded: LoadedProfile) -> None:
        self.loaded = loaded
        self.target = find_target_window(loaded.config.window_title_keywords)
        self.window_lookup_count = 1
        self._initial_rect = get_client_rect_on_screen(self.target)
        self._screen_capture = LazyMssCapture()
        self._closed = False

    def capture(self) -> FrameSnapshot:
        if self._closed:
            raise RuntimeError("持续采集源已经关闭")
        try:
            current_rect = get_client_rect_on_screen(self.target)
            if current_rect != self._initial_rect:
                raise LiveCaptureInterrupted(
                    "target window geometry changed; reopen live capture source"
                )
            frame = capture_standardized_client_frame(
                self.target,
                self._screen_capture,
                self.loaded.config,
            )
            if backend_uses_visible_screen(frame.backend):
                blockers = find_screen_occluders(self.target, current_rect)
                if blockers:
                    names = "、".join(
                        dict.fromkeys(blocker.title for blocker in blockers)
                    )
                    raise LiveCaptureInterrupted(
                        "屏幕采集已暂停：目标牌桌被其他窗口遮挡"
                        f"（{names}）。请把推荐浮窗和完整助手移到牌桌客户区外，"
                        "再点击继续；被遮挡帧不会进入识别或 DanZero。"
                    )
        except LiveCaptureInterrupted:
            raise
        except TargetWindowError as exc:
            raise LiveCaptureInterrupted(str(exc)) from exc
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

    def target_client_rect(self, profile_name: str):
        """Locate the target for companion-window placement without capturing."""

        loaded = self.load_profile(profile_name)
        target = find_target_window(loaded.config.window_title_keywords)
        return get_client_rect_on_screen(target)
