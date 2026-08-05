from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PROFILES_ROOT
from .image_io import save_capture_with_metadata, save_image_unicode
from .profiles import ProfileConfig, ProfilePaths, get_profile_paths, load_profile_config
from .storage import atomic_write_json
from .window_capture import (
    CapturedStandardizedFrame,
    LazyMssCapture,
    capture_standardized_client_frame,
    find_target_window,
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


@dataclass
class ScreenshotSession:
    profile_name: str
    directory: Path
    started_at: datetime
    interval_ms: int
    frame_count: int = 0
    finished_at: datetime | None = None
    first_capture: dict[str, Any] | None = None

    @property
    def metadata_path(self) -> Path:
        return self.directory / "session.json"

    def next_frame_path(self) -> Path:
        return self.directory / f"{self.frame_count + 1:06d}.png"


class CaptureService:
    """Qt-free service for Tencent DaGuandan window capture and recording."""

    def __init__(self, profiles_root: Path = PROFILES_ROOT):
        self.profiles_root = Path(profiles_root)

    def load_profile(self, name: str) -> LoadedProfile:
        paths = get_profile_paths(self.profiles_root, name)
        return LoadedProfile(paths, load_profile_config(paths))

    def recording_interval_seconds(self, profile_name: str) -> float:
        return float(self.load_profile(profile_name).config.recording_interval_sec)

    def update_recording_interval(self, profile_name: str, interval_sec: float) -> float:
        loaded = self.load_profile(profile_name)
        config = replace(
            loaded.config,
            recording_interval_sec=float(interval_sec),
        ).normalized()
        atomic_write_json(loaded.paths.profile_config_path, config.to_json_dict())
        return config.recording_interval_sec

    def capture_frame(self, profile_name: str) -> FrameSnapshot:
        loaded = self.load_profile(profile_name)
        target = find_target_window(loaded.config.window_title_keywords)
        with LazyMssCapture() as screen_capture:
            frame = capture_standardized_client_frame(
                target,
                screen_capture,
                loaded.config,
            )
        return FrameSnapshot(frame)

    @staticmethod
    def _capture_document(
        snapshot: FrameSnapshot,
        config: ProfileConfig,
    ) -> dict[str, Any]:
        result = snapshot.frame.standardization
        frame = snapshot.frame
        return {
            "captured_at": snapshot.captured_at.isoformat(),
            "source_size": list(result.source_size),
            "source_viewport": result.source_viewport.to_list(),
            "content_box": result.content_box.to_list(),
            "scale": result.scale,
            "padding": list(result.padding),
            "aspect_error": result.aspect_error,
            "aspect_compatible": result.aspect_compatible,
            "standardized_size": [result.image.shape[1], result.image.shape[0]],
            "profile_schema_version": config.schema_version,
            "window_title": frame.window_title,
            "client_rect": {
                "left": frame.rect.left,
                "top": frame.rect.top,
                "width": frame.rect.width,
                "height": frame.rect.height,
            },
            "window_dpi": frame.dpi,
            "capture_backend": frame.backend,
        }

    @staticmethod
    def _session_document(session: ScreenshotSession) -> dict[str, Any]:
        return {
            "format_version": 1,
            "profile": session.profile_name,
            "started_at": session.started_at.isoformat(),
            "finished_at": session.finished_at.isoformat() if session.finished_at else None,
            "capture_interval_ms": session.interval_ms,
            "frame_count": session.frame_count,
            "first_capture": session.first_capture,
        }

    def start_screenshot_session(
        self,
        profile_name: str,
        interval_ms: int,
    ) -> ScreenshotSession:
        if interval_ms <= 0:
            raise ValueError("录制间隔必须大于 0 毫秒")
        loaded = self.load_profile(profile_name)
        loaded.paths.ensure_dirs()
        started_at = datetime.now().astimezone()
        stem = f"game_{started_at.strftime('%Y%m%d_%H%M%S')}"
        directory = loaded.paths.screenshots_dir / stem
        suffix = 1
        while directory.exists():
            suffix += 1
            directory = loaded.paths.screenshots_dir / f"{stem}_{suffix:02d}"
        directory.mkdir(parents=True)
        session = ScreenshotSession(
            profile_name=profile_name,
            directory=directory.resolve(),
            started_at=started_at,
            interval_ms=int(interval_ms),
        )
        atomic_write_json(session.metadata_path, self._session_document(session))
        return session

    def save_session_frame(
        self,
        session: ScreenshotSession,
        snapshot: FrameSnapshot,
    ) -> Path:
        loaded = self.load_profile(session.profile_name)
        root = loaded.paths.screenshots_dir.resolve()
        try:
            session.directory.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError("录制目录必须位于当前 profile 的 screenshots 目录") from exc
        if session.finished_at is not None:
            raise RuntimeError("本局录制已经结束")
        output = session.next_frame_path()
        save_image_unicode(output, snapshot.image)
        session.frame_count += 1
        if session.first_capture is None:
            session.first_capture = self._capture_document(snapshot, loaded.config)
        atomic_write_json(session.metadata_path, self._session_document(session))
        return output

    def finish_screenshot_session(self, session: ScreenshotSession) -> ScreenshotSession:
        if session.finished_at is None:
            session.finished_at = datetime.now().astimezone()
            atomic_write_json(session.metadata_path, self._session_document(session))
        return session

    def save_frame(self, profile_name: str, snapshot: FrameSnapshot) -> Path:
        loaded = self.load_profile(profile_name)
        loaded.paths.ensure_dirs()
        stamp = snapshot.captured_at.strftime("screenshot_%Y%m%d_%H%M%S")
        output = loaded.paths.screenshots_dir / f"{stamp}.png"
        suffix = 1
        while output.exists():
            suffix += 1
            output = loaded.paths.screenshots_dir / f"{stamp}_{suffix:02d}.png"
        save_capture_with_metadata(
            output,
            snapshot.frame.standardization,
            self._capture_document(snapshot, loaded.config),
        )
        return output

    def screenshot_folder(self, profile_name: str) -> Path:
        loaded = self.load_profile(profile_name)
        loaded.paths.ensure_dirs()
        return loaded.paths.screenshots_dir
