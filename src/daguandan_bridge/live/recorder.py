from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np

from ..image_io import save_image_unicode
from ..storage import append_json_line


@dataclass(frozen=True)
class RecorderWarning:
    reason: str
    monotonic_ms: int
    details: str


@dataclass(frozen=True)
class RecordingResult:
    video_path: Path
    index_path: Path
    frame_count: int
    dropped_frames: int


@dataclass(frozen=True)
class IncidentMedia:
    clip_path: Path
    contact_sheet_path: Path
    trigger_frame_path: Path
    frame_count: int


class SessionRecorder:
    """Record standardized frames and their original capture timestamps."""

    def __init__(
        self,
        session_directory: Path,
        *,
        size: tuple[int, int],
        fps: float,
        buffer_seconds: float = 10.0,
        codec: str = "MJPG",
    ) -> None:
        self.session_directory = Path(session_directory)
        self.video_directory = self.session_directory / "video"
        self.video_directory.mkdir(parents=True, exist_ok=True)
        self.video_path = self.video_directory / "game.avi"
        self.index_path = self.video_directory / "frame_index.jsonl"
        self.index_path.touch(exist_ok=False)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps)
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("录像尺寸必须为正数")
        if self.fps <= 0:
            raise ValueError("录像帧率必须为正数")
        if len(codec) != 4:
            raise ValueError("视频编码 FourCC 必须是四个字符")
        self.codec = codec
        self._writer = self._open_writer(self.video_path)
        buffer_frames = max(1, int(math.ceil(self.fps * buffer_seconds)))
        self._buffer: deque[tuple[int, np.ndarray]] = deque(maxlen=buffer_frames)
        self._frame_count = 0
        self._dropped_frames = 0
        self._pending_drops = 0
        self._closed = False
        self._lock = RLock()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def _open_writer(self, path: Path) -> cv2.VideoWriter:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            self.size,
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法打开录像编码器：{self.codec} -> {path}")
        return writer

    def write_frame(
        self,
        frame: np.ndarray,
        captured_monotonic_ms: int,
        wall_time: str,
    ) -> RecorderWarning | None:
        with self._lock:
            self._ensure_open()
            expected_shape = (self.size[1], self.size[0], 3)
            if not isinstance(frame, np.ndarray) or frame.shape != expected_shape:
                return self._drop(
                    "frame_size_mismatch",
                    captured_monotonic_ms,
                    f"expected={expected_shape}, actual={getattr(frame, 'shape', None)}",
                )
            if frame.dtype != np.uint8:
                return self._drop(
                    "frame_dtype_mismatch",
                    captured_monotonic_ms,
                    f"expected=uint8, actual={frame.dtype}",
                )
            try:
                self._writer.write(frame)
            except cv2.error as exc:
                return self._drop("video_encode_failed", captured_monotonic_ms, str(exc))
            record = {
                "frame_index": self._frame_count,
                "monotonic_ms": int(captured_monotonic_ms),
                "wall_time": str(wall_time),
                "dropped_before": self._pending_drops,
            }
            append_json_line(self.index_path, record)
            self._buffer.append((int(captured_monotonic_ms), frame.copy()))
            self._frame_count += 1
            self._pending_drops = 0
            return None

    def _drop(self, reason: str, monotonic_ms: int, details: str) -> RecorderWarning:
        self._dropped_frames += 1
        self._pending_drops += 1
        return RecorderWarning(reason, int(monotonic_ms), details)

    def save_evidence_frame(self, path: Path, frame: np.ndarray) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".png":
            destination = destination.with_suffix(".png")
        save_image_unicode(destination, frame)
        return destination

    def save_incident_media(
        self,
        incident_directory: Path,
        *,
        trigger_ms: int,
        before_ms: int = 5_000,
        after_ms: int = 5_000,
    ) -> IncidentMedia:
        with self._lock:
            self._ensure_open()
            incident_directory = Path(incident_directory)
            incident_directory.mkdir(parents=True, exist_ok=True)
            selected = [
                (timestamp, frame.copy())
                for timestamp, frame in self._buffer
                if trigger_ms - before_ms <= timestamp <= trigger_ms + after_ms
            ]
            if not selected:
                raise RuntimeError("环形缓冲中没有事故时间范围内的帧")
            clip_path = incident_directory / "clip.avi"
            clip_writer = self._open_writer(clip_path)
            try:
                for _, frame in selected:
                    clip_writer.write(frame)
            finally:
                clip_writer.release()
            _, trigger_frame = min(
                selected, key=lambda item: abs(item[0] - int(trigger_ms))
            )
            frames_directory = incident_directory / "frames"
            trigger_path = self.save_evidence_frame(
                frames_directory / "trigger.png", trigger_frame
            )
            contact_sheet_path = incident_directory / "contact_sheet.png"
            save_image_unicode(
                contact_sheet_path,
                self._make_contact_sheet([frame for _, frame in selected]),
            )
            return IncidentMedia(
                clip_path=clip_path,
                contact_sheet_path=contact_sheet_path,
                trigger_frame_path=trigger_path,
                frame_count=len(selected),
            )

    def _make_contact_sheet(self, frames: list[np.ndarray]) -> np.ndarray:
        sample_count = min(9, len(frames))
        indexes = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        samples = [frames[int(index)] for index in indexes]
        columns = min(3, sample_count)
        rows = int(math.ceil(sample_count / columns))
        tile_width = min(self.size[0], 480)
        tile_height = max(1, int(round(tile_width * self.size[1] / self.size[0])))
        canvas = np.zeros(
            (rows * tile_height, columns * tile_width, 3), dtype=np.uint8
        )
        for index, frame in enumerate(samples):
            tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            row, column = divmod(index, columns)
            canvas[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = tile
        return canvas

    def close(self) -> RecordingResult:
        with self._lock:
            if not self._closed:
                self._writer.release()
                self._closed = True
            return RecordingResult(
                video_path=self.video_path,
                index_path=self.index_path,
                frame_count=self._frame_count,
                dropped_frames=self._dropped_frames,
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("录像器已经关闭")

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
