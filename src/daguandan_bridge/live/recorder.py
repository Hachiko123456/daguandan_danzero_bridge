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
from ..storage import append_json_line, atomic_write_json


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
    incident_media_failures: tuple["IncidentMediaFailure", ...] = ()


@dataclass(frozen=True)
class IncidentMedia:
    clip_path: Path
    contact_sheet_path: Path
    trigger_frame_path: Path
    frame_count: int


@dataclass(frozen=True)
class IncidentMediaFailure:
    reason: str
    incident_directory: Path
    trigger_ms: int
    details: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "incident_directory": str(self.incident_directory),
            "trigger_ms": self.trigger_ms,
            "details": self.details,
        }


@dataclass
class _PendingIncidentMedia:
    directory: Path
    trigger_ms: int
    deadline_ms: int
    frames: list[tuple[int, np.ndarray]]


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
        self._pending_incidents: list[_PendingIncidentMedia] = []
        self._incident_media_failures: list[IncidentMediaFailure] = []

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
            buffered_frame = frame.copy()
            self._buffer.append((int(captured_monotonic_ms), buffered_frame))
            self._advance_pending_incidents(
                int(captured_monotonic_ms),
                buffered_frame,
            )
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
                (timestamp, frame)
                for timestamp, frame in self._buffer
                if trigger_ms - before_ms <= timestamp <= trigger_ms + after_ms
            ]
            if not selected:
                raise RuntimeError("环形缓冲中没有事故时间范围内的帧")
            return self._write_incident_media(
                incident_directory,
                int(trigger_ms),
                selected,
            )

    def schedule_incident_media(
        self,
        incident_directory: Path,
        *,
        trigger_ms: int,
        before_ms: int = 5_000,
        after_ms: int = 5_000,
    ) -> None:
        """Keep pre-trigger frames now and finalize after future frames arrive."""

        with self._lock:
            self._ensure_open()
            directory = Path(incident_directory)
            directory.mkdir(parents=True, exist_ok=True)
            selected = [
                (timestamp, frame)
                for timestamp, frame in self._buffer
                if trigger_ms - before_ms <= timestamp <= trigger_ms
            ]
            pending = _PendingIncidentMedia(
                directory=directory,
                trigger_ms=int(trigger_ms),
                deadline_ms=int(trigger_ms + max(0, after_ms)),
                frames=selected,
            )
            if after_ms <= 0:
                if selected:
                    self._write_incident_media(directory, int(trigger_ms), selected)
                return
            self._pending_incidents.append(pending)

    def _advance_pending_incidents(
        self,
        monotonic_ms: int,
        frame: np.ndarray,
    ) -> None:
        completed: list[_PendingIncidentMedia] = []
        for pending in self._pending_incidents:
            if pending.trigger_ms < monotonic_ms <= pending.deadline_ms:
                pending.frames.append((monotonic_ms, frame))
            if monotonic_ms >= pending.deadline_ms:
                completed.append(pending)
        for pending in completed:
            self._finalize_pending_incident(pending)
            self._pending_incidents.remove(pending)

    def _finalize_pending_incident(self, pending: _PendingIncidentMedia) -> None:
        if not pending.frames:
            return
        try:
            self._write_incident_media(
                pending.directory,
                pending.trigger_ms,
                pending.frames,
            )
        except Exception as exc:
            failure = IncidentMediaFailure(
                reason="incident_media_finalize_failed",
                incident_directory=pending.directory,
                trigger_ms=pending.trigger_ms,
                details=str(exc),
            )
            self._incident_media_failures.append(failure)
            try:
                atomic_write_json(
                    pending.directory / "media_error.json",
                    {"schema_version": 1, **failure.to_dict()},
                )
            except Exception:
                # The in-memory failure is still returned to the session manifest.
                pass

    def _write_incident_media(
        self,
        incident_directory: Path,
        trigger_ms: int,
        selected: list[tuple[int, np.ndarray]],
    ) -> IncidentMedia:
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
        atomic_write_json(
            incident_directory / "media.json",
            {
                "schema_version": 1,
                "trigger_ms": int(trigger_ms),
                "first_frame_ms": int(selected[0][0]),
                "last_frame_ms": int(selected[-1][0]),
                "frame_count": len(selected),
                "clip": clip_path.relative_to(incident_directory).as_posix(),
                "contact_sheet": contact_sheet_path.relative_to(
                    incident_directory
                ).as_posix(),
                "trigger_frame": trigger_path.relative_to(
                    incident_directory
                ).as_posix(),
            },
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
                try:
                    self._writer.release()
                except Exception as exc:
                    self._incident_media_failures.append(
                        IncidentMediaFailure(
                            reason="main_video_release_failed",
                            incident_directory=self.video_directory,
                            trigger_ms=-1,
                            details=str(exc),
                        )
                    )
                finally:
                    self._closed = True
                for pending in tuple(self._pending_incidents):
                    self._finalize_pending_incident(pending)
                self._pending_incidents.clear()
            return RecordingResult(
                video_path=self.video_path,
                index_path=self.index_path,
                frame_count=self._frame_count,
                dropped_frames=self._dropped_frames,
                incident_media_failures=tuple(self._incident_media_failures),
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("录像器已经关闭")

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
