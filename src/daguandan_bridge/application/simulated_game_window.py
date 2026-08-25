"""Visible Qt replay window used by the phase-three Win32 capture checks."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QThread, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from ..live.replay import FrameIndexRecord
from ..models import TargetWindow
from ..storage import append_json_line, atomic_write_json
from ..window_capture import (
    get_client_rect_on_screen,
    get_window_dpi,
    resize_target_client,
)


DEFAULT_WINDOW_TITLE = "大掼蛋（腾讯）"


def _resolved(path: object, base: Path) -> Path:
    value = Path(str(path)).expanduser()
    return (base / value).resolve() if not value.is_absolute() else value.resolve()


@dataclass(frozen=True)
class SimulatedGameWindowConfig:
    video_path: Path
    frame_index_path: Path
    control_path: Path
    status_path: Path
    event_log_path: Path
    window_title: str = DEFAULT_WINDOW_TITLE
    client_size: tuple[int, int] = (1280, 764)
    header_height: int = 44
    time_scale: float = 1.0
    autoplay: bool = False
    max_frames: int | None = None
    command_poll_ms: int = 50
    position: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.video_path.is_file():
            raise ValueError(f"simulator video does not exist: {self.video_path}")
        if not self.frame_index_path.is_file():
            raise ValueError(
                f"simulator frame index does not exist: {self.frame_index_path}"
            )
        width, height = self.client_size
        if width <= 0 or height <= 0 or not 0 < self.header_height < height:
            raise ValueError("simulator client/header dimensions are invalid")
        if not math.isfinite(self.time_scale) or self.time_scale <= 0:
            raise ValueError("simulator time_scale must be positive and finite")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("simulator max_frames must be positive")
        if self.command_poll_ms < 10:
            raise ValueError("simulator command_poll_ms must be at least 10")
        if not self.window_title.strip():
            raise ValueError("simulator window_title must not be empty")

    @classmethod
    def from_path(cls, path: Path | str) -> "SimulatedGameWindowConfig":
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("simulator config must be a JSON object")
        base = config_path.parent
        session = _resolved(raw["session"], base) if raw.get("session") else None
        video = raw.get("video_path")
        index = raw.get("frame_index_path")
        if session is not None:
            video = video or session / "video" / "game.avi"
            index = index or session / "video" / "frame_index.jsonl"
        if video is None or index is None:
            raise ValueError("simulator config requires video_path and frame_index_path")
        client = raw.get("client_size", (1280, 764))
        position = raw.get("position")
        return cls(
            video_path=_resolved(video, base),
            frame_index_path=_resolved(index, base),
            control_path=_resolved(raw["control_path"], base),
            status_path=_resolved(raw["status_path"], base),
            event_log_path=_resolved(
                raw.get("event_log_path", "simulator_events.jsonl"), base
            ),
            window_title=str(raw.get("window_title", DEFAULT_WINDOW_TITLE)),
            client_size=(int(client[0]), int(client[1])),
            header_height=int(raw.get("header_height", 44)),
            time_scale=float(raw.get("time_scale", 1.0)),
            autoplay=bool(raw.get("autoplay", False)),
            max_frames=(
                int(raw["max_frames"]) if raw.get("max_frames") is not None else None
            ),
            command_poll_ms=int(raw.get("command_poll_ms", 50)),
            position=(
                (int(position[0]), int(position[1]))
                if position is not None
                else None
            ),
        )


class IndexedVideoPlayback:
    """Qt-free sequential decoder preserving explicit frame-index timestamps."""

    def __init__(self, config: SimulatedGameWindowConfig) -> None:
        self.config = config
        rows: list[FrameIndexRecord] = []
        for line in config.frame_index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(FrameIndexRecord.from_dict(json.loads(line)))
        if config.max_frames is not None:
            rows = rows[: config.max_frames]
        if not rows:
            raise ValueError("simulator frame index is empty")
        if any(
            current.frame_index <= previous.frame_index
            or current.monotonic_ms < previous.monotonic_ms
            for previous, current in zip(rows, rows[1:])
        ):
            raise ValueError("simulator frame index must be strictly ordered")
        self.records = tuple(rows)
        self._frame_numbers = tuple(record.frame_index for record in rows)
        self._capture = cv2.VideoCapture(str(config.video_path))
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"cannot open simulator video: {config.video_path}")
        self._next_decode_position: int | None = None

    @property
    def count(self) -> int:
        return len(self.records)

    def nearest_position(self, frame_index: int) -> int:
        position = bisect.bisect_left(self._frame_numbers, int(frame_index))
        return min(position, self.count - 1)

    def decode(self, position: int) -> tuple[FrameIndexRecord, np.ndarray]:
        if position < 0 or position >= self.count:
            raise IndexError("simulator frame position is out of range")
        record = self.records[position]
        if self._next_decode_position != position:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, record.frame_index):
                # OpenCV may report False even when an AVI seek succeeds; the
                # decoded position check below remains the authoritative gate.
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, record.frame_index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot decode simulator frame {record.frame_index}")
        self._next_decode_position = position + 1
        return record, frame

    def close(self) -> None:
        self._capture.release()


class SimulatedGameWindow(QWidget):
    """Thin GUI adapter; playback scheduling and commands stay deterministic."""

    def __init__(self, config: SimulatedGameWindowConfig) -> None:
        super().__init__()
        self.config = config
        self.playback = IndexedVideoPlayback(config)
        self.setWindowTitle(config.window_title)
        self.setObjectName("simulatedGameWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.header = QLabel(self)
        self.header.setObjectName("simulatedGameHeader")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header.setText("腾讯大掼蛋·窗口捕获验证")
        self.header.setStyleSheet(
            "background:#20252b;color:#f5f7fa;font-size:16px;font-weight:600;"
        )
        self.viewport = QLabel(self)
        self.viewport.setObjectName("simulatedGameViewport")
        self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport.setStyleSheet("background:#000000;")
        self.viewport.setScaledContents(True)
        self.resize(*config.client_size)
        self._position = 0
        self._record, self._frame = self.playback.decode(0)
        self._state = "paused"
        self._last_error: str | None = None
        self._last_request_id: str | None = None
        self._play_epoch = 0.0
        self._source_epoch_ms = self._record.monotonic_ms
        self._shown_count = 0
        self._closing = False
        self._ready = False
        self._occluder: QWidget | None = None
        self._render_frame(self._record, self._frame, reason="startup")

        self._play_timer = QTimer(self)
        self._play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._play_timer.setInterval(5)
        self._play_timer.timeout.connect(self._advance_due_frames)
        self._play_timer.start()
        self._command_timer = QTimer(self)
        self._command_timer.setInterval(config.command_poll_ms)
        self._command_timer.timeout.connect(self._poll_command)
        self._command_timer.start()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(100)
        self._status_timer.timeout.connect(self._write_status)
        self._status_timer.start()

    @property
    def hwnd(self) -> int:
        return int(self.winId())

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        width, height = self.width(), self.height()
        header_height = round(self.config.header_height / self.devicePixelRatioF())
        header_height = min(max(1, header_height), max(1, height - 1))
        self.header.setGeometry(0, 0, width, header_height)
        self.viewport.setGeometry(0, header_height, width, height - header_height)
        super().resizeEvent(event)

    def initialize_native_window(self) -> None:
        try:
            target = TargetWindow(self.hwnd, self.windowTitle())
            resize_target_client(target, self.config.client_size)
            if self.config.position is not None:
                self._move_native(*self.config.position)
            self._ready = True
        except Exception as exc:
            self._last_error = f"native initialization failed: {exc}"
        self._write_status()
        self._append_event(
            "ready" if self._ready else "initialization_failed",
            {} if self._ready else {"error": self._last_error},
        )
        if self._ready and self.config.autoplay:
            self.play()

    def play(self) -> None:
        if self._position >= self.playback.count - 1:
            self.reset()
        self._state = "playing"
        self._play_epoch = time.monotonic()
        self._source_epoch_ms = self._record.monotonic_ms
        self._append_event("play", {})
        self._write_status()

    def pause(self) -> None:
        if self._state == "playing":
            self._state = "paused"
            self._append_event("pause", {})
            self._write_status()

    def seek(self, frame_index: int) -> None:
        self._position = self.playback.nearest_position(frame_index)
        self._record, self._frame = self.playback.decode(self._position)
        self._state = "paused"
        self._render_frame(self._record, self._frame, reason="seek")
        self._write_status()

    def reset(self) -> None:
        self.seek(self.playback.records[0].frame_index)
        self._append_event("reset", {})

    def _advance_due_frames(self) -> None:
        if self._state != "playing":
            return
        elapsed_ms = (time.monotonic() - self._play_epoch) * 1000.0
        virtual_ms = elapsed_ms * self.config.time_scale
        while self._position + 1 < self.playback.count:
            next_position = self._position + 1
            next_record = self.playback.records[next_position]
            if next_record.monotonic_ms - self._source_epoch_ms > virtual_ms:
                break
            self._position = next_position
            self._record, self._frame = self.playback.decode(self._position)
            self._render_frame(self._record, self._frame, reason="playback")
        if self._position >= self.playback.count - 1:
            self._state = "eof"
            self._append_event("eof", {})
            self._write_status()

    def _render_frame(
        self,
        record: FrameIndexRecord,
        frame: np.ndarray,
        *,
        reason: str,
    ) -> None:
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("simulator frame render attempted outside GUI thread")
        bgr = np.ascontiguousarray(frame)
        height, width = bgr.shape[:2]
        image = QImage(
            bgr.data,
            width,
            height,
            int(bgr.strides[0]),
            QImage.Format.Format_BGR888,
        ).copy()
        self.viewport.setPixmap(QPixmap.fromImage(image))
        self._shown_count += 1
        self._append_event(
            "frame_shown",
            {
                "reason": reason,
                "frame_index": record.frame_index,
                "source_monotonic_ms": record.monotonic_ms,
                "frame_sha256": hashlib.sha256(bgr.tobytes()).hexdigest(),
            },
        )

    def _poll_command(self) -> None:
        if not self.config.control_path.is_file():
            return
        try:
            raw = json.loads(self.config.control_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("control document must be a JSON object")
            request_id = str(raw.get("request_id", ""))
            if not request_id or request_id == self._last_request_id:
                return
            self._last_request_id = request_id
            command = str(raw.get("command", "")).strip().lower()
            arguments = raw.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            self._execute_command(command, arguments)
            self._last_error = None
            self._append_event(
                "command_completed",
                {"request_id": request_id, "command": command},
            )
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._append_event("command_failed", {"error": self._last_error})
        self._write_status()

    def _execute_command(self, command: str, arguments: dict[str, object]) -> None:
        if command == "play":
            self.play()
        elif command == "pause":
            self.pause()
        elif command == "seek":
            self.seek(int(arguments["frame_index"]))
        elif command == "reset":
            self.reset()
        elif command == "close":
            self.close()
        elif command == "move":
            self._move_native(int(arguments["left"]), int(arguments["top"]))
        elif command == "resize_client":
            resize_target_client(
                TargetWindow(self.hwnd, self.windowTitle()),
                (int(arguments["width"]), int(arguments["height"])),
            )
        elif command == "minimize":
            self.showMinimized()
        elif command == "restore":
            self.showNormal()
            self.raise_()
        elif command == "occlude":
            self._set_occlusion(bool(arguments.get("visible", True)))
        else:
            raise ValueError(f"unsupported simulator command: {command}")

    def _move_native(self, left: int, top: int) -> None:
        if sys.platform != "win32":
            self.move(left, top)
            return
        import win32gui

        target = TargetWindow(self.hwnd, self.windowTitle())
        client = get_client_rect_on_screen(target)
        outer = win32gui.GetWindowRect(self.hwnd)
        outer_left = int(left) - (client.left - outer[0])
        outer_top = int(top) - (client.top - outer[1])
        win32gui.SetWindowPos(
            self.hwnd,
            0,
            outer_left,
            outer_top,
            outer[2] - outer[0],
            outer[3] - outer[1],
            0x0004 | 0x0010 | 0x0200,
        )

    def _set_occlusion(self, visible: bool) -> None:
        if self._occluder is None:
            overlay = QWidget(None, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
            overlay.setWindowTitle("Window E2E Occluder")
            overlay.setStyleSheet("background:#d32029;")
            overlay.setWindowOpacity(1.0)
            self._occluder = overlay
        if not visible:
            self._occluder.hide()
            return
        rect = get_client_rect_on_screen(TargetWindow(self.hwnd, self.windowTitle()))
        self._occluder.show()
        if sys.platform == "win32":
            import win32gui

            overlay_hwnd = int(self._occluder.winId())
            win32gui.SetWindowPos(
                overlay_hwnd,
                -1,
                rect.left + rect.width // 4,
                rect.top + rect.height // 4,
                rect.width // 2,
                rect.height // 2,
                0x0040,
            )
        self._occluder.raise_()

    def _status_payload(self) -> dict[str, object]:
        rect_payload: dict[str, int] | None = None
        dpi: int | None = None
        geometry_error: str | None = None
        try:
            target = TargetWindow(self.hwnd, self.windowTitle())
            rect = get_client_rect_on_screen(target)
            rect_payload = {
                "left": rect.left,
                "top": rect.top,
                "width": rect.width,
                "height": rect.height,
            }
            dpi = get_window_dpi(target)
        except Exception as exc:
            geometry_error = str(exc)
        return {
            "schema": "guandan.simulated-game-window-status/1",
            "ready": self._ready,
            "pid": os.getpid(),
            "hwnd": self.hwnd,
            "window_title": self.windowTitle(),
            "client_rect": rect_payload,
            "observed_dpi": dpi,
            "geometry_error": geometry_error,
            "viewport": {
                "x": 0,
                "y": self.config.header_height,
                "width": (
                    rect_payload["width"] if rect_payload is not None else self.width()
                ),
                "height": (
                    max(0, rect_payload["height"] - self.config.header_height)
                    if rect_payload is not None
                    else max(0, self.height() - self.config.header_height)
                ),
            },
            "state": self._state,
            "current_frame_index": self._record.frame_index,
            "current_source_monotonic_ms": self._record.monotonic_ms,
            "indexed_frame_count": self.playback.count,
            "shown_count": self._shown_count,
            "last_request_id": self._last_request_id,
            "last_error": self._last_error,
            "occluder_hwnd": (
                int(self._occluder.winId())
                if self._occluder is not None and self._occluder.isVisible()
                else None
            ),
            "gui_thread": QThread.currentThread() is self.thread(),
            "updated_at": datetime.now().astimezone().isoformat(),
        }

    def _write_status(self) -> None:
        try:
            atomic_write_json(self.config.status_path, self._status_payload())
        except Exception as exc:
            self._last_error = f"status write failed: {exc}"

    def _append_event(self, kind: str, payload: dict[str, object]) -> None:
        append_json_line(
            self.config.event_log_path,
            {
                "kind": kind,
                "wall_time": datetime.now().astimezone().isoformat(),
                "monotonic_ms": time.monotonic_ns() // 1_000_000,
                **payload,
            },
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if not self._closing:
            self._closing = True
            self._state = "closed"
            self._ready = False
            self._play_timer.stop()
            self._command_timer.stop()
            self._status_timer.stop()
            if self._occluder is not None:
                self._occluder.close()
            self.playback.close()
            self._append_event("closed", {})
            self._write_status()
        event.accept()
        QTimer.singleShot(0, QCoreApplication.quit)


def run_simulated_game_window(config_path: Path | str) -> int:
    """Run one visible simulator; status/control files are the process API."""

    config = SimulatedGameWindowConfig.from_path(config_path)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Daguandan Window E2E Simulator")
    window = SimulatedGameWindow(config)
    window.show()
    QTimer.singleShot(0, window.initialize_native_window)
    return int(app.exec())
