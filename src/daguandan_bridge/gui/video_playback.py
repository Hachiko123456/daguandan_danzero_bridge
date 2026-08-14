"""Reusable, worker-backed playback for recorded session videos."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QGridLayout, QWidget
from qfluentwidgets import BodyLabel, ComboBox, PrimaryPushButton, PushButton, SpinBox

from ..live.replay import VideoReplaySource


class SessionPlaybackToolbar(QWidget):
    """One Fluent playback surface shared by replay and template annotation.

    Pages keep ownership of their frame display and session-specific actions,
    while this widget owns the identical controls, labels and control signals.
    """

    play_pause_requested = Signal()
    step_requested = Signal()
    seek_requested = Signal(int)
    seek_seconds_requested = Signal(float)
    speed_changed = Signal(float)

    def __init__(self, parent=None, *, overlay_seek: bool = False) -> None:
        super().__init__(parent)
        self._overlay_seek = bool(overlay_seek)
        self.setObjectName("sessionPlaybackToolbar")
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(8)
        self._layout.setVerticalSpacing(6)
        self._compact_layout: bool | None = None

        self.play_button = PrimaryPushButton("播放")
        self.step_button = PushButton("下一帧")
        self.rewind_button = PushButton("后退 5 秒")
        self.forward_button = PushButton("前进 5 秒")
        self.frame_spin = SpinBox(self)
        self.frame_spin.setRange(0, 0)
        self.frame_spin.setPrefix("帧 ")
        self.frame_spin.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_spin.setMinimumWidth(156)
        self.frame_spin.setToolTip("跳转到指定录像帧")
        self.frame_jump_button = PushButton("跳转")
        # Compatibility aliases remain callable, but the replay surface now
        # exposes ±5 s as video overlays and Enter performs frame seeking.
        if self._overlay_seek:
            self.rewind_button.hide()
            self.forward_button.hide()
            self.frame_jump_button.hide()
        self.speed_combo = ComboBox()
        for label, speed in (("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0)):
            self.speed_combo.addItem(label, userData=speed)
        self.speed_combo.setCurrentIndex(1)
        self.frame_status = BodyLabel("帧：—")
        self.frame_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        for control in (
            self.play_button,
            self.step_button,
            self.rewind_button,
            self.forward_button,
            self.frame_spin,
            self.frame_jump_button,
            self.speed_combo,
        ):
            control.setMinimumHeight(34)
        self.play_button.setMinimumWidth(88)
        self.step_button.setMinimumWidth(76)
        self.rewind_button.setMinimumWidth(96)
        self.forward_button.setMinimumWidth(96)
        self.frame_jump_button.setMinimumWidth(72)
        self.speed_combo.setMinimumWidth(82)

        self._arrange_controls(compact=True)

        self.play_button.clicked.connect(self.play_pause_requested)
        self.step_button.clicked.connect(self.step_requested)
        self.rewind_button.clicked.connect(
            lambda: self.seek_seconds_requested.emit(-5.0)
        )
        self.forward_button.clicked.connect(
            lambda: self.seek_seconds_requested.emit(5.0)
        )
        self.frame_jump_button.clicked.connect(
            lambda: self.seek_requested.emit(self.frame_spin.value())
        )
        self.frame_spin.lineEdit().returnPressed.connect(
            lambda: self.seek_requested.emit(self.frame_spin.value())
        )
        self.speed_combo.currentIndexChanged.connect(
            lambda _index: self.speed_changed.emit(
                float(self.speed_combo.currentData() or 1.0)
            )
        )

    def set_playing(self, playing: bool) -> None:
        self.play_button.setText("暂停" if playing else "播放")

    def set_frame_count(self, frame_count: int) -> None:
        self.frame_spin.setRange(0, max(0, int(frame_count) - 1))

    def set_current_frame(self, frame_index: int) -> None:
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(max(0, int(frame_index)))
        self.frame_spin.blockSignals(False)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # The replay page may devote less than half of a 980px window to the
        # video pane.  At that size, use three short rows rather than allowing
        # any of the shared controls to overlap or disappear.
        compact_width = 720 if self._overlay_seek else 620
        self._arrange_controls(compact=self.width() < compact_width)

    def _arrange_controls(self, *, compact: bool) -> None:
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        for widget in (
            self.play_button,
            self.step_button,
            self.rewind_button,
            self.forward_button,
            self.frame_spin,
            self.frame_jump_button,
            self.speed_combo,
            self.frame_status,
        ):
            self._layout.removeWidget(widget)
        for column in range(5):
            self._layout.setColumnStretch(column, 0)
        if not self._overlay_seek:
            if compact:
                self._layout.addWidget(self.play_button, 0, 0)
                self._layout.addWidget(self.step_button, 0, 1)
                self._layout.addWidget(self.speed_combo, 0, 2)
                self._layout.addWidget(self.rewind_button, 1, 0)
                self._layout.addWidget(self.forward_button, 1, 1)
                self._layout.addWidget(self.frame_spin, 2, 0, 1, 2)
                self._layout.addWidget(self.frame_jump_button, 2, 2)
                self._layout.addWidget(self.frame_status, 2, 3)
                self._layout.setColumnStretch(3, 1)
                return
            self._layout.addWidget(self.play_button, 0, 0)
            self._layout.addWidget(self.step_button, 0, 1)
            self._layout.addWidget(self.rewind_button, 0, 2)
            self._layout.addWidget(self.forward_button, 0, 3)
            self._layout.addWidget(self.speed_combo, 0, 4)
            self._layout.addWidget(self.frame_spin, 1, 0, 1, 2)
            self._layout.addWidget(self.frame_jump_button, 1, 2)
            self._layout.addWidget(self.frame_status, 1, 3, 1, 2)
            self._layout.setColumnStretch(3, 1)
            return
        if compact:
            self._layout.addWidget(self.play_button, 0, 0)
            self._layout.addWidget(self.step_button, 0, 1)
            self._layout.addWidget(self.speed_combo, 0, 2)
            self._layout.addWidget(self.frame_spin, 1, 0, 1, 2)
            self._layout.addWidget(self.frame_status, 1, 2, 1, 2)
            self._layout.setColumnStretch(3, 1)
            return
        # Replay uses overlay seeking.  Keep its primary controls and frame
        # context on one row when there is room; a second sparse row wastes
        # vertical space directly below the video.
        self._layout.addWidget(self.play_button, 0, 0)
        self._layout.addWidget(self.step_button, 0, 1)
        self._layout.addWidget(self.speed_combo, 0, 2)
        self._layout.addWidget(self.frame_spin, 0, 3)
        self._layout.addWidget(self.frame_status, 0, 4)
        self._layout.setColumnStretch(4, 1)


class ReplayDecodeThread(QThread):
    """Decode a session video off the GUI thread and support paced stepping."""

    frame_ready = Signal(object, object)
    failed = Signal(str)

    def __init__(
        self,
        video_path: Path,
        index_path: Path,
        parent=None,
        *,
        start_frame: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.index_path = Path(index_path)
        self._condition = threading.Condition()
        self._playing = False
        self._step_requested = False
        self._stop_requested = False
        self._speed = 1.0
        self._start_frame = start_frame

    def play(self) -> None:
        with self._condition:
            self._playing = True
            self._condition.notify_all()

    def pause(self) -> None:
        with self._condition:
            self._playing = False

    def step(self) -> None:
        with self._condition:
            self._step_requested = True
            self._condition.notify_all()

    def set_speed(self, speed: float) -> None:
        with self._condition:
            self._speed = max(0.1, float(speed))
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    def run(self) -> None:
        previous_ms: int | None = None
        try:
            for record, frame in VideoReplaySource(
                self.video_path, self.index_path
            ).frames(start_frame=self._start_frame):
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_requested
                        or self._playing
                        or self._step_requested
                    )
                    if self._stop_requested:
                        return
                    stepping = self._step_requested and not self._playing
                    self._step_requested = False
                    speed = self._speed
                if previous_ms is not None and not stepping:
                    delay = max(
                        0.0,
                        (record.monotonic_ms - previous_ms) / 1000 / speed,
                    )
                    with self._condition:
                        self._condition.wait(timeout=min(delay, 2.0))
                        if self._stop_requested:
                            return
                previous_ms = record.monotonic_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(record, image)
                if stepping:
                    self.pause()
        except Exception as exc:
            self.failed.emit(str(exc))
