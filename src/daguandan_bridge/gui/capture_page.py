from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget


class CapturePage(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._session_active = False
        self._has_frame = False
        self.setWindowTitle("截图录制")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("大掼蛋截图录制"))
        self.preview = QLabel("尚未开始预览")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 360)
        layout.addWidget(self.preview)
        controls = QHBoxLayout()
        self.start_button = QPushButton("开始预览")
        self.stop_button = QPushButton("停止预览")
        self.start_session_button = QPushButton("开始本局录制")
        self.end_session_button = QPushButton("结束本局")
        self.open_folder_button = QPushButton("打开录制文件夹")
        self.interval_box = QSpinBox()
        self.interval_box.setRange(100, 60000)
        self.interval_box.setSingleStep(100)
        self.interval_box.setValue(controller.recording_interval_ms())
        self.interval_box.setSuffix(" ms")
        for widget in (self.start_button, self.stop_button, QLabel("录制间隔"), self.interval_box, self.start_session_button, self.end_session_button, self.open_folder_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.status = QLabel("状态：就绪")
        self.session_status = QLabel("对局录制：未开始")
        layout.addWidget(self.status)
        layout.addWidget(self.session_status)
        self.timer = QTimer(self)
        self.start_button.clicked.connect(controller.start_capture)
        self.stop_button.clicked.connect(controller.stop_capture)
        self.start_session_button.clicked.connect(self._start_session)
        self.end_session_button.clicked.connect(controller.finish_screenshot_session)
        self.open_folder_button.clicked.connect(controller.open_screenshot_folder)
        self.timer.timeout.connect(controller.save_current_frame)
        controller.frame_ready.connect(self._show_frame)
        controller.frame_cleared.connect(lambda: self.preview.setText("没有有效画面"))
        controller.state_changed.connect(lambda value: self.status.setText(f"状态：{value}"))
        controller.error.connect(self._show_error)
        controller.screenshot_session_updated.connect(self._session_updated)
        controller.screenshot_session_finished.connect(self._session_finished)
        self.end_session_button.setEnabled(False)

    def _start_session(self) -> None:
        self.controller.start_screenshot_session(self.interval_box.value())

    def _session_updated(self, session) -> None:
        self._session_active = True
        self.end_session_button.setEnabled(True)
        self.interval_box.setEnabled(False)
        self.timer.setInterval(session.interval_ms)
        if self._has_frame:
            self.timer.start()
        self.session_status.setText(f"对局录制：进行中，已保存 {session.frame_count} 帧")

    def _session_finished(self, session) -> None:
        self._session_active = False
        self.timer.stop()
        self.end_session_button.setEnabled(False)
        self.interval_box.setEnabled(True)
        self.session_status.setText(f"对局录制：已结束，共 {session.frame_count} 帧")

    def _show_error(self, message: str) -> None:
        self.timer.stop()
        self.status.setText(f"错误：{message}")

    def _show_frame(self, snapshot) -> None:
        image = snapshot.image
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, channels * width, QImage.Format.Format_BGR888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimage).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self._has_frame = True
        if self._session_active and not self.timer.isActive():
            self.timer.start()
