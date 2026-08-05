from __future__ import annotations

from PySide6.QtWidgets import QMainWindow

from .capture_page import CapturePage
from .controller import CaptureController


class DaguandanBridgeWindow(QMainWindow):
    def __init__(self, controller: CaptureController | None = None):
        super().__init__()
        self.controller = controller or CaptureController()
        self.capture_page = CapturePage(self.controller)
        self.setCentralWidget(self.capture_page)
        self.setWindowTitle("大掼蛋 DanZero 桥接器")
        self.resize(1100, 760)

    def closeEvent(self, event):
        self.controller.shutdown()
        super().closeEvent(event)
