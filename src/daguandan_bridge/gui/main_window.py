from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from ..annotation_service import AnnotationService
from .annotation_page import AnnotationPage
from .capture_page import CapturePage
from .controller import CaptureController


class DaguandanBridgeWindow(QMainWindow):
    def __init__(self, controller: CaptureController | None = None):
        super().__init__()
        self.controller = controller or CaptureController()
        self.capture_page = CapturePage(self.controller)
        self.annotation_page = AnnotationPage(AnnotationService())
        tabs = QTabWidget()
        tabs.addTab(self.capture_page, "截图录制")
        tabs.addTab(self.annotation_page, "区域标注")
        self.setCentralWidget(tabs)
        self.setWindowTitle("大掼蛋 DanZero 桥接器")
        self.resize(1100, 760)

    def closeEvent(self, event):
        self.controller.shutdown()
        super().closeEvent(event)
