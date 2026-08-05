import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from daguandan_bridge.capture_service import CaptureService
from daguandan_bridge.gui.controller import CaptureController
from daguandan_bridge.gui.main_window import DaguandanBridgeWindow


def test_capture_window_initializes_with_recording_controls():
    app = QApplication.instance() or QApplication([])
    controller = CaptureController(CaptureService())
    window = DaguandanBridgeWindow(controller)

    assert window.capture_page.start_button.text() == "开始预览"
    assert window.capture_page.start_session_button.text() == "开始本局录制"
    assert window.capture_page.end_session_button.text() == "结束本局"

    controller.shutdown()
    window.close()
    app.processEvents()
