from __future__ import annotations

from qfluentwidgets import FluentIcon, FluentWindow

from ..annotation_service import AnnotationService
from .annotation_page import AnnotationPage
from .capture_page import CapturePage
from .controller import CaptureController
from .live_assistant_page import LiveAssistantPage


class DaguandanBridgeWindow(FluentWindow):
    def __init__(
        self,
        controller: CaptureController | None = None,
        *,
        live_runtime=None,
    ) -> None:
        super().__init__()
        self.controller = controller or CaptureController()
        self.capture_page = CapturePage(self.controller)
        self.capture_page.setObjectName("capturePage")
        self.annotation_page = AnnotationPage(AnnotationService())
        self.annotation_page.setObjectName("annotationPage")
        self.live_assistant_page = LiveAssistantPage(live_runtime)

        self.addSubInterface(
            self.live_assistant_page,
            FluentIcon.ROBOT,
            "实时助手",
        )
        self.addSubInterface(
            self.capture_page,
            FluentIcon.CAMERA,
            "截图录制",
        )
        self.addSubInterface(
            self.annotation_page,
            FluentIcon.EDIT,
            "标记与模板",
        )
        self.setWindowTitle("大掼蛋 DanZero 桥接器")
        self.resize(1220, 820)
        self.setMinimumSize(980, 700)

    def closeEvent(self, event) -> None:
        self.live_assistant_page.shutdown()
        self.controller.shutdown()
        super().closeEvent(event)
