from __future__ import annotations

from qfluentwidgets import FluentIcon, FluentWindow

from ..annotation_service import AnnotationService
from .annotation_page import AnnotationPage
from .live_assistant_page import LiveAssistantPage
from .replay_page import ReplayPage


class DaguandanBridgeWindow(FluentWindow):
    def __init__(
        self,
        *,
        live_runtime=None,
    ) -> None:
        super().__init__()
        self.annotation_page = AnnotationPage(AnnotationService())
        self.annotation_page.setObjectName("annotationPage")
        self.live_assistant_page = LiveAssistantPage(live_runtime)
        self.replay_page = ReplayPage()

        self.addSubInterface(
            self.live_assistant_page,
            FluentIcon.ROBOT,
            "实时助手",
        )
        self.addSubInterface(
            self.annotation_page,
            FluentIcon.EDIT,
            "标记与模板",
        )
        self.addSubInterface(
            self.replay_page,
            FluentIcon.VIDEO,
            "对局回放",
        )
        self.setWindowTitle("大掼蛋 DanZero 桥接器")
        self.resize(1220, 820)
        self.setMinimumSize(980, 700)

    def closeEvent(self, event) -> None:
        self.replay_page.shutdown()
        self.live_assistant_page.shutdown()
        super().closeEvent(event)
