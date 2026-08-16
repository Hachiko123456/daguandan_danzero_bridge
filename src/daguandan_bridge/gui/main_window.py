from __future__ import annotations

from PySide6.QtCore import QRect
from qfluentwidgets import FluentIcon, FluentWindow

from .annotation_page import AnnotationPage
from .live_assistant_page import LiveAssistantPage
from .recommendation_window import RecommendationFloatWindow
from .replay_page import ReplayPage


class DaguandanBridgeWindow(FluentWindow):
    def __init__(
        self,
        *,
        dependencies=None,
        live_runtime=None,
    ) -> None:
        super().__init__()
        if dependencies is None:
            from ..bootstrap import build_application_dependencies

            dependencies = build_application_dependencies(live_runtime=live_runtime)
        self.live_runtime = live_runtime or dependencies.live_runtime
        self.annotation_page = AnnotationPage(dependencies.annotation_service)
        self.annotation_page.setObjectName("annotationPage")
        self.live_assistant_page = LiveAssistantPage(self.live_runtime)
        self.replay_page = ReplayPage(dependencies.sessions_root)
        self.recommendation_window = RecommendationFloatWindow(self.live_runtime)
        self.live_assistant_page.compact_mode_requested.connect(
            self.show_compact_recommendation
        )
        self.recommendation_window.open_full_assistant_requested.connect(
            self.show_full_assistant
        )

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
        self.setWindowTitle("大掼蛋智能助手")
        self.resize(1220, 820)
        self.setMinimumSize(980, 700)

    def show_compact_recommendation(self) -> None:
        try:
            rect = self.live_runtime.target_client_rect()
        except Exception:
            rect = None
        if rect is not None:
            self.recommendation_window.place_beside(
                QRect(
                    rect.left,
                    rect.top,
                    rect.width,
                    rect.height,
                )
            )
        self.recommendation_window.show()
        self.recommendation_window.raise_()
        self.showMinimized()

    def show_full_assistant(self) -> None:
        self.recommendation_window.hide()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        self.recommendation_window.hide()
        self.replay_page.shutdown()
        self.live_assistant_page.shutdown()
        super().closeEvent(event)
