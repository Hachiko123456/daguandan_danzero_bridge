from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from ..application_icon import install_application_icon
from ..bootstrap import build_application_dependencies
from .main_window import DaguandanBridgeWindow


def main(argv: Sequence[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("大掼蛋智能助手")
    icon_result = install_application_icon(app)
    window = DaguandanBridgeWindow(dependencies=build_application_dependencies())
    install_application_icon(window=window, result=icon_result)
    window.show()
    return int(app.exec())
