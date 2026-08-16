from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ..bootstrap import build_application_dependencies
from ..config import PROJECT_ROOT
from .main_window import DaguandanBridgeWindow


def main(argv: Sequence[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("大掼蛋智能助手")
    app.setWindowIcon(QIcon(str(PROJECT_ROOT / "app.ico")))
    window = DaguandanBridgeWindow(dependencies=build_application_dependencies())
    window.show()
    return int(app.exec())
