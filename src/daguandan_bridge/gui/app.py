from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from ..bootstrap import build_application_dependencies
from .main_window import DaguandanBridgeWindow


def main(argv: Sequence[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("大掼蛋 DanZero 桥接器")
    window = DaguandanBridgeWindow(dependencies=build_application_dependencies())
    window.show()
    return int(app.exec())
