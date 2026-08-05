from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daguandan_bridge.dpi import enable_windows_dpi_awareness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="腾讯大掼蛋截图录制与 DanZero 手动状态桥接器。",
    )
    parser.parse_args(argv)
    dpi_status = enable_windows_dpi_awareness()
    if not dpi_status.success:
        raise RuntimeError(
            "无法启用 Windows Per-Monitor DPI 感知："
            f"当前状态 {dpi_status.awareness}，方法 {dpi_status.method}"
        )
    from daguandan_bridge.gui.app import main as gui_main

    return int(gui_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
