from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daguandan_bridge.dpi import enable_windows_dpi_awareness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="腾讯大掼蛋智能助手：图片标记、模板管理、回放与模型建议。",
    )
    parser.add_argument(
        "--fabledan-fixed-benchmark",
        action="store_true",
        help="运行固定的 FableDan 无进贡基准评测，然后写入模型目录。",
    )
    args = parser.parse_args(argv)
    if args.fabledan_fixed_benchmark:
        from daguandan_bridge.application.fabledan_benchmark import (
            FableDanBenchmarkService,
        )

        result = FableDanBenchmarkService().run_fixed()
        print(json.dumps(result.payload, ensure_ascii=False, indent=2))
        print(f"结果文件：{result.output_path}")
        return 0
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
