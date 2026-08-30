from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daguandan_bridge.startup_diagnostics import (
    initialize_startup_diagnostics,
    record_startup_event,
)


STARTUP_DIAGNOSTICS = initialize_startup_diagnostics()
record_startup_event("entrypoint_loaded", {"entrypoint": "run.py"})

from daguandan_bridge.runtime_identity import write_runtime_identity_snapshot


RUNTIME_IDENTITY_SNAPSHOT = write_runtime_identity_snapshot(
    STARTUP_DIAGNOSTICS.run_directory
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="腾讯大掼蛋智能助手：图片标记、模板管理、回放与模型建议。",
    )
    parser.add_argument(
        "--fabledan-fixed-benchmark",
        action="store_true",
        help="运行固定的 FableDan 无进贡基准评测，然后写入模型目录。",
    )
    parser.add_argument(
        "--simulated-game-window-config",
        type=Path,
        help="使用 JSON 配置启动第三阶段可见模拟游戏窗口。",
    )
    parser.add_argument(
        "--window-e2e-validation-config",
        type=Path,
        help="使用 JSON 配置运行第三阶段窗口端到端验证。",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="在启动图形界面前运行安装、资源和依赖自检。",
    )
    parser.add_argument(
        "--doctor-output",
        type=Path,
        help="将 doctor JSON 另存到指定路径。",
    )
    parser.add_argument(
        "--_doctor-import-probe",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    selected_modes = sum(
        (
            bool(args.fabledan_fixed_benchmark),
            args.simulated_game_window_config is not None,
            args.window_e2e_validation_config is not None,
            bool(args.doctor),
            args._doctor_import_probe is not None,
        )
    )
    if selected_modes > 1:
        parser.error("基准、模拟窗口、窗口 E2E 验证和 doctor 模式不能同时启用")
    if args.doctor_output is not None and not (
        args.doctor or args._doctor_import_probe is not None
    ):
        parser.error("--doctor-output 只能与 --doctor 一起使用")
    if args._doctor_import_probe is not None:
        from daguandan_bridge.doctor import run_import_probe

        return int(run_import_probe(args._doctor_import_probe, args.doctor_output))
    if args.doctor:
        from daguandan_bridge.doctor import run_doctor

        return int(run_doctor(args.doctor_output))
    if args.fabledan_fixed_benchmark:
        from daguandan_bridge.application.fabledan_benchmark import (
            FableDanBenchmarkService,
        )

        result = FableDanBenchmarkService().run_fixed()
        print(json.dumps(result.payload, ensure_ascii=False, indent=2))
        print(f"结果文件：{result.output_path}")
        return 0
    from daguandan_bridge.dpi import enable_windows_dpi_awareness

    dpi_status = enable_windows_dpi_awareness()
    if not dpi_status.success:
        raise RuntimeError(
            "无法启用 Windows Per-Monitor DPI 感知："
            f"当前状态 {dpi_status.awareness}，方法 {dpi_status.method}"
        )
    if args.simulated_game_window_config is not None:
        from daguandan_bridge.application.simulated_game_window import (
            run_simulated_game_window,
        )

        return int(run_simulated_game_window(args.simulated_game_window_config))
    if args.window_e2e_validation_config is not None:
        from daguandan_bridge.application.window_e2e_validation import (
            run_window_e2e_validation,
        )

        return int(run_window_e2e_validation(args.window_e2e_validation_config))
    from daguandan_bridge.gui.app import main as gui_main

    return int(gui_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
