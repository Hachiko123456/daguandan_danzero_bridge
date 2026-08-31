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
        "--benchmark-output",
        type=Path,
        help="固定基准结果的精确 JSON 输出路径；只可与 --fabledan-fixed-benchmark 一起使用。",
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
        "--migrate-portable-data",
        type=Path,
        help=(
            "显式复制旧便携版的 data/profiles 到新的版本化用户目录；"
            "旧目录保持只读且不会被移动或覆盖。"
        ),
    )
    parser.add_argument(
        "--_doctor-import-probe",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--export-support",
        type=Path,
        metavar="ZIP",
        help="导出脱敏支持包；默认不包含截图，需显式 --include-support-images。",
    )
    parser.add_argument(
        "--support-run-dir",
        type=Path,
        help="指定诊断 run 目录；省略时自动选择最近的证据 run。",
    )
    parser.add_argument(
        "--support-session-dir",
        type=Path,
        help="可选的已封存 session 目录。",
    )
    parser.add_argument(
        "--include-support-images",
        action="store_true",
        help="明确同意把原始/标准化截图放入支持包。",
    )
    parser.add_argument(
        "--include-support-trace",
        action="store_true",
        help="明确同意把识别 trace 放入支持包。",
    )
    parser.add_argument(
        "--repro-support",
        type=Path,
        metavar="ZIP",
        help="使用支持包运行确定性复现。",
    )
    parser.add_argument("--repro-output", type=Path, metavar="JSON")
    parser.add_argument("--repro-truth", type=Path, metavar="JSON")
    parser.add_argument("--expected-level", type=str)
    parser.add_argument(
        "--expected-hand",
        type=str,
        help="独立真值手牌，使用逗号分隔，例如 2S,3H,...；不会修改支持包。",
    )
    parser.add_argument("--repro-repeats", type=int, default=20)
    parser.add_argument(
        "--repro-role",
        choices=("reference", "candidate", "unspecified"),
        default="unspecified",
        help="标记复现报告在修复门禁中的角色。",
    )
    parser.add_argument(
        "--repro-deterministic",
        action="store_true",
        help="复现时固定随机种子、OpenCV 单线程并关闭 OpenCL。",
    )
    parser.add_argument(
        "--_repro-probe",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--compare-repro",
        nargs=2,
        type=Path,
        metavar=("REFERENCE", "CANDIDATE"),
        help="比较两个 repro-report 并输出 guandan.repro-gate/1。",
    )
    parser.add_argument("--repro-gate-output", type=Path, metavar="JSON")
    parser.add_argument(
        "--annotate-repro-truth",
        type=Path,
        metavar="SUPPORT_ZIP",
        help="为支持包创建外部真值 annotation。",
    )
    parser.add_argument("--truth-output", type=Path, metavar="JSON")
    parser.add_argument(
        "--truth-input-sha256",
        type=str,
        help="把独立真值绑定到唯一的标准化输入像素 SHA256。",
    )
    parser.add_argument(
        "--truth-frame-seq",
        type=int,
        help="把独立真值绑定到 incident 中实际送达 opening gate 的帧序号。",
    )
    args = parser.parse_args(argv)
    selected_modes = sum(
        (
            bool(args.fabledan_fixed_benchmark),
            args.simulated_game_window_config is not None,
            args.window_e2e_validation_config is not None,
            bool(args.doctor),
            args.migrate_portable_data is not None,
            args._doctor_import_probe is not None,
            args.export_support is not None,
            args.repro_support is not None,
            args._repro_probe is not None,
            args.compare_repro is not None,
            args.annotate_repro_truth is not None,
        )
    )
    if selected_modes > 1:
        parser.error(
            "基准、模拟窗口、窗口 E2E、doctor、迁移、支持导出和复现模式不能同时启用"
        )
    if args.doctor_output is not None and not (
        args.doctor or args._doctor_import_probe is not None
    ):
        parser.error("--doctor-output 只能与 --doctor 一起使用")
    if args.benchmark_output is not None and not args.fabledan_fixed_benchmark:
        parser.error("--benchmark-output 只能与 --fabledan-fixed-benchmark 一起使用")
    if (
        args.truth_input_sha256 is not None or args.truth_frame_seq is not None
    ) and not (
        args.annotate_repro_truth is not None
        or args.repro_support is not None
        or args._repro_probe is not None
    ):
        parser.error("真值输入选择器只能用于真值标注或支持包复现")
    if args.repro_truth is not None and (
        args.truth_input_sha256 is not None or args.truth_frame_seq is not None
    ):
        parser.error("外部 --repro-truth 已包含输入绑定，不能被命令行选择器覆盖")
    if args._doctor_import_probe is not None:
        from daguandan_bridge.doctor import run_import_probe

        return int(run_import_probe(args._doctor_import_probe, args.doctor_output))
    if args.compare_repro is not None:
        from daguandan_bridge.support_repro import compare_repro_reports

        report = compare_repro_reports(
            args.compare_repro[0],
            args.compare_repro[1],
            output_path=args.repro_gate_output,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("status") == "PASS" else 2
    if args.annotate_repro_truth is not None:
        if args.truth_output is None:
            parser.error("--annotate-repro-truth 必须同时指定 --truth-output")
        if (args.truth_input_sha256 is None) == (args.truth_frame_seq is None):
            parser.error(
                "真值必须且只能指定 --truth-input-sha256 或 --truth-frame-seq 之一"
            )
        from daguandan_bridge.support_repro import write_truth_annotation

        expected_hand = _parse_expected_hand(args.expected_hand)
        truth = write_truth_annotation(
            args.truth_output,
            args.annotate_repro_truth,
            input_pixel_sha256=args.truth_input_sha256,
            input_frame_seq=args.truth_frame_seq,
            expected_level=args.expected_level,
            expected_hand=expected_hand,
        )
        print(json.dumps(truth, ensure_ascii=False, indent=2))
        return 0
    if args.repro_support is not None or args._repro_probe is not None:
        # A frozen repro run may use a brand-new isolated data root.  Seed its
        # immutable profile resources before constructing the production
        # recognizer, exactly as the GUI path does.
        from daguandan_bridge.runtime_layout import ensure_runtime_layout

        ensure_runtime_layout()
        from daguandan_bridge.support_repro import (
            reproduce_support_bundle,
            reproduce_support_suite,
        )

        support_path = args.repro_support or args._repro_probe
        output_path = args.repro_output
        expected_hand = _parse_expected_hand(args.expected_hand)
        inline_truth = bool(args.expected_level is not None or expected_hand is not None)
        if (
            args._repro_probe is None
            and args.repro_truth is None
            and inline_truth
            and (args.truth_input_sha256 is None) == (args.truth_frame_seq is None)
        ):
            parser.error(
                "命令行独立真值必须且只能指定 --truth-input-sha256 或 --truth-frame-seq 之一"
            )
        if args._repro_probe is not None:
            report = reproduce_support_bundle(
                support_path,
                output_path=output_path,
                truth_path=args.repro_truth,
                expected_level=args.expected_level,
                expected_hand=expected_hand,
                truth_input_pixel_sha256=args.truth_input_sha256,
                truth_input_frame_seq=args.truth_frame_seq,
                repeats=args.repro_repeats,
                deterministic=True,
                role=args.repro_role,
            )
        else:
            report = reproduce_support_suite(
                support_path,
                output_path=output_path,
                truth_path=args.repro_truth,
                expected_level=args.expected_level,
                expected_hand=expected_hand,
                truth_input_pixel_sha256=args.truth_input_sha256,
                truth_input_frame_seq=args.truth_frame_seq,
                repeats=args.repro_repeats,
                role=args.repro_role,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args._repro_probe is None:
            suite_gate = report.get("suite_gate")
            if args.repro_role == "candidate":
                truth = report.get("truth")
                if not isinstance(truth, dict) or truth.get("all_correct") is not True:
                    return 2
                if not isinstance(suite_gate, dict) or suite_gate.get("status") != "PASS":
                    return 2
            if not isinstance(suite_gate, dict) or suite_gate.get("status") != "PASS":
                return 2
        return 0
    if args.export_support is not None:
        from daguandan_bridge.support_export import (
            SupportExportRequest,
            export_collected_support_bundle,
        )

        runtime_layout = None
        try:
            from daguandan_bridge.runtime_layout import resolve_runtime_layout

            runtime_layout = resolve_runtime_layout()
        except Exception:
            runtime_layout = None
        current = STARTUP_DIAGNOSTICS.run_directory
        evidence_run = args.support_run_dir or _latest_evidence_run(
            STARTUP_DIAGNOSTICS.root,
            current,
        )
        result = export_collected_support_bundle(
            SupportExportRequest(
                destination=args.export_support,
                diagnostics_run_directory=current,
                evidence_run_directory=evidence_run,
                bundle_root=(runtime_layout.bundle_root if runtime_layout else PROJECT_ROOT),
                session_directory=args.support_session_dir,
                include_frames=bool(args.include_support_images),
                include_roi=bool(args.include_support_images),
                include_recognition_trace=bool(args.include_support_trace),
            )
        )
        print(json.dumps(result.bundle.manifest, ensure_ascii=False, indent=2))
        return 0
    if args.doctor:
        from daguandan_bridge.doctor import run_doctor

        return int(run_doctor(args.doctor_output))
    if args.migrate_portable_data is not None:
        from daguandan_bridge.portable_data_migration import migrate_portable_data

        result = migrate_portable_data(args.migrate_portable_data)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    from daguandan_bridge.runtime_layout import ensure_runtime_layout

    ensure_runtime_layout()
    if args.fabledan_fixed_benchmark:
        from daguandan_bridge.application.fabledan_benchmark import (
            FableDanBenchmarkService,
        )

        result = FableDanBenchmarkService().run_fixed(
            output_path=args.benchmark_output,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
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


def _parse_expected_hand(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--expected-hand cannot be empty")
    return values


def _latest_evidence_run(root: Path | None, current: Path | None) -> Path | None:
    """Select the newest run containing an opening incident, safely."""

    if current is not None and (current / "opening" / "incidents").is_dir():
        if any((current / "opening" / "incidents").iterdir()):
            return current
    if root is None:
        return current
    runs = root / "runs"
    if not runs.is_dir():
        return current
    candidates: list[tuple[int, Path]] = []
    for path in runs.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        incidents = path / "opening" / "incidents"
        if not incidents.is_dir() or not any(incidents.iterdir()):
            continue
        try:
            stamp = max(item.stat().st_mtime_ns for item in incidents.iterdir())
        except (OSError, ValueError):
            continue
        candidates.append((stamp, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else current


if __name__ == "__main__":
    raise SystemExit(main())
