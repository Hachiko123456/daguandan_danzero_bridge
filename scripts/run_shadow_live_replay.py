"""Run one recorded session through the asynchronous Shadow Live protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.shadow_live_replay import (  # noqa: E402
    FaultProfile,
    ShadowLiveReplayConfig,
    ShadowLiveReplayRunner,
)

_RECOMMENDED = (
    "game_20260814_004447_aab3dc（源码 canonical 稳定基准）",
    "game_20260823_125412_6b9274（发布数据结构）",
    "game_20260815_003925_66b329（已知收尾断链）",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="以生产异步协议执行 Qt-free 1× Shadow Live 回放。",
        epilog="推荐代表性会话（仅建议，不限制选择）：" + "；".join(_RECOMMENDED),
    )
    parser.add_argument("--session", type=Path, required=True, help="显式选择一个源 session 目录。")
    parser.add_argument("--output", type=Path, required=True, help="源 session 外的报告根目录。")
    parser.add_argument("--run-id", help="可重复引用的运行目录名。")
    parser.add_argument("--fault-profile", type=Path, help="FaultProfile JSON 文件。")
    parser.add_argument("--seed", type=int, default=0, help="故障计划随机种子，默认 0。")
    parser.add_argument("--time-scale", type=float, default=1.0, help="调度倍率，默认严格 1×；短测可显式加速。")
    parser.add_argument("--start-frame", type=int, help="片段起始 frame_index（含）。")
    parser.add_argument("--end-frame", type=int, help="片段结束 frame_index（含）。")
    parser.add_argument("--max-frames", type=int, help="最多消费的源帧数。")
    parser.add_argument("--drain-timeout", type=float, default=30.0, help="识别和建议 worker drain 秒数。")
    parser.add_argument("--baseline", type=Path, help="第一阶段 all_session_audit.json 或逐局 summary。")
    parser.add_argument("--compare-summary", type=Path, help="同计划上一次 Shadow summary.json。")
    parser.add_argument("--expected-plan-sha256", help="要求本次 fault plan 匹配的 SHA-256。")
    args = parser.parse_args(argv)

    try:
        profile = FaultProfile()
        if args.fault_profile is not None:
            raw = json.loads(args.fault_profile.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("fault profile must be a JSON object")
            profile = FaultProfile.from_dict(raw)
        result = ShadowLiveReplayRunner().run(
            ShadowLiveReplayConfig(
                session=args.session,
                output=args.output,
                fault_profile=profile,
                seed=args.seed,
                time_scale=args.time_scale,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                max_frames=args.max_frames,
                drain_timeout_sec=args.drain_timeout,
                run_id=args.run_id,
                baseline=args.baseline,
                compare_summary=args.compare_summary,
                expected_plan_sha256=args.expected_plan_sha256,
            )
        )
    except Exception as exc:
        print(f"shadow live failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(result.summary_path)
    return 0 if result.execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
