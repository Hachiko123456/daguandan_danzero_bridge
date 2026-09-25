from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

IDS = {
    "game_20260822_002135_9c2328", "game_20260829_192802_981e50",
    "game_20260821_200355_81b7ee", "game_20260817_002012_b523b5",
    "game_20260816_203403_00db35", "game_20260816_193531_f94cbf",
    "game_20260816_154444_392ea8", "game_20260815_001801_ba5066",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rescan the hard-coded selected unverified sessions with quiet percent-only progress.",
        epilog=(
            "Default behavior is unchanged: reads session evidence from the Tencent profile "
            "and writes a new batch under reports/session-corpus-validation/unverified-rescans."
        ),
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("data/profiles/tencent_daguandan/sessions"),
        help="Read-only input sessions root (default: %(default)s).",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path("data/profiles/tencent_daguandan"),
        help="Read-only profile root for templates/configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/session-corpus-validation/unverified-rescans"),
        help="Write output batch directory below this root (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Worker count used by the scan service (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from daguandan_bridge.application.session_workbench import inspect_sessions
    from daguandan_bridge.application.unverified_batch_scan import UnverifiedBatchScanService

    root = args.session_root.resolve()
    profile = args.profile_root.resolve()
    out = args.output_root.resolve()
    desc = tuple(item for item in inspect_sessions(root) if item.session_id in IDS)
    if {x.session_id for x in desc} != IDS:
        raise SystemExit("session selection mismatch")
    last = {"percent": -1}

    def progress(sid, done, total, frame, completed, percent):
        if percent != last["percent"]:
            last["percent"] = percent
            print(f"progress={percent}% completed={completed}/8 current={sid}", flush=True)

    result = UnverifiedBatchScanService().scan(
        desc,
        profile_root=profile,
        output_root=out,
        max_workers=args.max_workers,
        on_progress=progress,
    )
    print(json.dumps({
        "output_directory": str(result.output_directory),
        "summary_path": str(result.summary_path),
        "selected": result.selected_count,
        "completed": result.completed_count,
        "failed": result.failed_count,
        "cancelled": result.cancelled,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
