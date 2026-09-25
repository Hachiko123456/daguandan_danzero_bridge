from __future__ import annotations

"""Run the fixed five-session architecture acceptance set."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIVE_SESSIONS = (
    "game_20260816_202912_2fd064",      # healthy full advice path
    "game_20260821_135436_dac2f0",      # correction/desync/terminal path
    "game_20260815_001801_ba5066",      # advice failures
    "game_20260822_142942_e356d4",      # recovery/terminal-history gap
    "game_20260829_114157_a940ae",      # repeated recovery/advice target path
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed five-session live-v2 architecture acceptance set."
    )
    parser.add_argument("--sessions-root", type=Path, default=PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions")
    parser.add_argument("--profile-root", type=Path, default=PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "five-session-acceptance")
    parser.add_argument("--run-id", default="architecture-acceptance-20260921")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from run_listener_regression import main as listener_main

    forwarded = [
        "--sessions-root", str(args.sessions_root),
        "--profile-root", str(args.profile_root),
        "--output", str(args.output),
        "--run-id", str(args.run_id),
        "--workers", str(args.workers),
        "--session", *FIVE_SESSIONS,
        "--include-draft",
        "--include-no-truth",
    ]
    return int(listener_main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())

