"""Read-only validation for two historical double-down session timelines.

The original terminal events predate ``remaining_cards`` snapshots.  This
script reads each JSONL timeline, appends an in-memory terminal snapshot, and
asserts the new outcome inference without writing either source session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from daguandan_bridge.application.fabledan_training_data import _infer_outcome
from daguandan_bridge.live.session_store import read_json_lines


_CASES = {
    "game_20260814_004447_aab3dc": (
        {"self": 1, "right": 0, "opposite": 10, "left": 0},
        ["right", "left", "self", "opposite"],
    ),
    "game_20260815_014247_14ec80": (
        {"self": 6, "right": 0, "opposite": 2, "left": 0},
        ["right", "left", "opposite", "self"],
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only validation for historical terminal placement inference cases.",
        epilog=(
            "Reads timeline.jsonl from the selected historical sessions, appends only an in-memory "
            "terminal snapshot, and writes no files."
        ),
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions",
        help="Read-only sessions root containing the historical timelines (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sessions = args.sessions_root
    for session_id, (counts, expected_order) in _CASES.items():
        timeline = read_json_lines(sessions / session_id / "timeline.jsonl")
        outcome = _infer_outcome(
            [
                *timeline,
                {
                    "event_type": "game_end_detected",
                    "payload": {"remaining_cards": dict(counts)},
                },
            ]
        )
        assert outcome.get("status") == "complete", outcome
        assert outcome.get("finish_order") == expected_order, outcome
        print(f"{session_id}: {' > '.join(expected_order)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
