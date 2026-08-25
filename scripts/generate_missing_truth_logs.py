"""Stage headless visual TruthLog candidates; publishing is explicit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.visual_truth_generation import (  # noqa: E402
    VisualTruthGenerationService,
    write_visual_truth_generation_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Discover sessions under explicit roots and stage visual TruthLog drafts. "
            "Existing truth_log.json files are never overwritten."
        )
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        action="append",
        required=True,
        help="A source or release sessions directory (repeat for multiple stores).",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        required=True,
        help="Explicit external directory for the central batch report.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique staging ID; omit to generate one.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "After all gates pass, atomically create missing truth_log.json files. "
            "Without this flag the command only stages drafts."
        ),
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help=(
            "Resume by scanning only sessions that currently lack a root "
            "truth_log.json; existing formal logs are not replayed or staged."
        ),
    )
    parser.add_argument(
        "--trusted-staged-reference",
        type=Path,
        action="append",
        default=[],
        help=(
            "Explicit human-trusted staged truth_log.json. Only this exact path "
            "may corroborate an otherwise unresolved visual action; repeat as needed."
        ),
    )
    args = parser.parse_args(argv)

    service = VisualTruthGenerationService()
    run = service.generate(
        args.sessions_root,
        publish=bool(args.publish),
        only_missing=bool(args.missing_only),
        run_id=args.run_id,
        trusted_staged_references=args.trusted_staged_reference,
    )
    report = write_visual_truth_generation_report(args.report_root, run)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
