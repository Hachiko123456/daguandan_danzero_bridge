"""Read-only visual/FableDan audit for every session below explicit roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.session_replay_audit import (  # noqa: E402
    SessionReplayAuditService,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay every discovered session into an explicit report directory. "
            "Original session data is read-only."
        )
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        action="append",
        required=True,
        help="Source or release sessions directory; repeat to audit both stores.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Explicit external directory for all replay/audit artifacts.",
    )
    parser.add_argument(
        "--scan-run-id",
        action="append",
        help=(
            "When a canonical truth_log.json is absent, audit this staged "
            "derived/truth_scan_drafts/<id>/truth_log.json candidate. Repeat "
            "to select distinct explicit batches from multiple stores."
        ),
    )
    parser.add_argument("--run-id", help="Optional unique audit output ID.")
    args = parser.parse_args(argv)

    try:
        run = SessionReplayAuditService().audit(
            args.sessions_root,
            output=args.output,
            scan_run_id=args.scan_run_id,
            run_id=args.run_id,
            command=[str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        )
    except Exception as exc:
        print(f"audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(run.summary_path)
    if not run.execution_ok:
        if run.verification_path is not None and run.verification_path.is_file():
            verification = json.loads(run.verification_path.read_text(encoding="utf-8"))
            failed = [name for name, passed in verification.get("checks", {}).items() if not passed]
            print(f"verification failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
