"""Validate AVI recordings without trusting historical session logs.

Examples:
  python scripts/validate_session_corpus.py --target D:/sessions --mode smoke --output C:/DaguandanValidation
  python scripts/validate_session_corpus.py --target D:/sessions/game_x --mode single --output C:/DaguandanValidation --profile-root data/profiles/tencent_daguandan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.session_corpus_validation import (  # noqa: E402
    SessionCorpusValidationService,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run read-only AVI-first scanning, write external draft TruthLog "
            "artifacts from canonical scan actions, and optionally compare selected sessions."
        )
    )
    parser.add_argument("--target", type=Path, required=True, help="One session or a directory containing sessions.")
    parser.add_argument("--mode", choices=("smoke", "single", "corpus", "qualify"), default="corpus")
    parser.add_argument("--output", type=Path, required=True, help="External output directory; never place it below target.")
    parser.add_argument(
        "--profile-root",
        type=Path,
        help="Explicit selected profile directory (for example data/profiles/tencent_daguandan).",
    )
    parser.add_argument("--trusted-session", action="append", default=[], help="Session ID or path accepted as semantic truth for this run; source remains unchanged.")
    parser.add_argument("--scan-run-id", action="append", default=[], help="Explicit staged visual TruthLog run ID; repeat as needed.")
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Run inventory/preflight only; do not decode AVI files or produce scan/draft artifacts.",
    )
    parser.add_argument("--run-id", help="Stable output directory name for CI or repeated comparisons.")
    args = parser.parse_args(argv)
    try:
        run = SessionCorpusValidationService().validate(
            args.target,
            mode=args.mode,
            output=args.output,
            run_id=args.run_id,
            profile_root=args.profile_root,
            trusted_sessions=args.trusted_session,
            scan_run_id=args.scan_run_id,
            execute_replay=not args.no_replay,
            command=[str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(run.summary_path)
    return 0 if run.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
