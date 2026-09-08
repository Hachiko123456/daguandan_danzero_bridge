"""Create a deterministic, read-only session catalog and TruthLog case index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.session_dataset_catalog import (  # noqa: E402
    SessionDatasetCatalogBuilder,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Index sessions and emit external catalog/cases/coverage files. "
            "The source session tree is never modified and video is not copied."
        )
    )
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hash-video",
        action="store_true",
        help="Hash AVI files too (slower; default only records existence and size).",
    )
    args = parser.parse_args(argv)
    try:
        result = SessionDatasetCatalogBuilder().build(
            args.sessions_root,
            args.output,
            hash_video=bool(args.hash_video),
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(result.output_dir),
                "catalog": str(result.catalog_path),
                "cases": str(result.cases_path),
                "coverage_gaps": str(result.gaps_path),
                "session_count": result.session_count,
                "truth_session_count": result.truth_session_count,
                "draft_truth_session_count": result.draft_truth_session_count,
                "verified_truth_session_count": result.verified_truth_session_count,
                "case_count": result.case_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
