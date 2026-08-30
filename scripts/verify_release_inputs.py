from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.release_lock import ReleaseLockError, verify_release_inputs  # noqa: E402
from daguandan_bridge.storage import atomic_write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = verify_release_inputs(
            project_root=args.project_root,
            wheelhouse_root=args.wheelhouse,
            python_executable=args.python,
        )
    except (ReleaseLockError, OSError, ValueError) as exc:
        print(f"release input verification failed: {exc}", file=sys.stderr)
        return 3
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
