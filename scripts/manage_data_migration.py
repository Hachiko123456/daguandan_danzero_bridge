"""Migrate or restore frozen user-data generations with JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.portable_data_migration import (  # noqa: E402
    migrate_portable_data,
    restore_portable_migration,
)
from daguandan_bridge.runtime_layout import (  # noqa: E402
    RuntimeLayoutError,
    resolve_runtime_layout,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser("migrate")
    migrate.add_argument("portable_root", type=Path)
    restore = commands.add_parser("restore")
    restore.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    layout = resolve_runtime_layout(
        frozen=True,
        bundle_root=args.bundle_root,
        environ={"DAGUANDAN_DATA_ROOT": str(args.runtime_root.resolve())},
    )
    try:
        result = (
            migrate_portable_data(args.portable_root, layout=layout)
            if args.command == "migrate"
            else restore_portable_migration(args.receipt, layout=layout)
        )
    except (RuntimeLayoutError, OSError, ValueError) as exc:
        print(f"data migration error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
