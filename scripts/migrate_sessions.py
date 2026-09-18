from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.sessions_migration import (  # noqa: E402
    migrate_sessions, rollback_sessions, sessions_migration_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安全迁移掼蛋 sessions 证据目录")
    parser.add_argument("command", choices=("migrate", "rollback", "status"))
    parser.add_argument("--profiles-root", type=Path, default=PROJECT_ROOT / "data" / "profiles")
    parser.add_argument("--profile", default="tencent_daguandan")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "migrate":
            if args.target is None:
                parser.error("migrate 必须指定 --target，例如 D:\\sessions")
            result = migrate_sessions(
                profiles_root=args.profiles_root,
                profile_name=args.profile,
                source_root=args.source,
                target_root=args.target,
            ).to_dict()
        elif args.command == "rollback":
            result = rollback_sessions(
                profiles_root=args.profiles_root, profile_name=args.profile
            ).to_dict()
        else:
            result = sessions_migration_status(
                profiles_root=args.profiles_root, profile_name=args.profile
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
