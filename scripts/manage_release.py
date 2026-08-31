from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.release_manager import (  # noqa: E402
    ReleaseManagerError,
    activate_release,
    install_release,
    register_legacy_baseline,
    release_status,
    rollback_release,
    write_baseline_auth,
)
from daguandan_bridge.runtime_layout import RuntimeLayoutError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("install")
    install.add_argument("archive", type=Path)
    install.add_argument("--release-record", type=Path, required=True)
    install.add_argument("--checksum", type=Path, required=True)
    install.add_argument("--baseline", action="store_true")
    install.add_argument("--baseline-auth", type=Path)
    activate = sub.add_parser("activate")
    activate.add_argument("release_id")
    legacy = sub.add_parser("register-legacy-baseline")
    legacy.add_argument("directory", type=Path)
    legacy.add_argument("--baseline-auth", type=Path, required=True)
    auth = sub.add_parser("create-baseline-auth")
    auth.add_argument("directory", type=Path)
    auth.add_argument("--output", type=Path, required=True)
    auth.add_argument("--approve-stable-baseline", action="store_true")
    sub.add_parser("rollback")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            if args.baseline and args.baseline_auth is None:
                parser.error("--baseline requires --baseline-auth")
            if not args.baseline and args.baseline_auth is not None:
                parser.error("--baseline-auth requires --baseline")
            result = install_release(
                args.archive,
                release_record_path=args.release_record,
                checksum_path=args.checksum,
                runtime_root=args.runtime_root,
                baseline=args.baseline,
                baseline_auth_path=args.baseline_auth,
            ).to_dict()
        elif args.command == "activate":
            result = activate_release(args.release_id, runtime_root=args.runtime_root)
        elif args.command == "register-legacy-baseline":
            result = register_legacy_baseline(
                args.directory,
                baseline_auth_path=args.baseline_auth,
                runtime_root=args.runtime_root,
            ).to_dict()
        elif args.command == "create-baseline-auth":
            result = write_baseline_auth(
                args.directory,
                args.output,
                approved=args.approve_stable_baseline,
            )
        elif args.command == "rollback":
            result = rollback_release(runtime_root=args.runtime_root)
        else:
            result = release_status(runtime_root=args.runtime_root)
    except (ReleaseManagerError, RuntimeLayoutError, OSError, ValueError) as exc:
        print(f"release manager error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
