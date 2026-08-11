from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .application.dataset import DatasetUseCase


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guandan-dataset")
    parser.add_argument("command", choices=("validate", "export"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--root", action="store_true", help="path contains multiple session directories")
    args = parser.parse_args(argv)
    use_case = DatasetUseCase()
    try:
        if args.command == "validate":
            result = use_case.validate_root(args.path) if args.root else use_case.validate_session(args.path)
            print(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0 if result.valid else 2
        result = use_case.export_root(args.path) if args.root else use_case.export_session(args.path)
        print(json.dumps({"ok": True, **result.to_dict()}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
