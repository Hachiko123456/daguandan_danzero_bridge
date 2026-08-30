from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.frozen_bundle_audit import (  # noqa: E402
    FrozenBundleAuditError,
    audit_frozen_bundle,
    collect_pyinstaller_provenance,
    write_native_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        provenance = collect_pyinstaller_provenance(args.work_root)
        result = audit_frozen_bundle(
            args.bundle_root,
            provenance=provenance,
            allowed_venv_root=args.venv_root,
            allowed_python_root=args.python_root,
            project_root=args.project_root,
        )
        write_native_audit(args.output, result)
        print(json.dumps(result.document["summary"], ensure_ascii=False))
        return 0 if result.ok else 2
    except (FrozenBundleAuditError, OSError, ValueError) as exc:
        print(f"native bundle audit failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
