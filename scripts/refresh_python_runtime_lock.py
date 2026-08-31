"""Regenerate the complete CPython base-runtime inventory lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.build_manifest import sha256_file  # noqa: E402
from daguandan_bridge.release_lock import (  # noqa: E402
    create_python_runtime_lock,
)
from daguandan_bridge.storage import atomic_write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "python_runtime.lock.json",
    )
    parser.add_argument(
        "--toolchain",
        type=Path,
        default=PROJECT_ROOT / "release_toolchain.lock.json",
    )
    args = parser.parse_args(argv)
    completed = subprocess.run(
        [str(args.python.resolve()), "-I", "-c", "import sys; print(sys.base_prefix)"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise SystemExit("could not resolve CPython base prefix")
    base_root = Path(completed.stdout.strip()).resolve()
    inventory = create_python_runtime_lock(base_root)
    output = args.output.resolve()
    toolchain_path = args.toolchain.resolve()
    if output.parent != toolchain_path.parent:
        raise SystemExit("runtime inventory and toolchain locks must share one directory")
    atomic_write_json(output, inventory)
    toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
    if not isinstance(toolchain, dict):
        raise SystemExit("toolchain lock must be a JSON object")
    records = inventory["files"]
    python_dll = next(
        item for item in records if item["path"].casefold() == "python312.dll"
    )
    toolchain["schema"] = "guandan.release-toolchain-lock/3"
    toolchain["python_runtime"] = {
        "inventory_lock": {
            "path": output.name,
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "aggregate_sha256": inventory["aggregate_sha256"],
        "file_count": inventory["file_count"],
        "bytes": inventory["bytes"],
        "python_dll": python_dll,
    }
    locks = toolchain.setdefault("locks", {})
    if not isinstance(locks, dict):
        raise SystemExit("toolchain locks entry must be a JSON object")
    locks["python_runtime_lock_sha256"] = sha256_file(output)
    toolchain["acquisition"] = {
        "provider": "python.org",
        "artifact": "python-3.12.0-amd64.exe",
        "url": "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe",
        "install_scope": "per-user CPython 3.12.0 x64",
        "verification": (
            "Verify the official installer SHA256 below, install to a new directory, "
            "then regenerate and review the complete python_runtime.lock.json inventory."
        ),
        "sha256": "c6bdf93f4b2de6dfa1a3a847e7c24ae10edf7f6318653d452cd4381415700ada",
        "bytes": 26507904,
    }
    atomic_write_json(toolchain_path, toolchain)
    print(
        json.dumps(
            {
                "runtime_lock": str(output),
                "file_count": inventory["file_count"],
                "bytes": inventory["bytes"],
                "aggregate_sha256": inventory["aggregate_sha256"],
                "toolchain": str(toolchain_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
