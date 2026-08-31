"""Fail-closed audit for the no-site Python used to bootstrap release builds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping


SCHEMA = "guandan.bootstrap-python-audit/1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def audit_bootstrap_environment(
    python_root: Path,
    runtime_lock: Path,
    *,
    search_paths: Iterable[str] | None = None,
    loaded_modules: Mapping[str, object] | None = None,
    flags: object | None = None,
) -> dict[str, object]:
    root = python_root.resolve()
    lock_path = runtime_lock.resolve()
    errors: list[dict[str, object]] = []
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        lock = {}
        errors.append({"code": "BOOTSTRAP-RUNTIME-LOCK", "message": type(exc).__name__})
    if not isinstance(lock, dict):
        lock = {}
        errors.append({"code": "BOOTSTRAP-RUNTIME-LOCK", "message": "invalid document"})
    records = lock.get("files")
    locked_files = {
        str(record.get("path") or "").replace("\\", "/")
        for record in records or []
        if isinstance(record, dict)
    }
    if lock.get("schema") != "guandan.python-runtime-lock/1" or not locked_files:
        errors.append({"code": "BOOTSTRAP-RUNTIME-LOCK", "message": "invalid inventory"})

    version_zip = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    allowed_roots = {".", "Lib", "DLLs", version_zip}
    normalized_paths: list[str] = []
    for raw_path in list(sys.path if search_paths is None else search_paths):
        candidate = Path(raw_path or ".").resolve()
        try:
            relative = candidate.relative_to(root).as_posix() or "."
        except ValueError:
            normalized_paths.append(f"<external>/{candidate.name}")
            errors.append(
                {
                    "code": "BOOTSTRAP-SYSPATH-EXTERNAL",
                    "entry_name": candidate.name,
                    "entry_sha256": hashlib.sha256(str(candidate).encode("utf-8")).hexdigest(),
                }
            )
            continue
        normalized_paths.append("<python-root>" if relative == "." else f"<python-root>/{relative}")
        if relative not in allowed_roots:
            errors.append(
                {"code": "BOOTSTRAP-SYSPATH-UNLOCKED", "relative_path": relative}
            )
            continue
        if relative == "Lib" and not any(path.startswith("Lib/") for path in locked_files):
            errors.append({"code": "BOOTSTRAP-SYSPATH-UNLOCKED", "relative_path": relative})
        elif relative == "DLLs" and not any(path.startswith("DLLs/") for path in locked_files):
            errors.append({"code": "BOOTSTRAP-SYSPATH-UNLOCKED", "relative_path": relative})
        elif relative == "." and not any("/" not in path for path in locked_files):
            errors.append({"code": "BOOTSTRAP-SYSPATH-UNLOCKED", "relative_path": relative})
        elif relative == version_zip and candidate.exists() and relative not in locked_files:
            errors.append({"code": "BOOTSTRAP-SYSPATH-UNLOCKED", "relative_path": relative})

    active_flags = sys.flags if flags is None else flags
    required_flags = {
        "isolated": 1,
        "no_site": 1,
        "ignore_environment": 1,
        "no_user_site": 1,
        "safe_path": True,
    }
    observed_flags = {
        name: getattr(active_flags, name, None) for name in required_flags
    }
    for name, expected in required_flags.items():
        if observed_flags[name] != expected:
            errors.append(
                {
                    "code": "BOOTSTRAP-FLAG-MISSING",
                    "flag": name,
                    "expected": expected,
                    "actual": observed_flags[name],
                }
            )

    modules = sys.modules if loaded_modules is None else loaded_modules
    forbidden_modules = [
        name for name in ("site", "sitecustomize", "usercustomize") if name in modules
    ]
    if forbidden_modules:
        errors.append(
            {"code": "BOOTSTRAP-SITE-EXECUTED", "modules": forbidden_modules}
        )
    if root != Path(sys.base_prefix).resolve() or root != Path(sys.prefix).resolve():
        errors.append(
            {
                "code": "BOOTSTRAP-PREFIX-MISMATCH",
                "base_matches": root == Path(sys.base_prefix).resolve(),
                "prefix_matches": root == Path(sys.prefix).resolve(),
            }
        )

    return {
        "schema": SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "implementation": sys.implementation.name,
            "base_root_name": root.name,
        },
        "flags": observed_flags,
        "sys_path": normalized_paths,
        "site_modules_loaded": forbidden_modules,
        "runtime_lock_sha256": (
            hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else None
        ),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_bootstrap_environment(args.python_root, args.runtime_lock)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
