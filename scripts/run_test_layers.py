from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest


LAYERS: tuple[str, ...] = (
    "unit",
    "contract",
    "visual_fixture",
    "integration",
    "replay",
    "windows_integration",
    "slow",
    "benchmark",
    "legacy_compat",
    "data_quality",
)
REQUIRED_LAYERS: tuple[str, ...] = (
    "unit",
    "contract",
    "visual_fixture",
    "integration",
    "replay",
    "data_quality",
)
DEFAULT_EXCLUDED: frozenset[str] = frozenset(
    {"windows_integration", "benchmark", "legacy_compat"}
)
DEFAULT_SUMMARY_PATH = Path(tempfile.gettempdir()) / "daguandan-test-layers-summary.json"
SUMMARY_SCHEMA = "daguandan.test-layers-summary/1"
PLUGIN_SUMMARY_ENV = "DAGUAND_TEST_LAYER_SUMMARY"
PYTEST_COLLECTION_IGNORE = "--ignore=codex_project_analysis"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _layer_paths(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    root = root or _project_root()
    with (root / "pyproject.toml").open("rb") as handle:
        table = tomllib.load(handle).get("tool", {}).get("daguandan", {}).get("test_layers", {})
    result: dict[str, tuple[str, ...]] = {}
    for layer, paths in table.items():
        if layer not in LAYERS:
            raise ValueError(f"unknown configured test layer: {layer}")
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise ValueError(f"test layer {layer!r} must contain a list of paths")
        result[layer] = tuple(path.replace("\\", "/").rstrip("/") for path in paths)
    return result


def _relative_test_path(item: Any, root: Path) -> str:
    return Path(str(item.fspath)).resolve().relative_to(root).as_posix()


def _path_matches(relative_path: str, configured_path: str) -> bool:
    return relative_path == configured_path or relative_path.startswith(configured_path + "/")


def _known_item_markers(item: Any) -> set[str]:
    return {marker.name for marker in item.iter_markers() if marker.name in LAYERS}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Bridge legacy unmarked tests into explicit layers without editing them.

    Only the exact paths/directories declared in ``[tool.daguandan.test_layers]``
    receive a functional layer marker. Any other unmarked test is classified as
    ``legacy_compat`` so it cannot silently enter the fast/default layers.
    """
    root = Path(config.rootpath).resolve()
    mapping = _layer_paths(root)
    for item in items:
        relative_path = _relative_test_path(item, root)
        matched = {
            layer
            for layer, paths in mapping.items()
            if any(_path_matches(relative_path, configured) for configured in paths)
        }
        existing = _known_item_markers(item)
        for layer in sorted(matched):
            item.add_marker(layer)
        if not matched and not existing:
            item.add_marker("legacy_compat")


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    summary_path = os.environ.get(PLUGIN_SUMMARY_ENV)
    if not summary_path:
        return
    selected_count = int(getattr(session, "testscollected", 0) or 0)
    if selected_count == 0:
        selected_count = len(getattr(session, "items", ()))
    payload = {
        "selected_count": selected_count,
        "empty_layer": selected_count == 0,
    }
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if selected_count == 0 and exitstatus == 0:
        session.exitstatus = 5


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _marker_expression(layers: Sequence[str]) -> str:
    return " or ".join(layers)


def _active_layers(requested: Sequence[str] | None) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    return [layer for layer in LAYERS if layer not in DEFAULT_EXCLUDED]


def _summary_path(value: str | None) -> Path:
    path = Path(value) if value else DEFAULT_SUMMARY_PATH
    return path if path.is_absolute() else Path.cwd() / path


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pytest tests selected by marker layers.")
    parser.add_argument(
        "--layer",
        action="append",
        choices=LAYERS,
        help="Marker layer to run; repeat to select multiple layers.",
    )
    parser.add_argument("--list", action="store_true", help="List available layers and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pytest command and write a summary without running pytest.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Argument forwarded to pytest; repeat for multiple arguments.",
    )
    parser.add_argument(
        "--summary-json",
        "--summary",
        dest="summary_json",
        metavar="PATH",
        help=f"External JSON summary path (default: {DEFAULT_SUMMARY_PATH}).",
    )
    return parser


def _print_command(command: Sequence[str]) -> None:
    print("pytest command:")
    print(" ".join(repr(argument) for argument in command))


def _list_layers() -> None:
    print("Available pytest layers:")
    for layer in LAYERS:
        default = "no" if layer in DEFAULT_EXCLUDED else "yes"
        print(f"  {layer:20} default={default} marker={layer}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list:
        _list_layers()
        return 0

    layers = _active_layers(args.layer)
    expression = _marker_expression(layers)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "run_test_layers",
        PYTEST_COLLECTION_IGNORE,
        "-m",
        expression,
        *args.pytest_arg,
    ]
    summary_file = _summary_path(args.summary_json)
    started = _utc_now()
    start_monotonic = time.monotonic()
    if args.dry_run:
        _print_command(command)

    pytest_exit_code: int | None = None
    launch_error: str | None = None
    selected_count: int | None = None
    empty_layer = False
    status = "dry_run" if args.dry_run else "error"
    plugin_summary = summary_file.with_suffix(summary_file.suffix + ".plugin")
    if not args.dry_run:
        try:
            if plugin_summary.exists():
                plugin_summary.unlink()
            env = os.environ.copy()
            scripts_path = str(Path(__file__).resolve().parent)
            env["PYTHONPATH"] = scripts_path + os.pathsep + env.get("PYTHONPATH", "")
            env[PLUGIN_SUMMARY_ENV] = str(plugin_summary)
            completed = subprocess.run(command, cwd=_project_root(), env=env)
            pytest_exit_code = completed.returncode
            status = "passed" if pytest_exit_code == 0 else "failed"
        except OSError as exc:
            launch_error = f"could not launch pytest: {exc}"
            print(launch_error, file=sys.stderr)

        if plugin_summary.exists():
            try:
                plugin_data = json.loads(plugin_summary.read_text(encoding="utf-8"))
                selected_count = int(plugin_data.get("selected_count", 0))
                empty_layer = bool(plugin_data.get("empty_layer", selected_count == 0))
            except (OSError, ValueError, TypeError) as exc:
                launch_error = f"invalid layer bridge summary: {exc}"
                status = "bridge_error"
                empty_layer = True
            finally:
                try:
                    plugin_summary.unlink()
                except OSError:
                    pass
        else:
            launch_error = "pytest layer bridge did not report collection results"
            status = "bridge_error"
            empty_layer = True

        if empty_layer:
            status = "empty_layer"
            if pytest_exit_code in (None, 0):
                pytest_exit_code = 5

    summary: dict[str, object] = {
        "schema": SUMMARY_SCHEMA,
        "status": status,
        "layers": layers,
        "marker_expression": expression,
        "default_excluded_layers": sorted(DEFAULT_EXCLUDED),
        "command": command,
        "pytest_args": list(args.pytest_arg),
        "dry_run": args.dry_run,
        "pytest_exit_code": pytest_exit_code,
        "selected_count": selected_count,
        "empty_layer": empty_layer,
        "unclassified_layer": "legacy_compat",
        "launch_error": launch_error,
        "started_at": _isoformat(started),
        "finished_at": _isoformat(_utc_now()),
        "duration_seconds": round(time.monotonic() - start_monotonic, 3),
    }
    try:
        _write_summary(summary_file, summary)
    except OSError as exc:
        print(f"could not write summary {summary_file}: {exc}", file=sys.stderr)
        return 1 if pytest_exit_code in (None, 0) else pytest_exit_code

    print(f"summary: {summary_file}")
    if args.dry_run:
        return 0
    return pytest_exit_code if pytest_exit_code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
