from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.window_debug_report import WindowDebugReportService
from daguandan_bridge.window_debug import json_safe


def _hwnd(value: str) -> int:
    try:
        result = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("HWND must be decimal or 0x-prefixed hexadecimal") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("HWND must be positive")
    return result


def _print_json(payload: object) -> None:
    """Write UTF-8 JSON even when the Windows console uses a legacy code page."""

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is None:
            sys.stdout.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
        else:
            buffer.write(text.encode("utf-8"))
            buffer.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Win32 window diagnostics")
    parser.add_argument("--list-windows", action="store_true", help="list visible top-level windows")
    parser.add_argument("--hwnd", type=_hwnd, help="probe this explicit HWND")
    parser.add_argument("--capture", action="store_true", help="run one PrintWindow capture")
    parser.add_argument("--recognize", action="store_true", help="recognize the captured single frame")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--profile", default="tencent_daguandan", help="recognition profile name")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.capture or args.recognize) and args.hwnd is None:
        parser.error("--capture/--recognize require --hwnd")
    if not args.list_windows and args.hwnd is None:
        parser.error("select --list-windows or provide --hwnd")
    try:
        service = WindowDebugReportService(profile_name=args.profile)
        report = service.build(
            list_windows=args.list_windows,
            hwnd=args.hwnd,
            capture=args.capture or args.recognize,
            recognize=args.recognize,
        )
        service.write(report, args.output)
        _print_json(json_safe(report))
        return 1 if report.get("errors") else 0
    except Exception as exc:
        payload = {
            "schema": "guandan.window-debug-cli-error/v1",
            "error": {
                "code": str(getattr(exc, "code", type(exc).__name__)),
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        # Error reports use the same managed storage boundary when no
        # explicit export path was requested.
        try:
            WindowDebugReportService(profile_name=args.profile).write(payload, args.output)
        except Exception:
            pass
        _print_json(payload)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
