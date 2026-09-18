"""Validate canonical TruthLogs without reading screenshots.

With no arguments the command audits every canonical TruthLog under the
Tencent profile and exits non-zero only when a semantic error is found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.truth_log_semantic_validation import validate_truth_log_semantics
from daguandan_bridge.live.truth_log import load_truth_log

DEFAULT_ROOT = PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Validate TruthLog game logic; no image recognition is performed.")
    value.add_argument("sessions", nargs="*", type=Path, help="Session paths or IDs; default: all sessions with truth_log.json")
    value.add_argument("--sessions-root", type=Path, default=DEFAULT_ROOT)
    value.add_argument("--mode", choices=("logic", "publish"), default="logic")
    value.add_argument(
        "--allow-nonstandard-hand-sizes", action="store_true",
        help="Allow tribute-intermediate non-27 opponent counts; default validates normal playing phase.",
    )
    value.add_argument("--json", action="store_true", help="Print one machine-readable JSON document")
    return value


def resolve_sessions(root: Path, values: list[Path]) -> tuple[Path, ...]:
    if not values:
        return tuple(sorted(path.parent for path in root.glob("*/truth_log.json")))
    result = []
    for value in values:
        candidate = value if value.is_absolute() else root / value
        if candidate.name == "truth_log.json":
            candidate = candidate.parent
        if not (candidate / "truth_log.json").is_file():
            raise FileNotFoundError(f"TruthLog not found: {candidate / 'truth_log.json'}")
        result.append(candidate.resolve())
    return tuple(dict.fromkeys(result))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    try:
        sessions = resolve_sessions(args.sessions_root.resolve(), args.sessions)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rows = []
    for session in sessions:
        try:
            log = load_truth_log(session / "truth_log.json", session_id=session.name)
            report = validate_truth_log_semantics(
                log, mode=args.mode,
                standard_playing=not args.allow_nonstandard_hand_sizes,
            )
            row = report.to_dict()
        except Exception as exc:
            row = {
                "schema": "guandan.truth-validation/1",
                "session_id": session.name,
                "mode": args.mode,
                "valid": False,
                "error_count": 1,
                "warning_count": 0,
                "findings": [{"code": "TL-LOAD", "severity": "error", "message": f"{type(exc).__name__}: {exc}", "turn_id": None, "field": ""}],
            }
        rows.append(row)
    payload = {
        "schema": "guandan.truth-validation-batch/1",
        "mode": args.mode,
        "session_count": len(rows),
        "passed": sum(bool(row["valid"]) for row in rows),
        "failed": sum(not bool(row["valid"]) for row in rows),
        "sessions": rows,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{'PASS' if row['valid'] else 'FAIL'} {row['session_id']}")
            for item in row.get("findings", ()):
                marker = "ERROR" if item.get("severity") == "error" else "WARN"
                turn = f" turn={item.get('turn_id')}" if item.get("turn_id") is not None else ""
                print(f"  [{marker} {item.get('code')}]{turn} {item.get('message')}")
        print(f"\nsummary: passed={payload['passed']} failed={payload['failed']} total={payload['session_count']}")
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
