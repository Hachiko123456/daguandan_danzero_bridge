"""Safely repair derived trick_id fields across TruthLogs.

Dry-run is the default. Only logs whose sole findings are TL-TRICK-MISMATCH
warnings are eligible. Applying creates a new TruthRevision and never changes
cards, actors, pass flags, evidence, or turn count.
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

from daguandan_bridge.application.truth_log_semantic_validation import (  # noqa: E402
    normalize_truth_log_trick_ids,
    validate_truth_log_semantics,
)
from daguandan_bridge.application.truth_revision_store import (  # noqa: E402
    TruthRevisionStore,
    truth_log_changed_fields,
)
from daguandan_bridge.live.truth_log import load_truth_log  # noqa: E402

DEFAULT_ROOT = PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions"


def resolve_sessions(root: Path, values: list[Path]) -> tuple[Path, ...]:
    if not values:
        return tuple(sorted(path.parent for path in root.glob("*/truth_log.json")))
    result: list[Path] = []
    for value in values:
        candidate = value if value.is_absolute() else root / value
        if candidate.name == "truth_log.json":
            candidate = candidate.parent
        if not (candidate / "truth_log.json").is_file():
            raise FileNotFoundError(candidate / "truth_log.json")
        result.append(candidate.resolve())
    return tuple(dict.fromkeys(result))


def inspect(session: Path) -> dict[str, object]:
    try:
        log = load_truth_log(session / "truth_log.json", session_id=session.name)
        report = validate_truth_log_semantics(log, mode="logic", standard_playing=True)
        eligible = bool(report.warnings) and not report.errors and all(
            item.code == "TL-TRICK-MISMATCH" for item in report.warnings
        )
        normalized = normalize_truth_log_trick_ids(log, standard_playing=True)
        changed = truth_log_changed_fields(log, normalized)
        non_trick = tuple(
            field for field in changed if not field.endswith(".trick_id")
        )
        before_actions = [
            (turn.index, turn.actor, turn.is_pass, turn.cards, turn.move_semantics)
            for turn in log.turns
        ]
        after_actions = [
            (turn.index, turn.actor, turn.is_pass, turn.cards, turn.move_semantics)
            for turn in normalized.turns
        ]
        actions_unchanged = before_actions == after_actions
        if non_trick or not actions_unchanged:
            eligible = False
        return {
            "session_id": session.name,
            "status": "eligible" if eligible else "unchanged" if not changed else "blocked",
            "turn_count": len(log.turns),
            "warning_count": len(report.warnings),
            "error_count": len(report.errors),
            "changed_fields": list(changed),
            "non_trick_changes": list(non_trick),
            "actions_unchanged": actions_unchanged,
            "current_revision_id": TruthRevisionStore(session).manifest().current_revision_id,
            "_log": log,
            "_normalized": normalized,
        }
    except Exception as exc:
        return {
            "session_id": session.name,
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", type=Path)
    parser.add_argument("--sessions-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--apply", action="store_true", help="publish eligible repairs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    try:
        sessions = resolve_sessions(args.sessions_root.resolve(), args.sessions)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    rows: list[dict[str, object]] = []
    for session in sessions:
        row = inspect(session)
        log = row.pop("_log", None)
        normalized = row.pop("_normalized", None)
        if args.apply and row.get("status") == "eligible" and log is not None and normalized is not None:
            try:
                store = TruthRevisionStore(session)
                parent = store.manifest().current_revision_id
                revision = store.publish(
                    normalized,
                    changed_fields=row.get("changed_fields", []),
                    author="truth_trick_id_batch_repair",
                    expected_parent_revision_id=parent,
                )
                row["status"] = "repaired"
                row["revision_id"] = revision.revision_id
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    payload = {
        "schema": "guandan.truth-trick-id-repair/1",
        "apply": bool(args.apply),
        "session_count": len(rows),
        "rows": rows,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{str(row.get('status')).upper():9} {row.get('session_id')} "
                  f"warnings={row.get('warning_count', '-')}, "
                  f"changed={len(row.get('changed_fields', ())) if isinstance(row.get('changed_fields'), list) else '-'}")
    return 0 if not any(row.get("status") in {"blocked", "failed"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
