"""Classify recorded sessions before replay selection.

Exit codes: 0 has at least one strict candidate and no input error; 1 has no
strict candidate or one or more unqualified sessions; 2 has invalid parameters.
The JSON report is always written outside the immutable sessions tree.
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

from daguandan_bridge.application.session_qualification import qualify_sessions  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读预检 session 的 TruthLog、视频/索引、真实前段解码、开局证据和资源身份；不会把 manifest 的 confirmed 当作视频可观测开局。",
        epilog="退出码：0=存在严格候选且没有输入错误；1=没有严格候选或存在不合格 session；2=路径参数错误。",
    )
    parser.add_argument("--sessions-root", type=Path, required=True, help="包含 session 目录的目录。")
    parser.add_argument("--profile-root", type=Path, required=True, help="profile 目录，或包含 profile 子目录的 profiles 根目录。")
    parser.add_argument("--evidence-root", type=Path, help="可选的外部只读 opening evidence 根目录。")
    parser.add_argument("--output", type=Path, required=True, help="外部 JSON 输出目录；禁止放在 sessions-root 内。")
    parser.add_argument("--session", action="append", default=[], help="只检查指定目录名/ID/绝对路径；可重复。")
    return parser


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sessions_root = args.sessions_root.expanduser().resolve()
        profile_root = args.profile_root.expanduser().resolve()
        output = args.output.expanduser().resolve()
        evidence_root = args.evidence_root.expanduser().resolve() if args.evidence_root else None
        if not sessions_root.is_dir():
            raise ValueError(f"sessions-root 不存在或不是目录：{sessions_root}")
        if not profile_root.exists():
            raise ValueError(f"profile-root 不存在：{profile_root}")
        if evidence_root is not None and not evidence_root.is_dir():
            raise ValueError(f"evidence-root 不存在或不是目录：{evidence_root}")
        if _inside(output, sessions_root):
            raise ValueError("output 必须位于 sessions-root 外部，源数据保持不可变")
        paths = None
        if args.session:
            discovered = {str(p.resolve()): p for p in sessions_root.iterdir() if p.is_dir()} if sessions_root.is_dir() else {}
            paths = []
            for value in args.session:
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = discovered.get(str((sessions_root / value).resolve()), sessions_root / value)
                candidate = candidate.resolve()
                if not candidate.is_dir():
                    raise ValueError(f"session 不存在或不是目录：{candidate}")
                paths.append(candidate)
        records = qualify_sessions(sessions_root, profile_root, evidence_root=evidence_root, session_paths=paths)
        output.mkdir(parents=True, exist_ok=True)
        report = {
            "schema": "guandan.session-qualification/1",
            "sessions_root": str(sessions_root),
            "profile_root": str(profile_root),
            "evidence_root": str(evidence_root) if evidence_root else None,
            "records": list(records),
            "counts": {classification: sum(row["classification"] == classification for row in records) for classification in sorted({row["classification"] for row in records})},
            "strict_candidates": [row["session_id"] for row in records if row["strict_eligible"]],
        }
        report_path = output / "session_qualification.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(report_path), "strict_candidates": report["strict_candidates"], "records": len(records)}, ensure_ascii=False))
        return 0 if report["strict_candidates"] and not any(row["classification"] in {"invalid", "resource_mismatch", "source_not_observable"} for row in records) else 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"资格预检失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
