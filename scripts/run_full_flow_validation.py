"""One-click unit and recorded-session full-flow validation for PyCharm.

Run this file with no parameters to choose one replayable session by number,
or enter ``A`` to validate all sessions.  Command-line automation may use
``--session`` or ``--all`` instead of the interactive prompt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.shadow_live_replay import (  # noqa: E402
    ShadowLiveReplayConfig,
    ShadowLiveReplayRunner,
)

DEFAULT_ROOTS = (
    PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "full_flow_validation"
_ACTION_TYPES = frozenset(
    {"player_played", "player_passed", "manual_confirmed_event"}
)
_VERIFICATION_CLOSE_TOLERANCE_MS = 1_200


def discover_sessions(roots: Iterable[Path]) -> tuple[Path, ...]:
    """Return replayable session directories, newest first."""

    sessions: dict[Path, float] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve()
        manifests = (root / "manifest.json",) if root.is_dir() else ()
        if root.is_dir() and not manifests[0].is_file():
            manifests = tuple(root.rglob("manifest.json"))
        for manifest in manifests:
            try:
                relative_parts = manifest.relative_to(root).parts
            except ValueError:
                relative_parts = manifest.parts
            if "_quarantine" in relative_parts:
                continue
            session = manifest.parent.resolve()
            if not (session / "video" / "game.avi").is_file():
                continue
            if not (session / "video" / "frame_index.jsonl").is_file():
                continue
            sessions[session] = manifest.stat().st_mtime
    return tuple(
        session
        for session, _mtime in sorted(
            sessions.items(),
            key=lambda item: (item[1], str(item[0]).lower()),
            reverse=True,
        )
    )


def select_interactively(
    sessions: tuple[Path, ...],
    *,
    input_fn: Callable[[str], str] = input,
) -> tuple[Path, ...]:
    """Prompt for one numeric session or all sessions."""

    if not sessions:
        raise ValueError("没有找到可回放的 session")
    print("\n可回放牌局（最新在前）：")
    for index, session in enumerate(sessions, start=1):
        stamp = datetime.fromtimestamp(
            (session / "manifest.json").stat().st_mtime
        ).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {index:>3}. {session.name}  [{stamp}]")
    answer = input_fn("\n输入序号选择一局，或输入 A 进行全量测试：").strip()
    if answer.upper() == "A":
        return sessions
    try:
        selected = int(answer)
    except ValueError as exc:
        raise ValueError("请输入有效序号或 A") from exc
    if selected < 1 or selected > len(sessions):
        raise ValueError(f"序号超出范围：1-{len(sessions)}")
    return (sessions[selected - 1],)


def _resolve_session(value: Path, discovered: tuple[Path, ...]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
    else:
        matches = [session for session in discovered if session.name == str(value)]
        if len(matches) != 1:
            raise ValueError(f"无法唯一找到 session：{value}")
        resolved = matches[0]
    if not (resolved / "manifest.json").is_file():
        raise ValueError(f"不是有效 session：{resolved}")
    if not (resolved / "video" / "game.avi").is_file():
        raise ValueError(f"session 缺少录像：{resolved}")
    return resolved


def _safe_run_id(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    ).strip("._")
    if not cleaned:
        raise ValueError("run-id 无效")
    return cleaned


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def audit_previous_action_verifications(timeline_path: Path) -> dict[str, object]:
    """Fail a replay if a previous-action advice hold remains open too long."""

    rows: list[dict[str, object]] = []
    with Path(timeline_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                return {
                    "execution_ok": False,
                    "targets": 0,
                    "closed": 0,
                    "unresolved": [],
                    "errors": [f"timeline 第 {line_number} 行不是有效 JSON：{exc}"],
                }
            if isinstance(value, dict):
                rows.append(value)

    targets: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if (
            row.get("event_type") != "advice_withheld"
            or payload.get("reason") != "previous_action_reread_pending"
        ):
            continue
        target_id = str(payload.get("target_event_id", ""))
        opened_ms = int(row.get("monotonic_ms", 0))
        closure: dict[str, object] | None = None
        for candidate in rows[row_index + 1 :]:
            candidate_payload = candidate.get("payload", {})
            if not isinstance(candidate_payload, dict):
                candidate_payload = {}
            event_type = str(candidate.get("event_type", ""))
            matching_lifecycle = bool(
                event_type
                in {
                    "previous_action_verified",
                    "previous_action_verification_expired",
                }
                and str(candidate_payload.get("target_event_id", "")) == target_id
            )
            if not matching_lifecycle:
                continue
            closed_ms = int(candidate.get("monotonic_ms", opened_ms))
            latency_ms = max(0, closed_ms - opened_ms)
            closure = {
                "event_type": event_type,
                "event_id": candidate.get("event_id"),
                "latency_ms": latency_ms,
                "within_tolerance": latency_ms <= _VERIFICATION_CLOSE_TOLERANCE_MS,
            }
            break
        target = {
            "target_event_id": target_id,
            "followup_event_id": payload.get("followup_event_id"),
            "withheld_event_id": row.get("event_id"),
            "withheld_monotonic_ms": opened_ms,
            "closure": closure,
        }
        targets.append(target)
        if closure is None or not closure["within_tolerance"]:
            unresolved.append(target)

    return {
        "execution_ok": not unresolved,
        "tolerance_ms": _VERIFICATION_CLOSE_TOLERANCE_MS,
        "targets": len(targets),
        "closed": len(targets) - len(unresolved),
        "unresolved": unresolved,
        "errors": [],
    }


def resolved_business_actions(
    rows: Iterable[dict[str, object]],
) -> tuple[tuple[str, bool, tuple[str, ...]], ...]:
    """Resolve corrections and return event-ID-independent formal actions."""

    timeline = tuple(rows)
    corrections: dict[str, dict[str, object]] = {}
    for row in timeline:
        if row.get("event_type") != "event_correction":
            continue
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        target_id = str(payload.get("target_event_id", ""))
        if target_id:
            corrections[target_id] = payload

    actions: list[tuple[str, bool, tuple[str, ...]]] = []
    for row in timeline:
        event_type = str(row.get("event_type", ""))
        if event_type not in _ACTION_TYPES:
            continue
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        effective = corrections.get(str(row.get("event_id", "")), payload)
        is_pass = bool(
            effective.get("is_pass", event_type == "player_passed")
        )
        cards = () if is_pass else tuple(
            sorted(str(card) for card in effective.get("cards", ()))
        )
        actions.append((str(row.get("actor", "")), is_pass, cards))
    return tuple(actions)


def audit_business_coverage(
    source_rows: Iterable[dict[str, object]],
    runtime_rows: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Compare complete ordered gameplay after applying timeline corrections."""

    source = resolved_business_actions(source_rows)
    runtime = resolved_business_actions(runtime_rows)
    divergence_index: int | None = None
    for index, (source_action, runtime_action) in enumerate(zip(source, runtime)):
        if source_action != runtime_action:
            divergence_index = index
            break
    if divergence_index is None and len(source) != len(runtime):
        divergence_index = min(len(source), len(runtime))

    def serialized(
        action: tuple[str, bool, tuple[str, ...]] | None,
    ) -> dict[str, object] | None:
        if action is None:
            return None
        actor, is_pass, cards = action
        return {"actor": actor, "is_pass": is_pass, "cards": list(cards)}

    first_divergence: dict[str, object] | None = None
    if divergence_index is not None:
        first_divergence = {
            "action_index": divergence_index,
            "action_number": divergence_index + 1,
            "source": serialized(
                source[divergence_index]
                if divergence_index < len(source)
                else None
            ),
            "runtime": serialized(
                runtime[divergence_index]
                if divergence_index < len(runtime)
                else None
            ),
        }
    return {
        "execution_ok": divergence_index is None,
        "source_action_count": len(source),
        "runtime_action_count": len(runtime),
        "counts_match": len(source) == len(runtime),
        "first_divergence": first_divergence,
    }


def _read_timeline_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"timeline 第 {line_number} 行不是有效 JSON：{path}: {exc}"
                ) from exc
            if isinstance(value, dict):
                rows.append(value)
    return tuple(rows)


def _write_reports(run_directory: Path, summary: dict[str, object]) -> None:
    json_path = run_directory / "full_flow_summary.json"
    markdown_path = run_directory / "full_flow_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    unit = summary.get("unit_tests", {})
    replays = summary.get("replays", [])
    lines = [
        "# 全流程验证报告",
        "",
        f"- 状态：{'通过' if summary.get('execution_ok') else '失败'}",
        f"- Run ID：`{summary.get('run_id')}`",
        f"- 单元测试：`{unit.get('status', 'unknown')}`",
        f"- 选择牌局：{summary.get('selected_session_count', 0)}",
        f"- 回放完成：{len(replays)}",
        f"- 时间倍率：{summary.get('time_scale')}×",
        "",
        "## 逐局结果",
        "",
    ]
    if not replays:
        lines.append("- 无（单元测试失败或尚未开始）")
    for replay in replays:
        status = "通过" if replay.get("execution_ok") else "失败"
        lines.append(
            f"- {status} `{replay.get('session_name')}`："
            f"`{replay.get('summary_path', replay.get('error', ''))}`"
        )
    errors = summary.get("errors", [])
    lines.extend(("", "## 错误", ""))
    lines.extend(f"- {error}" for error in errors)
    if not errors:
        lines.append("- 无")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    sessions: tuple[Path, ...],
    *,
    output: Path,
    run_id: str,
    time_scale: float,
    skip_unit: bool,
    subprocess_run: Callable[..., object] = subprocess.run,
    replay_runner_factory: Callable[[], ShadowLiveReplayRunner] = ShadowLiveReplayRunner,
) -> tuple[int, Path]:
    """Run the gate and return ``(process_exit_code, aggregate_report)``."""

    if not sessions:
        raise ValueError("至少选择一局 session")
    if time_scale <= 0:
        raise ValueError("time-scale 必须大于 0")
    output_root = Path(output).expanduser().resolve()
    aggregate_id = _safe_run_id(run_id)
    run_directory = output_root / aggregate_id
    for session in sessions:
        if _is_relative_to(run_directory, session.parent.resolve()):
            raise ValueError("报告目录必须位于源 sessions 目录之外")
    run_directory.mkdir(parents=True, exist_ok=False)

    started = datetime.now().astimezone().isoformat()
    summary: dict[str, object] = {
        "schema_version": 1,
        "run_id": aggregate_id,
        "started_at": started,
        "project_root": str(PROJECT_ROOT),
        "output_directory": str(run_directory),
        "selected_session_count": len(sessions),
        "selected_sessions": [str(session) for session in sessions],
        "time_scale": float(time_scale),
        "unit_tests": {"status": "skipped" if skip_unit else "pending"},
        "replays": [],
        "errors": [],
        "execution_ok": False,
    }
    _write_reports(run_directory, summary)

    if not skip_unit:
        command = [sys.executable, "-m", "pytest"]
        print("\n[1/2] 运行完整 pytest：", " ".join(command))
        unit_started = time.monotonic()
        completed = subprocess_run(command, cwd=PROJECT_ROOT)
        returncode = int(getattr(completed, "returncode", 1))
        summary["unit_tests"] = {
            "status": "passed" if returncode == 0 else "failed",
            "command": command,
            "returncode": returncode,
            "duration_seconds": round(time.monotonic() - unit_started, 3),
        }
        if returncode != 0:
            summary["errors"].append(f"完整 pytest 失败，退出码 {returncode}")
            summary["finished_at"] = datetime.now().astimezone().isoformat()
            _write_reports(run_directory, summary)
            return 1, run_directory / "full_flow_summary.json"
    else:
        print("\n[1/2] 已显式跳过完整 pytest（仅用于调试）")

    print(f"\n[2/2] 开始回放 {len(sessions)} 局")
    replays: list[dict[str, object]] = summary["replays"]  # type: ignore[assignment]
    errors: list[str] = summary["errors"]  # type: ignore[assignment]
    for index, session in enumerate(sessions, start=1):
        replay_id = _safe_run_id(f"{index:03d}_{session.name}")
        print(f"  [{index}/{len(sessions)}] {session.name}")
        try:
            result = replay_runner_factory().run(
                ShadowLiveReplayConfig(
                    session=session,
                    output=run_directory,
                    time_scale=float(time_scale),
                    run_id=replay_id,
                )
            )
            runtime_directory = Path(str(result.summary["runtime_directory"]))
            verification_audit = audit_previous_action_verifications(
                runtime_directory / "timeline.jsonl"
            )
            business_coverage_audit = audit_business_coverage(
                _read_timeline_rows(session / "timeline.jsonl"),
                _read_timeline_rows(runtime_directory / "timeline.jsonl"),
            )
            replay_ok = bool(result.execution_ok) and bool(
                verification_audit["execution_ok"]
            ) and bool(business_coverage_audit["execution_ok"])
            replays.append(
                {
                    "session": str(session),
                    "session_name": session.name,
                    "execution_ok": replay_ok,
                    "run_directory": str(result.run_directory),
                    "summary_path": str(result.summary_path),
                    "previous_action_verification_audit": verification_audit,
                    "business_coverage_audit": business_coverage_audit,
                }
            )
            if not result.execution_ok:
                errors.append(f"回放门禁失败：{session.name}")
            if not verification_audit["execution_ok"]:
                errors.append(f"上一手复核存在超时未关闭 target：{session.name}")
            if not business_coverage_audit["execution_ok"]:
                errors.append(f"源会话与回放业务动作不完整或不一致：{session.name}")
        except Exception as exc:  # keep all-session validation progressing
            message = f"{type(exc).__name__}: {exc}"
            replays.append(
                {
                    "session": str(session),
                    "session_name": session.name,
                    "execution_ok": False,
                    "error": message,
                }
            )
            errors.append(f"回放异常 {session.name}：{message}")
        _write_reports(run_directory, summary)

    summary["execution_ok"] = not errors
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    _write_reports(run_directory, summary)
    report = run_directory / "full_flow_summary.json"
    print(f"\n汇总报告：{report}")
    return (0 if summary["execution_ok"] else 1), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一键运行完整 pytest 和真实录像 Shadow Live 全流程回放。",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--session",
        type=Path,
        help="选择一个 session 目录，或给出可唯一匹配的 session 名称。",
    )
    selection.add_argument("--all", action="store_true", help="回放发现的全部 session。")
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="session 搜索根目录，可提供多个；默认扫描腾讯掼蛋 sessions。",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="汇总报告根目录。")
    parser.add_argument("--time-scale", type=float, default=1.0, help="回放时间倍率，默认 1×。")
    parser.add_argument("--run-id", help="本次汇总目录名。")
    parser.add_argument(
        "--skip-unit",
        action="store_true",
        help="仅调试：跳过完整 pytest，不应用作交付验证。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        discovered = discover_sessions(tuple(args.roots))
        if args.session is not None:
            selected = (_resolve_session(args.session, discovered),)
        elif args.all:
            if not discovered:
                raise ValueError("指定 roots 下没有可回放 session")
            selected = discovered
        else:
            selected = select_interactively(discovered)
        run_id = args.run_id or datetime.now().astimezone().strftime(
            "full_flow_%Y%m%dT%H%M%S"
        )
        code, _report = run_validation(
            selected,
            output=args.output,
            run_id=run_id,
            time_scale=args.time_scale,
            skip_unit=args.skip_unit,
        )
        return code
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"全流程验证无法启动：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
