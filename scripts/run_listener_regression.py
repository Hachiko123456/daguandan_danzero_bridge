"""Run the focused recorded-session regression for the visual listener core.

The command intentionally uses the production visual replay path:

    AVI -> FrameEnvelope -> production OpeningTracker/page gate
        -> LiveV2SessionRuntime
        -> LiveV2 Vision Runtime -> LiveEngine -> ProductionRuleSession
        -> LiveV2 Advice/FableDan -> post-replay TruthLog comparison

The legacy LiveOrchestrator is not part of the main regression path; it is
retained only for compatibility/unit-test paths.

A verified TruthLog is an expected-result baseline only; it is never injected
as the visual action input for the primary listener replay. The same TruthLog is
also replayed through the advisor-only path so FableDan can be tested with a
complete, known-good game state independently of visual recognition errors.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import sys
import time
from threading import Lock
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.session_replay_audit import (  # noqa: E402
    SessionReplayAuditService,
    _fabledan_blocking_reasons,
)
from daguandan_bridge.application.session_workbench import (  # noqa: E402
    SessionDescriptor,
    inspect_sessions,
)
from daguandan_bridge.storage import atomic_write_json  # noqa: E402


DEFAULT_SESSIONS_ROOT = (
    PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan" / "sessions"
)
DEFAULT_PROFILE_ROOT = PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "listener-core-regression"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run historical sessions through the production visual listener "
            "core and FableDan; source sessions are read-only."
        ),
        epilog=(
            "测试链路：\n"
            "  1. 从 sessions-root 选择 session；默认只选 verified TruthLog。\n"
            "  2. 逐帧读取 AVI，进入生产 OpeningTracker/page gate -> "
            "LiveV2SessionRuntime。\n"
            "     主链包含 LiveV2 Vision Runtime -> LiveEngine -> "
            "ProductionRuleSession -> LiveV2 Advice/FableDan。\n"
            "  3. 用 TruthLog 比较级牌、手牌、首家、出牌、PASS、顺序和终局。\n"
            "  4. 视觉驱动 FableDan 与 TruthLog 隔离驱动 FableDan 分开统计。\n"
            "     旧 LiveOrchestrator 仅保留兼容/单元测试路径，不是主验收链。\n\n"
            "选择优先级：--session 先过滤；--random-count 再从过滤后的集合随机抽取。\n"
            "--session 可以一次指定多个 ID，也可以重复写多个 --session。\n"
            "默认同时处理 3 局；--workers 可在 1～3 之间调整。\n"
            "随机测试建议配合 --seed 使用，以便之后复现同一批 session。\n\n"
            "退出码：0=严格回归通过；1=发现质量失败；2=参数、路径或执行错误。\n\n"
            "示例：\n"
            "  python scripts/run_listener_regression.py\n"
            "  python scripts/run_listener_regression.py --random-count 3 --seed 20260912\n"
            "  python scripts/run_listener_regression.py --session game_xxx --run-id smoke_001"
        ),
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=DEFAULT_SESSIONS_ROOT,
        help="Session directory or directory containing sessions.",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_PROFILE_ROOT,
        help="Selected profile directory containing profile.json/templates/models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="External directory for regression reports; never place it below sessions-root.",
    )
    parser.add_argument(
        "--session",
        action="extend",
        nargs="+",
        default=[],
        metavar="SESSION",
        help=(
            "按 session ID、目录名或绝对路径过滤；一次可指定多个，"
            "也可重复 --session。不指定时使用 sessions-root 下的全部候选。"
        ),
    )
    parser.add_argument(
        "--random-count",
        "--random-n",
        dest="random_count",
        type=int,
        help=(
            "从过滤后的候选中随机抽取 N 局；默认不抽样。"
            "N 必须大于 0 且不超过候选数量。"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "随机抽样种子。只有指定 --random-count 时生效；"
            "指定后可复现同一批 session。"
        ),
    )
    parser.add_argument(
        "--include-draft",
        action="store_true",
        help=(
            "把 draft TruthLog 也加入参考诊断；它们会执行但不作为严格阻断。"
        ),
    )
    parser.add_argument(
        "--include-no-truth",
        action="store_true",
        help=(
            "把无 TruthLog 但有可用 timeline 初始状态的 session 加入参考诊断；"
            "不能进行严格动作真值比较。"
        ),
    )
    parser.add_argument(
        "--include-ineligible",
        action="store_true",
        help=(
            "仅用于诊断：允许选择 source/resource/metadata 不满足严格资格的 session；"
            "这些结果不会阻断严格回归。"
        ),
    )
    parser.add_argument(
        "--workers",
        "--max-workers",
        dest="workers",
        type=int,
        default=3,
        help=(
            "同时处理的 session 数量，默认 3；范围 1～3。"
            "线程只按 session 并发，不会把一局的帧全部加载到内存。"
        ),
    )
    parser.add_argument(
        "--run-id",
        help="报告目录名；不指定时自动生成。重复名称会报错，避免覆盖旧报告。",
    )
    return parser


def _normalize_session_filters(values: Iterable[object]) -> tuple[str, ...]:
    if isinstance(values, (str, Path)):
        values = (values,)
    result: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.extend(_normalize_session_filters(value))
        else:
            normalized = str(value).strip()
            if normalized:
                result.append(normalized)
    return tuple(dict.fromkeys(result))


def _validate_worker_count(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("--workers 必须是 1～3 之间的整数")
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--workers 必须是 1～3 之间的整数") from exc
    if not 1 <= workers <= 3:
        raise ValueError("--workers 必须是 1～3 之间的整数")
    return workers


def _read_manifest_for_descriptor(descriptor: SessionDescriptor) -> dict[str, object]:
    try:
        raw = json.loads(descriptor.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _normalized_status(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        return normalized or None
    return None


def qualify_descriptor(descriptor: SessionDescriptor) -> dict[str, object]:
    """Return a conservative, pre-replay qualification for one session.

    Missing newer provenance fields are intentionally unknown.  We never turn
    an old/manual descriptor into a strict candidate merely because it has a
    video and a verified TruthLog.
    """
    manifest = _read_manifest_for_descriptor(descriptor)
    reasons: list[str] = []
    source_value = manifest.get("source_observability")
    if source_value is None:
        source_value = manifest.get("initial_state_observability")
    if isinstance(source_value, dict):
        source_value = source_value.get("status") or source_value.get("classification")
    source_status = _normalized_status(source_value)
    if source_status is None:
        # Existing live manifests have an explicit, auditable pair of fields.
        # Do not apply this compatibility inference to manual recordings.
        if (
            manifest.get("recording_phase") == "live"
            and manifest.get("initial_state_status") == "confirmed"
            and manifest.get("session_kind") != "manual_recording"
        ):
            source_status = "strict"
        elif str(manifest.get("session_kind", "")).lower() == "manual_recording" or descriptor.session_id.startswith("manual_"):
            source_status = "unknown"
        else:
            source_status = "unknown"
    if source_status in {"observable", "confirmed", "strict_replay", "strict"}:
        source_status = "strict"
    elif source_status in {"not_observable", "unobservable", "missing_initial_state", "diagnostic_only"}:
        source_status = "not_observable"
    else:
        source_status = "unknown"

    resource_value = manifest.get("resource_identity_match")
    if resource_value is None:
        resource_value = manifest.get("profile_resource_match")
    if isinstance(resource_value, dict):
        resource_value = resource_value.get("status")
    if isinstance(resource_value, bool):
        resource_status = "match" if resource_value else "mismatch"
    else:
        resource_status = _normalized_status(resource_value)
        if resource_status in {"matched", "match", "same", "strict"}:
            resource_status = "match"
        elif resource_status in {"mismatch", "different", "failed"}:
            resource_status = "mismatch"
        else:
            resource_status = "unknown"

    if descriptor.truth_status != "verified":
        reasons.append(f"truth_status_{descriptor.truth_status}")
    if not descriptor.has_video:
        reasons.append("missing_video")
    if not descriptor.manifest_readable:
        reasons.append("manifest_unknown")
    if source_status != "strict":
        reasons.append(f"source_observability_{source_status}")
    if resource_status != "match":
        reasons.append(f"resource_identity_{resource_status}")
    strict_eligible = not reasons
    return {
        "session_id": descriptor.session_id,
        "source": str(descriptor.root),
        "strict_eligible": strict_eligible,
        "source_observability": source_status,
        "resource_identity": resource_status,
        "reasons": reasons,
    }


def select_descriptors(
    descriptors: Iterable[SessionDescriptor],
    *,
    session_filters: Iterable[str] = (),
    include_draft: bool = False,
    include_no_truth: bool = False,
    include_ineligible: bool = False,
    strict_only: bool = True,
    random_count: int | None = None,
    seed: int | None = None,
) -> tuple[SessionDescriptor, ...]:
    filters = _normalize_session_filters(session_filters)
    available = tuple(descriptors)
    if filters:
        ordered: list[SessionDescriptor] = []
        used: set[Path] = set()
        for value in filters:
            match = next(
                (
                    item
                    for item in available
                    if item.session_id == value
                    or item.root.name == value
                    or str(item.root) == value
                ),
                None,
            )
            if match is not None and match.root not in used:
                ordered.append(match)
                used.add(match.root)
        available = tuple(ordered)
    statuses = {"verified"}
    if include_draft:
        statuses.add("draft")
    if include_no_truth:
        statuses.update({"missing", "invalid"})
    selected = [item for item in available if item.truth_status in statuses and item.has_video]
    if seed is not None and random_count is None:
        raise ValueError("--seed 必须与 --random-count 一起使用")
    if random_count is not None:
        if random_count <= 0:
            raise ValueError("--random-count 必须大于 0")
        if strict_only and not include_ineligible:
            selected = [item for item in selected if qualify_descriptor(item)["strict_eligible"]]
        if random_count > len(selected):
            raise ValueError(
                f"--random-count={random_count} 超过可选 strict session 数量 {len(selected)}"
            )
        rng = random.Random(seed)
        selected = rng.sample(selected, random_count)
        selected.sort(key=lambda item: item.session_id.lower())
    return tuple(selected)


def _row_source_observability(row: dict[str, object]) -> str:
    value = row.get("source_observability") or row.get("initial_state_observability")
    if isinstance(value, dict):
        value = value.get("status") or value.get("classification")
    status = _normalized_status(value)
    if status in {"strict", "observable", "confirmed", "strict_replay"}:
        return "strict"
    if status in {"not_observable", "unobservable", "missing_initial_state", "diagnostic_only"}:
        return "not_observable"
    session_id = str(row.get("session_id") or "")
    if session_id.startswith("manual_") or row.get("opening_status") == "opening_not_observable":
        return "not_observable"
    return "unknown"


def _row_resource_identity(row: dict[str, object]) -> str:
    value = row.get("resource_identity_match")
    if value is None:
        value = row.get("profile_resource_match")
    if isinstance(value, dict):
        value = value.get("status")
    if isinstance(value, bool):
        return "match" if value else "mismatch"
    status = _normalized_status(value)
    if status in {"match", "matched", "same", "strict"}:
        return "match"
    if status in {"mismatch", "different", "failed"}:
        return "mismatch"
    return "unknown"


def evaluate_run(
    summary: dict[str, object],
    *,
    include_ineligible: bool = False,
) -> dict[str, object]:
    rows = tuple(row for row in summary.get("sessions", ()) if isinstance(row, dict))
    blocking: list[dict[str, object]] = []
    strict_passes: list[dict[str, object]] = []
    advisory: list[dict[str, object]] = []
    classified: list[dict[str, object]] = []
    required = {
        "execution_status": "completed",
        "frame_replay_status": "complete",
        "listener_status": "complete",
        "opening_status": "recognized",
        "truth_quality": "passed",
        "visual_quality": "passed",
        "comparison_status": "passed",
    }
    for row in rows:
        truth_kind = str((row.get("truth_log") or {}).get("kind", "none")) if isinstance(row.get("truth_log"), dict) else "none"
        truth_qualification = row.get("truth_qualification")
        truth_strict = truth_kind in {"canonical", "verified"} and truth_qualification in {"verified_label", "verified", "trusted_for_run"}
        source_status = _row_source_observability(row)
        resource_status = _row_resource_identity(row)
        failure_reasons = [f"{key}_{row.get(key)}" for key, expected in required.items() if row.get(key) != expected]
        # Keep the existing trusted FableDan channel checks for strict truth
        # sessions.  Advisory FableDan output remains explicitly non-blocking;
        # actual failed/timeout trusted-channel evidence remains a quality
        # failure and therefore cannot earn strict_pass.
        failure_reasons.extend(_fabledan_blocking_reasons(row, strict=truth_strict))
        first_divergence = row.get("first_divergence")
        if source_status == "not_observable":
            classification = "source_not_observable"
        elif resource_status == "mismatch":
            classification = "resource_mismatch"
        elif not truth_strict or source_status == "unknown" or resource_status == "unknown":
            classification = "invalid"
            if not truth_strict:
                failure_reasons.append("truth_qualification_unknown_or_unverified")
            if source_status == "unknown":
                failure_reasons.append("source_observability_unknown")
            if resource_status == "unknown":
                failure_reasons.append("resource_identity_unknown")
        elif failure_reasons or first_divergence:
            classification = "listener_divergence"
        else:
            classification = "strict_pass"
        item = {
            "session_id": row.get("session_id"),
            "source": row.get("source"),
            "classification": classification,
            # --include-ineligible is an explicit diagnostic mode: provenance
            # failures are retained in the report but do not block the run.
            # Without it, a selected ineligible session is a failed gate.
            "blocking": classification == "listener_divergence" or (
                classification in {"resource_mismatch", "source_not_observable", "invalid"}
                and not include_ineligible
            ),
            "strict_eligible": classification == "strict_pass",
            "source_observability": source_status,
            "resource_identity": resource_status,
            "quality": {key: row.get(key) for key in required},
            "failure_reasons": failure_reasons,
            "first_divergence": first_divergence,
            "error": row.get("error"),
        }
        classified.append(item)
        if classification == "strict_pass":
            strict_passes.append(item)
        elif item["blocking"]:
            blocking.append(item)
        else:
            item["failed"] = True
            advisory.append(item)
    counts = {name: sum(item["classification"] == name for item in classified) for name in (
        "strict_pass", "source_not_observable", "resource_mismatch", "listener_divergence", "invalid"
    )}
    return {
        "status": "PASS" if not blocking and bool(rows) else "FAIL",
        "session_count": len(rows),
        "classification_counts": counts,
        "strict_passes": strict_passes,
        "blocking_failures": blocking,
        "advisory_results": advisory,
        "results": classified,
    }

def _write_markdown_report(
    path: Path,
    report: dict[str, object],
    *,
    source_summary: Path,
    fabledan: object,
) -> None:
    failures = report.get("blocking_failures", ())
    advisory = report.get("advisory_results", ())
    lines = [
        "# 识别监听核心回归报告",
        "",
        f"- 结论：**{report.get('status', 'UNKNOWN')}**",
        f"- 测试局数：{report.get('session_count', 0)}",
        f"- 严格失败：{len(failures) if isinstance(failures, list) else 0}",
        f"- 参考结果：{len(advisory) if isinstance(advisory, list) else 0}",
        f"- 完整审计：`{source_summary}`",
        "",
        "## FableDan",
        "",
        f"```json\n{json.dumps(fabledan, ensure_ascii=False, indent=2)}\n```",
        "",
        "## 严格失败",
        "",
    ]
    if isinstance(failures, list) and failures:
        for item in failures:
            if not isinstance(item, dict):
                continue
            lines.extend([
                f"### {item.get('session_id')}",
                "",
                f"- 质量：`{json.dumps(item.get('quality'), ensure_ascii=False)}`",
                f"- 错误：{item.get('error') or '无'}",
                f"- 首个分歧：`{json.dumps(item.get('first_divergence'), ensure_ascii=False)}`",
                "",
            ])
    else:
        lines.extend(["无。", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _ConsoleProgress:
    """Render one aggregate progress bar for concurrent session workers."""

    _PHASE_LABELS = {
        "prepare": "准备",
        "opening": "开局识别",
        "visual": "视觉监听",
        "fabledan_truth": "FableDan",
        "session_start": "开始会话",
        "session_done": "会话完成",
    }

    def __init__(self, total_sessions: int) -> None:
        self.total_sessions = max(1, int(total_sessions))
        self.completed = 0
        self.session_id = ""
        self.phase = "准备"
        self.detail = ""
        self._states: dict[str, dict[str, object]] = {}
        self._lock = Lock()
        self._last_length = 0
        self._last_draw_at = 0.0
        self._last_percent = -1.0
        self._last_phase = ""

    def __call__(
        self,
        phase: str,
        session_id: str,
        processed: int,
        total: int,
        detail: str,
    ) -> None:
        with self._lock:
            sid = str(session_id or "unknown")
            state = self._states.setdefault(sid, {"fraction": 0.0, "phase": "准备"})
            phase_label = self._PHASE_LABELS.get(phase, phase)
            current_fraction = float(state.get("fraction", 0.0) or 0.0)
            next_fraction = current_fraction
            if phase == "session_start":
                next_fraction = current_fraction
            elif phase == "session_done":
                next_fraction = 1.0
            elif phase == "visual":
                next_fraction = 0.5 * self._ratio(processed, total)
            elif phase == "fabledan_truth":
                next_fraction = 0.5 + 0.5 * self._ratio(processed, total)
            state["fraction"] = max(current_fraction, next_fraction)
            state["phase"] = phase_label
            state["detail"] = detail
            self.session_id = sid
            self.phase = phase_label
            self.detail = detail
            self.completed = sum(
                float(item.get("fraction", 0.0) or 0.0) >= 1.0
                for item in self._states.values()
            )
            overall = min(
                1.0,
                sum(float(item.get("fraction", 0.0) or 0.0) for item in self._states.values())
                / self.total_sessions,
            )
            percent = overall * 100.0
            now = time.monotonic()
            phase_changed = self.phase != self._last_phase
            is_terminal = phase == "session_done"
            if (
                not phase_changed
                and not is_terminal
                and percent < self._last_percent + 0.5
                and now - self._last_draw_at < 0.5
            ):
                return
            self._last_phase = self.phase
            self._last_percent = percent
            self._last_draw_at = now
            width = 28
            filled = min(width, max(0, int(percent / 100.0 * width)))
            bar = "#" * filled + "-" * (width - filled)
            suffix = f" | {detail}" if detail else ""
            text = (
                f"\r[{bar}] {percent:6.2f}% | 局 "
                f"{self.completed}/{self.total_sessions} | {self.phase} | {sid}{suffix}"
            )
            padding = max(0, self._last_length - len(text))
            sys.stdout.write(text + (" " * padding))
            sys.stdout.flush()
            self._last_length = len(text)

    @staticmethod
    def _ratio(processed: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, int(processed) / int(total)))

    def finish(self) -> None:
        with self._lock:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _print_console_report(report: dict[str, object], *, summary_path: Path) -> None:
    print(f"核心监听回归：{report['status']}")
    print(f"测试局数：{report['session_count']}")
    print(f"汇总：{summary_path}")
    failures = report.get("blocking_failures", ())
    if isinstance(failures, list) and failures:
        print("失败会话：", file=sys.stderr)
        for item in failures:
            if not isinstance(item, dict):
                continue
            print(
                f"  {item.get('session_id')}: "
                f"{item.get('quality')}；首个分歧：{item.get('first_divergence')}",
                file=sys.stderr,
            )
    advisory = report.get("advisory_results", ())
    if isinstance(advisory, list) and advisory:
        print(f"参考会话：{len(advisory)}（不作为严格阻断）")


def _artifact_role(relative: str) -> tuple[str, str]:
    name = relative.replace("\\", "/")
    if name == "00_llm_summary.md":
        return "大模型首要入口：测试结论、首个失败和读取建议", "P0"
    if name == "03_failures.json":
        return "机器可读失败列表", "P0"
    if "first_divergence" in name:
        if name.endswith("description.md"):
            return "首个分歧的人类可读说明", "P0"
        if name.endswith("context.json"):
            return "首个分歧的状态机上下文", "P0"
        if name.endswith("expected.json") or name.endswith("actual.json"):
            return "首个分歧的预期/实际动作", "P0"
        if name.endswith(".png"):
            return "首个分歧的帧或 ROI 证据", "P0"
    if name.endswith("recognition_trace.jsonl"):
        return "逐帧识别诊断日志；按 frame_index 局部读取", "P2"
    if name.endswith("observations.jsonl.gz"):
        return "压缩逐帧观察；默认不要全量读取", "P3"
    if name.endswith("advice.jsonl"):
        return "FableDan 推荐明细", "P1"
    if name.endswith("decisions.jsonl"):
        return "FableDan 决策输入和状态版本", "P1"
    if name.endswith("timeline.jsonl"):
        return "视觉监听或 TruthLog 驱动的事件时间线", "P1"
    if name.endswith("summary.json"):
        return "会话或 FableDan 汇总", "P1"
    if name.endswith(".avi"):
        return "原始录像；只在需要人工复核时读取", "P3"
    return "运行元数据或审计产物", "P2"



def _write_artifact_manifest(run_directory: Path) -> None:
    files: list[dict[str, object]] = []
    for path in sorted(run_directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_directory).as_posix()
        role, priority = _artifact_role(relative)
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "role": role,
                "priority": priority,
                "read_hint": (
                    "优先读取"
                    if priority == "P0"
                    else "按 session/frame/turn 局部读取"
                    if priority == "P1"
                    else "需要深入诊断时读取"
                    if priority == "P2"
                    else "默认不要全量加载"
                ),
            }
        )
    atomic_write_json(
        run_directory / "02_artifact_manifest.json",
        {
            "schema": "guandan.listener-core-artifacts/1",
            "entrypoint": "00_llm_summary.md",
            "read_order": [
                "00_llm_summary.md",
                "03_failures.json",
                "sessions/<session>/00_session_summary.md",
                "sessions/<session>/first_divergence/context.json",
                "sessions/<session>/first_divergence/trigger.png",
                "sessions/<session>/visual_driven/runtime/timeline.jsonl",
                "sessions/<session>/truth_driven/*/summary.json",
            ],
            "do_not_load_whole": [
                "*.avi",
                "observations.jsonl.gz",
                "recognition_trace.jsonl",
                "decisions.jsonl",
            ],
            "files": files,
        },
    )


def _write_llm_artifacts(
    run_directory: Path,
    *,
    raw_summary: dict[str, object],
    quality: dict[str, object],
) -> None:
    """Write a small, deterministic index so an LLM need not guess artifacts."""

    failures = quality.get("blocking_failures", [])
    advisory = quality.get("advisory_results", [])
    fabledan = raw_summary.get("fabledan", {})
    session_rows = tuple(
        row for row in raw_summary.get("sessions", ()) if isinstance(row, dict)
    )
    summary_path = run_directory / "01_run_summary.json"
    atomic_write_json(summary_path, raw_summary)
    failures_path = run_directory / "03_failures.json"
    atomic_write_json(
        failures_path,
        {
            "schema": "guandan.listener-core-failures/1",
            "status": quality.get("status"),
            "blocking_failures": failures,
            "advisory_results": advisory,
        },
    )

    lines = [
        "# 识别监听核心回归：大模型摘要",
        "",
        f"- 总结：**{quality.get('status', 'UNKNOWN')}**",
        f"- 测试局数：{quality.get('session_count', 0)}",
        f"- 严格失败：{len(failures) if isinstance(failures, list) else 0}",
        f"- 参考结果：{len(advisory) if isinstance(advisory, list) else 0}",
        "",
        "## 测试链路",
        "",
        "```text",
        "历史 AVI -> FrameEnvelope -> 生产 OpeningTracker/page gate",
        "  -> LiveV2SessionRuntime -> LiveV2 Vision Runtime",
        "  -> LiveEngine -> ProductionRuleSession -> LiveV2 Advice/FableDan",
        "  -> 回放完成后 TruthLog 比较",
        "```",
        "",
        "说明：旧 LiveOrchestrator 仅保留兼容路径，不属于主验收链。",
        "",
        "## TruthLog 基线版本",
        "",
    ]
    for row in session_rows:
        truth = row.get("truth_log")
        truth = truth if isinstance(truth, dict) else {}
        lines.append(
            f"- `{row.get('session_id')}`：revision="
            f"`{truth.get('revision_id') or 'unversioned'}`，semantic_sha256="
            f"`{truth.get('semantic_sha256') or 'unknown'}`，文件匹配="
            f"`{truth.get('revision_matches_truth')}`"
        )
    lines.extend([
        "",
        "## 首先读取",
        "",
        "1. `00_llm_summary.md`（本文件）",
        "2. `03_failures.json`",
        "3. 失败 session 的 `00_session_summary.md`",
        "4. `first_divergence/context.json` 和对应 PNG",
        "5. 需要深入时再按 frame/turn 局部读取 JSONL",
        "",
        "## FableDan 汇总",
        "",
        f"```json\n{json.dumps(fabledan, ensure_ascii=False, indent=2)}\n```",
        "",
        "## 严格失败",
        "",
    ])
    if isinstance(failures, list) and failures:
        for item in failures:
            if not isinstance(item, dict):
                continue
            lines.extend(
                [
                    f"### {item.get('session_id')}",
                    "",
                    f"- 质量：`{json.dumps(item.get('quality'), ensure_ascii=False)}`",
                    f"- 首个分歧：`{json.dumps(item.get('first_divergence'), ensure_ascii=False)}`",
                    f"- 错误：{item.get('error') or '无'}",
                    "",
                ]
            )
    else:
        lines.extend(["无。", ""])
    lines.extend(
        [
            "## 产物说明",
            "",
            "详细文件角色和读取优先级见 `02_artifact_manifest.json`。",
            "大模型默认不要全量读取 AVI、`observations.jsonl.gz`、"
            "`recognition_trace.jsonl` 和 `decisions.jsonl`。",
            "",
        ]
    )
    (run_directory / "00_llm_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    for row in raw_summary.get("sessions", []):
        if not isinstance(row, dict):
            continue
        source = Path(str(row.get("evidence_paths", {}).get("session_audit", "")))
        if not source.is_dir():
            continue
        session_summary = {
            "schema": "guandan.listener-core-session-summary/1",
            "session_id": row.get("session_id"),
            "quality": {
                "execution_status": row.get("execution_status"),
                "truth_quality": row.get("truth_quality"),
                "visual_quality": row.get("visual_quality"),
                "fabledan_quality": row.get("fabledan_quality"),
            },
            "first_divergence": row.get("first_divergence"),
            "fabledan": row.get("fabledan"),
            "read_next": [
                "first_divergence/context.json",
                "first_divergence/trigger.png",
                "visual_driven/runtime/timeline.jsonl",
                "truth_driven/*/summary.json",
            ],
        }
        atomic_write_json(source / "01_session_summary.json", session_summary)
        (source / "00_session_summary.md").write_text(
            "\n".join(
                [
                    f"# Session {row.get('session_id')}",
                    "",
                    f"- execution：{row.get('execution_status')}",
                    f"- truth：{row.get('truth_quality')}",
                    f"- visual：{row.get('visual_quality')}",
                    f"- FableDan：{row.get('fabledan_quality')}",
                    f"- 首个分歧：{json.dumps(row.get('first_divergence'), ensure_ascii=False)}",
                    "",
                    "优先查看 `first_divergence/`；不要直接全量加载大日志。",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    # Refresh after session summaries so the manifest contains every artifact.
    _write_artifact_manifest(run_directory)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sessions_root = args.sessions_root.expanduser().resolve()
    profile_root = args.profile_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not sessions_root.exists():
        print(f"sessions-root 不存在：{sessions_root}", file=sys.stderr)
        return 2
    if not profile_root.is_dir():
        print(f"profile-root 不存在：{profile_root}", file=sys.stderr)
        return 2
    try:
        workers = _validate_worker_count(args.workers)
        descriptors = inspect_sessions(sessions_root)
        effective_seed = (
            secrets.randbits(64)
            if args.random_count is not None and args.seed is None
            else args.seed
        )
        selected = select_descriptors(
            descriptors,
            session_filters=args.session,
            include_draft=args.include_draft,
            include_no_truth=args.include_no_truth,
            include_ineligible=args.include_ineligible,
            random_count=args.random_count,
            strict_only=args.random_count is not None,
            seed=effective_seed,
        )
        if not selected:
            print("没有找到符合条件且包含录像的 session", file=sys.stderr)
            return 2
        print(
            f"准备开始核心监听回归：已选择 {len(selected)} 局；"
            f"随机种子={effective_seed if args.random_count is not None else '不适用'}；"
            f"并发线程={min(workers, len(selected))}；"
            "视觉监听和 FableDan 会按 session 并发执行。",
            flush=True,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        service = SessionReplayAuditService(profile_root=profile_root)
        progress = _ConsoleProgress(len(selected))
        # Use only explicit selected session paths here. Passing the parent
        # sessions directory as an audit root would add every session again and
        # defeat the default verified-only selection.
        run = service.audit(
            [],
            output=output_root,
            run_id=args.run_id,
            session_paths=[item.root for item in selected],
            command=[str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
            on_progress=progress,
            max_workers=workers,
        )
        progress.finish()
        raw_summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
        quality = evaluate_run(raw_summary, include_ineligible=args.include_ineligible)
        wrapper = {
            "schema": "guandan.listener-core-regression/1",
            "status": quality["status"],
            "source_summary": str(run.summary_path),
            "verification": str(run.verification_path) if run.verification_path else None,
            "sessions_root": str(sessions_root),
            "profile_root": str(profile_root),
            "selection": {
                "verified": sum(item.truth_status == "verified" for item in selected),
                "draft": sum(item.truth_status == "draft" for item in selected),
                "missing_or_invalid": sum(item.truth_status in {"missing", "invalid"} for item in selected),
                "random_count": args.random_count,
                "random_seed": effective_seed,
                "include_ineligible": args.include_ineligible,
                "qualifications": [qualify_descriptor(item) for item in selected],
                "session_ids": [item.session_id for item in selected],
                "workers": min(workers, len(selected)),
            },
            "quality": quality,
            "fabledan": raw_summary.get("fabledan", {}),
        }
        wrapper_path = run.run_directory / "listener_core_regression.json"
        atomic_write_json(wrapper_path, wrapper)
        _write_llm_artifacts(
            run.run_directory,
            raw_summary=raw_summary,
            quality=quality,
        )
        markdown_path = run.run_directory / "listener_core_regression.md"
        _write_markdown_report(
            markdown_path,
            quality,
            source_summary=run.summary_path,
            fabledan=raw_summary.get("fabledan", {}),
        )
        _write_artifact_manifest(run.run_directory)
        _print_console_report(quality, summary_path=markdown_path)
        return 0 if quality["status"] == "PASS" and run.execution_ok else 1
    except Exception as exc:
        if "progress" in locals():
            progress.finish()
        print(f"核心监听回归执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
