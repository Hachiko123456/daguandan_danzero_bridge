from __future__ import annotations

"""Read-only dry-run contract checker for a fixed WeChat capture window.

This module deliberately has no dependency on the application runtime.  It reads
JSON configuration and runtime-layout evidence, then emits one JSON document.  It
never finds a HWND, moves/resizes a window, captures pixels, starts WeChat, or
removes files.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


SCHEMA = "guandan.fixed-window-portability-dry-run/1"
_ALLOWED_BACKENDS = {"auto", "printwindow", "screen", "gdi_screen"}
_RUNTIME_ABSOLUTE_KEYS = {
    "bundle_root",
    "resource_data_dir",
    "runtime_root",
    "generation_root",
    "data_dir",
    "profiles_root",
    "logs_root",
    "diagnostics_root",
    "preferences_root",
    "cache_root",
    "active_generation_path",
}
_PATH_KEY_RE = re.compile(
    r"(?:^|_)(?:path|file|dir|root|model|resource|template|profile|session|data)(?:$|_)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"^(?:\\\\|//)")


def _json_read(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable: {exc}"


def _is_absolute_string(value: str) -> bool:
    return bool(
        _WINDOWS_ABSOLUTE_RE.match(value)
        or _UNC_RE.match(value)
        or value.startswith("/")
    )


def _looks_like_path(key: str, value: str) -> bool:
    return bool(
        _PATH_KEY_RE.search(key)
        or "/" in value
        or "\\" in value
        or _WINDOWS_ABSOLUTE_RE.match(value)
        or _UNC_RE.match(value)
    )


def _walk_strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _walk_strings(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _finding(
    code: str,
    severity: str,
    message: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": dict(evidence or {}),
    }


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": status,
        "message": message,
        "evidence": dict(evidence or {}),
    }


def _discover_profile(project_root: Path, profile: Path | None) -> Path | None:
    if profile is not None:
        return profile if profile.is_absolute() else project_root / profile
    candidates = sorted((project_root / "data" / "profiles").glob("*/profile.json"))
    return candidates[0] if candidates else None


def _runtime_candidates(project_root: Path, runtime_root: Path | None) -> list[Path]:
    root = runtime_root or project_root
    return [
        root / "runtime_layout.json",
        root / ".daguandan-user-data-root.json",
        root / "data" / "v1" / "runtime_layout.json",
        root / "data" / "v1" / "active.json",
    ]


def _path_risks(documents: Iterable[tuple[str, Path, Any]]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for document_kind, source, document in documents:
        if document is None:
            continue
        for key, value in _walk_strings(document):
            if not _looks_like_path(key, value):
                continue
            absolute = _is_absolute_string(value)
            runtime_structural = (
                document_kind == "runtime_layout"
                and key.split(".")[-1] in _RUNTIME_ABSOLUTE_KEYS
            )
            if absolute:
                risks.append(
                    {
                        "kind": "absolute_path",
                        "severity": "warning" if runtime_structural else "blocker",
                        "document": document_kind,
                        "source": str(source),
                        "field": key,
                        "value": value,
                        "portable": False,
                        "reason": (
                            "runtime layout root is machine-local and must be derived"
                            if runtime_structural
                            else "persisted absolute path binds evidence to one machine"
                        ),
                    }
                )
            else:
                safe_relative = not any(part in {"", ".", ".."} for part in value.replace("\\", "/").split("/"))
                risks.append(
                    {
                        "kind": "relative_path",
                        "severity": "info" if safe_relative else "blocker",
                        "document": document_kind,
                        "source": str(source),
                        "field": key,
                        "value": value,
                        "portable": safe_relative,
                        "reason": (
                            "relative path is portable only when resolved below the selected runtime root"
                            if safe_relative
                            else "relative path contains traversal or an empty segment"
                        ),
                    }
                )
    return risks


def build_report(
    *,
    project_root: Path,
    profile_path: Path | None = None,
    runtime_root: Path | None = None,
    frozen: bool | None = None,
) -> dict[str, Any]:
    """Build a deterministic, read-only report for one profile/layout pair.

    Source checkouts intentionally do not have frozen runtime-layout evidence.
    The evidence is mandatory for frozen/published runs; callers may pass
    ``frozen=True`` to make that release-mode contract explicit.
    """
    project_root = project_root.resolve()
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
    source_profiles_root = project_root / "data" / "profiles"
    is_source_checkout = source_profiles_root.is_dir() and not is_frozen
    selected_profile = _discover_profile(project_root, profile_path)
    profile_doc: Any | None = None
    profile_error: str | None = "missing profile path"
    if selected_profile is not None:
        profile_doc, profile_error = _json_read(selected_profile)

    runtime_docs: list[dict[str, Any]] = []
    for candidate in _runtime_candidates(project_root, runtime_root):
        document, error = _json_read(candidate)
        if document is not None or error != "missing":
            runtime_docs.append(
                {
                    "path": str(candidate),
                    "present": document is not None,
                    "error": error,
                    "document": document,
                }
            )
    layout_doc = next(
        (item["document"] for item in runtime_docs if item["document"] is not None),
        None,
    )

    findings: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    profile = profile_doc if isinstance(profile_doc, Mapping) else {}

    allow_resize = profile.get("allow_resize")
    if not isinstance(allow_resize, bool):
        checks.append(_check("allow_resize", "FAIL", "profile 必须显式声明布尔值 allow_resize=false", evidence={"value": allow_resize}))
        findings.append(_finding("profile.allow_resize_missing_or_invalid", "blocker", "固定窗口契约要求显式禁止 resize", evidence={"value": allow_resize}))
    elif allow_resize:
        checks.append(_check("allow_resize", "FAIL", "allow_resize=true 会破坏固定客户区契约", evidence={"value": True}))
        findings.append(_finding("profile.resize_enabled", "blocker", "固定窗口不得允许自动调整大小"))
    else:
        checks.append(_check("allow_resize", "PASS", "allow_resize=false", evidence={"value": False}))

    backend = profile.get("capture_backend")
    normalized_backend = str(backend).strip().lower() if isinstance(backend, str) else ""
    if normalized_backend not in _ALLOWED_BACKENDS:
        checks.append(_check("capture_backend", "FAIL", "capture_backend 缺失或不受支持", evidence={"value": backend, "allowed": sorted(_ALLOWED_BACKENDS)}))
        findings.append(_finding("profile.capture_backend_invalid", "blocker", "必须选择受支持的捕获后端", evidence={"value": backend}))
    elif normalized_backend != "printwindow":
        checks.append(_check("capture_backend", "FAIL", "固定窗口遮挡验收要求 capture_backend=printwindow", evidence={"value": normalized_backend}))
        findings.append(_finding("profile.capture_backend_not_printwindow", "blocker", "screen/gdi_screen 只读取可见桌面；auto 允许回退，不能证明遮挡场景", evidence={"value": normalized_backend}))
    else:
        checks.append(_check("capture_backend", "PASS", "capture_backend=printwindow", evidence={"value": normalized_backend}))

    fallback = profile.get("allow_screen_fallback")
    if fallback is True:
        checks.append(_check("allow_screen_fallback", "FAIL", "遮挡验收不得允许回退到屏幕采集", evidence={"value": True}))
        findings.append(_finding("profile.screen_fallback_enabled", "blocker", "allow_screen_fallback=true 会掩盖 PrintWindow 失败并引入遮挡依赖"))
    elif fallback is False:
        checks.append(_check("allow_screen_fallback", "PASS", "allow_screen_fallback=false", evidence={"value": False}))
    else:
        checks.append(_check("allow_screen_fallback", "WARN", "未显式声明 allow_screen_fallback；建议声明 false", evidence={"value": fallback}))

    base_size = profile.get("base_size")
    target_size = profile.get("target_client_size")
    geometry_ok = (
        isinstance(base_size, list) and len(base_size) == 2
        and all(isinstance(item, int) and item > 0 for item in base_size)
        and isinstance(target_size, list) and len(target_size) == 2
        and all(isinstance(item, int) and item > 0 for item in target_size)
        and target_size[0] >= base_size[0] and target_size[1] >= base_size[1]
    )
    if geometry_ok:
        checks.append(_check("fixed_client_geometry", "PASS", "target_client_size 覆盖 base_size 且为正整数", evidence={"base_size": base_size, "target_client_size": target_size}))
    else:
        checks.append(_check("fixed_client_geometry", "FAIL", "base_size/target_client_size 缺失或不满足固定窗口几何约束", evidence={"base_size": base_size, "target_client_size": target_size}))
        findings.append(_finding("profile.client_geometry_invalid", "blocker", "无法证明固定客户区尺寸", evidence={"base_size": base_size, "target_client_size": target_size}))

    if selected_profile is None or profile_error:
        checks.append(_check("profile_readable", "FAIL", "profile.json 不存在或不可解析", evidence={"path": str(selected_profile) if selected_profile else None, "error": profile_error}))
        findings.append(_finding("profile.unreadable", "blocker", "缺少可审计的 profile.json", evidence={"error": profile_error}))
    else:
        checks.append(_check("profile_readable", "PASS", "profile.json 可解析", evidence={"path": str(selected_profile)}))

    if layout_doc is None:
        if is_source_checkout:
            checks.append(
                _check(
                    "runtime_layout_evidence",
                    "WARN",
                    "source checkout 不要求 runtime-layout evidence；发布/冻结运行仍必须提供",
                    evidence={
                        "mode": "source_checkout",
                        "checked": [item["path"] for item in runtime_docs],
                        "evidence_required_for_frozen": True,
                    },
                )
            )
        else:
            checks.append(
                _check(
                    "runtime_layout_evidence",
                    "FAIL",
                    "发布/冻结运行缺少 runtime layout evidence；无法完成跨电脑 dry-run",
                    evidence={
                        "mode": "frozen_or_publish",
                        "checked": [item["path"] for item in runtime_docs],
                    },
                )
            )
            findings.append(
                _finding(
                    "runtime_layout.missing",
                    "blocker",
                    "发布/冻结运行需要 runtime_layout.json、runtime-root marker 或 active.json 之一",
                )
            )
    else:
        checks.append(
            _check(
                "runtime_layout_evidence",
                "PASS",
                "找到 runtime layout evidence",
                evidence={
                    "mode": "frozen_or_publish" if is_frozen else "source_checkout",
                    "documents": [
                        item["path"]
                        for item in runtime_docs
                        if item["document"] is not None
                    ],
                },
            )
        )

    documents: list[tuple[str, Path, Any]] = []
    if selected_profile is not None:
        documents.append(("profile", selected_profile, profile_doc))
    for item in runtime_docs:
        if item["document"] is not None:
            documents.append(("runtime_layout", Path(item["path"]), item["document"]))
    path_risks = _path_risks(documents)
    path_blockers = [item for item in path_risks if item["severity"] == "blocker"]
    if path_blockers:
        checks.append(_check("path_portability", "FAIL", "发现会绑定当前电脑的路径风险", evidence={"blocker_count": len(path_blockers), "risk_count": len(path_risks)}))
        findings.append(_finding("paths.non_portable", "blocker", "配置或证据中存在不可移植路径", evidence={"blockers": path_blockers}))
    else:
        checks.append(_check("path_portability", "PASS", "没有发现配置级绝对路径或危险相对路径", evidence={"risk_count": len(path_risks), "risks": path_risks}))

    failed = any(item["status"] == "FAIL" for item in checks)
    return {
        "schema": SCHEMA,
        "report_type": "read_only_dry_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "execution_mode": "source_checkout" if is_source_checkout else "frozen_or_publish",
        "inputs": {
            "profile_path": str(selected_profile) if selected_profile else None,
            "runtime_root": str((runtime_root or project_root).resolve()),
            "runtime_documents": runtime_docs,
        },
        "observed_profile": {
            "allow_resize": allow_resize,
            "capture_backend": normalized_backend or None,
            "allow_screen_fallback": fallback,
            "base_size": base_size,
            "target_client_size": target_size,
            "viewport_mode": profile.get("viewport_mode"),
            "viewport_aspect_ratio": profile.get("viewport_aspect_ratio"),
        },
        "checks": checks,
        "path_risks": path_risks,
        "findings": findings,
        "summary": {
            "status": "FAIL" if failed else "PASS",
            "passed_checks": sum(item["status"] == "PASS" for item in checks),
            "failed_checks": sum(item["status"] == "FAIL" for item in checks),
            "warning_count": sum(item["status"] == "WARN" for item in checks),
            "finding_count": len(findings),
        },
        "side_effects": {
            "controls_wechat": False,
            "captures_screen": False,
            "resizes_window": False,
            "deletes_files": False,
            "writes_files_by_default": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定微信窗口与跨电脑运行的只读 dry-run 验收报告")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="工程根目录")
    parser.add_argument("--profile", type=Path, help="profile.json 路径；默认读取 data/profiles/*/profile.json")
    parser.add_argument("--runtime-root", type=Path, help="运行时根目录；默认使用工程根目录并检查标准 marker")
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="按发布/冻结运行检查；此模式缺少 runtime-layout evidence 会失败",
    )
    parser.add_argument("--output", type=Path, help="可选：将同一份 JSON 写入指定文件；不指定时只输出 stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        project_root=args.root,
        profile_path=args.profile,
        runtime_root=args.runtime_root,
        frozen=args.frozen,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0 if report["summary"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
