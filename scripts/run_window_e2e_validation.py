"""Host orchestrator for source plus isolated frozen-window validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daguandan_bridge.application.window_e2e_validation import snapshot_paths
from daguandan_bridge.storage import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在同一个可见模拟窗口上验证源码和独立打包 EXE。"
    )
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "window_e2e_validation",
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--profile-source",
        type=Path,
        default=PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan",
    )
    parser.add_argument(
        "--baseline-summary",
        type=Path,
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--run-timeout", type=float, default=3600.0)
    parser.add_argument("--drain-timeout", type=float, default=60.0)
    parser.add_argument(
        "--scenarios",
        default="initial_capture,move_recovery,resize_recovery,minimize_recovery,occlusion,dpi,full_chain",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="仅供开发短测；跳过打包和 EXE 验证，不构成第三阶段验收。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = _safe_name(
        args.run_id
        or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    )
    output_root = args.output.resolve()
    run_directory = output_root / "runs" / run_id
    build_root = output_root / "build" / run_id
    if run_directory.exists() or build_root.exists():
        raise SystemExit(f"run/build directory already exists for run-id {run_id}")
    run_directory.mkdir(parents=True)
    control_path = run_directory / "simulator_control.json"
    status_path = run_directory / "simulator_status.json"
    simulator_config_path = run_directory / "simulator_config.json"
    simulator_log = run_directory / "simulator_process.log"
    source_log = run_directory / "source_validator_process.log"
    bundle_log = run_directory / "bundle_validator_process.log"
    package_log = run_directory / "package_process.log"
    session = args.session.resolve()
    profile_source = args.profile_source.resolve()
    existing_release = PROJECT_ROOT / "release"
    protected = (session, existing_release)
    integrity_before = snapshot_paths(protected)
    existing_targets = _exact_title_hwnds("大掼蛋（腾讯）")
    if existing_targets:
        atomic_write_json(
            run_directory / "host_summary.json",
            {
                "execution_ok": False,
                "errors": [
                    "preflight found an existing exact-title target; close it before validation"
                ],
                "conflicting_hwnds": existing_targets,
            },
        )
        return 2
    scenarios = tuple(item.strip() for item in args.scenarios.split(",") if item.strip())
    simulator_config = {
        "session": str(session),
        "control_path": str(control_path),
        "status_path": str(status_path),
        "event_log_path": str(run_directory / "simulator_events.jsonl"),
        "window_title": "大掼蛋（腾讯）",
        "client_size": [1280, 764],
        "header_height": 44,
        "time_scale": args.time_scale,
        "autoplay": False,
        "max_frames": args.max_frames,
    }
    atomic_write_json(simulator_config_path, simulator_config)
    simulator: subprocess.Popen[bytes] | None = None
    source_exit: int | None = None
    package_exit: int | None = None
    bundle_exit: int | None = None
    source_summary: dict[str, object] = {}
    bundle_summary: dict[str, object] = {}
    host_errors: list[str] = []
    forced_simulator_termination = False
    simulator_handle = simulator_log.open("wb")
    try:
        # This process is intentionally visible: do not use CREATE_NO_WINDOW
        # or Start-Process -WindowStyle Hidden for the simulator.
        simulator = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "run.py"),
                "--simulated-game-window-config",
                str(simulator_config_path),
            ],
            cwd=PROJECT_ROOT,
            stdout=simulator_handle,
            stderr=subprocess.STDOUT,
        )
        status = _wait_status_ready(status_path, simulator, timeout=30.0)
        hwnd = int(status["hwnd"])
        common = {
            "session": str(session),
            "reference_profile": str(profile_source),
            "simulator_control_path": str(control_path),
            "simulator_status_path": str(status_path),
            "expected_hwnd": hwnd,
            "scenarios": list(scenarios),
            "max_frames": args.max_frames,
            "simulator_time_scale": args.time_scale,
            "run_timeout_sec": args.run_timeout,
            "drain_timeout_sec": args.drain_timeout,
            "protected_paths": [str(path) for path in protected],
            "baseline_summary": (
                str(args.baseline_summary.resolve()) if args.baseline_summary else None
            ),
            "danzero_checkpoint_source": str(
                PROJECT_ROOT
                / "src"
                / "daguandan_bridge"
                / "danzero"
                / "_vendor"
                / "guandan_rlcard"
                / "baselines"
                / "danzero"
                / "q_network.ckpt"
            ),
        }
        source_config = {
            **common,
            "output": str(run_directory / "source"),
            "profile_source": str(profile_source),
            "run_kind": "source",
            "executable_path": sys.executable,
        }
        source_config_path = run_directory / "source_validation_config.json"
        atomic_write_json(source_config_path, source_config)
        source_exit = _run_logged(
            [
                sys.executable,
                str(PROJECT_ROOT / "run.py"),
                "--window-e2e-validation-config",
                str(source_config_path),
            ],
            source_log,
        )
        source_summary = _read_json(run_directory / "source" / "summary.json")
        if source_exit != 0:
            host_errors.append(f"source validator exited {source_exit}")
        if not args.source_only:
            _simulator_command(control_path, status_path, "reset", timeout=15.0)
            build_root.parent.mkdir(parents=True, exist_ok=True)
            package_exit = _run_logged(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(PROJECT_ROOT / "scripts" / "package_release.ps1"),
                    "-ReleaseRoot",
                    str(build_root),
                ],
                package_log,
            )
            executable = (
                build_root
                / "dist"
                / "DaguandanAssistant"
                / "DaguandanAssistant.exe"
            )
            bundle_profile = (
                executable.parent / "data" / "profiles" / "tencent_daguandan"
            )
            if package_exit != 0 or not executable.is_file():
                host_errors.append(
                    f"isolated package failed (exit={package_exit}, exe={executable.is_file()})"
                )
            else:
                bundle_config = {
                    **common,
                    "output": str(run_directory / "bundle"),
                    "profile_source": str(bundle_profile),
                    "run_kind": "frozen_exe",
                    "executable_path": str(executable),
                }
                bundle_config_path = run_directory / "bundle_validation_config.json"
                atomic_write_json(bundle_config_path, bundle_config)
                bundle_exit = _run_logged(
                    [
                        str(executable),
                        "--window-e2e-validation-config",
                        str(bundle_config_path),
                    ],
                    bundle_log,
                )
                bundle_summary = _read_json(run_directory / "bundle" / "summary.json")
                if bundle_exit != 0:
                    host_errors.append(f"bundle validator exited {bundle_exit}")
    except Exception as exc:
        host_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if simulator is not None and simulator.poll() is None:
            try:
                _simulator_command(control_path, status_path, "close", timeout=10.0)
                simulator.wait(timeout=15.0)
            except Exception:
                forced_simulator_termination = True
                simulator.terminate()
                try:
                    simulator.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    simulator.kill()
                    simulator.wait(timeout=5.0)
        simulator_handle.close()
    integrity_after = snapshot_paths(protected)
    integrity_unchanged = integrity_before == integrity_after
    if not integrity_unchanged:
        host_errors.append("existing release or source session changed")
    same_hwnd = bool(
        source_summary
        and (
            args.source_only
            or bundle_summary
            and source_summary.get("observed_hwnd") == bundle_summary.get("observed_hwnd")
        )
    )
    if not same_hwnd:
        host_errors.append("source and bundle did not validate the same simulator HWND")
    complete_matrix = not args.source_only
    qualification_run = bool(
        complete_matrix
        and args.baseline_summary is not None
        and args.max_frames is None
        and abs(float(args.time_scale) - 1.0) <= 1e-9
        and set(scenarios)
        == {
            "initial_capture",
            "move_recovery",
            "resize_recovery",
            "minimize_recovery",
            "occlusion",
            "dpi",
            "full_chain",
        }
    )
    execution_ok = bool(
        not host_errors
        and complete_matrix
        and source_exit == 0
        and package_exit == 0
        and bundle_exit == 0
        and source_summary.get("execution_ok")
        and bundle_summary.get("execution_ok")
        and integrity_unchanged
        and same_hwnd
        and not forced_simulator_termination
    )
    acceptance_passed = bool(
        execution_ok
        and qualification_run
        and source_summary.get("acceptance_passed")
        and bundle_summary.get("acceptance_passed")
    )
    host_summary = {
        "schema": "guandan.window-e2e-host-summary/1",
        "run_id": run_id,
        "execution_ok": execution_ok,
        "acceptance_eligible": qualification_run,
        "acceptance_passed": acceptance_passed,
        "complete_source_and_bundle_matrix": complete_matrix,
        "source_only_debug_run": bool(args.source_only),
        "source_exit_code": source_exit,
        "package_exit_code": package_exit,
        "bundle_exit_code": bundle_exit,
        "source_summary": str(run_directory / "source" / "summary.json"),
        "bundle_summary": str(run_directory / "bundle" / "summary.json"),
        "build_root": str(build_root),
        "same_hwnd": same_hwnd,
        "integrity_unchanged": integrity_unchanged,
        "forced_simulator_termination": forced_simulator_termination,
        "errors": host_errors,
    }
    atomic_write_json(run_directory / "host_summary.json", host_summary)
    (run_directory / "host_summary.md").write_text(
        _markdown(host_summary), encoding="utf-8"
    )
    print(run_directory / "host_summary.json")
    return 0 if execution_ok else 1


def _run_logged(command: list[str], log_path: Path) -> int:
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _wait_status_ready(
    status_path: Path,
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"simulator exited before ready: {process.returncode}")
        last = _read_json(status_path)
        if last.get("ready"):
            return last
        if last.get("last_error"):
            raise RuntimeError(f"simulator initialization failed: {last['last_error']}")
        time.sleep(0.05)
    raise TimeoutError(f"simulator ready timeout; last_status={last}")


def _simulator_command(
    control_path: Path,
    status_path: Path,
    command: str,
    arguments: dict[str, object] | None = None,
    *,
    timeout: float,
) -> dict[str, object]:
    request_id = f"HOST-{os.getpid()}-{time.monotonic_ns()}"
    atomic_write_json(
        control_path,
        {
            "request_id": request_id,
            "command": command,
            "arguments": dict(arguments or {}),
        },
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _read_json(status_path)
        if status.get("last_request_id") == request_id:
            if status.get("last_error"):
                raise RuntimeError(str(status["last_error"]))
            return status
        time.sleep(0.05)
    raise TimeoutError(f"simulator command timeout: {command}")


def _exact_title_hwnds(title: str) -> list[int]:
    if sys.platform != "win32":
        return []
    import win32gui

    matches: list[int] = []

    def callback(hwnd: int, _value: object) -> bool:
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd).strip() == title:
            matches.append(int(hwnd))
        return True

    win32gui.EnumWindows(callback, None)
    return matches


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _safe_name(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(value)
    ).strip("_")
    if not normalized:
        raise ValueError("run-id must contain a safe character")
    return normalized[:100]


def _markdown(summary: dict[str, object]) -> str:
    acceptance_result = (
        "PASS"
        if summary.get("acceptance_passed")
        else "NOT_ACCEPTANCE"
        if summary.get("execution_ok")
        else "FAIL"
    )
    return "\n".join(
        (
            "# 第三阶段 Host 验收汇总",
            "",
            f"- 第三阶段验收：{acceptance_result}",
            f"- 运行执行结果：{'PASS' if summary.get('execution_ok') else 'FAIL'}",
            f"- 验收资格：{summary.get('acceptance_eligible')}",
            f"- 源码退出码：{summary.get('source_exit_code')}",
            f"- 打包退出码：{summary.get('package_exit_code')}",
            f"- EXE 退出码：{summary.get('bundle_exit_code')}",
            f"- 同一 HWND：{summary.get('same_hwnd')}",
            f"- 原有 release/sessions 未变：{summary.get('integrity_unchanged')}",
            "",
            "## 错误",
            "",
            *(
                f"- {error}" for error in summary.get("errors", [])
            ),
            "",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
