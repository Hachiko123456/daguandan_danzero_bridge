"""Phase-three Win32 window-capture and frozen-bundle validation runner."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication

from ..bootstrap import build_live_controller_dependencies
from ..capture_service import CaptureService, FrameSnapshot, LiveCaptureInterrupted
from ..gui.live_controller import LiveAssistantController
from ..live.session_store import read_json_lines
from ..models import ClientRect, TargetWindow
from ..storage import append_json_line, atomic_write_json
from ..window_capture import (
    find_target_window,
    get_client_rect_on_screen,
    get_window_dpi,
)
from .shadow_live_replay import summarize_advice_lifecycle


_ACTION_TYPES = frozenset(
    {"player_played", "player_passed", "manual_confirmed_event"}
)
_TERMINAL_ADVICE = frozenset(
    {"ready", "failed", "stale", "withheld", "timeout", "cancelled"}
)
_DEFAULT_SCENARIOS = (
    "initial_capture",
    "move_recovery",
    "resize_recovery",
    "minimize_recovery",
    "occlusion",
    "dpi",
    "full_chain",
)


def _resolve(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class WindowE2EValidationConfig:
    session: Path
    output: Path
    profile_source: Path
    reference_profile: Path
    simulator_control_path: Path
    simulator_status_path: Path
    profile_name: str = "tencent_daguandan"
    target_title: str = "大掼蛋（腾讯）"
    expected_hwnd: int | None = None
    run_kind: str = "source"
    scenarios: tuple[str, ...] = _DEFAULT_SCENARIOS
    max_frames: int | None = None
    simulator_time_scale: float = 1.0
    run_timeout_sec: float = 3600.0
    drain_timeout_sec: float = 60.0
    command_timeout_sec: float = 10.0
    pixel_mae_limit: float = 8.0
    perceptual_hash_distance_limit: int = 8
    baseline_summary: Path | None = None
    danzero_checkpoint_source: Path | None = None
    executable_path: Path | None = None
    protected_paths: tuple[Path, ...] = ()
    required_dpis: tuple[int, ...] = (96, 120)
    optional_dpis: tuple[int, ...] = (144,)
    capture_interval_sec: float = 0.02

    def __post_init__(self) -> None:
        if not (self.session / "video" / "game.avi").is_file():
            raise ValueError(f"invalid validation session: {self.session}")
        if not (self.session / "video" / "frame_index.jsonl").is_file():
            raise ValueError(f"validation session has no frame index: {self.session}")
        for profile in (self.profile_source, self.reference_profile):
            for name in ("profile.json", "regions_config.json", "templates_config.json"):
                if not (profile / name).is_file():
                    raise ValueError(f"profile resource is missing: {profile / name}")
        if _is_relative_to(self.output, self.session.parent):
            raise ValueError("window E2E output must be outside source sessions")
        if not math.isfinite(self.simulator_time_scale) or self.simulator_time_scale <= 0:
            raise ValueError("simulator_time_scale must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.baseline_summary is not None and not self.baseline_summary.is_file():
            raise ValueError(f"baseline summary does not exist: {self.baseline_summary}")
        if min(self.run_timeout_sec, self.drain_timeout_sec, self.command_timeout_sec) <= 0:
            raise ValueError("validation timeouts must be positive")
        if not math.isfinite(self.capture_interval_sec) or self.capture_interval_sec <= 0:
            raise ValueError("capture_interval_sec must be positive")
        unknown = set(self.scenarios) - set(_DEFAULT_SCENARIOS)
        if unknown:
            raise ValueError(f"unknown window E2E scenarios: {sorted(unknown)}")

    @classmethod
    def from_path(cls, path: Path | str) -> "WindowE2EValidationConfig":
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("window E2E config must be a JSON object")
        base = config_path.parent
        profile_source = _resolve(raw["profile_source"], base)
        reference_profile = _resolve(
            raw.get("reference_profile", profile_source), base
        )
        baseline = raw.get("baseline_summary")
        checkpoint = raw.get("danzero_checkpoint_source")
        executable = raw.get("executable_path")
        return cls(
            session=_resolve(raw["session"], base),
            output=_resolve(raw["output"], base),
            profile_source=profile_source,
            reference_profile=reference_profile,
            simulator_control_path=_resolve(raw["simulator_control_path"], base),
            simulator_status_path=_resolve(raw["simulator_status_path"], base),
            profile_name=str(raw.get("profile_name", "tencent_daguandan")),
            target_title=str(raw.get("target_title", "大掼蛋（腾讯）")),
            expected_hwnd=(
                int(raw["expected_hwnd"])
                if raw.get("expected_hwnd") is not None
                else None
            ),
            run_kind=str(raw.get("run_kind", "source")),
            scenarios=tuple(str(item) for item in raw.get("scenarios", _DEFAULT_SCENARIOS)),
            max_frames=(
                int(raw["max_frames"]) if raw.get("max_frames") is not None else None
            ),
            simulator_time_scale=float(raw.get("simulator_time_scale", 1.0)),
            run_timeout_sec=float(raw.get("run_timeout_sec", 3600.0)),
            drain_timeout_sec=float(raw.get("drain_timeout_sec", 60.0)),
            command_timeout_sec=float(raw.get("command_timeout_sec", 10.0)),
            pixel_mae_limit=float(raw.get("pixel_mae_limit", 8.0)),
            perceptual_hash_distance_limit=int(
                raw.get("perceptual_hash_distance_limit", 8)
            ),
            baseline_summary=_resolve(baseline, base) if baseline else None,
            danzero_checkpoint_source=(
                _resolve(checkpoint, base) if checkpoint else None
            ),
            executable_path=_resolve(executable, base) if executable else None,
            protected_paths=tuple(
                _resolve(item, base) for item in raw.get("protected_paths", ())
            ),
            required_dpis=tuple(int(item) for item in raw.get("required_dpis", (96, 120))),
            optional_dpis=tuple(int(item) for item in raw.get("optional_dpis", (144,))),
            capture_interval_sec=float(raw.get("capture_interval_sec", 0.02)),
        )

    @property
    def acceptance_eligible(self) -> bool:
        """Whether this run is allowed to claim phase-three acceptance."""

        return bool(
            self.baseline_summary is not None
            and self.max_frames is None
            and math.isclose(self.simulator_time_scale, 1.0, abs_tol=1e-9)
            and set(self.scenarios) == set(_DEFAULT_SCENARIOS)
        )


class SimulatorControlClient:
    """File-based simulator API; validators never depend on child stdout."""

    def __init__(
        self,
        control_path: Path,
        status_path: Path,
        *,
        timeout: float,
        pump: Callable[[], None],
    ) -> None:
        self.control_path = Path(control_path)
        self.status_path = Path(status_path)
        self.timeout = float(timeout)
        self.pump = pump
        self._sequence = 0

    def status(self) -> dict[str, object]:
        if not self.status_path.is_file():
            return {}
        try:
            value = json.loads(self.status_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def wait_ready(self) -> dict[str, object]:
        return self.wait_for(lambda row: bool(row.get("ready")), "simulator ready")

    def command(
        self,
        command: str,
        arguments: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self._sequence += 1
        request_id = f"VAL-{os.getpid()}-{self._sequence:04d}"
        atomic_write_json(
            self.control_path,
            {
                "request_id": request_id,
                "command": str(command),
                "arguments": dict(arguments or {}),
            },
        )
        result = self.wait_for(
            lambda row: row.get("last_request_id") == request_id,
            f"simulator command {command}",
        )
        if result.get("last_error"):
            raise RuntimeError(
                f"simulator command {command} failed: {result['last_error']}"
            )
        return result

    def wait_for(
        self,
        predicate: Callable[[dict[str, object]], bool],
        description: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, object]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        last: dict[str, object] = {}
        while time.monotonic() < deadline:
            self.pump()
            last = self.status()
            if predicate(last):
                return last
            time.sleep(0.02)
        raise TimeoutError(f"timed out waiting for {description}; last_status={last}")


def perceptual_hash(image: np.ndarray) -> str:
    """Return a compact gradient perceptual hash for capture evidence."""

    gray = cv2.cvtColor(_as_bgr(image), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (resized[:, 1:] >= resized[:, :-1]).flatten()
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def perceptual_hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def compare_pixels(
    captured: np.ndarray,
    expected: np.ndarray,
    *,
    mae_limit: float,
    hash_distance_limit: int,
) -> dict[str, object]:
    actual = _as_bgr(captured)
    source = _as_bgr(expected)
    same_size = actual.shape == source.shape
    if not same_size:
        return {
            "passed": False,
            "same_size": False,
            "captured_shape": list(actual.shape),
            "expected_shape": list(source.shape),
            "mae": None,
            "perceptual_hash_distance": None,
            "mae_limit": mae_limit,
            "perceptual_hash_distance_limit": hash_distance_limit,
        }
    mae = float(np.mean(np.abs(actual.astype(np.float32) - source.astype(np.float32))))
    actual_hash = perceptual_hash(actual)
    expected_hash = perceptual_hash(source)
    distance = perceptual_hash_distance(actual_hash, expected_hash)
    return {
        "passed": mae <= mae_limit and distance <= hash_distance_limit,
        "same_size": True,
        "captured_shape": list(actual.shape),
        "expected_shape": list(source.shape),
        "mae": mae,
        "perceptual_hash": actual_hash,
        "expected_perceptual_hash": expected_hash,
        "perceptual_hash_distance": distance,
        "mae_limit": mae_limit,
        "perceptual_hash_distance_limit": hash_distance_limit,
    }


def build_resource_manifest(
    reference_profile: Path,
    candidate_profile: Path,
    *,
    reference_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Compare every required profile/template/model resource by file hash."""

    reference_profile = Path(reference_profile)
    candidate_profile = Path(candidate_profile)
    names = {"profile.json", "regions_config.json", "templates_config.json", "models/best.npz"}
    names.update(
        path.relative_to(reference_profile).as_posix()
        for path in (reference_profile / "templates").rglob("*")
        if path.is_file()
    )
    names.add("models/danzero/q_network.ckpt")
    rows: list[dict[str, object]] = []
    for name in sorted(names):
        reference = reference_profile / name
        if name == "models/danzero/q_network.ckpt" and not reference.is_file():
            reference = Path(reference_checkpoint) if reference_checkpoint else reference
        candidate = candidate_profile / name
        reference_hash = _file_hash(reference)
        candidate_hash = _file_hash(candidate)
        rows.append(
            {
                "path": name,
                "reference_path": str(reference),
                "candidate_path": str(candidate),
                "reference_exists": reference.is_file(),
                "candidate_exists": candidate.is_file(),
                "reference_sha256": reference_hash,
                "candidate_sha256": candidate_hash,
                "matches": bool(reference_hash and reference_hash == candidate_hash),
            }
        )
    return {
        "schema": "guandan.window-e2e-resource-manifest/1",
        "reference_profile": str(reference_profile),
        "candidate_profile": str(candidate_profile),
        "files": rows,
        "all_match": bool(rows) and all(bool(row["matches"]) for row in rows),
        "mismatches": [row["path"] for row in rows if not row["matches"]],
    }


def probe_monitors() -> list[dict[str, object]]:
    """Report physical monitor geometry and effective DPI without changing it."""

    if sys.platform != "win32":
        return []
    import win32api

    monitors: list[dict[str, object]] = []
    for handle, _dc, rect in win32api.EnumDisplayMonitors():
        info = win32api.GetMonitorInfo(handle)
        dpi_x = dpi_y = None
        error = None
        try:
            value_x = ctypes.c_uint()
            value_y = ctypes.c_uint()
            result = ctypes.windll.shcore.GetDpiForMonitor(  # type: ignore[attr-defined]
                int(handle),
                0,
                ctypes.byref(value_x),
                ctypes.byref(value_y),
            )
            if int(result) != 0:
                raise OSError(f"GetDpiForMonitor HRESULT={int(result)}")
            dpi_x, dpi_y = int(value_x.value), int(value_y.value)
        except Exception as exc:
            error = str(exc)
        work = tuple(int(item) for item in info.get("Work", rect))
        monitors.append(
            {
                "handle": int(handle),
                "device": str(info.get("Device", "")),
                "primary": bool(info.get("Flags", 0) & 1),
                "geometry": [int(item) for item in rect],
                "work_area": list(work),
                "dpi_x": dpi_x,
                "dpi_y": dpi_y,
                "dpi_error": error,
            }
        )
    return monitors


class WindowE2EValidator:
    def __init__(self, config: WindowE2EValidationConfig) -> None:
        self.config = config
        self.output = config.output
        self.scenario_directory = self.output / "scenarios"
        self.evidence_directory = self.output / "evidence"
        self.runtime_profiles_root = self.output / "runtime_profiles"
        self.runtime_profile = self.runtime_profiles_root / config.profile_name
        self.errors: list[str] = []
        self.scenario_results: dict[str, dict[str, object]] = {}
        self._app = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
        self.control = SimulatorControlClient(
            config.simulator_control_path,
            config.simulator_status_path,
            timeout=config.command_timeout_sec,
            pump=self._pump,
        )

    def run(self) -> dict[str, object]:
        self.output.mkdir(parents=True, exist_ok=False)
        self.scenario_directory.mkdir()
        self.evidence_directory.mkdir()
        protected_before = snapshot_paths(self.config.protected_paths)
        source_before = snapshot_paths((self.config.session,))
        started_at = datetime.now().astimezone().isoformat()
        resource_manifest: dict[str, object] = {}
        runtime: dict[str, object] = {"executed": False}
        try:
            status = self.control.wait_ready()
            self._assert_target(status)
            self._stage_runtime_profile()
            resource_manifest = build_resource_manifest(
                self.config.reference_profile,
                self.runtime_profile,
                reference_checkpoint=self._reference_checkpoint(),
            )
            atomic_write_json(self.output / "resource_manifest.json", resource_manifest)
            capture = CaptureService(self.runtime_profiles_root)
            for name in self.config.scenarios:
                if name == "full_chain":
                    runtime = self._record_scenario(name, self._run_full_chain)
                elif name == "initial_capture":
                    self._record_scenario(
                        name, lambda service=capture: self._initial_capture(service)
                    )
                elif name == "move_recovery":
                    self._record_scenario(
                        name, lambda service=capture: self._move_recovery(service)
                    )
                elif name == "resize_recovery":
                    self._record_scenario(
                        name, lambda service=capture: self._resize_recovery(service)
                    )
                elif name == "minimize_recovery":
                    self._record_scenario(
                        name, lambda service=capture: self._minimize_recovery(service)
                    )
                elif name == "occlusion":
                    self._record_scenario(name, self._occlusion)
                elif name == "dpi":
                    self._record_scenario(
                        name, lambda service=capture: self._dpi_matrix(service)
                    )
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
        source_after = snapshot_paths((self.config.session,))
        protected_after = snapshot_paths(self.config.protected_paths)
        source_unchanged = source_before == source_after
        protected_unchanged = protected_before == protected_after
        if not source_unchanged:
            self.errors.append("source session changed during validation")
        if not protected_unchanged:
            self.errors.append("protected release/session paths changed during validation")
        required_scenarios = set(self.config.scenarios) - {"dpi"}
        scenarios_pass = all(
            bool(self.scenario_results.get(name, {}).get("passed"))
            for name in required_scenarios
        )
        dpi_result = self.scenario_results.get("dpi")
        if dpi_result is not None:
            scenarios_pass = scenarios_pass and bool(dpi_result.get("passed"))
        resources_pass = bool(resource_manifest.get("all_match"))
        execution_ok = bool(
            not self.errors
            and scenarios_pass
            and resources_pass
            and source_unchanged
            and protected_unchanged
        )
        acceptance_passed = bool(execution_ok and self.config.acceptance_eligible)
        summary = {
            "schema": "guandan.window-e2e-summary/1",
            "run_kind": self.config.run_kind,
            "started_at": started_at,
            "finished_at": datetime.now().astimezone().isoformat(),
            "source_session": str(self.config.session),
            "target_title": self.config.target_title,
            "expected_hwnd": self.config.expected_hwnd,
            "observed_hwnd": self.control.status().get("hwnd"),
            "execution_ok": execution_ok,
            "acceptance_eligible": self.config.acceptance_eligible,
            "acceptance_passed": acceptance_passed,
            "scenarios": self.scenario_results,
            "runtime": runtime,
            "resources": {
                "manifest": str(self.output / "resource_manifest.json"),
                "all_match": resources_pass,
                "mismatches": resource_manifest.get("mismatches", []),
            },
            "runtime_identity": self._runtime_identity(),
            "integrity": {
                "source_session_unchanged": source_unchanged,
                "protected_paths_unchanged": protected_unchanged,
                "protected_paths": [str(path) for path in self.config.protected_paths],
            },
            "errors": self.errors,
            "required_artifacts": {},
        }
        required = (
            self.output / "summary.json",
            self.output / "summary.md",
            self.output / "resource_manifest.json",
        )
        summary["required_artifacts"] = {str(path): True for path in required}
        atomic_write_json(self.output / "summary.json", summary)
        (self.output / "summary.md").write_text(
            _summary_markdown(summary), encoding="utf-8"
        )
        return summary

    def _pump(self) -> None:
        self._app.processEvents()

    def _assert_target(self, status: dict[str, object]) -> None:
        hwnd = int(status.get("hwnd", 0) or 0)
        if not hwnd:
            raise RuntimeError("simulator status has no HWND")
        if self.config.expected_hwnd is not None and hwnd != self.config.expected_hwnd:
            raise RuntimeError(
                f"simulator HWND mismatch: expected {self.config.expected_hwnd}, got {hwnd}"
            )
        if status.get("window_title") != self.config.target_title:
            raise RuntimeError("simulator title does not exactly match the validation target")
        target = find_target_window((self.config.target_title,))
        if target.hwnd != hwnd:
            raise RuntimeError("Win32 target lookup did not resolve the simulator HWND")

    def _stage_runtime_profile(self) -> None:
        if self.runtime_profile.exists():
            raise FileExistsError(f"runtime profile already exists: {self.runtime_profile}")
        shutil.copytree(
            self.config.profile_source,
            self.runtime_profile,
            ignore=shutil.ignore_patterns(
                "sessions",
                "truth_log_batch_reports",
                "_quarantine*",
                "__pycache__",
            ),
        )
        checkpoint = self.config.danzero_checkpoint_source
        bundled = self.config.profile_source / "models" / "danzero" / "q_network.ckpt"
        source = bundled if bundled.is_file() else checkpoint or self._reference_checkpoint()
        target = self.runtime_profile / "models" / "danzero" / "q_network.ckpt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source is not None and Path(source).is_file():
            shutil.copy2(source, target)

    def _reference_checkpoint(self) -> Path | None:
        if self.config.danzero_checkpoint_source is not None:
            return self.config.danzero_checkpoint_source
        bundled = self.config.reference_profile / "models" / "danzero" / "q_network.ckpt"
        if bundled.is_file():
            return bundled
        candidate = (
            Path(__file__).resolve().parents[1]
            / "danzero"
            / "_vendor"
            / "guandan_rlcard"
            / "baselines"
            / "danzero"
            / "q_network.ckpt"
        )
        return candidate if candidate.is_file() else None

    def _record_scenario(
        self,
        name: str,
        operation: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        started = time.perf_counter()
        try:
            result = operation()
            result = {"passed": bool(result.get("passed")), **result}
        except Exception as exc:
            result = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self.errors.append(f"scenario {name}: {result['error']}")
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        self.scenario_results[name] = result
        atomic_write_json(self.scenario_directory / f"{name}.json", result)
        if not result["passed"] and result.get("error"):
            (self.scenario_directory / f"{name}.error.txt").write_text(
                str(result["error"]), encoding="utf-8"
            )
        return result

    def _initial_capture(self, service: CaptureService) -> dict[str, object]:
        status = self.control.command("reset")
        source = service.open_live_source(self.config.profile_name)
        try:
            snapshot = source.capture()
        finally:
            source.close()
        similarity = self._pixel_evidence(snapshot, status, "initial_capture")
        metadata = _snapshot_metadata(snapshot)
        expected_rect = {"width": 1280, "height": 764}
        rect = status.get("client_rect") or {}
        geometry_ok = all(rect.get(key) == value for key, value in expected_rect.items())
        return {
            "passed": bool(similarity["passed"] and geometry_ok),
            "simulator_status": status,
            "capture": metadata,
            "pixel_similarity": similarity,
            "geometry_ok": geometry_ok,
        }

    def _move_recovery(self, service: CaptureService) -> dict[str, object]:
        status = self.control.command("reset")
        rect = dict(status["client_rect"])  # type: ignore[arg-type]
        source = service.open_live_source(self.config.profile_name)
        source.capture()
        moved = self.control.command(
            "move",
            {"left": int(rect["left"]) + 80, "top": int(rect["top"]) + 50},
        )
        interrupted = _capture_is_interrupted(source)
        source.close()
        restored = self.control.command(
            "move", {"left": int(rect["left"]), "top": int(rect["top"])}
        )
        reopened = service.open_live_source(self.config.profile_name)
        try:
            recovered = _capture_succeeds(reopened)
        finally:
            reopened.close()
        return {
            "passed": interrupted and recovered,
            "old_source_interrupted": interrupted,
            "reopened_source_captured": recovered,
            "moved_status": moved,
            "restored_status": restored,
        }

    def _resize_recovery(self, service: CaptureService) -> dict[str, object]:
        self.control.command("reset")
        source = service.open_live_source(self.config.profile_name)
        source.capture()
        resized = self.control.command(
            "resize_client", {"width": 1100, "height": 700}
        )
        interrupted = _capture_is_interrupted(source)
        source.close()
        locked = service.lock_target_client_size(self.config.profile_name)
        reopened = service.open_live_source(self.config.profile_name)
        try:
            recovered = _capture_succeeds(reopened)
        finally:
            reopened.close()
        return {
            "passed": interrupted
            and recovered
            and (locked.width, locked.height) == (1280, 764),
            "old_source_interrupted": interrupted,
            "resized_status": resized,
            "locked_client_rect": _rect_dict(locked),
            "reopened_source_captured": recovered,
        }

    def _minimize_recovery(self, service: CaptureService) -> dict[str, object]:
        self.control.command("reset")
        source = service.open_live_source(self.config.profile_name)
        source.capture()
        minimized = self.control.command("minimize")
        interrupted = _capture_is_interrupted(source)
        source.close()
        self.control.command("restore")
        restored = self.control.wait_for(
            lambda row: isinstance(row.get("client_rect"), dict),
            "restored simulator geometry",
        )
        service.lock_target_client_size(self.config.profile_name)
        reopened = service.open_live_source(self.config.profile_name)
        try:
            recovered = _capture_succeeds(reopened)
        finally:
            reopened.close()
        return {
            "passed": interrupted and recovered,
            "old_source_interrupted": interrupted,
            "minimized_status": minimized,
            "restored_status": restored,
            "reopened_source_captured": recovered,
        }

    def _occlusion(self) -> dict[str, object]:
        self.control.command("reset")
        services = {
            backend: CaptureService(self._backend_profile(backend))
            for backend in ("printwindow", "screen", "gdi_screen")
        }
        occluded = self.control.command("occlude", {"visible": True})
        results: dict[str, object] = {}
        try:
            print_source = services["printwindow"].open_live_source(
                self.config.profile_name
            )
            try:
                snapshot = print_source.capture()
                print_kept = True
                similarity = self._pixel_evidence(snapshot, occluded, "occluded_printwindow")
            except Exception as exc:
                print_kept = False
                similarity = {"passed": False, "error": str(exc)}
            finally:
                print_source.close()
            results["printwindow"] = {
                "capture_kept": print_kept,
                "pixel_similarity": similarity,
            }
            for backend in ("screen", "gdi_screen"):
                source = services[backend].open_live_source(self.config.profile_name)
                try:
                    interrupted = _capture_is_interrupted(source)
                finally:
                    source.close()
                results[backend] = {"occlusion_interrupted": interrupted}
        finally:
            self.control.command("occlude", {"visible": False})
        for backend in ("screen", "gdi_screen"):
            reopened = services[backend].open_live_source(self.config.profile_name)
            try:
                recovered = _capture_succeeds(reopened)
            finally:
                reopened.close()
            results[backend]["reopened_source_captured"] = recovered  # type: ignore[index]
        passed = bool(
            results["printwindow"]["capture_kept"]  # type: ignore[index]
            and results["printwindow"]["pixel_similarity"]["passed"]  # type: ignore[index]
            and all(
                results[name]["occlusion_interrupted"]  # type: ignore[index]
                and results[name]["reopened_source_captured"]  # type: ignore[index]
                for name in ("screen", "gdi_screen")
            )
        )
        return {"passed": passed, "backends": results, "occluded_status": occluded}

    def _backend_profile(self, backend: str) -> Path:
        profiles_root = self.output / "scenario_profiles" / backend
        target = profiles_root / self.config.profile_name
        if not target.exists():
            shutil.copytree(
                self.runtime_profile,
                target,
                ignore=shutil.ignore_patterns("sessions"),
            )
            path = target / "profile.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["capture_backend"] = backend
            raw["allow_screen_fallback"] = False
            atomic_write_json(path, raw)
        return profiles_root

    def _dpi_matrix(self, service: CaptureService) -> dict[str, object]:
        monitors = probe_monitors()
        initial = self.control.status()
        initial_rect = dict(initial.get("client_rect") or {})
        results: dict[str, dict[str, object]] = {}
        requested = tuple(dict.fromkeys((*self.config.required_dpis, *self.config.optional_dpis)))
        for desired in requested:
            matches = [
                row
                for row in monitors
                if row.get("dpi_x") is not None
                and abs(int(row["dpi_x"]) - desired) <= 2
            ]
            if not matches:
                results[str(desired)] = {
                    "status": "ENVIRONMENT_UNAVAILABLE",
                    "required": desired in self.config.required_dpis,
                    "reason": f"no physical monitor reports effective DPI {desired}",
                }
                continue
            monitor = matches[0]
            work = monitor["work_area"]
            try:
                self.control.command(
                    "move", {"left": int(work[0]) + 20, "top": int(work[1]) + 20}
                )
                service.lock_target_client_size(self.config.profile_name)
                source = service.open_live_source(self.config.profile_name)
                try:
                    snapshot = source.capture()
                finally:
                    source.close()
                actual = int(snapshot.frame.dpi)
                results[str(desired)] = {
                    "status": "VERIFIED" if abs(actual - desired) <= 2 else "FAILED",
                    "required": desired in self.config.required_dpis,
                    "requested_dpi": desired,
                    "observed_dpi": actual,
                    "monitor": monitor,
                    "capture": _snapshot_metadata(snapshot),
                }
            except Exception as exc:
                results[str(desired)] = {
                    "status": "FAILED",
                    "required": desired in self.config.required_dpis,
                    "error": f"{type(exc).__name__}: {exc}",
                    "monitor": monitor,
                }
        if initial_rect:
            self.control.command(
                "move",
                {"left": int(initial_rect["left"]), "top": int(initial_rect["top"])},
            )
            service.lock_target_client_size(self.config.profile_name)
        required_verified = all(
            results.get(str(dpi), {}).get("status") == "VERIFIED"
            for dpi in self.config.required_dpis
        )
        failed = any(row.get("status") == "FAILED" for row in results.values())
        return {
            "passed": required_verified and not failed,
            "required_verified": required_verified,
            "monitors": monitors,
            "dpi_results": results,
            "environment_unavailable": [
                int(key)
                for key, row in results.items()
                if row.get("status") == "ENVIRONMENT_UNAVAILABLE"
            ],
        }

    def _run_full_chain(self) -> dict[str, object]:
        status = self.control.command("reset")
        expected_count = int(status.get("indexed_frame_count", 0) or 0)
        dependencies = build_live_controller_dependencies(
            profile_name=self.config.profile_name,
            capture=CaptureService(self.runtime_profiles_root),
            advisor_strategy="fabledan",
        )
        controller = LiveAssistantController(
            dependencies.capture,
            profile_name=self.config.profile_name,
            recognition_service=dependencies.recognizer,
            advisor=dependencies.advisor,
            session_factory=dependencies.session_factory,
            capture_interval_sec=self.config.capture_interval_sec,
            deduplicate_analysis_frames=True,
        )
        # Hand preselection is an input sidecar, not part of the capture/state/
        # advisor validation chain.  Disable it so E2E never sends mouse input.
        try:
            controller.update_ready.disconnect(controller._schedule_hand_preselection)
        except Exception:
            pass
        frame_log = self.output / "capture_frames.jsonl"
        update_log = self.output / "live_updates.jsonl"
        errors: list[str] = []
        frame_count = 0
        updates = 0
        recognition_events = 0
        advice_statuses: Counter[str] = Counter()
        last_snapshot: dict[str, object] | None = None
        finished: list[object] = []

        def on_frame(value: object) -> None:
            nonlocal frame_count
            if not isinstance(value, FrameSnapshot):
                return
            frame_count += 1
            append_json_line(
                frame_log,
                {"capture_seq": frame_count, **_snapshot_metadata(value)},
            )

        def on_update(value: object) -> None:
            nonlocal updates, recognition_events, last_snapshot
            updates += 1
            events = tuple(getattr(value, "events", ()) or ())
            event = getattr(value, "event", None)
            if event is not None and all(event is not item for item in events):
                events = (*events, event)
            recognition_events += len(events)
            advice = getattr(value, "advice", None)
            advice_status = getattr(advice, "status", None)
            if advice_status:
                advice_statuses[str(advice_status)] += 1
            snapshot = getattr(value, "snapshot", None)
            semantic = getattr(snapshot, "semantic_dict", None)
            if callable(semantic):
                last_snapshot = semantic()
            append_json_line(
                update_log,
                {
                    "update_seq": updates,
                    "status": getattr(value, "status", None),
                    "event_types": [getattr(item, "event_type", None) for item in events],
                    "advice_status": advice_status,
                    "revision": getattr(snapshot, "revision", None),
                    "turn_id": getattr(snapshot, "turn_id", None),
                },
            )

        controller.frame_ready.connect(on_frame)
        controller.update_ready.connect(on_update)
        controller.error.connect(errors.append)
        controller.session_finished.connect(finished.append)
        initial = _initial_state(self.config.session)
        if not controller.start_session(
            round_level=str(initial["round_level"]),
            hand=tuple(initial["hand"]),
            lead_player=initial["lead_player"],
            recognition_strategy="two_valid_streak",
        ):
            controller.shutdown()
            return {
                "passed": False,
                "executed": True,
                "errors": errors or ["controller rejected source initial state"],
            }
        orchestrator = controller.orchestrator
        runtime_directory = Path(orchestrator.store.directory)  # type: ignore[union-attr]
        advisor_info = _advisor_info(dependencies.advisor)
        rule_engine_probe = _rule_engine_probe()
        analysis_worker = None
        capture_worker = None
        started = time.monotonic()
        self.control.command("play")
        while time.monotonic() - started < self.config.run_timeout_sec:
            self._pump()
            simulator = self.control.status()
            if simulator.get("state") == "eof":
                break
            if errors and getattr(orchestrator, "status", None) == "paused":
                break
            time.sleep(0.01)
        else:
            errors.append("simulator playback timed out")
        controller._resume_requested = False
        capture_worker = controller._capture_worker
        controller._stop_capture_worker()
        if capture_worker is not None:
            capture_worker.wait(5_000)
        self._pump()
        analysis_worker = controller._analysis_worker
        analysis_drained = bool(
            analysis_worker is None
            or analysis_worker.wait_idle(timeout=self.config.drain_timeout_sec)
        )
        advice_drained = bool(
            orchestrator is not None
            and orchestrator.wait_for_advice_idle(timeout=self.config.drain_timeout_sec)
        )
        analysis_stats = analysis_worker.stats if analysis_worker is not None else {}
        controller._stop_analysis_worker()
        controller.finish()
        finish_deadline = time.monotonic() + self.config.drain_timeout_sec
        while time.monotonic() < finish_deadline and not finished:
            self._pump()
            time.sleep(0.01)
        finish_completed = bool(finished)
        controller.shutdown()
        self._pump()
        advice = summarize_advice_lifecycle(
            runtime_directory,
            drained=advice_drained,
        )
        actual_actions = _action_signatures(runtime_directory / "timeline.jsonl")
        baseline = self._baseline_comparison(actual_actions)
        baseline_gate = bool(
            baseline.get("available") and baseline.get("actions_identical")
            if self.config.acceptance_eligible
            else not baseline.get("available") or baseline.get("actions_identical")
        )
        simulator = self.control.status()
        frame_target = min(
            expected_count,
            self.config.max_frames or expected_count,
        )
        thread_state = {
            "capture_worker_stopped": bool(
                capture_worker is None or not capture_worker.is_running
            ),
            "analysis_worker_stopped": bool(
                analysis_worker is None or not analysis_worker.is_running
            ),
            "finish_completed": finish_completed,
            "analysis_drained": analysis_drained,
            "advice_drained": advice_drained,
            "python_threads": [
                thread.name
                for thread in threading.enumerate()
                if thread is not threading.current_thread()
            ],
        }
        terminal_count = sum(
            int(count)
            for status_name, count in advice.get("terminal_counts", {}).items()
            if status_name in _TERMINAL_ADVICE
        )
        advisor_terminal = terminal_count == int(advice.get("requested", 0))
        passed = bool(
            not errors
            and simulator.get("state") == "eof"
            and frame_count > 0
            and int(simulator.get("current_frame_index", -1)) >= frame_target - 1
            and analysis_drained
            and advice_drained
            and finish_completed
            and advisor_terminal
            and bool(rule_engine_probe.get("passed"))
            and len(actual_actions) > 0
            and thread_state["capture_worker_stopped"]
            and thread_state["analysis_worker_stopped"]
            and baseline_gate
        )
        return {
            "passed": passed,
            "executed": True,
            "same_hwnd": int(simulator.get("hwnd", 0) or 0)
            == int(self.config.expected_hwnd or simulator.get("hwnd", 0) or 0),
            "simulator": simulator,
            "expected_source_frames": frame_target,
            "captured_frames": frame_count,
            "updates": updates,
            "recognition_events": recognition_events,
            "advice_update_statuses": dict(advice_statuses),
            "analysis_worker": analysis_stats,
            "thread_and_drain": thread_state,
            "advice": advice,
            "advisor": advisor_info,
            "rule_engine_probe": rule_engine_probe,
            "baseline_comparison": baseline,
            "confirmed_action_count": len(actual_actions),
            "final_snapshot": last_snapshot,
            "runtime_directory": str(runtime_directory),
            "frame_log": str(frame_log),
            "update_log": str(update_log),
            "errors": errors,
        }

    def _baseline_comparison(
        self, actual: list[dict[str, object]]
    ) -> dict[str, object]:
        path = self.config.baseline_summary
        if path is None or not path.is_file():
            return {"available": False, "reason": "no phase-two baseline supplied"}
        raw = json.loads(path.read_text(encoding="utf-8"))
        runtime = raw.get("runtime_directory") if isinstance(raw, dict) else None
        timeline = Path(str(runtime)) / "timeline.jsonl" if runtime else None
        if timeline is None or not timeline.is_file():
            return {
                "available": False,
                "reason": "phase-two baseline has no readable runtime timeline",
                "summary": str(path),
            }
        expected = _action_signatures(timeline)
        first = None
        for index in range(max(len(expected), len(actual))):
            left = expected[index] if index < len(expected) else None
            right = actual[index] if index < len(actual) else None
            if left != right:
                first = {"position": index + 1, "expected": left, "actual": right}
                break
        return {
            "available": True,
            "summary": str(path),
            "expected_action_count": len(expected),
            "actual_action_count": len(actual),
            "actions_identical": expected == actual,
            "first_divergence": first,
        }

    def _pixel_evidence(
        self,
        snapshot: FrameSnapshot,
        simulator_status: dict[str, object],
        name: str,
    ) -> dict[str, object]:
        frame_index = int(simulator_status["current_frame_index"])
        expected = _read_video_frame(self.config.session, frame_index)
        captured_path = self.evidence_directory / f"{name}_captured.png"
        source_path = self.evidence_directory / f"{name}_source.png"
        if not cv2.imwrite(str(captured_path), snapshot.image):
            raise RuntimeError(f"failed to write capture evidence: {captured_path}")
        if not cv2.imwrite(str(source_path), expected):
            raise RuntimeError(f"failed to write source evidence: {source_path}")
        result = compare_pixels(
            snapshot.image,
            expected,
            mae_limit=self.config.pixel_mae_limit,
            hash_distance_limit=self.config.perceptual_hash_distance_limit,
        )
        result.update(
            {
                "source_frame_index": frame_index,
                "captured_path": str(captured_path),
                "source_path": str(source_path),
            }
        )
        return result

    def _runtime_identity(self) -> dict[str, object]:
        executable = self.config.executable_path
        if executable is None and getattr(sys, "frozen", False):
            executable = Path(sys.executable)
        try:
            pyinstaller = importlib.metadata.version("PyInstaller")
        except importlib.metadata.PackageNotFoundError:
            pyinstaller = None
        return {
            "frozen": bool(getattr(sys, "frozen", False)),
            "python": sys.version,
            "executable": str(executable) if executable else sys.executable,
            "executable_sha256": _file_hash(executable) if executable else None,
            "pyinstaller_version": pyinstaller,
            "pid": os.getpid(),
        }


def run_window_e2e_validation(config_path: Path | str) -> int:
    """Run from source or frozen EXE; result files, not stdout, are the API."""

    try:
        config = WindowE2EValidationConfig.from_path(config_path)
    except Exception:
        return 2
    try:
        summary = WindowE2EValidator(config).run()
    except Exception as exc:
        # A config with a valid output path still receives a machine-readable
        # bootstrap failure even if the normal report construction could not run.
        output = config.output
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "guandan.window-e2e-summary/1",
            "execution_ok": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        atomic_write_json(output / "summary.json", failure)
        (output / "summary.md").write_text(_summary_markdown(failure), encoding="utf-8")
        return 2
    return 0 if summary.get("execution_ok") else 1


def snapshot_paths(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for root in paths:
        root = Path(root).resolve()
        if root.is_file():
            result[str(root)] = {
                "size": root.stat().st_size,
                "sha256": _file_hash(root),
            }
        elif root.is_dir():
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                result[str(path)] = {
                    "size": path.stat().st_size,
                    "sha256": _file_hash(path),
                }
        else:
            result[str(root)] = {"missing": True}
    return result


def _file_hash(path: Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bgr(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    if value.ndim == 3 and value.shape[2] == 4:
        return cv2.cvtColor(value, cv2.COLOR_BGRA2BGR)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("pixel evidence must be grayscale, BGR, or BGRA")
    return np.ascontiguousarray(value)


def _read_video_frame(session: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(session / "video" / "game.avi"))
    try:
        if not capture.isOpened():
            raise RuntimeError("cannot open source video for evidence")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot decode source evidence frame {frame_index}")
        return frame
    finally:
        capture.release()


def _capture_is_interrupted(source: Any) -> bool:
    try:
        source.capture()
    except LiveCaptureInterrupted:
        return True
    return False


def _capture_succeeds(source: Any) -> bool:
    try:
        source.capture()
    except Exception:
        return False
    return True


def _snapshot_metadata(snapshot: FrameSnapshot) -> dict[str, object]:
    frame = snapshot.frame
    standard = frame.standardization
    return {
        "captured_at": snapshot.captured_at.isoformat(),
        "backend": frame.backend,
        "dpi": frame.dpi,
        "window_title": frame.window_title,
        "client_rect": _rect_dict(frame.rect),
        "standardized_size": [int(snapshot.image.shape[1]), int(snapshot.image.shape[0])],
        "source_viewport": standard.source_viewport.to_list(),
        "content_box": standard.content_box.to_list(),
        "scale": standard.scale,
        "padding": list(standard.padding),
        "aspect_error": standard.aspect_error,
        "aspect_compatible": standard.aspect_compatible,
    }


def _rect_dict(rect: ClientRect) -> dict[str, int]:
    return {
        "left": rect.left,
        "top": rect.top,
        "width": rect.width,
        "height": rect.height,
    }


def _initial_state(session: Path) -> dict[str, object]:
    for event in read_json_lines(session / "timeline.jsonl"):
        if event.get("event_type") != "initial_state_confirmed":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        lead = payload.get("lead_player", event.get("actor"))
        return {
            "round_level": str(payload.get("round_level", "")),
            "hand": tuple(str(card) for card in payload.get("hand", ())),
            "lead_player": lead if lead in {"self", "right", "opposite", "left"} else None,
        }
    raise ValueError("source timeline lacks initial_state_confirmed")


def _action_signatures(path: Path) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for event in read_json_lines(path):
        if event.get("event_type") not in _ACTION_TYPES:
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        cards = payload.get("cards", ())
        if isinstance(cards, str):
            cards = (cards,)
        actions.append(
            {
                "actor": event.get("actor"),
                "is_pass": bool(payload.get("is_pass", event.get("event_type") == "player_passed")),
                "cards": [str(card) for card in cards or ()],
                "turn_id": event.get("turn_id"),
            }
        )
    return actions


def _rule_engine_probe() -> dict[str, object]:
    """Prove the frozen runtime can validate a simple legal pair."""

    try:
        from ..danzero.rules import actions_for_cards

        actions = actions_for_cards(("5C", "5H"), "6")
        normalized = [
            {
                "play_type": str(action[0]),
                "key_rank": str(action[1]),
                "cards": [str(card) for card in action[2]],
            }
            for action in actions
        ]
        return {
            "passed": any(row["play_type"] == "Pair" for row in normalized),
            "input": {"cards": ["5C", "5H"], "level_rank": "6"},
            "actions": normalized,
            "error": None,
        }
    except Exception as exc:
        return {
            "passed": False,
            "input": {"cards": ["5C", "5H"], "level_rank": "6"},
            "actions": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _advisor_info(advisor: object) -> dict[str, object]:
    audit = getattr(advisor, "audit_info", None)
    if callable(audit):
        return dict(audit())
    return {"backend": type(advisor).__name__}


def _summary_markdown(summary: dict[str, object]) -> str:
    scenarios = summary.get("scenarios", {})
    lines = [
        "# 第三阶段窗口 E2E 验证",
        "",
        f"- 执行结果：{'PASS' if summary.get('execution_ok') else 'FAIL'}",
        f"- 正式验收资格：{summary.get('acceptance_eligible', False)}",
        f"- 正式验收结果：{'PASS' if summary.get('acceptance_passed') else 'NOT_PASS'}",
        f"- 运行类型：{summary.get('run_kind', 'unknown')}",
        f"- HWND：{summary.get('observed_hwnd', 'N/A')}",
        "",
        "## 场景",
        "",
    ]
    if isinstance(scenarios, dict):
        for name, value in scenarios.items():
            passed = isinstance(value, dict) and value.get("passed")
            lines.append(f"- {name}：{'PASS' if passed else 'FAIL'}")
    errors = summary.get("errors", [])
    if errors:
        lines.extend(("", "## 错误", ""))
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines) + "\n"
