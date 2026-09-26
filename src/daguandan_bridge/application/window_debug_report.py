"""Application service for one-shot, read-only window debug reports."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
from datetime import datetime
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from ..annotation_service import AnnotationService
from ..config import PROFILES_ROOT
from ..opening_gate import evaluate_opening_gate
from ..profiles import ProfileConfig, get_profile_paths, load_profile_config
from ..recognition_service import ScreenshotRecognitionService
from .opening_readiness import report_for_error, report_for_phase
from .diagnostic_presentation import build_detailed_diagnostic, build_user_view
from .window_debug_storage import WindowDebugStorage
from .session_diagnostic_frames import SessionDiagnosticFrameStore
from ..window_debug import (
    SCHEMA as WINDOW_DEBUG_SCHEMA,
    WindowInfo,
    capture_printwindow_frame,
    enumerate_visible_windows,
    inspect_window,
    json_safe,
    portable_window_identity,
    standardize_frame,
)

REPORT_SCHEMA = "guandan.window-debug-report/v1"


class WindowDebugReportService:
    """Collect window/capture/recognition evidence without controlling a window."""

    def __init__(
        self,
        *,
        profiles_root: Path | str = PROFILES_ROOT,
        profile_name: str = "tencent_daguandan",
        window_api: Any | None = None,
        process_api: Any | None = None,
        process_name_resolver: Callable[[int], str | None] | Mapping[int, str | None] | None = None,
        dpi_getter: Callable[[int], int] | None = None,
        capture_function: Callable[..., Any] | None = None,
        profile_config: ProfileConfig | None = None,
        recognizer: Any | None = None,
        storage: WindowDebugStorage | None = None,
    ) -> None:
        self.profiles_root = Path(profiles_root)
        self.profile_name = str(profile_name)
        self.window_api = window_api
        self.process_api = process_api
        self.process_name_resolver = process_name_resolver
        self.dpi_getter = dpi_getter
        self.capture_function = capture_function
        self._profile_config = profile_config
        self._recognizer = recognizer
        # Keep storage lazy: report construction and explicit user exports
        # must not create diagnostic runs as a side effect.
        self.storage = storage

    def _window_kwargs(self) -> dict[str, object]:
        return {
            "window_api": self.window_api,
            "process_api": self.process_api,
            "process_name_resolver": self.process_name_resolver,
            "dpi_getter": self.dpi_getter,
        }

    def _config(self) -> ProfileConfig:
        if self._profile_config is not None:
            return self._profile_config
        return load_profile_config(get_profile_paths(self.profiles_root, self.profile_name))

    def _recognition_service(self) -> Any:
        if self._recognizer is None:
            annotation = AnnotationService(self.profiles_root, self.profile_name)
            self._recognizer = ScreenshotRecognitionService(
                annotation_service=annotation,
                diagnostic_tracing=True,
            )
        return self._recognizer

    @staticmethod
    def _window_payload(window: WindowInfo) -> dict[str, object]:
        return {
            **window.to_dict(),
            "hwnd_scope": "current-machine-session-only",
            "portable_identity": portable_window_identity(window),
        }

    def _recognize_standardized(
        self,
        standardized: Any,
        *,
        capture: Mapping[str, object] | None = None,
        errors: object | None = None,
    ) -> dict[str, object]:
        """Run recognition and the shared evidence-only presentation adapter."""

        recognizer = self._recognition_service()
        roi_validation = json_safe(recognizer.validate_configuration(standardized))
        result = recognizer.recognize(standardized, allow_unknown_suit=True)
        # Snapshot the production pass now: later page/opening probes may
        # replace or mutate the recognizer's thread-local diagnostic trace.
        trace_json = json_safe(
            recognizer.get_last_diagnostic_trace()
            if hasattr(recognizer, "get_last_diagnostic_trace")
            else None
        )
        input_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(standardized))).hexdigest()
        metadata = (capture or {}).get("metadata")
        captured_context = (
            json_safe(metadata.get("diagnostic_context"))
            if isinstance(metadata, Mapping)
            else None
        )
        anchor_scores = (
            recognizer.recognize_page_anchor_scores(standardized)
            if hasattr(recognizer, "recognize_page_anchor_scores")
            else {}
        )
        opening_signal = (
            recognizer.recognize_opening_signal(standardized)
            if hasattr(recognizer, "recognize_opening_signal")
            else None
        )
        anchor_score = max((float(v) for v in anchor_scores.values()), default=0.0)
        gate = evaluate_opening_gate(result, anchor_score=anchor_score)
        readiness = report_for_phase(
            gate.reason,
            hand_count=len(tuple(getattr(result, "my_hand", ()) or ())),
            details={
                "anchor_score": anchor_score,
                "anchor_scores": dict(anchor_scores),
                "opening_signal": json_safe(opening_signal),
                "lead_player": getattr(result, "lead_player", None),
                "input_sha256": input_sha256,
                "captured_context": captured_context,
            },
        )
        result_json = json_safe(result)
        readiness_json = readiness.to_dict()
        gate_json = json_safe(gate)
        detailed = build_detailed_diagnostic(
            result=result_json,
            trace=trace_json,
            readiness=readiness_json,
            gate=gate_json,
            roi_validation=roi_validation,
            capture=capture,
            errors=errors,
        )
        return {
            "roi_validation": roi_validation,
            "recognition": {"result": result_json, "trace": trace_json, "input_sha256": input_sha256},
            "detailed_diagnostic": detailed,
            "opening_readiness_inputs": {
                "input_sha256": input_sha256,
                "captured_context": captured_context,
                "anchor_scores": json_safe(anchor_scores),
                "recognition": result_json,
                "opening_signal": json_safe(opening_signal),
                "gate": gate_json,
                "readiness": readiness_json,
            },
        }

    @staticmethod
    def _decode_saved_frame(item: object) -> tuple[np.ndarray | None, dict[str, object]]:
        """Decode ``media.items.current_frame`` without writing a temporary file."""

        if item is None:
            return None, {
                "status": "MEDIA_NOT_PRESENT",
                "media_key": "current_frame",
                "message": "report.json 未包含 media.items.current_frame 截图",
            }

        encoding = "base64"
        value: object = item
        if isinstance(item, Mapping):
            encoding = str(item.get("encoding") or "base64").lower()
            for key in ("content", "data", "base64", "value"):
                if key in item:
                    value = item[key]
                    break
        if encoding != "base64" or not isinstance(value, str) or not value.strip():
            return None, {
                "status": "MEDIA_INVALID",
                "media_key": "current_frame",
                "error_code": "MEDIA_BASE64_INVALID",
                "message": "current_frame 不是有效的 Base64 媒体项",
            }
        try:
            content = base64.b64decode(value.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError):
            return None, {
                "status": "MEDIA_INVALID",
                "media_key": "current_frame",
                "error_code": "MEDIA_BASE64_INVALID",
                "message": "current_frame Base64 解码失败",
            }
        if not content:
            return None, {
                "status": "MEDIA_INVALID",
                "media_key": "current_frame",
                "error_code": "MEDIA_EMPTY",
                "message": "current_frame 解码后为空",
            }
        try:
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            image = None
        if image is None or image.size == 0:
            return None, {
                "status": "MEDIA_INVALID",
                "media_key": "current_frame",
                "error_code": "MEDIA_IMAGE_INVALID",
                "message": "current_frame 不是可读取的图片",
                "bytes": len(content),
            }
        return image, {
            "status": "PRESENT",
            "media_key": "current_frame",
            "encoding": "base64",
            "bytes": len(content),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "image_restored_in_memory": True,
            "image_persisted": False,
        }

    def build_from_listener_frame(
        self,
        image_path: Path | str,
        *,
        metadata_path: Path | str | None = None,
        recognize: bool = True,
    ) -> dict[str, object]:
        """Diagnose one exact frame previously captured by the live listener.

        The PNG is decoded without window capture, resizing, black-bar
        detection, or any other standardization.  It is therefore the same
        profile-standardized pixel array that the listener submitted to the
        recognition service.
        """

        store = SessionDiagnosticFrameStore()
        record = store.read_record(image_path, metadata_path=metadata_path)
        image = store.load_image(record)
        metadata = dict(record.metadata)
        source_kind = str(metadata.get("source") or "live_listener_frame")
        source_phase = str(metadata.get("source_phase") or source_kind)
        is_manual_capture = source_kind == "manual_window_capture"
        diagnosis_mode = (
            "saved_manual_window_frame"
            if is_manual_capture
            else "saved_live_listener_frame"
        )
        diagnosis_label = (
            "独立窗口截图诊断（同一实时采集链路）"
            if is_manual_capture
            else "实时监听保存帧诊断（非实时窗口捕获）"
        )
        source_label = "manual_window_capture" if is_manual_capture else "live_listener_frame"
        height, width = image.shape[:2]
        capture = {
            "backend": metadata.get("backend"),
            "size": [int(width), int(height)],
            "dpi": metadata.get("dpi"),
            "rect": metadata.get("rect"),
            "image_persisted": True,
            "source": source_kind,
            "source_phase": source_phase,
            "standardization": {
                "applied": False,
                "source_standardized": True,
                "standardized_size": [int(width), int(height)],
            },
            "metadata": json_safe(metadata),
        }
        output: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "window_debug_schema": WINDOW_DEBUG_SCHEMA,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "profile_name": self.profile_name,
            "read_only": True,
            "diagnosis_mode": diagnosis_mode,
            "diagnostic_mode": diagnosis_mode,
            "diagnosis_label": diagnosis_label,
            "live_window_diagnosis": False,
            "source_metadata": json_safe(metadata),
            "source_frame": {
                "sequence": record.sequence,
                "session_id": metadata.get("session_id"),
                "image_name": record.image_path.name,
                "metadata_name": record.metadata_path.name,
            },
            "capture": capture,
            "errors": [],
            "events": [],
        }
        if recognize:
            output.update(self._recognize_standardized(image, capture=capture))
            output["recognition_status"] = "PASS"
        else:
            output["roi_validation"] = json_safe(
                self._recognition_service().validate_configuration(image)
            )
            output["recognition_status"] = "NOT_REQUESTED"
        presentation = {
            **output,
            "capture": capture,
            "roi_validation": output.get("roi_validation", {}),
            "recognition": output.get("recognition"),
            "opening_readiness_inputs": output.get("opening_readiness_inputs", {}),
        }
        view = build_user_view(presentation).to_dict()
        view.update({
            "诊断模式": diagnosis_label,
            "实时窗口诊断": False,
            "来源": source_label,
            "来源阶段": source_phase,
            "截图序号": int(record.sequence),
        })
        output["user_view"] = view
        return json_safe(output)  # type: ignore[return-value]

    @staticmethod
    def _source_media_item(source_report: Mapping[str, object]) -> object | None:
        media = source_report.get("media")
        if not isinstance(media, Mapping):
            return None
        items = media.get("items")
        if not isinstance(items, Mapping):
            return None
        return items.get("current_frame")

    @staticmethod
    def _first_play_evidence(report: Mapping[str, object]) -> dict[str, object]:
        """Keep the original opening/first-play evidence visible in replay output."""

        inputs = report.get("opening_readiness_inputs")
        if not isinstance(inputs, Mapping):
            inputs = {}
        recognition = report.get("recognition")
        result = recognition.get("result") if isinstance(recognition, Mapping) else None
        result = result if isinstance(result, Mapping) else {}
        return {
            "opening_signal": json_safe(inputs.get("opening_signal")),
            "anchor_scores": json_safe(inputs.get("anchor_scores")),
            "first_play": json_safe(result.get("first_play")),
            "lead_player": json_safe(result.get("lead_player")),
            "source_readiness": json_safe(inputs.get("readiness")),
        }

    @staticmethod
    def _value_differences(source: object, replay: object, prefix: str = "") -> list[dict[str, object]]:
        if isinstance(source, Mapping) and isinstance(replay, Mapping):
            differences: list[dict[str, object]] = []
            keys = sorted(set(source) | set(replay), key=str)
            for key in keys:
                name = f"{prefix}.{key}" if prefix else str(key)
                if key not in source or key not in replay:
                    differences.append({"field": name, "source": json_safe(source.get(key)), "re_diagnosed": json_safe(replay.get(key))})
                else:
                    differences.extend(WindowDebugReportService._value_differences(source[key], replay[key], name))
            return differences
        if source != replay:
            return [{"field": prefix or "value", "source": json_safe(source), "re_diagnosed": json_safe(replay)}]
        return []

    @classmethod
    def _comparison(cls, source_report: Mapping[str, object], replay: Mapping[str, object]) -> dict[str, object]:
        source_recognition = source_report.get("recognition")
        source_result = source_recognition.get("result") if isinstance(source_recognition, Mapping) else None
        replay_recognition = replay.get("recognition")
        replay_result = replay_recognition.get("result") if isinstance(replay_recognition, Mapping) else None
        if source_result is None and replay_result is None:
            recognition_status = "NOT_AVAILABLE"
            recognition_differences: list[dict[str, object]] = []
        elif source_result is None:
            recognition_status = "SOURCE_NOT_PRESENT"
            recognition_differences = [{"field": "result", "source": None, "re_diagnosed": json_safe(replay_result)}]
        else:
            recognition_differences = cls._value_differences(source_result, replay_result)
            recognition_status = "CHANGED" if recognition_differences else "UNCHANGED"
        source_inputs = source_report.get("opening_readiness_inputs")
        replay_inputs = replay.get("opening_readiness_inputs")
        source_readiness = source_inputs.get("readiness") if isinstance(source_inputs, Mapping) else None
        replay_readiness = replay_inputs.get("readiness") if isinstance(replay_inputs, Mapping) else None
        readiness_differences = cls._value_differences(source_readiness, replay_readiness)
        changed = bool(recognition_differences or readiness_differences)
        return {
            "status": "CHANGED" if changed else "UNCHANGED" if recognition_status == "UNCHANGED" else recognition_status,
            "changed": changed,
            "recognition_changed": bool(recognition_differences),
            "readiness_changed": bool(readiness_differences),
            "recognition": {
                "status": recognition_status,
                "source": json_safe(source_result),
                "re_diagnosed": json_safe(replay_result),
                "differences": recognition_differences,
            },
            "readiness": {
                "source": json_safe(source_readiness),
                "re_diagnosed": json_safe(replay_readiness),
                "differences": readiness_differences,
            },
            "first_play_evidence": {
                "source": cls._first_play_evidence(source_report),
                "re_diagnosed": cls._first_play_evidence(replay),
            },
        }

    def list_windows(self) -> list[dict[str, object]]:
        return [
            self._window_payload(item)
            for item in enumerate_visible_windows(**self._window_kwargs())
        ]

    def probe(self, hwnd: int) -> dict[str, object]:
        return self._window_payload(inspect_window(hwnd, **self._window_kwargs()))

    def build(
        self,
        *,
        list_windows: bool = False,
        hwnd: int | None = None,
        capture: bool = False,
        recognize: bool = False,
        include_media: bool = False,
    ) -> dict[str, object]:
        """Build a JSON-safe report. Recognition implies a PrintWindow capture."""

        report: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "window_debug_schema": WINDOW_DEBUG_SCHEMA,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "profile_name": self.profile_name,
            "read_only": True,
            "control_policy": {
                "resize": False,
                "move": False,
                "focus": False,
                "click": False,
                "restore": False,
            },
            "errors": [],
            "events": [],
            "media": {"media_saved": False, "items": {}},
        }
        if list_windows:
            report["windows"] = self.list_windows()
        if hwnd is None:
            if not list_windows:
                raise ValueError("hwnd is required unless list_windows is true")
            report["user_view"] = build_user_view(report).to_dict()
            return json_safe(report)  # type: ignore[return-value]

        window = inspect_window(hwnd, **self._window_kwargs())
        report["window"] = self._window_payload(window)
        if not (capture or recognize):
            return report

        try:
            frame = capture_printwindow_frame(
                window,
                capture_function=self.capture_function,
                **self._window_kwargs(),
            )
            config = self._config()
            standardized, standardization = standardize_frame(
                frame.image,
                base_size=config.base_size,
                aspect_ratio_tolerance=config.aspect_ratio_tolerance,
                detect_black_bars=config.detect_black_bars,
                viewport_mode=config.viewport_mode,
                viewport_aspect_ratio=config.viewport_aspect_ratio,
            )
            report["capture"] = {
                **frame.to_dict(),
                "image_persisted": bool(include_media),
                "standardization": standardization,
            }
            if include_media:
                ok, encoded = cv2.imencode(
                    ".jpg", frame.image, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                if not ok:
                    raise RuntimeError("当前画面无法编码为诊断图片")
                report["media"] = {
                    "media_saved": True,
                    "format": "jpeg",
                    "items": {
                        "current_frame": {
                            "encoding": "base64",
                            "content": base64.b64encode(encoded.tobytes()).decode("ascii"),
                        }
                    },
                }
            if recognize:
                report.update(self._recognize_standardized(standardized, capture=report.get("capture")))
            else:
                report["roi_validation"] = json_safe(
                    self._recognition_service().validate_configuration(standardized)
                )
        except Exception as exc:
            error = {
                "code": str(getattr(exc, "code", type(exc).__name__)),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            report["errors"] = [error]
            report["opening_readiness_inputs"] = {
                "readiness": report_for_error(exc, stage="capture").to_dict()
            }
        report["user_view"] = build_user_view(report).to_dict()
        return json_safe(report)  # type: ignore[return-value]

    def build_from_report_file(
        self, path: Path | str, *, recognize: bool = True
    ) -> dict[str, object]:
        """Restore ``current_frame`` from report.json and diagnose it in memory.

        This is deliberately not a live-window operation: it never resolves an
        HWND, captures a window, writes media, or changes the source report.
        ``source_report`` keeps the original window, recognition, and opening
        evidence alongside the new result for direct comparison.
        """

        report_path = Path(path).expanduser()
        source_value = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(source_value, Mapping):
            raise ValueError("report.json 顶层必须是 JSON 对象")
        source_report = dict(json_safe(source_value))
        profile_name = str(source_report.get("profile_name") or self.profile_name)
        output: dict[str, object] = {
            "schema": REPORT_SCHEMA,
            "window_debug_schema": WINDOW_DEBUG_SCHEMA,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "profile_name": self.profile_name,
            "read_only": True,
            "diagnosis_mode": "saved_snapshot_re_diagnosis",
            "diagnostic_mode": "saved_screenshot",
            "diagnosis_label": "保存截图重诊（非实时窗口诊断）",
            "live_window_diagnosis": False,
            "source_report": {
                "path": str(report_path),
                "schema": source_report.get("schema"),
                "source_schema": source_report.get("schema"),
                "source_path_name": report_path.name,
                "storage_run_id": source_report.get("storage_run_id"),
                "storage_ref": source_report.get("storage_ref"),
                "profile_name": profile_name,
                "report": source_report,
                "window": source_report.get("window"),
                "recognition": source_report.get("recognition"),
                "opening_readiness_inputs": source_report.get("opening_readiness_inputs"),
                "first_play_evidence": self._first_play_evidence(source_report),
            },
            "captured_snapshot": {},
            "re_diagnosis": {
                "mode": "saved_snapshot",
                "recognize_requested": bool(recognize),
                "status": "NOT_STARTED",
            },
            "errors": [],
            "events": [],
        }

        item = self._source_media_item(source_report)
        image, snapshot = self._decode_saved_frame(item)
        output["captured_snapshot"] = snapshot
        output["capture"] = {
            "backend": "saved_report",
            "image_persisted": False,
            **snapshot,
        }
        replay: dict[str, object] = {
            "mode": "saved_snapshot",
            "recognize_requested": bool(recognize),
            "status": str(snapshot.get("status", "MEDIA_INVALID")),
            "profile_name": self.profile_name,
            "source_media": "media.items.current_frame",
        }
        if image is None:
            code = str(snapshot.get("status") or "MEDIA_INVALID")
            replay["recognition"] = None
            replay["opening_readiness_inputs"] = {
                "readiness": {
                    "status": "WAIT",
                    "primary_reason": code,
                    "error_code": code,
                    "message": str(snapshot.get("message") or code),
                    "recoverable": code == "MEDIA_NOT_PRESENT",
                    "suggested_action": "在 report.json 的 media.items.current_frame 中提供有效截图",
                }
            }
            output["re_diagnosis"] = replay
            output["errors"] = [{
                "code": code,
                "type": "SavedSnapshotMediaError",
                "message": str(snapshot.get("message") or code),
            }]
            output["recognition"] = None
            output["comparison"] = self._comparison(source_report, replay)
            presentation = {
                **output,
                "window": source_report.get("window"),
                "capture": snapshot,
                "roi_validation": {},
                "opening_readiness_inputs": replay["opening_readiness_inputs"],
            }
            view = build_user_view(presentation).to_dict()
            view.update({
                "诊断模式": "保存截图重诊（非实时窗口诊断）",
                "实时窗口诊断": False,
            })
            output["user_view"] = view
            return json_safe(output)  # type: ignore[return-value]

        try:
            config = self._config()
            standardized, standardization = standardize_frame(
                image,
                base_size=config.base_size,
                aspect_ratio_tolerance=config.aspect_ratio_tolerance,
                detect_black_bars=config.detect_black_bars,
                viewport_mode=config.viewport_mode,
                viewport_aspect_ratio=config.viewport_aspect_ratio,
            )
            output["captured_snapshot"] = {**snapshot, "standardization": standardization}
            output["capture"] = {
                "backend": "saved_report",
                "image_persisted": False,
                **output["captured_snapshot"],
            }
            if recognize:
                replay.update(self._recognize_standardized(standardized, capture=output.get("capture")))
                replay["status"] = "PASS"
            else:
                replay["roi_validation"] = json_safe(
                    self._recognition_service().validate_configuration(standardized)
                )
                replay["status"] = "CAPTURED"
        except Exception as exc:
            error = {
                "code": str(getattr(exc, "code", type(exc).__name__)),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            output["errors"] = [error]
            replay["status"] = "FAILED"
            replay["error"] = error
            replay["opening_readiness_inputs"] = {
                "readiness": report_for_error(exc, stage="saved_snapshot_re_diagnosis").to_dict()
            }
        output["re_diagnosis"] = replay
        output["recognition"] = replay.get("recognition")
        output["comparison"] = self._comparison(source_report, replay)
        presentation = {
            **output,
            "window": source_report.get("window"),
            "capture": output["captured_snapshot"],
            "roi_validation": replay.get("roi_validation", {}),
            "recognition": replay.get("recognition"),
            "detailed_diagnostic": replay.get("detailed_diagnostic", {}),
            "opening_readiness_inputs": replay.get("opening_readiness_inputs", {}),
        }
        view = build_user_view(presentation).to_dict()
        view.update({
            "诊断模式": "保存截图重诊（非实时窗口诊断）",
            "实时窗口诊断": False,
        })
        output["user_view"] = view
        return json_safe(output)  # type: ignore[return-value]

    @staticmethod
    def _validate_explicit_output(output: Path | str) -> Path:
        """Allow explicit user exports, but reject traversal and EXE paths."""

        raw = Path(output).expanduser()
        if ".." in raw.parts:
            raise ValueError("explicit diagnostic output must not contain '..' path traversal")
        if raw.name in {"", ".", ".."} or raw.suffix.lower() != ".json":
            raise ValueError("explicit diagnostic output must be a .json file")
        path = raw.resolve(strict=False)
        executable_dir = Path(sys.executable).resolve(strict=False).parent
        if path == executable_dir or path.is_relative_to(executable_dir):
            raise ValueError("explicit diagnostic output must not be inside the executable directory")
        for ancestor in (path.parent, *path.parent.parents):
            try:
                if stat.S_ISLNK(ancestor.lstat().st_mode):
                    raise ValueError("explicit diagnostic output must not use a symlinked parent")
            except FileNotFoundError:
                continue
        if path.exists() and path.is_symlink():
            raise ValueError("explicit diagnostic output must not replace a symlink")
        return path

    def _storage(self) -> WindowDebugStorage:
        if self.storage is None:
            self.storage = WindowDebugStorage()
        return self.storage

    def write(
        self,
        report: Mapping[str, object],
        output: Path | str | None = None,
    ) -> Path:
        """Persist to bounded diagnostics by default or to an explicit safe export."""

        payload = dict(json_safe(report))
        if output is None:
            storage = self._storage()
            run = storage.create_run()
            payload["storage_run_id"] = run.run_id
            payload["storage_ref"] = f"window_debug/{run.run_id}/report.json"
            storage.write_report(run, payload)
            storage.cleanup()
            if isinstance(report, dict):
                report.update({
                    "storage_run_id": run.run_id,
                    "storage_ref": payload["storage_ref"],
                })
            return run.path / "report.json"

        # Explicit exports are user-selected copies, not the managed default
        # run. They still pass through the EXE/path-traversal safety check.
        path = self._validate_explicit_output(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def _service_from_kwargs(kwargs: dict[str, object]) -> WindowDebugReportService:
    service_keys = {
        "profiles_root", "profile_name", "window_api", "process_api",
        "process_name_resolver", "dpi_getter", "capture_function",
        "profile_config", "recognizer", "storage",
    }
    service_args = {key: kwargs.pop(key) for key in tuple(kwargs) if key in service_keys}
    return WindowDebugReportService(**service_args)  # type: ignore[arg-type]


def build_window_debug_report(**kwargs: object) -> dict[str, object]:
    """Convenience entry point using the default service."""

    return _service_from_kwargs(kwargs).build(**kwargs)  # type: ignore[arg-type]


def build_from_report_file(
    path: Path | str, *, recognize: bool = True, **kwargs: object
) -> dict[str, object]:
    """Public report.json replay entry point; it never writes a new file."""

    service = _service_from_kwargs(kwargs)
    return service.build_from_report_file(path, recognize=recognize)


__all__ = [
    "REPORT_SCHEMA",
    "WindowDebugReportService",
    "build_window_debug_report",
    "build_from_report_file",
]

