"""Read-only Windows desktop smoke test for the DaGuandan capture path.

The command deliberately does not launch, focus, move, resize, click, or type
into the game client.  It only checks that an interactive desktop is available,
locates the configured target, and runs the same one-frame
``CaptureService.open_live_source().capture()`` path used by live listening.
An optional duration reuses that source for bounded repeated samples.  The
only intentional filesystem write is the JSON report requested by ``--output``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_RUN = "NOT_RUN"
STATUS_NOT_CHECKED = "NOT_CHECKED"
EXIT_CODES = {STATUS_PASS: 0, STATUS_FAIL: 1, STATUS_NOT_RUN: 3}
DEFAULT_PROFILE_ROOT = PROJECT_ROOT / "data" / "profiles" / "tencent_daguandan"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "windows_smoke"

_NOT_RUN_CAPTURE_CODES = frozenset(
    {
        "CAPTURE-UNSUPPORTED",
        "WINDOW-NOT-FOUND",
        "WINDOW-MINIMIZED",
        "WINDOW-AMBIGUOUS",
    }
)


def _duration_seconds(value: str) -> float:
    """Parse a bounded sample duration; zero means one single-frame sample."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--duration must be a finite number in the range [0, 300]"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed > 300:
        raise argparse.ArgumentTypeError(
            "--duration must be a finite number in the range [0, 300]"
        )
    return parsed


# Backward-compatible helper name for callers that used the initial draft.
_positive_duration = _duration_seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Windows smoke test for the DaGuandan target window and "
            "CaptureService single-frame path."
        )
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_PROFILE_ROOT,
        help="Profile directory or profiles parent containing profile.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Report file (.json) or an output directory containing report.json.",
    )
    parser.add_argument(
        "--duration",
        type=_duration_seconds,
        help="Optionally sample the same source for this many seconds (max 300).",
    )
    parser.add_argument(
        "--session",
        type=Path,
        help="Existing session directory to inspect read-only for diagnostic frame paths.",
    )
    return parser


def _print_json(payload: object) -> None:
    """Print UTF-8 JSON even when a Windows console uses a legacy code page."""

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        try:
            buffer.write(text.encode("utf-8"))
            buffer.flush()
            return
        except (AttributeError, OSError, TypeError):
            pass
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def _json_value(value: Any) -> Any:
    """Convert common runtime and NumPy values to JSON-safe data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_value(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_value(value.tolist())
        except Exception:
            pass
    return str(value)


def _check_visible_desktop(
    *,
    platform: str | None = None,
    user32: Any | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Return whether this process has a usable interactive Windows desktop.

    ``user32`` and ``platform`` are injectable so the environment gate can be
    unit-tested without opening a real desktop.  A service session, headless
    runner, or non-Windows host is a valid ``NOT_RUN`` result, not a failure of
    the capture implementation.
    """

    current_platform = sys.platform if platform is None else platform
    if current_platform != "win32":
        return False, {
            "reason": "non_windows_platform",
            "platform": current_platform,
            "message": "Windows desktop smoke test is only runnable on Windows",
        }

    try:
        api = user32 if user32 is not None else ctypes.windll.user32  # type: ignore[attr-defined]
        width = int(api.GetSystemMetrics(0))
        height = int(api.GetSystemMetrics(1))
        if width <= 0 or height <= 0:
            return False, {
                "reason": "screen_metrics_unavailable",
                "screen_size": [width, height],
                "message": "No visible physical desktop metrics are available",
            }

        # DESKTOP_READOBJECTS | DESKTOP_SWITCHDESKTOP.  A handle proves that
        # the process is attached to an interactive input desktop, rather than
        # merely running under Windows without a visible GUI session.
        desktop = api.OpenInputDesktop(0, False, 0x0001 | 0x0100)
        if not desktop:
            return False, {
                "reason": "visible_desktop_unavailable",
                "screen_size": [width, height],
                "message": "The Windows input desktop is not available or visible",
            }
        try:
            return True, {
                "reason": "visible_desktop_available",
                "screen_size": [width, height],
            }
        finally:
            close_desktop = getattr(api, "CloseDesktop", None)
            if callable(close_desktop):
                close_desktop(desktop)
    except Exception as exc:  # pragma: no cover - Windows boundary
        return False, {
            "reason": "visible_desktop_check_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "message": "The Windows visible-desktop check could not be completed",
        }


def _resolve_profile(profile_root: Path) -> tuple[Path, str]:
    """Resolve either a concrete profile directory or its profiles parent."""

    root = Path(profile_root).expanduser().resolve()
    if (root / "profile.json").is_file():
        return root.parent, root.name
    if not root.is_dir():
        raise FileNotFoundError(f"profile root is not a directory: {profile_root}")

    candidates = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "profile.json").is_file()
    )
    preferred = root / "tencent_daguandan"
    if preferred in candidates:
        return root, preferred.name
    if len(candidates) == 1:
        return root, candidates[0].name
    if not candidates:
        raise FileNotFoundError(
            f"profile.json not found below profile root: {profile_root}"
        )
    raise ValueError(
        "profile root contains multiple profiles; pass the concrete profile directory: "
        + ", ".join(path.name for path in candidates)
    )


def _check_diagnostic_paths(session: Path | None) -> dict[str, Any]:
    """Check existing diagnostic screenshot/metadata pairs without creating anything."""

    if session is None:
        return {
            "status": STATUS_NOT_CHECKED,
            "reason": "no_session_argument",
            "read_only": True,
        }

    root = Path(session).expanduser().resolve()
    if not root.is_dir():
        return {
            "status": STATUS_FAIL,
            "reason": "session_directory_missing",
            "session": str(root),
            "read_only": True,
        }

    diagnostic = root / "diagnostic_frames"
    if not diagnostic.exists():
        return {
            "status": STATUS_FAIL,
            "reason": "diagnostic_frames_directory_missing",
            "session": str(root),
            "diagnostic_frames": str(diagnostic),
            "read_only": True,
        }
    if not diagnostic.is_dir():
        return {
            "status": STATUS_FAIL,
            "reason": "diagnostic_frames_not_directory",
            "diagnostic_frames": str(diagnostic),
            "read_only": True,
        }

    try:
        entries = tuple(diagnostic.iterdir())
    except OSError as exc:
        return {
            "status": STATUS_FAIL,
            "reason": "diagnostic_frames_unreadable",
            "diagnostic_frames": str(diagnostic),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "read_only": True,
        }

    images = sorted(
        path
        for path in entries
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    metadata = sorted(
        path for path in entries if path.is_file() and path.suffix.lower() == ".json"
    )
    missing_metadata = sorted(
        path.name for path in images if not (path.with_suffix(".json")).is_file()
    )
    orphan_metadata = sorted(
        path.name
        for path in metadata
        if not any(image.with_suffix(".json") == path for image in images)
    )
    reparse_entries = sorted(
        path.name for path in entries if path.is_symlink()
    )
    valid = not missing_metadata and not orphan_metadata and not reparse_entries
    return {
        "status": STATUS_PASS if valid else STATUS_FAIL,
        "reason": "diagnostic_frame_pairs_valid"
        if valid
        else "diagnostic_frame_paths_invalid",
        "session": str(root),
        "diagnostic_frames": str(diagnostic),
        "image_count": len(images),
        "metadata_count": len(metadata),
        "missing_metadata": missing_metadata,
        "orphan_metadata": orphan_metadata,
        "reparse_entries": reparse_entries,
        "read_only": True,
    }


def _rect_payload(rect: Any) -> list[int] | dict[str, Any] | None:
    if rect is None:
        return None
    if isinstance(rect, Mapping):
        return _json_value(dict(rect))
    try:
        return [
            int(getattr(rect, "left")),
            int(getattr(rect, "top")),
            int(getattr(rect, "width")),
            int(getattr(rect, "height")),
        ]
    except (AttributeError, TypeError, ValueError):
        return None


def _snapshot_metadata(
    snapshot: Any,
    source_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract stable FrameSnapshot and capture metadata without persisting pixels."""

    source = dict(source_state or {})
    image = getattr(snapshot, "image", None)
    frame = getattr(snapshot, "frame", None)
    shape_value = getattr(image, "shape", ()) if image is not None else ()
    shape = [int(item) for item in shape_value] if shape_value else []
    size = int(getattr(image, "size", 0) or 0) if image is not None else 0
    captured_at = getattr(snapshot, "captured_at", None)
    if captured_at is not None and hasattr(captured_at, "isoformat"):
        captured_at = captured_at.isoformat()
    standardization = getattr(frame, "standardization", None)
    standardization_metadata: dict[str, Any] = {}
    for name in (
        "source_size",
        "source_viewport",
        "content_box",
        "scale",
        "padding",
        "aspect_error",
        "aspect_compatible",
    ):
        if standardization is not None and hasattr(standardization, name):
            value = getattr(standardization, name)
            if hasattr(value, "to_list") and callable(value.to_list):
                value = value.to_list()
            standardization_metadata[name] = _json_value(value)

    backend = getattr(frame, "backend", None) or source.get("backend")
    dpi = getattr(frame, "dpi", None) or source.get("dpi")
    rect = _rect_payload(getattr(frame, "rect", None)) or source.get("rect")
    window_title = getattr(frame, "window_title", None) or source.get("window_title")
    return {
        "evidence_frame_id": getattr(snapshot, "evidence_frame_id", None),
        "captured_at": captured_at,
        "captured_monotonic_ms": getattr(snapshot, "captured_monotonic_ms", None),
        "shape": shape,
        "size": size,
        "dtype": str(getattr(image, "dtype", "")) if image is not None else None,
        "backend": backend,
        "dpi": dpi,
        "rect": rect,
        "window_title": window_title,
        "non_empty": bool(size > 0 and len(shape) >= 2 and all(item > 0 for item in shape[:2])),
        "standardization": standardization_metadata,
    }


def _check(name: str, status: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "reason": reason, **_json_value(details)}


def _source_state(source: Any) -> dict[str, Any]:
    try:
        value = source.diagnostic_state()
    except Exception:
        value = {}
    if isinstance(value, Mapping):
        return dict(_json_value(value))
    return {}


def _snapshot_source_state(snapshot: Any) -> dict[str, Any]:
    """Derive target metadata when the one-shot CaptureService path is used."""

    frame = getattr(snapshot, "frame", None)
    return {
        "rect": _rect_payload(getattr(frame, "rect", None)),
        "dpi": getattr(frame, "dpi", None),
        "backend": getattr(frame, "backend", None),
        "window_title": getattr(frame, "window_title", None),
    }


def _occlusion_payload(backend: str | None) -> dict[str, Any]:
    normalized = str(backend or "").strip().lower()
    if normalized in {"screen", "gdi_screen"}:
        return {
            "status": "clear",
            "occluded": False,
            "checked_by_capture_source": True,
            "reason": "no_occlusion_error",
        }
    if normalized:
        return {
            "status": "not_applicable",
            "occluded": None,
            "checked_by_capture_source": False,
            "reason": "backend_does_not_read_visible_screen",
        }
    return {
        "status": STATUS_NOT_CHECKED,
        "occluded": None,
        "checked_by_capture_source": False,
        "reason": "capture_backend_not_reported",
    }


def _is_not_run_code(code: str) -> bool:
    return str(code).strip().upper() in _NOT_RUN_CAPTURE_CODES


def _finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    statuses = [item.get("status") for item in report.get("checks", [])]
    report["summary"] = {
        "passed_checks": sum(status == STATUS_PASS for status in statuses),
        "failed_checks": sum(status == STATUS_FAIL for status in statuses),
        "not_run_checks": sum(status == STATUS_NOT_RUN for status in statuses),
        "not_checked_checks": sum(status == STATUS_NOT_CHECKED for status in statuses),
    }
    report["exit_code"] = EXIT_CODES.get(str(report.get("status")), 1)
    return _json_value(report)


def run_smoke(
    *,
    profile_root: Path,
    output: Path | None = None,
    duration: float | None = None,
    session: Path | None = None,
    desktop_checker: Callable[[], tuple[bool, dict[str, Any]]] | None = None,
    service_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Run the bounded smoke test and return its JSON acceptance report.

    ``desktop_checker`` and ``service_factory`` are intentionally injectable so
    CI can test the report and error classification without controlling a real
    desktop or constructing the heavyweight application graph.
    """

    report: dict[str, Any] = {
        "schema": "daguandan.windows-smoke/1",
        "status": STATUS_NOT_RUN,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "side_effects": {
            "launches_target": False,
            "focuses_target": False,
            "moves_or_resizes_target": False,
            "sends_input": False,
            "writes_profile_or_session": False,
            "writes_report": True,
        },
        "inputs": {
            "profile_root": str(profile_root),
            "output": str(output) if output is not None else None,
            "duration_sec": duration,
            "session": str(session) if session is not None else None,
        },
        "environment": {},
        "target_window": {},
        "capture": {},
        "frame_snapshot": {},
        "diagnostic_paths": _check_diagnostic_paths(session),
        "checks": [],
        "errors": [],
    }

    diagnostic = report["diagnostic_paths"]
    if diagnostic.get("status") in {STATUS_PASS, STATUS_FAIL}:
        report["checks"].append(
            _check(
                "diagnostic_screenshot_paths",
                str(diagnostic["status"]),
                str(diagnostic.get("reason", "diagnostic_path_check")),
                diagnostic_frames=diagnostic.get("diagnostic_frames"),
            )
        )

    checker = desktop_checker or _check_visible_desktop
    try:
        visible, environment = checker()
    except Exception as exc:
        visible = False
        environment = {
            "reason": "visible_desktop_check_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "message": "The visible-desktop check could not be completed",
        }
    report["environment"] = {
        "platform": sys.platform,
        "visible_desktop": bool(visible),
        **_json_value(environment),
    }
    if not visible:
        report["status"] = STATUS_NOT_RUN
        report["errors"] = [
            {
                "code": "ENVIRONMENT-NOT-RUN",
                "reason": environment.get("reason", "environment_unavailable"),
                "message": environment.get(
                    "message", "No usable visible Windows desktop is available"
                ),
            }
        ]
        return _finalize_report(report)

    source: Any | None = None
    try:
        profiles_root, profile_name = _resolve_profile(Path(profile_root))
        if service_factory is None:
            from daguandan_bridge.capture_service import (  # noqa: PLC0415
                CaptureService,
            )

            service = CaptureService(profiles_root=profiles_root)
        else:
            service = service_factory(profiles_root)

        sample_duration = 0.0 if duration is None else float(duration)
        samples: list[dict[str, Any]] = []
        source_state: dict[str, Any] = {}

        if sample_duration <= 0 and callable(getattr(service, "capture_frame", None)):
            # Prefer the public one-shot API for the default smoke check.  It
            # creates and closes the same validated live source internally,
            # while remaining easy to replace with a small test double.
            snapshot = service.capture_frame(profile_name)
            source_state = _snapshot_source_state(snapshot)
            samples.append(_snapshot_metadata(snapshot, source_state))
        else:
            # A bounded duration intentionally keeps one source open so the
            # target identity, client geometry, DPI and backend remain stable.
            source = service.open_live_source(profile_name)
            source_state = _source_state(source)
            deadline = time.monotonic() + sample_duration
            while True:
                snapshot = source.capture()
                samples.append(_snapshot_metadata(snapshot, source_state))
                if sample_duration <= 0 or time.monotonic() >= deadline:
                    break
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))

        report["target_window"] = {
            "status": STATUS_PASS,
            "profile_name": profile_name,
            "profile_root": str(profiles_root / profile_name),
            **source_state,
        }
        report["checks"].append(
            _check("target_window_discovery", STATUS_PASS, "target_window_discovered")
        )

        latest = samples[-1]
        backend = str(latest.get("backend") or source_state.get("backend") or "")
        occlusion = _occlusion_payload(backend)
        report["capture"] = {
            "status": STATUS_PASS,
            "sample_count": len(samples),
            "backend": backend or None,
            "dpi": latest.get("dpi"),
            "rect": latest.get("rect"),
            "occlusion": occlusion,
        }
        snapshot_ok = bool(latest.get("non_empty"))
        report["frame_snapshot"] = {
            "status": STATUS_PASS if snapshot_ok else STATUS_FAIL,
            "latest": latest,
            "samples": samples,
        }
        report["checks"].extend(
            [
                _check(
                    "capture_service_single_frame",
                    STATUS_PASS,
                    "capture_succeeded",
                ),
                _check(
                    "frame_snapshot_metadata",
                    STATUS_PASS if snapshot_ok else STATUS_FAIL,
                    "frame_metadata_and_pixels_valid"
                    if snapshot_ok
                    else "empty_or_invalid_frame",
                ),
                _check(
                    "dpi_metadata",
                    STATUS_PASS
                    if isinstance(latest.get("dpi"), (int, float))
                    and int(latest["dpi"]) > 0
                    else STATUS_FAIL,
                    "dpi_present"
                    if isinstance(latest.get("dpi"), (int, float))
                    and int(latest["dpi"]) > 0
                    else "dpi_missing",
                    dpi=latest.get("dpi"),
                ),
                _check(
                    "occlusion_check",
                    STATUS_PASS
                    if occlusion["status"] in {"clear", "not_applicable"}
                    else STATUS_NOT_CHECKED,
                    str(occlusion["reason"]),
                ),
            ]
        )
    except (FileNotFoundError, ValueError) as exc:
        report["status"] = STATUS_FAIL
        report["errors"] = [
            {"code": "PROFILE_INVALID", "type": type(exc).__name__, "message": str(exc)}
        ]
        return _finalize_report(report)
    except Exception as exc:
        code = str(getattr(exc, "code", "CAPTURE-FAILED"))
        error = {
            "code": code,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        report["errors"] = [error]
        not_run = _is_not_run_code(code)
        status = STATUS_NOT_RUN if not_run else STATUS_FAIL
        report["target_window"] = {
            **report.get("target_window", {}),
            "status": status,
            "error_code": code,
        }
        report["capture"] = {
            **report.get("capture", {}),
            "status": status,
        }
        details = getattr(exc, "details", None)
        if isinstance(details, Mapping):
            report["capture"]["failure_details"] = _json_value(details)
        if code.upper() == "CAPTURE-OCCLUDED":
            report["capture"]["occlusion"] = {
                "status": "blocked",
                "occluded": True,
                "checked_by_capture_source": True,
                "reason": "capture_source_reported_occlusion",
            }
        report["checks"].append(
            _check(
                "target_window_discovery",
                status,
                "target_window_unavailable" if not_run else "capture_failed",
                error_code=code,
            )
        )
        report["status"] = status
        return _finalize_report(report)
    finally:
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                report.setdefault("errors", []).append(
                    {
                        "code": "CAPTURE-SOURCE-CLOSE-FAILED",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                report["checks"].append(
                    _check(
                        "capture_source_close",
                        STATUS_FAIL,
                        "capture_source_close_failed",
                    )
                )

    applicable = [
        item
        for item in report["checks"]
        if item["status"] in {STATUS_PASS, STATUS_FAIL}
    ]
    report["status"] = (
        STATUS_PASS
        if applicable and all(item["status"] == STATUS_PASS for item in applicable)
        else STATUS_FAIL
    )
    return _finalize_report(report)


def _report_path(output: Path) -> Path:
    resolved = Path(output).expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        return resolved
    return resolved / "windows_smoke_report.json"


def write_report(report: Mapping[str, Any], output: Path) -> Path:
    """Write exactly the acceptance report requested by the caller."""

    path = _report_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        # argparse uses status 2 for invalid arguments and 0 for --help.
        return int(exc.code) if isinstance(exc.code, int) else 2

    report = run_smoke(
        profile_root=args.profile_root,
        output=args.output,
        duration=args.duration,
        session=args.session,
    )
    report["report_path"] = str(_report_path(args.output))
    try:
        report_path = write_report(report, args.output)
        report["report_path"] = str(report_path)
    except Exception as exc:
        report["status"] = STATUS_FAIL
        report["errors"] = [
            *list(report.get("errors", [])),
            {
                "code": "REPORT-WRITE-FAILED",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        ]
        report["exit_code"] = EXIT_CODES[STATUS_FAIL]
        _print_json(report)
        return EXIT_CODES[STATUS_FAIL]

    # Emit the complete machine-readable acceptance report, not only a human
    # summary.  The report file and stdout intentionally share the same shape.
    _print_json(report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
