"""Portable window calibration and semantic binding service.

This module is deliberately an observation/persistence boundary.  It records
how a target window was observed; it never resizes, focuses, moves, or otherwise
controls a window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .models import ClientRect
from .runtime_layout import (
    RuntimeLayout,
    CALIBRATION_SCHEMA,
    load_calibration,
    load_window_binding,
    save_calibration,
    save_window_binding,
    stable_window_binding_id,
)


WINDOW_CALIBRATION_SCHEMA = "guandan.window-calibration/1"


@dataclass(frozen=True)
class WindowCalibrationArtifacts:
    """The two portable documents and their stable storage locations."""

    binding_id: str
    binding: dict[str, object]
    calibration: dict[str, object]
    binding_path: Path
    calibration_path: Path


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _size(value: Sequence[object], *, field: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must contain width and height")
    return (
        _positive_int(value[0], field=f"{field}.width"),
        _positive_int(value[1], field=f"{field}.height"),
    )


def _rect(value: ClientRect | Mapping[str, object] | Sequence[object]) -> dict[str, int]:
    if isinstance(value, ClientRect):
        values = (value.left, value.top, value.width, value.height)
    elif isinstance(value, Mapping):
        try:
            values = (value["left"], value["top"], value["width"], value["height"])
        except KeyError as exc:
            raise ValueError("client_rect requires left, top, width, and height") from exc
    else:
        if isinstance(value, (str, bytes)) or len(value) != 4:
            raise ValueError("client_rect must contain left, top, width, and height")
        values = tuple(value)
    left, top = int(values[0]), int(values[1])
    width = _positive_int(values[2], field="client_rect.width")
    height = _positive_int(values[3], field="client_rect.height")
    return {"left": left, "top": top, "width": width, "height": height}


def _text(value: str, *, field: str, required: bool = False) -> str:
    result = str(value).strip()
    if required and not result:
        raise ValueError(f"{field} is required")
    return result


def build_window_binding(
    profile_name: str,
    application_id: str,
    window_class: str,
    title_role: str,
    client_rect: ClientRect | Mapping[str, object] | Sequence[object],
    dpi: int,
    viewport_size: Sequence[object],
    base_size: Sequence[object],
) -> tuple[str, dict[str, object]]:
    """Build a portable semantic binding without touching the filesystem."""

    profile = _text(profile_name, field="profile_name", required=True)
    application = _text(application_id, field="application_id", required=True)
    window_class_value = _text(window_class, field="window_class")
    role = _text(title_role, field="title_role")
    rect = _rect(client_rect)
    viewport = _size(viewport_size, field="viewport_size")
    base = _size(base_size, field="base_size")
    dpi_value = _positive_int(dpi, field="dpi")
    binding_id = stable_window_binding_id(
        application_id=application,
        window_class=window_class_value,
        title_role=role,
        client_size=(rect["width"], rect["height"]),
    )
    binding: dict[str, object] = {
        "schema": "guandan.window-binding/1",
        "profile": profile,
        "binding_id": binding_id,
        "application_id": application,
        "window_class": window_class_value,
        "title_role": role,
        "client_rect": rect,
        "dpi": dpi_value,
        "viewport_size": list(viewport),
        "base_size": list(base),
        "control_policy": "observe-and-bind-only",
    }
    return binding_id, binding


def build_calibration(
    binding: Mapping[str, object],
) -> dict[str, object]:
    """Derive a JSON-safe calibration document from a binding."""

    viewport = _size(binding["viewport_size"], field="viewport_size")  # type: ignore[arg-type]
    base = _size(binding["base_size"], field="base_size")  # type: ignore[arg-type]
    rect = binding["client_rect"]
    if not isinstance(rect, Mapping):
        raise ValueError("binding.client_rect must be an object")
    result: dict[str, object] = {
        "schema": CALIBRATION_SCHEMA,
        "profile": binding["profile"],
        "binding_id": binding["binding_id"],
        "application_id": binding["application_id"],
        "client_rect": dict(rect),
        "dpi": binding["dpi"],
        "viewport_size": list(viewport),
        "base_size": list(base),
        "scale": {
            "x": viewport[0] / base[0],
            "y": viewport[1] / base[1],
        },
        "control_policy": "no-resize-no-window-control",
    }
    return result


def save_window_calibration(
    layout: RuntimeLayout,
    profile_name: str,
    application_id: str,
    window_class: str,
    title_role: str,
    client_rect: ClientRect | Mapping[str, object] | Sequence[object],
    dpi: int,
    viewport_size: Sequence[object],
    base_size: Sequence[object],
) -> WindowCalibrationArtifacts:
    """Build and persist the binding/calibration pair in stable app storage."""

    binding_id, binding = build_window_binding(
        profile_name,
        application_id,
        window_class,
        title_role,
        client_rect,
        dpi,
        viewport_size,
        base_size,
    )
    calibration = build_calibration(binding)
    binding_path = save_window_binding(layout, profile_name, binding, binding_id=binding_id)
    calibration_path = save_calibration(
        layout, profile_name, calibration, binding_id=binding_id
    )
    return WindowCalibrationArtifacts(
        binding_id=binding_id,
        binding=binding,
        calibration=calibration,
        binding_path=binding_path,
        calibration_path=calibration_path,
    )


def load_window_calibration(
    layout: RuntimeLayout,
    profile_name: str,
    binding_id: str,
) -> WindowCalibrationArtifacts | None:
    """Load a previously saved pair, or return ``None`` if neither exists."""

    binding = load_window_binding(layout, profile_name, binding_id=binding_id)
    calibration = load_calibration(layout, profile_name, binding_id=binding_id)
    if binding is None and calibration is None:
        return None
    if binding is None or calibration is None:
        raise ValueError("window binding and calibration must be saved as a pair")
    from .runtime_layout import calibration_path, window_binding_path

    return WindowCalibrationArtifacts(
        binding_id=binding_id,
        binding=binding,
        calibration=calibration,
        binding_path=window_binding_path(layout, profile_name, binding_id=binding_id),
        calibration_path=calibration_path(layout, profile_name, binding_id=binding_id),
    )


class WindowCalibrationService:
    """Small callable service facade; it performs no window control."""

    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout

    def save(self, **kwargs: object) -> WindowCalibrationArtifacts:
        return save_window_calibration(self.layout, **kwargs)  # type: ignore[arg-type]

    def load(self, profile_name: str, binding_id: str) -> WindowCalibrationArtifacts | None:
        return load_window_calibration(self.layout, profile_name, binding_id)


__all__ = [
    "WINDOW_CALIBRATION_SCHEMA",
    "WindowCalibrationArtifacts",
    "WindowCalibrationService",
    "build_calibration",
    "build_window_binding",
    "load_window_calibration",
    "save_window_calibration",
]



