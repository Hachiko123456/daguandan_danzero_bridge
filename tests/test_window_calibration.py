from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.models import ClientRect
from daguandan_bridge.runtime_layout import resolve_runtime_layout
from daguandan_bridge.window_calibration import (
    WindowCalibrationService,
    build_window_binding,
    load_window_calibration,
    save_window_calibration,
)


def _args(size=(1280, 764), rect=None):
    return dict(
        profile_name="tencent_daguandan",
        application_id="wechat",
        window_class="WeChatAppEx",
        title_role="game",
        client_rect=rect or ClientRect(100, 200, *size),
        dpi=144,
        viewport_size=size,
        base_size=(1280, 764),
    )


def test_semantic_binding_id_is_same_across_machines_and_has_no_hwnd_or_path():
    left_id, left = build_window_binding(**_args(rect=ClientRect(10, 20, 1280, 764)))
    right_id, right = build_window_binding(**_args(rect=ClientRect(1900, 80, 1280, 764)))
    assert left_id == right_id
    assert left["binding_id"] == right["binding_id"]
    encoded = json.dumps(left)
    assert "hwnd" not in encoded.lower()
    assert "path" not in encoded.lower()


def test_different_client_sizes_get_different_binding_ids():
    first, _ = build_window_binding(**_args(size=(1280, 764)))
    second, _ = build_window_binding(**_args(size=(1600, 900)))
    assert first != second


def test_save_load_round_trip_and_paths_stay_below_app_data_root(tmp_path: Path):
    layout = resolve_runtime_layout(frozen=False, bundle_root=tmp_path / "app")
    artifacts = save_window_calibration(layout, **_args())
    loaded = load_window_calibration(layout, "tencent_daguandan", artifacts.binding_id)
    assert loaded is not None
    assert loaded.binding == artifacts.binding
    assert loaded.calibration == artifacts.calibration
    assert artifacts.binding_path.is_relative_to(layout.app_data_root)
    assert artifacts.calibration_path.is_relative_to(layout.app_data_root)


def test_service_facade_is_callable_without_window_control(tmp_path: Path):
    layout = resolve_runtime_layout(frozen=False, bundle_root=tmp_path / "app")
    service = WindowCalibrationService(layout)
    saved = service.save(**_args())
    assert service.load("tencent_daguandan", saved.binding_id) is not None
    assert saved.binding["control_policy"] == "observe-and-bind-only"
    assert saved.calibration["control_policy"] == "no-resize-no-window-control"
