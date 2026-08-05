import json
import shutil

import cv2
import numpy as np
import pytest

from daguandan_bridge.annotation_service import (
    AnnotationService,
    RegionRecord,
    REGION_DISPLAY_NAMES,
    ROLE_DISPLAY_NAMES,
)
from daguandan_bridge.models import Box


def _temp_service(tmp_path):
    from daguandan_bridge.config import PROFILES_ROOT

    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots"),
    )
    return AnnotationService(root)


def test_migrated_profile_contains_all_named_regions():
    service = AnnotationService()
    regions = service.list_regions()

    assert len(regions) == 20
    assert {region.name for region in regions} >= {
        "my_hand",
        "my_play",
        "left_play",
        "opposite_play",
        "right_play",
        "table_anchor_1",
        "table_anchor_2",
    }


def test_recorded_image_discovery_is_recursive_and_root_limited(tmp_path):
    service = _temp_service(tmp_path)
    root = service.screenshots_root
    (root / "game_001" / "nested").mkdir(parents=True)
    (root / "game_001" / "000001.png").write_bytes(b"png")
    (root / "game_001" / "nested" / "000002.jpg").write_bytes(b"jpg")
    (tmp_path / "outside.png").write_bytes(b"outside")

    images = service.list_recorded_images()

    assert [path.name for path in images] == ["000001.png", "000002.jpg"]
    assert all(path.is_relative_to(root) for path in images)


def test_update_region_persists_absolute_and_ratio_boxes(tmp_path):
    service = _temp_service(tmp_path)
    original = service.list_regions()[0]

    updated = service.update_region(
        original.name,
        name=original.name,
        role="play",
        box=Box(10, 20, 300, 120),
    )
    reloaded = next(item for item in service.list_regions() if item.name == original.name)
    payload = json.loads(service.regions_path.read_text(encoding="utf-8"))

    assert updated.abs_box == Box(10, 20, 300, 120)
    assert reloaded.role == "play"
    assert reloaded.ratio_box == (10 / 1280, 20 / 720, 300 / 1280, 120 / 720)
    assert any(item["name"] == original.name for item in payload["regions"])
    assert all("source_image" not in item for item in payload["regions"])


def test_region_and_role_labels_are_chinese_but_keep_internal_keys():
    assert REGION_DISPLAY_NAMES["my_hand"] == "我的手牌"
    assert REGION_DISPLAY_NAMES["table_anchor_1"] == "牌桌锚点一"
    assert ROLE_DISPLAY_NAMES == {
        "hand": "手牌",
        "play": "出牌",
        "anchor": "锚点",
        "generic": "通用区域",
    }


def test_profile_region_config_has_no_source_image_field():
    service = AnnotationService()
    payload = json.loads(service.regions_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert len(payload["regions"]) == 20
    assert all("source_image" not in item for item in payload["regions"])


def test_legacy_source_image_is_ignored_and_not_written(tmp_path):
    service = _temp_service(tmp_path)
    original = service.list_regions()[0]

    updated = service.update_region(
        original.name,
        name=original.name,
        role="play",
        box=Box(10, 20, 300, 120),
    )
    payload = json.loads(service.regions_path.read_text(encoding="utf-8"))

    assert not hasattr(updated, "source_image")
    assert all("source_image" not in item for item in payload["regions"])


def test_update_region_can_rename_and_reject_duplicate_names(tmp_path):
    service = _temp_service(tmp_path)
    payload = json.loads(service.regions_path.read_text(encoding="utf-8"))
    payload["regions"] = [item for item in payload["regions"] if item["name"] != "my_hand"]
    service.regions_path.write_text(json.dumps(payload), encoding="utf-8")
    original = service.list_regions()[0]
    target = service.list_regions()[1]

    renamed = service.update_region(
        original.name,
        name="my_hand",
        role="play",
        box=original.abs_box,
    )

    assert renamed.name == "my_hand"
    with pytest.raises(ValueError, match="重复"):
        service.update_region(
            target.name,
            name="my_hand",
            role=target.role,
            box=target.abs_box,
        )


def test_overlay_uses_distinct_colors_for_selected_regions():
    service = AnnotationService()
    regions = (
        RegionRecord("my_hand", "hand", Box(10, 10, 40, 40), (0.0, 0.0, 0.0, 0.0)),
        RegionRecord("my_play", "play", Box(80, 80, 40, 40), (0.0, 0.0, 0.0, 0.0)),
    )
    image = np.zeros((160, 160, 3), dtype=np.uint8)

    overlay = service.overlay_regions(image, regions)

    assert tuple(overlay[10, 10]) != (0, 0, 0)
    assert tuple(overlay[80, 80]) != (0, 0, 0)
    assert tuple(overlay[10, 10]) != tuple(overlay[80, 80])
