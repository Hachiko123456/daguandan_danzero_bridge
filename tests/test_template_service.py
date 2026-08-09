import json
import shutil

import numpy as np
import pytest

from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.models import Box
from daguandan_bridge.template_service import TemplateService


def _service(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots"),
    )
    return TemplateService(root)


def test_template_service_saves_typed_crop_and_metadata(tmp_path):
    service = _service(tmp_path)
    initial_template_count = len(service.list_templates())
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    image[100:180, 100:180] = (0, 0, 0)

    sample = service.save_template(
        image,
        kind="anchor",
        label="table_anchor_new",
        box=Box(90, 90, 100, 100),
        source_image="game_test/000001.png",
        source_role="generic",
    )
    payload = json.loads(service.templates_path.read_text(encoding="utf-8"))

    assert sample.sample_id == "anchor_table_anchor_new_001"
    assert (service.profile_root / sample.file).is_file()
    assert len(payload["templates"]) == initial_template_count + 1
    assert payload["templates"][-1]["source_image"] == "game_test/000001.png"


def test_template_service_rejects_unknown_template_kind(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="模板类型"):
        service.save_template(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            kind="unknown",
            label="x",
            box=Box(1, 1, 20, 20),
            source_image="000001.png",
            source_role="generic",
        )


def test_template_service_deletes_managed_sample_and_metadata(tmp_path):
    service = _service(tmp_path)
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    sample = service.save_template(
        image,
        kind="anchor",
        label="temporary_anchor",
        box=Box(10, 10, 30, 30),
        source_image="game_test/000001.png",
    )

    deleted = service.delete_templates((sample.sample_id,))
    payload = json.loads(service.templates_path.read_text(encoding="utf-8"))

    assert deleted[0]["sample_id"] == sample.sample_id
    assert not (service.profile_root / sample.file).exists()
    assert all(item.get("sample_id") != sample.sample_id for item in payload["templates"])


def test_auto_registers_copied_png_and_persists_to_config(tmp_path):
    import cv2

    service = _service(tmp_path)
    before = len(service.list_templates())
    destination = service.templates_root / "rank" / "9_play_auto_001.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.full((30, 24, 3), 0, np.uint8))

    templates = service.list_templates()

    assert len(templates) == before + 1
    registered = next(
        item for item in templates if item.get("sample_id") == "rank_9_play_auto_001"
    )
    assert registered["label"] == "9"
    assert registered["source_role"] == "play"
    assert registered["abs_box"] == [0, 0, 24, 30]
    # 已落盘：再次列出不会重复注册
    payload = json.loads(service.templates_path.read_text(encoding="utf-8"))
    assert any(
        item.get("sample_id") == "rank_9_play_auto_001"
        for item in payload["templates"]
    )
    assert len(service.list_templates()) == before + 1


def test_template_service_normalizes_rank_to_white_background(tmp_path):
    service = _service(tmp_path)
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    image[120:175, 120:145] = (0, 0, 0)

    sample = service.save_template(
        image,
        kind="rank",
        label="A",
        box=Box(100, 100, 80, 100),
        source_image="game_test/000001.png",
        source_role="hand",
    )
    import cv2

    pixels = cv2.imdecode(np.fromfile(str(service.profile_root / sample.file), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert pixels is not None
    assert pixels.shape[0] < 100 and pixels.shape[1] < 80
    assert int(pixels.max()) == 255
    assert int(pixels.min()) == 0
