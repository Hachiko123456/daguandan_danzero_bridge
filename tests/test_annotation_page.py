import os
import shutil
import tempfile

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QAbstractItemView

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.gui.annotation_page import AnnotationPage


def test_annotation_page_displays_regions_and_supports_multi_select():
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService())

    assert page.region_table.rowCount() == 20
    assert page.region_table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert page.show_selected_button.text() == "标注选中区域"
    assert page.save_button.text() == "保存修改"

    page.close()
    app.processEvents()


def test_annotation_page_uses_chinese_dropdowns_and_removes_source_field():
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService())

    assert not hasattr(page, "source_edit")
    assert [page.region_table.horizontalHeaderItem(i).text() for i in range(4)] == [
        "名称", "角色", "绝对坐标", "比例坐标"
    ]
    assert page.name_combo.itemText(page.name_combo.findData("my_hand")) == "我的手牌"
    assert page.role_combo.itemText(page.role_combo.findData("hand")) == "手牌"
    assert page.region_table.item(0, 0).text() == "左侧首出牌提示"
    assert page.region_table.item(0, 1).text() == "通用区域"

    page.close()
    app.processEvents()


def test_annotation_page_loads_first_recorded_image(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))

    assert page.image_combo.count() == 1
    assert page.current_image is not None
    assert page.canvas.pixmap() is not None
    assert not page.canvas.pixmap().isNull()
    page.close()
    app.processEvents()
