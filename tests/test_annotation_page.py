import os
import json
import pytest
import shutil
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QAbstractItemView, QComboBox
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.gui.annotation_page import AnnotationPage
from daguandan_bridge.live.recorder import SessionRecorder
from daguandan_bridge.models import Box


def _wait_until(predicate, timeout_ms=3000):
    deadline = time.monotonic() + timeout_ms / 1000
    app = QApplication.instance()
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
    app.processEvents()
    return predicate()


def test_annotation_page_displays_regions_and_supports_multi_select():
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService())
    page.region_config_button.click()
    config = page.region_config_page

    assert config.region_table.rowCount() == 22
    assert config.region_table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert page.show_selected_button.text() == "标注选中区域"
    assert config.preview_selected_button.text() == "标注选中区域"
    assert config.save_button.text() == "保存修改"
    assert page.template_kind_combo.count() == 7
    assert page.crop_template_button.text() == "裁剪并保存模板"

    config.close()
    page.close()
    app.processEvents()


def test_annotation_page_uses_chinese_dropdowns_and_removes_source_field():
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService())
    page.region_config_button.click()
    config = page.region_config_page

    assert not hasattr(page, "source_edit")
    assert [config.region_table.horizontalHeaderItem(i).text() for i in range(3)] == [
        "名称", "绝对坐标", "比例坐标"
    ]
    assert config.name_combo.itemText(config.name_combo.findData("my_hand")) == "我的手牌"
    assert config.role_combo.itemText(config.role_combo.findData("hand")) == "手牌"
    assert config.role_combo.isHidden()
    assert page.region_role_combo.isHidden()
    assert (
        config.name_combo.itemText(config.name_combo.findData("button_actions"))
        == "按钮区域"
    )
    assert config.region_table.item(0, 0).text() == "左侧首出牌提示"
    assert config.region_table.item(0, 1).text() == "[128, 210, 120, 77]"

    config.close()
    page.close()
    app.processEvents()


def test_annotation_page_loads_first_recorded_image(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
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


def test_annotation_page_does_not_repeat_preview_path_in_status(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))

    assert "已预览" not in page.status.text()
    page.close()
    app.processEvents()


def test_template_label_is_an_editable_dropdown(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))

    assert isinstance(page.template_label_edit, QComboBox)
    assert page.template_label_edit.isEditable()
    page.template_label_edit.setText("new_manual_label")
    assert page.template_label_edit.currentText() == "new_manual_label"

    page.close()
    app.processEvents()


def test_template_filter_uses_template_folder_kinds_and_label_field_opens_popup():
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService())
    page.resize(1200, 800)
    page.show()
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))
    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("effect"))
    app.processEvents()

    assert [
        page.template_filter_combo.itemData(index)
        for index in range(page.template_filter_combo.count())
    ] == ["", "anchor", "button", "effect", "rank", "suit", "status", "timer"]
    QTest.mouseClick(
        page.template_label_edit.lineEdit(), Qt.MouseButton.LeftButton
    )
    app.processEvents()
    assert page.template_label_edit.view().isVisible()

    page.template_label_edit.hidePopup()
    page.close()
    app.processEvents()


def test_template_editor_uses_coordinate_table_and_selectable_button_values(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.resize(1100, 900)
    page.show()
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))
    app.processEvents()

    assert [
        page.template_table.horizontalHeaderItem(index).text()
        for index in range(page.template_table.columnCount())
    ] == ["模板类型", "模板标签", "来源角色", "绝对坐标", "比例坐标", "文件"]
    assert page.template_table.rowCount() > 0
    assert page.template_table.item(0, 3).text().startswith("[")
    assert page.template_table.item(0, 4).text().startswith("[")
    assert not page.template_table.horizontalScrollBar().isVisible()

    assert page.template_kind_combo.isEnabled()
    assert page.template_source_role_combo.isEnabled()
    assert page.template_label_edit.isEnabled()
    assert page.template_kind_combo.findData("button") >= 0
    assert page.template_source_role_combo.findData("generic") >= 0

    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("button"))
    button_labels = {
        page.template_label_edit.itemData(index)
        for index in range(page.template_label_edit.count())
    }
    assert {"super_double", "double", "cannot_beat"}.issubset(button_labels)
    assert page.template_label_edit.itemText(
        page.template_label_edit.findData("super_double")
    ) == "超级加倍"
    assert page.template_label_edit.itemText(
        page.template_label_edit.findData("double")
    ) == "加倍"
    assert page.template_label_edit.itemText(
        page.template_label_edit.findData("cannot_beat")
    ) == "要不起"

    page.template_source_role_combo.setCurrentIndex(
        page.template_source_role_combo.findData("generic")
    )
    page.template_label_edit.setCurrentIndex(
        page.template_label_edit.findData("double")
    )
    assert page.template_kind_combo.currentData() == "button"
    assert page.template_source_role_combo.currentData() == "generic"
    assert page.template_label_edit.currentText() == "加倍"
    assert page.template_label_edit.currentData() == "double"

    page.close()
    app.processEvents()


def test_template_label_options_follow_template_kind_and_expose_level_suit_crop(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))

    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("rank"))
    rank_labels = {
        page.template_label_edit.itemData(index)
        for index in range(page.template_label_edit.count())
    }
    assert rank_labels == {
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
        "small_joker", "big_joker",
    }

    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("suit"))
    suit_labels = {
        page.template_label_edit.itemData(index)
        for index in range(page.template_label_edit.count())
    }
    assert suit_labels == {"spade", "heart", "club", "diamond"}
    assert page.template_source_role_combo.findData("level") >= 0
    assert page.template_source_role_combo.itemText(
        page.template_source_role_combo.findData("level")
    ) == "级牌 / 逢人配"

    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("effect"))
    effect_labels = {
        page.template_label_edit.itemData(index)
        for index in range(page.template_label_edit.count())
    }
    assert {
        "straight", "straight_flush", "bomb", "joker_bomb",
        "three_with_two", "two_trips", "triple_pair", "consecutive_pairs",
    }.issubset(effect_labels)
    assert page.template_label_edit.itemText(
        page.template_label_edit.findData("straight_flush")
    ) == "同花顺特效"

    page.close()
    app.processEvents()


def test_region_mode_hides_template_editor_and_shows_selected_region_coordinates(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.resize(1100, 900)
    page.show()
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("region"))
    app.processEvents()

    assert page.mode_detail_stack.currentIndex() == 0
    assert page.mode_detail_stack.height() <= 300
    assert page.status.height() <= 60
    assert page.image_navigation_layout.geometry().top() < 100
    assert page.template_detail_widget.isVisible() is False
    assert page.template_kind_combo.isVisible() is False
    assert page.template_label_edit.isVisible() is False
    assert page.template_source_role_combo.isVisible() is False
    assert [
        page.region_coordinate_table.horizontalHeaderItem(index).text()
        for index in range(page.region_coordinate_table.columnCount())
    ] == ["区域", "x", "y", "w", "h"]

    page.region_config_button.click()
    page.region_config_page.region_table.selectRow(0)
    app.processEvents()

    assert page.region_coordinate_table.rowCount() == 1
    assert page.region_coordinate_table.item(0, 0).text() == "左侧首出牌提示"
    assert [
        page.region_coordinate_table.item(0, column).text()
        for column in range(1, 5)
    ] == ["128", "210", "120", "77"]

    page.region_config_page.close()
    page.close()
    app.processEvents()


def test_region_mode_supports_choose_drag_and_save_region_annotation(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("region"))
    page.region_name_combo.setCurrentIndex(page.region_name_combo.findData("my_hand"))
    page.region_role_combo.setCurrentIndex(page.region_role_combo.findData("hand"))
    page._set_roi(Box(50, 100, 200, 90))
    page.save_region_button.click()

    saved = next(
        record for record in AnnotationService(root).list_regions()
        if record.name == "my_hand"
    )
    assert saved.role == "hand"
    assert saved.abs_box == Box(50, 100, 200, 90)
    assert "已保存区域" in page.status.text()
    assert [
        page.region_coordinate_table.item(0, column).text()
        for column in range(1, 5)
    ] == ["50", "100", "200", "90"]

    page.close()
    app.processEvents()


def test_region_mode_canvas_drag_updates_region_coordinates_before_save(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(
        str(image_dir / "000001.png"),
        np.full((720, 1280, 3), 255, dtype=np.uint8),
    )
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.show()
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("region"))
    page.region_name_combo.setCurrentIndex(page.region_name_combo.findData("my_hand"))
    app.processEvents()
    pixmap_rect = page.canvas._pixmap_rect()
    start = QPoint(pixmap_rect.left() + 20, pixmap_rect.top() + 20)
    end = QPoint(pixmap_rect.left() + 120, pixmap_rect.top() + 80)

    QTest.mousePress(page.canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(page.canvas, end, delay=20)
    QTest.mouseRelease(page.canvas, Qt.MouseButton.LeftButton, pos=end)
    app.processEvents()

    assert page.current_roi is not None
    assert page.current_roi.w > 0 and page.current_roi.h > 0
    assert page.region_coordinate_table.item(0, 3).text() == str(page.current_roi.w)
    assert page.region_coordinate_table.item(0, 4).text() == str(page.current_roi.h)

    page.close()
    app.processEvents()


def test_template_roi_updates_coordinates_and_remains_visible_in_status(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.full((720, 1280, 3), 255, dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))
    box = Box(100, 100, 120, 80)
    page._set_roi(box)

    assert (page.x_spin.value(), page.y_spin.value(), page.w_spin.value(), page.h_spin.value()) == (100, 100, 120, 80)
    assert "已选模板框：x=100, y=100, w=120, h=80" in page.status.text()

    page.close()
    app.processEvents()


def test_single_image_state_form_builds_confirmed_game_state(tmp_path):
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    page.round_level_combo.setCurrentIndex(page.round_level_combo.findData("2"))
    page.wild_rank_combo.setCurrentIndex(page.wild_rank_combo.findData("2"))
    page.current_player_combo.setCurrentIndex(page.current_player_combo.findData("self"))
    page.lead_player_combo.setCurrentIndex(page.lead_player_combo.findData("self"))
    page.my_hand_edit.setText("3S 4H 5D")

    page.event_player_combo.setCurrentIndex(page.event_player_combo.findData("opposite"))
    page.event_action_combo.setCurrentIndex(page.event_action_combo.findData("play"))
    page.event_cards_edit.setText("6S 6H")
    page.add_event_button.click()
    page.event_player_combo.setCurrentIndex(page.event_player_combo.findData("right"))
    page.event_action_combo.setCurrentIndex(page.event_action_combo.findData("pass"))
    page.event_cards_edit.clear()
    page.add_event_button.click()

    state = page.build_state()

    assert state.round_level == "2"
    assert state.my_hand == ("3S", "4H", "5D")
    assert [(event.player, event.cards, event.is_pass) for event in state.trick_plays] == [
        ("opposite", ("6H", "6S"), False),
        ("right", (), True),
    ]
    page.close()
    app.processEvents()


def test_single_image_state_form_previews_source_image(tmp_path):
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage

    image_path = tmp_path / "single.png"
    cv2.imwrite(str(image_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage(image_path)

    assert page.image_preview.pixmap() is not None
    assert not page.image_preview.pixmap().isNull()
    page.close()
    app.processEvents()


def test_single_image_state_form_has_scrollable_full_image_preview(tmp_path):
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage

    image_path = tmp_path / "single.png"
    cv2.imwrite(str(image_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage(image_path)

    assert page.content_scroll.widgetResizable() is True
    assert page.image_preview.minimumHeight() >= 360
    pixmap = page.image_preview.pixmap()
    assert pixmap is not None and abs(pixmap.width() / pixmap.height() - 16 / 9) < 0.02
    page.close()
    app.processEvents()


def test_single_image_state_form_applies_template_recognition_result():
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage
    from daguandan_bridge.recognition_service import RecognizedEvent, RecognitionResult

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    result = RecognitionResult(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="opposite",
        my_hand=("3S", "4H"),
        events=(
            RecognizedEvent("opposite", ("5D",), False, 0.91, "template:cards"),
            RecognizedEvent("right", (), True, 0.88, "template:status"),
        ),
        field_confidences={"my_hand": 0.91},
        sources={"my_hand": "template:cards"},
        unresolved_fields=(),
        diagnostics=(),
    )

    page.apply_recognition(result)

    assert page.my_hand_edit.text() == "3S 4H"
    assert page.round_level_combo.currentData() == "2"
    assert page.lead_player_combo.currentData() == "opposite"
    assert page.event_table.rowCount() == 2
    assert "模板" in page.recognition_status.text()
    page.close()
    app.processEvents()


def test_single_image_page_displays_colored_cards_and_annotation_boxes():
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage
    from daguandan_bridge.recognition_service import (
        RecognitionAnnotation,
        RecognitionResult,
    )

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    result = RecognitionResult(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="self",
        my_hand=("5H", "9S"),
        events=(),
        field_confidences={"my_hand": 0.95},
        sources={"my_hand": "template:cards"},
        unresolved_fields=(),
        diagnostics=(),
        annotations=(
            RecognitionAnnotation("5H", (20, 500, 40, 60), 0.95, "hand"),
        ),
    )

    page.apply_recognition(result)

    assert len(page.hand_card_widgets) == 2
    assert page.hand_card_widgets[0].suit_label.text() == "♥"
    assert page.hand_card_widgets[0].suit_label.styleSheet()
    assert page.annotation_legend.text().find("5H") >= 0
    assert len(page.preview_annotations) == 1
    page.close()
    app.processEvents()


def test_single_image_page_exposes_hand_as_copyable_danzero_code():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    page.my_hand_edit.setText("3S 4H small_joker")
    page.copy_hand_code_button.click()

    assert "DanZero" in page.copy_hand_code_button.toolTip()
    assert QGuiApplication.clipboard().text() == "3S 4H small_joker"
    assert "mono" in page.my_hand_edit.font().family().lower()
    assert page.form_splitter.orientation() == Qt.Orientation.Horizontal

    page.resize(900, 800)
    page.show()
    app.processEvents()
    assert page.form_splitter.orientation() == Qt.Orientation.Vertical
    page.close()
    app.processEvents()


def test_single_image_page_displays_detected_button_values():
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage
    from daguandan_bridge.recognition_service import (
        RecognitionAnnotation,
        RecognitionResult,
    )

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    result = RecognitionResult(
        round_level=None,
        wild_rank=None,
        current_player=None,
        lead_player=None,
        my_hand=(),
        events=(),
        field_confidences={},
        sources={},
        unresolved_fields=(),
        diagnostics=(),
        annotations=(
            RecognitionAnnotation(
                "cannot_beat", (590, 235, 210, 80), 0.99, "button"
            ),
        ),
        buttons=("cannot_beat", "super_double"),
    )

    page.apply_recognition(result)

    assert "要不起" in page.button_detection_label.text()
    assert "超级加倍" in page.button_detection_label.text()
    assert "要不起" in page.annotation_legend.text()
    page.close()
    app.processEvents()


def test_single_image_page_wheel_does_not_change_recognition_combos():
    from PySide6.QtCore import QPoint, QPointF, QEvent, Qt
    from PySide6.QtGui import QWheelEvent
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    combo = page.round_level_combo
    combo.setCurrentIndex(combo.findData("2"))
    event = QWheelEvent(
        QPointF(4, 4),
        QPointF(4, 4),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    combo.wheelEvent(event)

    assert combo.currentData() == "2"
    page.close()
    app.processEvents()


def test_single_image_page_copies_image_or_path_from_preview_and_double_clicks_key_info(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage

    image_path = tmp_path / "copy-me.png"
    cv2.imwrite(str(image_path), np.full((80, 120, 3), 255, dtype=np.uint8))
    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage(image_path)
    page.recognition_status.setText("需要复制的关键识别信息")

    assert page.image_preview.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    page._copy_image_to_clipboard()
    assert not QGuiApplication.clipboard().image().isNull()
    page._copy_image_path_to_clipboard()
    assert QGuiApplication.clipboard().text() == str(image_path)

    page.recognition_status.mouseDoubleClickEvent(None)
    assert QGuiApplication.clipboard().text() == "需要复制的关键识别信息"
    page.close()
    app.processEvents()


def test_annotation_page_drops_stale_recognition_result_after_image_switch(tmp_path):
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage
    from daguandan_bridge.recognition_service import RecognitionResult

    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    first_path = image_dir / "000001.png"
    second_path = image_dir / "000002.png"
    cv2.imwrite(str(first_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    cv2.imwrite(str(second_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.current_image_path = second_path.resolve()
    page.single_image_danzero_page = SingleImageDanzeroPage(second_path)
    page._recognition_request_id = 2
    stale = RecognitionResult(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="self",
        my_hand=("AH",),
        events=(),
        field_confidences={},
        sources={},
        unresolved_fields=(),
        diagnostics=(),
    )

    page._template_recognition_succeeded(
        stale,
        request_id=1,
        image_path=first_path.resolve(),
    )

    assert page.single_image_danzero_page.my_hand_edit.text() == ""
    page.single_image_danzero_page.close()
    page.close()
    app.processEvents()


def test_single_image_state_form_rejects_invalid_cards(tmp_path):
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    page.round_level_combo.setCurrentIndex(page.round_level_combo.findData("2"))
    page.wild_rank_combo.setCurrentIndex(page.wild_rank_combo.findData("2"))
    page.current_player_combo.setCurrentIndex(page.current_player_combo.findData("self"))
    page.lead_player_combo.setCurrentIndex(page.lead_player_combo.findData("self"))
    page.my_hand_edit.setText("not_a_card")

    with pytest.raises(ValueError):
        page.build_state()
    assert "无效" in page.status.text()
    page.close()
    app.processEvents()


def test_annotation_page_opens_single_image_danzero_test_page(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    assert page.single_image_test_button.text() == "单图标注 / 测试 DanZero"
    page.single_image_test_button.click()
    app.processEvents()

    assert page.single_image_danzero_page is not None
    assert page.single_image_danzero_page.isVisible()
    assert page.single_image_danzero_page.image_path.name == "000001.png"

    page.single_image_danzero_page.close()
    page.close()
    app.processEvents()


def test_annotation_page_auto_fills_single_image_from_template_recognition(tmp_path):
    from daguandan_bridge.recognition_service import RecognitionResult

    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    recognition_options = []

    class FakeRecognition:
        def recognize(self, image, *, allow_unknown_suit=False):
            recognition_options.append(allow_unknown_suit)
            return RecognitionResult(
                round_level="2",
                wild_rank="2",
                current_player="self",
                lead_player="self",
                my_hand=("3S", "4H"),
                events=(),
                field_confidences={"my_hand": 0.95},
                sources={"my_hand": "template:cards"},
                unresolved_fields=(),
                diagnostics=(),
            )

    page = AnnotationPage(AnnotationService(root))
    page.recognition_service = FakeRecognition()
    page.single_image_test_button.click()
    dialog = page.single_image_danzero_page

    assert _wait_until(lambda: dialog.recognition_status.text().startswith("模板识别完成"))
    assert dialog.my_hand_edit.text() == "3S 4H"
    assert dialog.current_player_combo.currentData() == "self"
    assert recognition_options == [True]

    dialog.close()
    page.close()
    app.processEvents()


def test_single_image_page_displays_recognition_elapsed_time():
    from daguandan_bridge.gui.single_image_danzero_page import SingleImageDanzeroPage
    from daguandan_bridge.recognition_service import RecognitionResult

    app = QApplication.instance() or QApplication([])
    page = SingleImageDanzeroPage()
    result = RecognitionResult(
        round_level="2",
        wild_rank="2",
        current_player="self",
        lead_player="self",
        my_hand=("3S",),
        events=(),
        field_confidences={},
        sources={},
        unresolved_fields=(),
        diagnostics=(),
        elapsed_ms=123.45,
    )

    page.apply_recognition(result)

    assert page.recognition_elapsed_ms == 123.45
    assert "123.45" in page.recognition_status.text()
    assert "123.45" in page.state_summary.toPlainText()
    page.close()
    app.processEvents()


def test_single_image_danzero_test_runs_async_and_shows_advice(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    captured = []

    class FakeAdvisor:
        def recommend(self, state, *, request_id=""):
            captured.append((state, request_id))
            time.sleep(0.05)
            return SimpleNamespace(
                cards=("3S",),
                play_type="Single",
                is_pass=False,
                state_revision=state.revision,
                elapsed_ms=12.5,
                request_id=request_id,
                engine_input={"project_snapshot": {"my_hand": ["3S"]}},
                timings={"create_agent": 100.0, "agent_step": 12.0},
            )

    page = AnnotationPage(AnnotationService(root))
    page.danzero_advisor = FakeAdvisor()
    page.single_image_test_button.click()
    dialog = page.single_image_danzero_page
    dialog.round_level_combo.setCurrentIndex(dialog.round_level_combo.findData("2"))
    dialog.wild_rank_combo.setCurrentIndex(dialog.wild_rank_combo.findData("2"))
    dialog.current_player_combo.setCurrentIndex(dialog.current_player_combo.findData("self"))
    dialog.lead_player_combo.setCurrentIndex(dialog.lead_player_combo.findData("self"))
    dialog.my_hand_edit.setText("3S 4H 5D")

    dialog.test_button.click()

    assert dialog.test_button.isEnabled() is False
    assert _wait_until(lambda: dialog.test_button.isEnabled())
    assert captured and captured[0][0].my_hand == ("3S", "4H", "5D")
    assert "3S" in dialog.result_text.toPlainText()
    assert "engine_input" in dialog.result_text.toPlainText()
    assert "模型初始化" in dialog.result_text.toPlainText()
    assert "模型推理" in dialog.result_text.toPlainText()

    dialog.close()
    page.close()
    app.processEvents()


def test_annotation_page_preloads_danzero_in_background(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])

    class FakeAdvisor:
        def __init__(self):
            self.initialized = False

        def initialize(self):
            self.initialized = True

    page = AnnotationPage(AnnotationService(root))
    fake = FakeAdvisor()
    page.danzero_advisor = fake
    page._start_danzero_warmup()

    assert _wait_until(lambda: fake.initialized and page._danzero_warmup_elapsed_ms is not None)
    assert page._danzero_warmup_elapsed_ms is not None
    page.close()
    app.processEvents()


def test_single_image_danzero_test_shows_error_and_restores_button(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    class FailingAdvisor:
        def recommend(self, state, *, request_id=""):
            raise RuntimeError("模拟 DanZero 异常")

    page = AnnotationPage(AnnotationService(root))
    page.danzero_advisor = FailingAdvisor()
    page.single_image_test_button.click()
    dialog = page.single_image_danzero_page
    dialog.round_level_combo.setCurrentIndex(dialog.round_level_combo.findData("2"))
    dialog.wild_rank_combo.setCurrentIndex(dialog.wild_rank_combo.findData("2"))
    dialog.current_player_combo.setCurrentIndex(dialog.current_player_combo.findData("self"))
    dialog.lead_player_combo.setCurrentIndex(dialog.lead_player_combo.findData("self"))
    dialog.my_hand_edit.setText("3S")
    dialog.test_button.click()

    assert _wait_until(lambda: dialog.test_button.isEnabled())
    assert "模拟 DanZero 异常" in dialog.result_text.toPlainText()
    assert dialog.status.text() == "DanZero 测试失败"

    dialog.close()
    page.close()
    app.processEvents()


def test_annotation_page_navigates_images_with_previous_and_next_buttons(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    cv2.imwrite(str(image_dir / "000002.png"), np.full((720, 1280, 3), 255, dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))

    assert page.previous_image_button.isEnabled() is False
    assert page.next_image_button.isEnabled() is True
    page.next_image_button.click()
    assert page.current_image_path.name == "000002.png"
    assert page.previous_image_button.isEnabled() is True
    assert page.next_image_button.isEnabled() is False
    page.previous_image_button.click()
    assert page.current_image_path.name == "000001.png"

    page.close()
    app.processEvents()


def test_annotation_page_opens_standalone_region_config_and_places_arrows_around_canvas(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    assert not hasattr(page, "region_table")
    assert page.region_config_button.text() == "查看区域配置"
    canvas_index = page.image_navigation_layout.indexOf(page.canvas)
    assert canvas_index > 0
    assert page.image_navigation_layout.indexOf(page.previous_image_button) == canvas_index - 1
    assert page.image_navigation_layout.indexOf(page.next_image_button) == canvas_index + 1

    page.region_config_button.click()
    app.processEvents()
    assert page.region_config_page is not None
    assert page.region_config_page.isVisible()

    page.region_config_page.close()
    page.close()
    app.processEvents()


def test_region_config_selection_updates_preview_regions(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "000001.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.region_config_button.click()
    config = page.region_config_page
    config.region_table.selectRow(0)
    app.processEvents()

    assert [region.name for region in page.preview_regions] == ["first_play_left"]
    assert page.overlay_visible is False
    assert page.show_selected_button.text() == "标注选中区域"

    config.close()
    page.close()
    app.processEvents()


def test_region_mode_roi_is_visible_and_syncs_config_editor(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    cv2.imwrite(
        str(image_dir / "000001.png"),
        np.full((720, 1280, 3), 255, dtype=np.uint8),
    )
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.region_config_button.click()
    config = page.region_config_page
    config.region_table.selectRow(0)
    box = Box(100, 100, 120, 80)
    page._set_roi(box)
    app.processEvents()

    assert page.current_roi == box
    assert config.x_spin.value() == 100
    assert config.y_spin.value() == 100
    pixmap = page.canvas.pixmap()
    assert pixmap is not None and not pixmap.isNull()
    orange_pixels = 0
    image = pixmap.toImage()
    for x in range(image.width()):
        for y in range(image.height()):
            red, green, blue, _ = image.pixelColor(x, y).getRgb()
            if red > 245 and 130 <= green <= 190 and blue < 10:
                orange_pixels += 1
    assert orange_pixels > 20

    config.close()
    page.close()
    app.processEvents()


def test_annotation_page_crops_template_from_recorded_image(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    image[100:180, 100:180] = (0, 0, 0)
    cv2.imwrite(str(image_dir / "000001.png"), image)
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))
    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("anchor"))
    page.template_label_edit.setText("table_anchor_new")
    page._set_template_roi(Box(90, 90, 100, 100))
    page.crop_template_button.click()

    assert _wait_until(lambda: page.crop_template_button.isEnabled())
    payload = json.loads((root / "tencent_daguandan" / "templates_config.json").read_text(encoding="utf-8"))
    assert payload["templates"][-1]["sample_id"] == "anchor_table_anchor_new_001"
    assert (root / "tencent_daguandan" / payload["templates"][-1]["file"]).is_file()
    assert "anchor_table_anchor_new_001" in page.status.text()
    assert "table_anchor_new" in [
        page.template_label_edit.itemText(index)
        for index in range(page.template_label_edit.count())
    ]
    page.close()
    app.processEvents()


def test_template_crop_runs_without_blocking_the_gui(tmp_path, monkeypatch):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    image_dir = root / "tencent_daguandan" / "screenshots" / "game_test"
    image_dir.mkdir(parents=True)
    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    image[100:180, 100:180] = (0, 0, 0)
    cv2.imwrite(str(image_dir / "000001.png"), image)
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))
    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("anchor"))
    page.template_label_edit.setText("async_anchor")
    page._set_template_roi(Box(90, 90, 100, 100))
    original_save = page.template_service.save_template

    def slow_save(image, **kwargs):
        time.sleep(0.2)
        return original_save(image, **kwargs)

    monkeypatch.setattr(page.template_service, "save_template", slow_save)
    started = time.perf_counter()
    page.crop_template_button.click()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.1
    assert page.crop_template_button.isEnabled() is False
    assert page.status.text() == "正在裁剪模板……"
    assert _wait_until(lambda: page.crop_template_button.isEnabled())
    assert "模板已保存：" in page.status.text()

    page.close()
    app.processEvents()


def test_region_config_page_displays_chinese_fields_and_emits_selection(tmp_path):
    from daguandan_bridge.gui.region_config_page import RegionConfigPage

    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])
    service = AnnotationService(root)
    page = RegionConfigPage(service, list(service.list_regions()))
    selected = []
    requested = []
    page.region_selected.connect(selected.append)
    page.preview_requested.connect(requested.append)

    assert [page.region_table.horizontalHeaderItem(i).text() for i in range(3)] == [
        "名称", "绝对坐标", "比例坐标"
    ]
    assert page.name_combo.itemText(page.name_combo.findData("my_hand")) == "我的手牌"
    assert page.role_combo.itemText(page.role_combo.findData("hand")) == "手牌"
    assert page.role_combo.isHidden()
    assert (
        page.name_combo.itemText(page.name_combo.findData("button_actions"))
        == "按钮区域"
    )

    page.region_table.selectRow(0)
    app.processEvents()
    assert selected[-1].name == "first_play_left"
    page.preview_selected_button.click()
    assert [record.name for record in requested[-1]] == ["first_play_left"]

    page.close()
    app.processEvents()


def test_region_config_page_save_emits_updated_record(tmp_path):
    from daguandan_bridge.gui.region_config_page import RegionConfigPage

    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])
    service = AnnotationService(root)
    page = RegionConfigPage(service, list(service.list_regions()))
    updated = []
    page.region_updated.connect(lambda old_name, record: updated.append((old_name, record)))

    page.region_table.selectRow(0)
    page.x_spin.setValue(25)
    page.y_spin.setValue(35)
    page.w_spin.setValue(45)
    page.h_spin.setValue(55)
    page.save_button.click()
    app.processEvents()

    assert updated[-1][0] == "first_play_left"
    assert updated[-1][1].abs_box == Box(25, 35, 45, 55)
    assert service.list_regions()[0].abs_box == Box(25, 35, 45, 55)

    page.close()
    app.processEvents()


def test_annotation_page_browses_any_local_image_folder_with_arrow_navigation(tmp_path):
    image_dir = tmp_path / "external_samples" / "nested"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "01.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    cv2.imwrite(str(image_dir / "02.png"), np.full((720, 1280, 3), 255, dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService())
    page.set_image_folder(image_dir.parent)

    assert page.image_folder == image_dir.parent.resolve()
    assert page.image_combo.count() == 2
    assert page.current_image_path.name == "01.png"
    assert page.previous_image_button.isEnabled() is False
    assert page.next_image_button.isEnabled() is True
    assert page.image_position_label.text() == "1 / 2"

    page.next_image_button.click()
    assert page.current_image_path.name == "02.png"
    assert page.image_position_label.text() == "2 / 2"
    page.close()
    app.processEvents()


def test_annotation_page_can_step_a_recorded_session_without_saving_a_screenshot(tmp_path):
    root = tmp_path / "profiles"
    profile_root = root / "tencent_daguandan"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        profile_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    session = profile_root / "sessions" / "game_annotation_test"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session.name,
                "status": "sealed",
                "frame_count": 2,
                "dropped_frames": 0,
            }
        ),
        encoding="utf-8",
    )
    recorder = SessionRecorder(session, size=(64, 32), fps=10)
    recorder.write_frame(
        np.zeros((32, 64, 3), dtype=np.uint8),
        captured_monotonic_ms=100,
        wall_time="t0",
    )
    recorder.write_frame(
        np.full((32, 64, 3), 255, dtype=np.uint8),
        captured_monotonic_ms=200,
        wall_time="t1",
    )
    recorder.close()
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService(root))
    assert not hasattr(page, "source_mode_combo")
    session_index = page.session_combo.findData(str(session.resolve()))
    assert session_index >= 0
    page.session_combo.setCurrentIndex(session_index)
    page.session_step_button.click()

    assert _wait_until(lambda: page._session_record is not None)
    assert page.current_image is not None
    assert page.current_image_path is None
    assert page._session_record.frame_index == 0
    assert page._session_frame_source().endswith("game.avi#frame=0")
    assert page.session_play_button.isEnabled()
    assert page.session_step_button.isEnabled()
    assert page.session_rewind_button.isEnabled()
    assert page.session_forward_button.isEnabled()
    assert page.session_frame_spin.minimumWidth() >= 156
    assert page.session_playback_toolbar.play_button is page.session_play_button
    assert page.session_playback_toolbar.frame_spin is page.session_frame_spin

    page.close()
    app.processEvents()


def test_annotation_template_kind_filters_saved_templates_and_backlinks(tmp_path):
    root = tmp_path / "profiles"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        root / "tencent_daguandan",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService(root))

    effect_index = page.template_kind_combo.findData("effect")
    page.template_kind_combo.setCurrentIndex(effect_index)
    app.processEvents()

    assert page.template_filter_combo.currentData() == "effect"
    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))
    assert all(
        page.template_table.item(row, 0).text() == "牌型特效"
        for row in range(page.template_table.rowCount())
    )
    assert page.template_source_role_combo.currentData() == "generic"

    anchor_index = page.template_filter_combo.findData("anchor")
    page.template_filter_combo.setCurrentIndex(anchor_index)
    app.processEvents()
    assert page.template_kind_combo.currentData() == "anchor"

    page.close()
    app.processEvents()


def test_annotation_page_captures_one_effect_key_frame(tmp_path):
    root = tmp_path / "profiles"
    profile_root = root / "tencent_daguandan"
    shutil.copytree(
        PROFILES_ROOT / "tencent_daguandan",
        profile_root,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("templates", "screenshots", "sessions"),
    )
    session = profile_root / "sessions" / "game_effect_capture"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": session.name,
                "status": "sealed",
                "frame_count": 2,
                "dropped_frames": 0,
            }
        ),
        encoding="utf-8",
    )
    recorder = SessionRecorder(session, size=(64, 32), fps=10)
    for index, value in enumerate((40, 180)):
        recorder.write_frame(
            np.full((32, 64, 3), value, dtype=np.uint8),
            captured_monotonic_ms=(index + 1) * 100,
            wall_time=f"t{index}",
        )
    recorder.close()
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService(root))
    session_index = page.session_combo.findData(str(session.resolve()))
    page.session_combo.setCurrentIndex(session_index)
    page.session_step_button.click()
    assert _wait_until(lambda: page.current_image is not None)

    page.mode_combo.setCurrentIndex(page.mode_combo.findData("template"))
    page.template_kind_combo.setCurrentIndex(page.template_kind_combo.findData("effect"))
    page.template_label_edit.setText("royal_flush")
    page._set_template_roi(Box(12, 6, 28, 20))
    page.crop_template_button.click()

    assert _wait_until(lambda: page.crop_template_button.isEnabled())
    saved = [
        record
        for record in page.template_service.list_templates()
        if record.get("kind") == "effect" and record.get("label") == "royal_flush"
    ]
    assert len(saved) == 1
    assert saved[0]["source_image"] == "sessions/game_effect_capture/video/game.avi#frame=0"
    assert "模板已保存" in page.status.text()

    page.close()
    app.processEvents()


def test_annotation_region_overlay_button_toggles_selected_boxes(tmp_path):
    image_dir = tmp_path / "samples"
    image_dir.mkdir()
    cv2.imwrite(str(image_dir / "sample.png"), np.zeros((720, 1280, 3), dtype=np.uint8))
    app = QApplication.instance() or QApplication([])

    page = AnnotationPage(AnnotationService())
    page.set_image_folder(image_dir)
    page.region_config_button.click()
    config = page.region_config_page
    config.region_table.selectRow(0)
    app.processEvents()

    assert page.overlay_visible is False
    config.preview_selected_button.click()
    assert page.overlay_visible is True
    assert page.show_selected_button.text() == "隐藏选中区域"
    assert config.preview_selected_button.text() == "隐藏选中区域"

    config.preview_selected_button.click()
    assert page.overlay_visible is False
    assert page.show_selected_button.text() == "标注选中区域"
    assert config.preview_selected_button.text() == "标注选中区域"

    config.close()
    page.close()
    app.processEvents()
