from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import uuid

import cv2

from PySide6.QtCore import QPoint, QRect, QThread, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QLineEdit,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRubberBand,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ..annotation_service import (
    REGION_DISPLAY_NAMES,
    ROLE_DISPLAY_NAMES,
    AnnotationService,
    RegionRecord,
    display_region_name,
)
from ..danzero import DanzeroAdvisor
from ..danzero.advisor import format_engine_input_summary
from ..image_io import read_image_unicode
from ..models import Box
from ..recognition_service import RecognitionResult, ScreenshotRecognitionService
from ..template_service import SOURCE_ROLES, TEMPLATE_KINDS, TemplateService
from .region_config_page import RegionConfigPage
from .single_image_danzero_page import SingleImageDanzeroPage
from .workers import OneShotThread

TEMPLATE_KIND_LABELS = {
    "rank": "点数",
    "suit": "花色",
    "anchor": "锚点",
    "button": "按钮",
    "status": "状态",
    "effect": "牌型特效",
    "timer": "倒计时",
}
SOURCE_ROLE_LABELS = {
    "hand": "我的手牌",
    "hand_partial": "部分手牌",
    "play": "出牌区域",
    "generic": "通用样本",
    "level": "级牌 / 逢人配",
}
# 裁剪 rank/suit 模板时，按当前选中的区域名自动推断来源角色，
# 避免手牌/出牌模板分类错误导致识别不到。
_REGION_ROLE_HINTS = {
    "my_hand": "hand",
    "my_play": "play",
    "left_play": "play",
    "right_play": "play",
    "opposite_play": "play",
    "level_rank": "level",
}
TEMPLATE_LABEL_LABELS = {
    "super_double": "超级加倍",
    "double": "加倍",
    "cannot_beat": "要不起",
    "pass": "不出",
    "hint": "提示",
    "play_cards": "出牌",
    "first_play": "首出标记",
    "passed": "过牌标记",
    "active": "行动中",
    "arrange": "整理手牌",
    "quick_arrange": "一键整理",
    "chat": "聊天",
    "more": "更多",
    "rules": "规则",
    "change_table": "换桌",
    "continue_game": "继续游戏",
    "game_logo_anchor": "牌桌标志锚点",
    "table_anchor_1": "牌桌锚点一",
    "table_anchor_2": "牌桌锚点二",
    "bomb": "炸弹特效",
    "triple_pair": "三连对特效",
    "consecutive_pairs": "连对特效",
    "small_joker": "小王",
    "big_joker": "大王",
    "spade": "黑桃",
    "heart": "红桃",
    "club": "梅花",
    "diamond": "方块",
}
TEMPLATE_LABEL_OPTIONS_BY_KIND = {
    "rank": (
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
        "small_joker", "big_joker",
    ),
    "suit": ("spade", "heart", "club", "diamond"),
    "button": (
        "super_double", "double", "cannot_beat", "pass", "hint", "play_cards",
        "arrange", "quick_arrange", "chat", "more", "rules", "change_table",
        "continue_game",
    ),
    "status": ("first_play", "passed", "active"),
    "effect": ("bomb", "triple_pair", "consecutive_pairs"),
    "timer": ("active",),
    "anchor": ("game_logo_anchor", "table_anchor_1", "table_anchor_2"),
}
TEMPLATE_TABLE_HEADERS = ("模板类型", "模板标签", "来源角色", "模板文件")
REGION_COORDINATE_HEADERS = ("区域", "x", "y", "w", "h")


DANZERO_TIMING_LABELS = {
    "build_engine_state": "参数构建",
    "create_agent": "模型初始化",
    "prime_danzero": "状态准备",
    "agent_step": "模型推理",
    "validate_action": "动作校验",
}


def _format_danzero_timings(timings: object) -> str:
    if not isinstance(timings, dict) or not timings:
        return "暂无分段计时"
    parts: list[str] = []
    for key, value in timings.items():
        try:
            milliseconds = float(value)
        except (TypeError, ValueError):
            continue
        label = DANZERO_TIMING_LABELS.get(str(key), str(key))
        parts.append(f"{label} {milliseconds:.2f} ms")
    return "；".join(parts) or "暂无分段计时"


class EditableTemplateLabelComboBox(QComboBox):
    """Editable history combo with the old line-edit ``setText`` API."""

    def setText(self, value: str) -> None:
        index = self.findData(str(value))
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            self.setEditText(str(value))

    def wheelEvent(self, event) -> None:
        event.ignore()


class ScrollSafeComboBox(QComboBox):
    """Prevent the page scroll gesture from changing template metadata."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class RoiCanvas(QLabel):
    roi_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_start: QPoint | None = None
        self._image_size: tuple[int, int] | None = None
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rubber_band.hide()
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_source_size(self, image) -> None:
        self._image_size = (int(image.shape[1]), int(image.shape[0]))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        point = event.position().toPoint()
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.pixmap() is not None
            and self._pixmap_rect().contains(point)
        ):
            self._drag_start = point
            self._rubber_band.setGeometry(QRect(point, point))
            self._rubber_band.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None:
            current = self._clamp_to_pixmap(event.position().toPoint())
            self._rubber_band.setGeometry(
                QRect(self._drag_start, current).normalized().intersected(self._pixmap_rect())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            start = self._to_image_point(self._drag_start)
            end = self._to_image_point(event.position().toPoint())
            self._drag_start = None
            self._rubber_band.hide()
            if start and end:
                x, right = sorted((start[0], end[0]))
                y, bottom = sorted((start[1], end[1]))
                if right - x >= 5 and bottom - y >= 5:
                    self.roi_changed.emit(Box(x, y, right - x, bottom - y))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _pixmap_rect(self) -> QRect:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.width() <= 0 or pixmap.height() <= 0:
            return QRect()
        left = (self.width() - pixmap.width()) // 2
        top = (self.height() - pixmap.height()) // 2
        return QRect(left, top, pixmap.width(), pixmap.height())

    def _clamp_to_pixmap(self, point: QPoint) -> QPoint:
        rect = self._pixmap_rect()
        if rect.isNull():
            return point
        return QPoint(
            max(rect.left(), min(rect.right(), point.x())),
            max(rect.top(), min(rect.bottom(), point.y())),
        )

    def _to_image_point(self, point: QPoint) -> tuple[int, int] | None:
        pixmap = self.pixmap()
        if pixmap is None or self._image_size is None or pixmap.width() <= 0 or pixmap.height() <= 0:
            return None
        left = (self.width() - pixmap.width()) // 2
        top = (self.height() - pixmap.height()) // 2
        x = round((point.x() - left) * self._image_size[0] / pixmap.width())
        y = round((point.y() - top) * self._image_size[1] / pixmap.height())
        return max(0, min(self._image_size[0] - 1, x)), max(0, min(self._image_size[1] - 1, y))


class AnnotationPage(QWidget):
    """Edit and overlay configured regions on recorded screenshots only."""

    HEADERS = ("名称", "角色", "绝对坐标", "比例坐标")

    def __init__(self, service: AnnotationService | None = None):
        super().__init__()
        self.service = service or AnnotationService()
        self.regions = list(self.service.list_regions())
        self.preview_regions: tuple[RegionRecord, ...] = ()
        self.image_paths: tuple[Path, ...] = ()
        self.current_image_path: Path | None = None
        self.current_image = None
        self.current_roi: Box | None = None
        self.region_config_page: RegionConfigPage | None = None
        self.single_image_danzero_page: SingleImageDanzeroPage | None = None
        self.danzero_advisor = DanzeroAdvisor()
        self._crop_thread: QThread | None = None
        self._recognition_thread: QThread | None = None
        self._danzero_warmup_thread: QThread | None = None
        self._danzero_warmup_elapsed_ms: float | None = None
        self._recognition_request_id = 0
        self._recognition_restart_pending = False
        self._closing = False
        self._danzero_thread: QThread | None = None
        self.template_service = TemplateService(self.service.profiles_root)
        self.recognition_service = ScreenshotRecognitionService(
            self.service,
            self.template_service,
        )
        self.setObjectName("annotationPage")
        self._build_ui()
        self._refresh_images()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignmentFlag.AlignTop)
        header = QHBoxLayout()
        header.addWidget(QLabel("区域标注（仅可选择 screenshots 目录内图片）"))
        header.addStretch(1)
        self.region_config_button = QPushButton("查看区域配置")
        self.region_config_button.clicked.connect(self._open_region_config)
        header.addWidget(self.region_config_button)
        self.show_selected_button = QPushButton("标注选中区域")
        self.show_selected_button.clicked.connect(self._show_selected_regions)
        header.addWidget(self.show_selected_button)
        self.single_image_test_button = QPushButton("单图标注 / 测试 DanZero")
        self.single_image_test_button.setEnabled(False)
        self.single_image_test_button.clicked.connect(self._open_single_image_danzero)
        header.addWidget(self.single_image_test_button)
        root.addLayout(header)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("区域配置", "region")
        self.mode_combo.addItem("模板裁剪", "template")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        root.addWidget(self.mode_combo)

        self.image_navigation_layout = QHBoxLayout()
        self.previous_image_button = QToolButton()
        self.previous_image_button.setText("◀")
        self.previous_image_button.setToolTip("上一张图片")
        self.previous_image_button.setAccessibleName("上一张图片")
        self.previous_image_button.clicked.connect(self._select_previous_image)
        self.next_image_button = QToolButton()
        self.next_image_button.setText("▶")
        self.next_image_button.setToolTip("下一张图片")
        self.next_image_button.setAccessibleName("下一张图片")
        self.next_image_button.clicked.connect(self._select_next_image)
        self.canvas = RoiCanvas()
        self.canvas.setText("请选择录制截图")
        self.canvas.setMinimumSize(520, 300)
        self.canvas.setMaximumHeight(405)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.canvas.roi_changed.connect(self._set_roi)
        self.image_navigation_layout.addWidget(self.previous_image_button)
        self.image_navigation_layout.addWidget(self.canvas, 1)
        self.image_navigation_layout.addWidget(self.next_image_button)
        root.addLayout(self.image_navigation_layout)

        image_selector = QHBoxLayout()
        image_selector.addWidget(QLabel("当前图片"))
        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self._image_changed)
        image_selector.addWidget(self.image_combo, 1)
        root.addLayout(image_selector)

        self.status = QLabel("区域：—")
        self.status.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self.status.setMaximumHeight(48)
        root.addWidget(self.status)

        right = QVBoxLayout()
        self.mode_detail_stack = QStackedWidget()

        self.region_detail_widget = QWidget()
        region_detail_layout = QVBoxLayout(self.region_detail_widget)
        region_detail_layout.setContentsMargins(0, 0, 0, 0)
        region_detail_layout.addWidget(QLabel("区域标注：选择区域名称，然后在图片上框选并保存"))
        region_form = QFormLayout()
        self.region_name_combo = ScrollSafeComboBox()
        for name, label in REGION_DISPLAY_NAMES.items():
            self.region_name_combo.addItem(label, name)
        self.region_role_combo = ScrollSafeComboBox()
        for role, label in ROLE_DISPLAY_NAMES.items():
            self.region_role_combo.addItem(label, role)
        self.region_role_combo.setVisible(False)
        region_form.addRow("区域名称", self.region_name_combo)
        region_detail_layout.addLayout(region_form)
        region_detail_layout.addWidget(QLabel("已选区域坐标"))
        self.region_coordinate_table = QTableWidget(
            0, len(REGION_COORDINATE_HEADERS)
        )
        self.region_coordinate_table.setHorizontalHeaderLabels(
            REGION_COORDINATE_HEADERS
        )
        self.region_coordinate_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.region_coordinate_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.region_coordinate_table.setMinimumHeight(92)
        self.region_coordinate_table.setMaximumHeight(140)
        self.region_coordinate_table.horizontalHeader().setStretchLastSection(True)
        region_detail_layout.addWidget(self.region_coordinate_table)
        self.save_region_button = QPushButton("保存区域坐标")
        self.save_region_button.clicked.connect(self._save_region_annotation)
        region_detail_layout.addWidget(self.save_region_button)
        region_detail_layout.addWidget(
            QLabel("请点击“查看区域配置”选择区域；拖动图片框选后坐标会同步更新。")
        )
        self.region_name_combo.currentIndexChanged.connect(self._region_name_changed)

        self.template_detail_widget = QWidget()
        template_detail_layout = QVBoxLayout(self.template_detail_widget)
        template_detail_layout.setContentsMargins(0, 0, 0, 0)

        self.x_spin, self.y_spin, self.w_spin, self.h_spin = (QSpinBox() for _ in range(4))
        for spin, maximum in (
            (self.x_spin, 1279),
            (self.y_spin, 719),
            (self.w_spin, 1280),
            (self.h_spin, 720),
        ):
            spin.setRange(0, maximum)

        grid = QGridLayout()
        for column, (label, spin) in enumerate(
            (("x", self.x_spin), ("y", self.y_spin), ("w", self.w_spin), ("h", self.h_spin))
        ):
            grid.addWidget(QLabel(label), 0, column)
            grid.addWidget(spin, 1, column)
        template_detail_layout.addWidget(QLabel("模板裁剪坐标"))
        template_detail_layout.addLayout(grid)

        template_form = QFormLayout()
        self.template_kind_combo = ScrollSafeComboBox()
        for kind in sorted(TEMPLATE_KINDS):
            self.template_kind_combo.addItem(TEMPLATE_KIND_LABELS.get(kind, kind), kind)
        self.template_kind_combo.currentIndexChanged.connect(
            self._template_kind_changed
        )
        self.template_label_edit = EditableTemplateLabelComboBox()
        self.template_label_edit.setEditable(True)
        self.template_label_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.template_source_role_combo = ScrollSafeComboBox()
        for role in sorted(SOURCE_ROLES):
            self.template_source_role_combo.addItem(SOURCE_ROLE_LABELS.get(role, role), role)
        template_form.addRow("模板类型", self.template_kind_combo)
        template_form.addRow("模板标签", self.template_label_edit)
        template_form.addRow("来源角色", self.template_source_role_combo)
        template_detail_layout.addLayout(template_form)
        self.crop_template_button = QPushButton("裁剪并保存模板")
        self.delete_template_button = QPushButton("删除选中模板")
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("筛选角色"))
        self.template_filter_combo = ScrollSafeComboBox()
        self.template_filter_combo.addItem("全部角色", "")
        for role in sorted(SOURCE_ROLES):
            self.template_filter_combo.addItem(
                SOURCE_ROLE_LABELS.get(role, role), role
            )
        filter_row.addWidget(self.template_filter_combo)
        self.template_search_edit = QLineEdit()
        self.template_search_edit.setPlaceholderText(
            "搜索模板（标签/类型/角色/文件名，支持中文）"
        )
        self.template_search_edit.setClearButtonEnabled(True)
        filter_row.addWidget(self.template_search_edit, 1)
        self.template_count_label = QLabel("")
        filter_row.addWidget(self.template_count_label)
        self.template_table = QTableWidget(0, len(TEMPLATE_TABLE_HEADERS))
        self.template_table.setHorizontalHeaderLabels(TEMPLATE_TABLE_HEADERS)
        self.template_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.template_table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection
        )
        self.template_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.template_table.setMinimumHeight(170)
        self.template_table.setWordWrap(False)
        for column, width in enumerate((90, 130, 110, 190)):
            self.template_table.setColumnWidth(column, width)
        self.template_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        # Keep the old attribute as a compatibility alias for integrations that
        # only use it to read or select saved templates.
        self.template_list = self.template_table
        template_detail_layout.addWidget(self.crop_template_button)
        template_detail_layout.addWidget(self.delete_template_button)
        template_detail_layout.addLayout(filter_row)
        template_detail_layout.addWidget(self.template_table)
        self.template_filter_combo.currentIndexChanged.connect(
            self._refresh_template_status
        )
        self.template_search_edit.textChanged.connect(
            self._refresh_template_status
        )

        self.mode_detail_stack.addWidget(self.region_detail_widget)
        self.mode_detail_stack.addWidget(self.template_detail_widget)
        self.mode_detail_stack.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        right.addWidget(self.mode_detail_stack)

        self.crop_template_button.clicked.connect(self._crop_template)
        self.delete_template_button.clicked.connect(self._delete_selected_templates)
        root.addLayout(right)
        self._mode_changed()
        self._refresh_template_status()

    def _mode_changed(self, *_args) -> None:
        is_template = self.mode_combo.currentData() == "template"
        self.mode_detail_stack.setCurrentIndex(1 if is_template else 0)
        if is_template:
            self.mode_detail_stack.setMinimumHeight(0)
            self.mode_detail_stack.setMaximumHeight(16777215)
            self.mode_detail_stack.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Preferred,
            )
        else:
            self.mode_detail_stack.setFixedHeight(280)
        self.show_selected_button.setVisible(not is_template)
        self._refresh_preview()

    def _open_region_config(self) -> None:
        if self.region_config_page is None:
            self.region_config_page = RegionConfigPage(self.service, self.regions, self)
            self.region_config_page.region_selected.connect(self._region_selected)
            self.region_config_page.preview_requested.connect(self._preview_selected_regions)
            self.region_config_page.region_updated.connect(self._region_updated)
        self.region_config_page.show()
        self.region_config_page.raise_()
        self.region_config_page.activateWindow()

    def _set_region_editor_values(self, region: RegionRecord) -> None:
        self.region_name_combo.blockSignals(True)
        self.region_role_combo.blockSignals(True)
        self.region_name_combo.setCurrentIndex(
            max(0, self.region_name_combo.findData(region.name))
        )
        self.region_role_combo.setCurrentIndex(
            max(0, self.region_role_combo.findData(region.role))
        )
        self.region_name_combo.blockSignals(False)
        self.region_role_combo.blockSignals(False)

    def _region_name_changed(self, *_args) -> None:
        name = str(self.region_name_combo.currentData() or "")
        region = next((item for item in self.regions if item.name == name), None)
        if region is None:
            return
        self.region_role_combo.blockSignals(True)
        self.region_role_combo.setCurrentIndex(
            max(0, self.region_role_combo.findData(region.role))
        )
        self.region_role_combo.blockSignals(False)
        self.preview_regions = (region,)
        self.current_roi = region.abs_box
        self._refresh_region_coordinate_table()
        self._refresh_preview()

    def _save_region_annotation(self) -> None:
        name = str(self.region_name_combo.currentData() or "")
        region = next((item for item in self.regions if item.name == name), None)
        role = region.role if region is not None else "generic"
        box = self.current_roi
        if not name:
            self.status.setText("请先选择区域名称")
            return
        if box is None:
            self.status.setText("请先在图片上拖动框选区域")
            return
        try:
            updated = self.service.update_region(
                name,
                name=name,
                role=role,
                box=box,
            )
        except Exception as exc:
            self.status.setText(f"保存区域失败：{exc}")
            return
        self.regions = list(self.service.list_regions())
        self._set_region_editor_values(updated)
        self.preview_regions = (updated,)
        self.current_roi = updated.abs_box
        self._refresh_region_coordinate_table()
        self._refresh_preview()
        if self.region_config_page is not None:
            self.region_config_page.regions = list(self.regions)
            self.region_config_page._refresh_regions()
            row = next(
                (index for index, item in enumerate(self.regions) if item.name == updated.name),
                -1,
            )
            if row >= 0:
                self.region_config_page.region_table.selectRow(row)
        self.status.setText(f"已保存区域：{display_region_name(updated.name)}")

    def _region_selected(self, region: RegionRecord) -> None:
        self._set_region_editor_values(region)
        self.preview_regions = (region,)
        self.current_roi = region.abs_box
        self._refresh_region_coordinate_table()
        self._refresh_preview()

    def _preview_selected_regions(self, regions: object) -> None:
        selected = tuple(regions) if regions else ()
        self.preview_regions = selected
        self.current_roi = selected[0].abs_box if selected else None
        if selected:
            self._set_region_editor_values(selected[0])
        self._refresh_region_coordinate_table()
        self._refresh_preview()
        if not selected:
            self.status.setText("请先选择一个或多个区域")

    def _region_updated(self, old_name: str, updated: RegionRecord) -> None:
        self.regions = list(self.service.list_regions())
        self.preview_regions = tuple(
            updated if region.name == old_name else region
            for region in self.preview_regions
        )
        if self.preview_regions and self.preview_regions[0].name == updated.name:
            self._set_region_editor_values(updated)
            self.current_roi = updated.abs_box
        self._refresh_region_coordinate_table()
        self._refresh_preview()

    def _refresh_region_coordinate_table(self) -> None:
        self.region_coordinate_table.setRowCount(len(self.preview_regions))
        for row, region in enumerate(self.preview_regions):
            box = self.current_roi if row == 0 and self.current_roi is not None else region.abs_box
            values = (region.name, box.x, box.y, box.w, box.h)
            for column, value in enumerate(values):
                item = QTableWidgetItem(
                    str(value) if column else display_region_name(region.name)
                )
                self.region_coordinate_table.setItem(row, column, item)
        self.region_coordinate_table.resizeColumnsToContents()

    def _set_roi(self, box: Box) -> None:
        self.current_roi = box
        if self.mode_combo.currentData() == "template":
            self.x_spin.setValue(box.x)
            self.y_spin.setValue(box.y)
            self.w_spin.setValue(box.w)
            self.h_spin.setValue(box.h)
            self.status.setText(f"已选模板框：x={box.x}, y={box.y}, w={box.w}, h={box.h}")
        else:
            if self.region_config_page is not None:
                self.region_config_page.set_box(box)
            self._refresh_region_coordinate_table()
            self.status.setText(f"已选区域框：x={box.x}, y={box.y}, w={box.w}, h={box.h}")
        self._refresh_preview()

    def _set_template_roi(self, box: Box) -> None:
        """兼容旧调用方，统一转入 ROI 更新路径。"""
        self._set_roi(box)

    def _template_matches(
        self,
        record: dict[str, object],
        role_filter: str,
        keyword: str,
    ) -> bool:
        if role_filter and str(record.get("source_role", "")) != role_filter:
            return False
        if not keyword:
            return True
        kind = str(record.get("kind", ""))
        label = str(record.get("label", ""))
        role = str(record.get("source_role", ""))
        filename = Path(str(record.get("file", ""))).name
        haystack = " ".join(
            (
                kind,
                TEMPLATE_KIND_LABELS.get(kind, ""),
                label,
                TEMPLATE_LABEL_LABELS.get(label, ""),
                role,
                SOURCE_ROLE_LABELS.get(role, ""),
                filename,
            )
        ).lower()
        return keyword.lower() in haystack

    def _refresh_template_status(self, *_args) -> None:
        records = self.template_service.list_templates()
        self._refresh_template_labels(records)
        selected_ids = self._selected_template_ids()
        role_filter = str(self.template_filter_combo.currentData() or "")
        keyword = self.template_search_edit.text().strip()
        filtered = [
            record
            for record in records
            if self._template_matches(record, role_filter, keyword)
        ]
        self.template_table.setRowCount(0)
        self.template_table.setToolTip(
            f"已保存模板：{len(records)} 条（显示 {len(filtered)} 条）"
        )
        self.template_table.setRowCount(len(filtered))
        for row, record in enumerate(filtered):
            sample_id = str(record.get("sample_id", ""))
            kind = str(record.get("kind", ""))
            label = str(record.get("label", ""))
            source_role = str(record.get("source_role", ""))
            relative = str(record.get("file", ""))
            values = (
                TEMPLATE_KIND_LABELS.get(kind, kind),
                TEMPLATE_LABEL_LABELS.get(label, label),
                SOURCE_ROLE_LABELS.get(source_role, source_role),
                Path(relative).name,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, sample_id)
                    item.setToolTip(sample_id)
                if column == 1:
                    item.setToolTip(f"{label}（{sample_id}）")
                if column == 3:
                    item.setToolTip(relative)
                self.template_table.setItem(row, column, item)
            if sample_id in selected_ids:
                self.template_table.selectRow(row)
        total = len(records)
        if len(filtered) == total:
            self.template_count_label.setText(f"共 {total} 条")
        else:
            self.template_count_label.setText(f"显示 {len(filtered)} / {total} 条")

    def _selected_template_ids(self) -> set[str]:
        ids: set[str] = set()
        for model_index in self.template_table.selectionModel().selectedRows():
            item = self.template_table.item(model_index.row(), 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole):
                ids.add(str(item.data(Qt.ItemDataRole.UserRole)))
        return ids

    def _refresh_template_labels(self, records=()) -> None:
        current = self._template_label_value()
        kind = str(self.template_kind_combo.currentData() or "")
        labels: list[str] = list(TEMPLATE_LABEL_OPTIONS_BY_KIND.get(kind, ()))
        for record in records:
            if str(record.get("kind", "")) != kind:
                continue
            label = str(record.get("label", "")).strip()
            if label and label not in labels:
                labels.append(label)
        self.template_label_edit.blockSignals(True)
        self.template_label_edit.clear()
        for label in labels:
            self.template_label_edit.addItem(
                TEMPLATE_LABEL_LABELS.get(label, label),
                label,
            )
        if current:
            index = self.template_label_edit.findData(current)
            if index >= 0:
                self.template_label_edit.setCurrentIndex(index)
            else:
                self.template_label_edit.setCurrentIndex(-1)
                self.template_label_edit.setEditText("")
        else:
            self.template_label_edit.setCurrentIndex(-1)
            self.template_label_edit.setEditText("")
        self.template_label_edit.blockSignals(False)

    def _template_kind_changed(self, *_args) -> None:
        self._refresh_template_labels(self.template_service.list_templates())

    def _template_label_value(self) -> str:
        data = self.template_label_edit.currentData()
        if data is not None and self.template_label_edit.currentIndex() >= 0:
            return str(data).strip()
        return self.template_label_edit.currentText().strip()

    def _refresh_images(self) -> None:
        self.image_paths = self.service.list_recorded_images()
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for path in self.image_paths:
            self.image_combo.addItem(
                str(path.relative_to(self.service.screenshots_root)),
                str(path),
            )
        self.image_combo.blockSignals(False)
        if self.image_paths:
            self._image_changed(0)
        else:
            self._image_changed(-1)

    def _update_image_navigation(self) -> None:
        index = self.image_combo.currentIndex()
        count = self.image_combo.count()
        self.previous_image_button.setEnabled(index > 0)
        self.next_image_button.setEnabled(0 <= index < count - 1)

    def _select_previous_image(self) -> None:
        index = self.image_combo.currentIndex()
        if index > 0:
            self.image_combo.setCurrentIndex(index - 1)

    def _select_next_image(self) -> None:
        index = self.image_combo.currentIndex()
        if index < self.image_combo.count() - 1:
            self.image_combo.setCurrentIndex(index + 1)

    def _image_changed(self, index: int) -> None:
        self._recognition_request_id += 1
        if self._recognition_thread is not None and self._recognition_thread.isRunning():
            self._recognition_restart_pending = True
        if index < 0 or index >= len(self.image_paths):
            self.current_image_path = None
            self.current_image = None
            self.current_roi = None
            self.single_image_test_button.setEnabled(False)
            if self.single_image_danzero_page is not None:
                self.single_image_danzero_page.set_image_path(None)
            self._update_image_navigation()
            self._refresh_preview()
            return

        self.current_image_path = self.image_paths[index]
        try:
            self.current_image = read_image_unicode(self.current_image_path)
        except Exception as exc:
            self.current_image = None
            self.single_image_test_button.setEnabled(False)
            self.canvas.clear()
            if self.single_image_danzero_page is not None:
                self.single_image_danzero_page.set_image_path(None)
            self.canvas.setText(f"图片读取失败：{exc}")
            self.status.setText("图片读取失败")
            self._update_image_navigation()
            return
        self.canvas.set_source_size(self.current_image)
        self.current_roi = None
        self.single_image_test_button.setEnabled(True)
        self._update_image_navigation()
        self._refresh_preview()

        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.set_image_path(self.current_image_path)
            self._start_template_recognition()

    def _pixmap_for_image(self, image) -> QPixmap:
        height, width, channels = image.shape
        qimage = QImage(
            image.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_BGR888,
        ).copy()
        return QPixmap.fromImage(qimage).scaled(
            self.canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _refresh_preview(self) -> None:
        if self.current_image is None:
            self.canvas.clear()
            self.canvas.setText("暂无可预览的录制图片")
            return

        regions = self.preview_regions if self.mode_combo.currentData() == "region" else ()
        image = (
            self.service.overlay_regions(self.current_image, regions)
            if regions
            else self.current_image.copy()
        )
        if self.current_roi is not None and self.current_roi.fits_within((image.shape[1], image.shape[0])):
            box = self.current_roi
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.w, box.y + box.h),
                (0, 165, 255),
                3,
            )
        self.canvas.setPixmap(self._pixmap_for_image(image))
        if regions:
            self.status.setText(f"已标注 {len(regions)} 个区域")
        elif self.current_roi is not None:
            box = self.current_roi
            if self.mode_combo.currentData() == "template":
                self.status.setText(
                    f"已选模板框：x={box.x}, y={box.y}, w={box.w}, h={box.h}"
                )
            else:
                self.status.setText(
                    f"已选区域框：x={box.x}, y={box.y}, w={box.w}, h={box.h}"
                )
        else:
            self.status.clear()

    def _show_selected_regions(self) -> None:
        if self.current_image is None:
            self.status.setText("请先从 screenshots 目录选择图片")
            return
        regions = self.preview_regions
        if not regions:
            self.status.setText("请先选择一个或多个区域")
            return
        self._refresh_preview()

    def _open_single_image_danzero(self) -> None:
        if self.current_image_path is None or self.current_image is None:
            self.status.setText("请先选择一张截图")
            return
        if self.single_image_danzero_page is None:
            self.single_image_danzero_page = SingleImageDanzeroPage(
                self.current_image_path,
                self,
            )
            self.single_image_danzero_page.test_requested.connect(self._run_danzero_test)
            self.single_image_danzero_page.recognize_requested.connect(
                self._start_template_recognition
            )
        else:
            self.single_image_danzero_page.set_image_path(self.current_image_path)
        self.single_image_danzero_page.show()
        self.single_image_danzero_page.raise_()
        self.single_image_danzero_page.activateWindow()
        self._start_danzero_warmup()
        self._start_template_recognition()

    def _start_danzero_warmup(self) -> None:
        if self._danzero_warmup_thread is not None and self._danzero_warmup_thread.isRunning():
            return
        initializer = getattr(self.danzero_advisor, "initialize", None)
        if not callable(initializer):
            return
        def operation() -> float:
            started = perf_counter()
            initializer()
            return (perf_counter() - started) * 1000

        thread = OneShotThread(operation, self)
        thread.result.connect(self._danzero_warmup_succeeded)
        thread.error.connect(self._danzero_warmup_failed)
        thread.finished.connect(
            lambda: setattr(self, "_danzero_warmup_thread", None)
        )
        self._danzero_warmup_thread = thread
        thread.start()

    def _danzero_warmup_succeeded(self, elapsed_ms: float) -> None:
        self._danzero_warmup_elapsed_ms = float(elapsed_ms)

    def _danzero_warmup_failed(self, message: str) -> None:
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.status.setText(
                f"DanZero 模型预加载失败，测试时会重试：{message}"
            )

    def _start_template_recognition(self) -> None:
        if self._recognition_thread is not None and self._recognition_thread.isRunning():
            return
        if (
            self.single_image_danzero_page is None
            or self.current_image is None
        ):
            return
        self.single_image_danzero_page.set_recognition_busy(True)
        image = self.current_image.copy()
        image_path = self.current_image_path
        request_id = self._recognition_request_id
        self._recognition_restart_pending = False
        thread = OneShotThread(
            lambda: self.recognition_service.recognize(image),
            self,
        )
        thread.result.connect(
            lambda result, request_id=request_id, image_path=image_path: self._template_recognition_succeeded(
                result,
                request_id=request_id,
                image_path=image_path,
            )
        )
        thread.error.connect(
            lambda message, request_id=request_id, image_path=image_path: self._template_recognition_failed(
                message,
                request_id=request_id,
                image_path=image_path,
            )
        )
        thread.finished.connect(
            lambda request_id=request_id, image_path=image_path: self._template_recognition_finished(
                request_id=request_id,
                image_path=image_path,
            )
        )
        self._recognition_thread = thread
        thread.start()

    def _template_recognition_succeeded(
        self,
        result: RecognitionResult,
        *,
        request_id: int | None = None,
        image_path: Path | None = None,
    ) -> None:
        if request_id is not None and request_id != self._recognition_request_id:
            return
        if image_path is not None and image_path != self.current_image_path:
            return
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.apply_recognition(result)

    def _template_recognition_failed(
        self,
        message: str,
        *,
        request_id: int | None = None,
        image_path: Path | None = None,
    ) -> None:
        if request_id is not None and request_id != self._recognition_request_id:
            return
        if image_path is not None and image_path != self.current_image_path:
            return
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.show_recognition_error(message)

    def _template_recognition_finished(
        self,
        *,
        request_id: int | None = None,
        image_path: Path | None = None,
    ) -> None:
        if self.single_image_danzero_page is not None:
            if request_id is None or request_id == self._recognition_request_id:
                self.single_image_danzero_page.set_recognition_busy(False)
        self._recognition_thread = None
        stale = request_id is not None and request_id != self._recognition_request_id
        if (
            stale
            and self._recognition_restart_pending
            and not self._closing
            and self.current_image is not None
            and self.current_image_path is not None
        ):
            self._recognition_restart_pending = False
            self._start_template_recognition()

    def _run_danzero_test(self, state: object) -> None:
        if self._danzero_thread is not None and self._danzero_thread.isRunning():
            if self.single_image_danzero_page is not None:
                self.single_image_danzero_page.status.setText("正在调用 DanZero……")
            return
        if self.single_image_danzero_page is None:
            return
        request_id = f"single-image-{uuid.uuid4().hex[:12]}"
        operation = lambda: self.danzero_advisor.recommend(
            state,
            request_id=request_id,
        )
        self.single_image_danzero_page.set_test_busy(True)
        thread = OneShotThread(operation, self)
        thread.result.connect(self._danzero_succeeded)
        thread.error.connect(self._danzero_failed)
        thread.finished.connect(self._danzero_finished)
        self._danzero_thread = thread
        thread.start()

    def _danzero_succeeded(self, advice) -> None:
        cards = "、".join(str(card) for card in advice.cards) or "不出"
        engine_input = advice.engine_input
        timings = _format_danzero_timings(getattr(advice, "timings", {}))
        recognition_elapsed = getattr(
            self.single_image_danzero_page,
            "recognition_elapsed_ms",
            None,
        )
        recognition_line = (
            f"图片识别耗时：{float(recognition_elapsed):.2f} ms"
            if recognition_elapsed is not None
            else "图片识别耗时：暂无"
        )
        warmup_elapsed = getattr(
            self,
            "_danzero_warmup_elapsed_ms",
            None,
        )
        warmup_line = (
            f"后台模型初始化耗时：{float(warmup_elapsed):.2f} ms"
            if warmup_elapsed is not None
            else "后台模型初始化耗时：未预加载"
        )
        result = "\n".join(
            (
                f"推荐动作：{cards}",
                f"牌型：{advice.play_type}",
                f"是否不出：{'是' if advice.is_pass else '否'}",
                f"耗时：{advice.elapsed_ms:.2f} ms",
                "",
                "参数摘要：",
                format_engine_input_summary(engine_input),
                "",
                "engine_input JSON：",
                json.dumps(engine_input, ensure_ascii=False, indent=2),
            )
        )
        result = (
            recognition_line
            + f"\n{warmup_line}\n耗时分解：{timings}\n\n"
            + result
        )
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.show_test_result(result)

    def _danzero_failed(self, message: str) -> None:
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.show_test_error(f"调用失败：{message}")

    def _danzero_finished(self) -> None:
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.set_test_busy(False)
        self._danzero_thread = None

    def _delete_selected_templates(self) -> None:
        sample_ids = tuple(self._selected_template_ids())
        if not sample_ids:
            self.status.setText("请先在模板列表中选择一个或多个模板")
            return
        try:
            deleted = self.template_service.delete_templates(sample_ids)
        except Exception as exc:
            self.status.setText(f"删除失败：{exc}")
            return
        self.recognition_service.reload_templates()
        self._refresh_template_status()
        self.status.setText(f"已删除 {len(deleted)} 个模板，原始截图未删除")

    def _set_crop_busy(self, busy: bool) -> None:
        for widget in (
            self.mode_combo,
            self.image_combo,
            self.previous_image_button,
            self.next_image_button,
            self.single_image_test_button,
            self.template_kind_combo,
            self.template_label_edit,
            self.template_source_role_combo,
            self.crop_template_button,
            self.delete_template_button,
            self.template_table,
        ):
            widget.setEnabled(not busy)

    def _crop_succeeded(self, sample) -> None:
        self.recognition_service.reload_templates()
        self._refresh_template_status()
        self.status.setText(f"模板已保存：{sample.sample_id}")

    def _crop_failed(self, message: str) -> None:
        self.status.setText(f"裁剪保存失败：{message}")

    def _crop_finished(self) -> None:
        self._set_crop_busy(False)
        self._crop_thread = None

    def _crop_template(self) -> None:
        if self._crop_thread is not None and self._crop_thread.isRunning():
            self.status.setText("正在裁剪模板……")
            return
        if self.current_image is None or self.current_image_path is None:
            self.status.setText("请先从 screenshots 目录选择图片")
            return
        label = self._template_label_value()
        if not label:
            self.status.setText("请先输入模板标签")
            return
        box = self.current_roi or Box(
            self.x_spin.value(),
            self.y_spin.value(),
            self.w_spin.value(),
            self.h_spin.value(),
        )
        source_image = self.current_image_path.relative_to(self.service.screenshots_root).as_posix()
        image = self.current_image.copy()
        kind = str(self.template_kind_combo.currentData())
        source_role = str(self.template_source_role_combo.currentData())
        if source_role == "generic" and kind in ("rank", "suit"):
            source_role = _REGION_ROLE_HINTS.get(
                str(self.region_name_combo.currentData() or ""),
                "generic",
            )
        operation = lambda: self.template_service.save_template(
            image,
            kind=kind,
            label=label,
            box=box,
            source_image=source_image,
            source_role=source_role,
        )

        self._set_crop_busy(True)
        self.status.setText("正在裁剪模板……")
        thread = OneShotThread(operation, self)
        thread.result.connect(self._crop_succeeded)
        thread.error.connect(self._crop_failed)
        thread.finished.connect(self._crop_finished)
        self._crop_thread = thread
        thread.start()

    def closeEvent(self, event) -> None:
        self._closing = True
        self._recognition_request_id += 1
        if self._crop_thread is not None and self._crop_thread.isRunning():
            self._crop_thread.wait(5000)
        if self._recognition_thread is not None and self._recognition_thread.isRunning():
            self._recognition_thread.wait(10000)
        if self._danzero_warmup_thread is not None and self._danzero_warmup_thread.isRunning():
            self._danzero_warmup_thread.wait(10000)
        if self._danzero_thread is not None and self._danzero_thread.isRunning():
            self._danzero_thread.wait(10000)
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.close()
        super().closeEvent(event)
