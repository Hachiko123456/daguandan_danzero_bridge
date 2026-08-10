from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import uuid

import cv2
import numpy as np

from PySide6.QtCore import QPoint, QRect, QThread, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QLineEdit,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRubberBand,
    QStackedWidget,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from qfluentwidgets import (
    CardWidget,
    CaptionLabel,
    PrimaryPushButton,
    PushButton,
    SearchLineEdit,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
    ToolButton,
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
from ..live.replay import FrameIndexRecord, VideoReplaySource
from ..live.session_store import read_json_lines
from ..models import Box
from ..recognition_service import RecognitionResult, ScreenshotRecognitionService
from ..template_service import SOURCE_ROLES, TEMPLATE_KINDS, TemplateService
from .region_config_page import RegionConfigPage
from .single_image_danzero_page import SingleImageDanzeroPage
from .video_playback import ReplayDecodeThread, SessionPlaybackToolbar
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
    "continue_game": "再来一局",
    "game_logo_anchor": "牌桌标志锚点",
    "table_anchor_1": "牌桌锚点一",
    "table_anchor_2": "牌桌锚点二",
    "single": "单张特效",
    "pair": "对子特效",
    "trips": "三张特效",
    "three_with_two": "三带二特效",
    "two_trips": "钢板特效",
    "bomb": "炸弹特效",
    "joker_bomb": "天王炸特效",
    "triple_pair": "三连对特效",
    "consecutive_pairs": "连对特效",
    "straight": "顺子特效",
    "straight_flush": "同花顺特效",
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
    "effect": (
        "single", "pair", "trips", "three_with_two", "two_trips",
        "triple_pair", "consecutive_pairs", "straight", "bomb",
        "straight_flush", "joker_bomb",
    ),
    "timer": ("active",),
    "anchor": ("game_logo_anchor", "table_anchor_1", "table_anchor_2"),
}
TEMPLATE_TABLE_HEADERS = (
    "模板类型",
    "模板标签",
    "来源角色",
    "绝对坐标",
    "比例坐标",
    "文件",
)
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


class _PopupLineEdit(QLineEdit):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class EditableTemplateLabelComboBox(QComboBox):
    """Editable history combo with the old line-edit ``setText`` API."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        line_edit = _PopupLineEdit(self)
        self.setLineEdit(line_edit)
        line_edit.clicked.connect(self.showPopup)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.showPopup()

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
    """A responsive workspace for region annotation and template management."""

    HEADERS = ("名称", "角色", "绝对坐标", "比例坐标")

    def __init__(self, service: AnnotationService | None = None):
        super().__init__()
        self.service = service or AnnotationService()
        self.regions = list(self.service.list_regions())
        self.preview_regions: tuple[RegionRecord, ...] = ()
        self.overlay_visible = False
        self._show_draft_roi = False
        self.image_paths: tuple[Path, ...] = ()
        self.image_folder: Path | None = (
            self.service.screenshots_root
            if self.service.screenshots_root.is_dir()
            else None
        )
        self.sessions_root = self.service.profile_root / "sessions"
        self.current_session: Path | None = None
        self._session_decode_thread: ReplayDecodeThread | None = None
        self._session_record = None
        self._session_start_frame: int | None = None
        self._session_playing = False
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
        # A fresh installation without recordings still supports the legacy
        # programmatic image API.  Normal profiles begin with the focused
        # session workflow and do not preload an unrelated screenshot.
        if not self._has_playable_session():
            self._refresh_images()
        self._refresh_sessions()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 20)
        root.setSpacing(14)

        header_card = CardWidget(self)
        header = QHBoxLayout(header_card)
        header.setContentsMargins(20, 16, 20, 16)
        header.setSpacing(12)
        heading = QVBoxLayout()
        heading.setSpacing(3)
        heading.addWidget(SubtitleLabel("标记与模板"))
        heading.addWidget(CaptionLabel("从已录制对局定位单帧，配置识别区域并裁剪可复用模板。"))
        header.addLayout(heading, 1)
        self.region_config_button = PushButton("查看区域配置")
        self.region_config_button.setToolTip("查看、编辑并多选区域配置")
        self.region_config_button.clicked.connect(self._open_region_config)
        header.addWidget(self.region_config_button)
        self.show_selected_button = PushButton("标注选中区域")
        self.show_selected_button.setToolTip("再次点击可隐藏当前选中区域的标记框")
        self.show_selected_button.clicked.connect(self._show_selected_regions)
        header.addWidget(self.show_selected_button)
        self.single_image_test_button = PrimaryPushButton("单图标注 / 测试 DanZero")
        self.single_image_test_button.setEnabled(False)
        self.single_image_test_button.clicked.connect(self._open_single_image_danzero)
        for control in (
            self.region_config_button,
            self.show_selected_button,
            self.single_image_test_button,
        ):
            control.setMinimumHeight(34)
        header.addWidget(self.single_image_test_button)
        root.addWidget(header_card)

        source_card = CardWidget(self)
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(20, 14, 20, 14)
        source_layout.setSpacing(8)
        source_layout.addWidget(StrongBodyLabel("已录制对局"))
        source_layout.addWidget(
            CaptionLabel("从 tencent_daguandan/sessions 选择对局，播放或定位到单帧后直接框选。")
        )

        self.folder_source_widget = QWidget(self)
        folder_source_layout = QVBoxLayout(self.folder_source_widget)
        folder_source_layout.setContentsMargins(0, 0, 0, 0)
        folder_source_layout.setSpacing(8)
        folder_row = QHBoxLayout()
        self.folder_path_label = CaptionLabel("未选择图片文件夹")
        self.folder_path_label.setWordWrap(True)
        self.folder_path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        folder_row.addWidget(self.folder_path_label, 1)
        self.choose_folder_button = PushButton("选择文件夹")
        self.choose_folder_button.clicked.connect(self._choose_image_folder)
        folder_row.addWidget(self.choose_folder_button)
        self.refresh_images_button = ToolButton()
        self.refresh_images_button.setText("↻")
        self.refresh_images_button.setToolTip("刷新当前文件夹")
        self.refresh_images_button.clicked.connect(self._refresh_images)
        folder_row.addWidget(self.refresh_images_button)
        folder_source_layout.addLayout(folder_row)
        image_selector = QHBoxLayout()
        image_selector.addWidget(CaptionLabel("当前图片"))
        self.image_combo = QComboBox()
        self.image_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.image_combo.currentIndexChanged.connect(self._image_changed)
        image_selector.addWidget(self.image_combo, 1)
        self.image_position_label = CaptionLabel("0 / 0")
        image_selector.addWidget(self.image_position_label)
        folder_source_layout.addLayout(image_selector)
        # Keep the legacy image-folder controls alive for programmatic callers,
        # but do not expose a second source in the focused session workflow.
        self.folder_source_widget.hide()

        session_selector = QHBoxLayout()
        session_selector.addWidget(CaptionLabel("对局"))
        self.session_combo = ScrollSafeComboBox()
        self.session_combo.setToolTip("来自 tencent_daguandan/sessions 的已录制对局")
        self.session_combo.setMinimumHeight(34)
        self.session_combo.currentIndexChanged.connect(self._session_changed)
        session_selector.addWidget(self.session_combo, 1)
        self.refresh_sessions_button = ToolButton()
        self.refresh_sessions_button.setText("↻")
        self.refresh_sessions_button.setToolTip("刷新对局列表")
        self.refresh_sessions_button.setFixedSize(34, 34)
        self.refresh_sessions_button.clicked.connect(self._refresh_sessions)
        session_selector.addWidget(self.refresh_sessions_button)
        source_layout.addLayout(session_selector)

        self.session_playback_toolbar = SessionPlaybackToolbar(self)
        # Stable aliases keep the older page API working while both pages now
        # render and signal through the exact same toolbar component.
        self.session_play_button = self.session_playback_toolbar.play_button
        self.session_step_button = self.session_playback_toolbar.step_button
        self.session_rewind_button = self.session_playback_toolbar.rewind_button
        self.session_forward_button = self.session_playback_toolbar.forward_button
        self.session_frame_spin = self.session_playback_toolbar.frame_spin
        self.session_jump_button = self.session_playback_toolbar.frame_jump_button
        self.session_speed_combo = self.session_playback_toolbar.speed_combo
        self.session_status = CaptionLabel("选择对局后，可播放、逐帧定位并直接框选模板。")
        self.session_status.setWordWrap(True)
        self.session_playback_toolbar.play_pause_requested.connect(
            self._toggle_session_playback
        )
        self.session_playback_toolbar.step_requested.connect(self._step_session_frame)
        self.session_playback_toolbar.seek_requested.connect(
            self._jump_to_session_frame
        )
        self.session_playback_toolbar.seek_seconds_requested.connect(
            self._seek_session_by_seconds
        )
        self.session_playback_toolbar.speed_changed.connect(self._session_speed_changed)
        source_layout.addWidget(self.session_playback_toolbar)
        source_layout.addWidget(self.session_status)
        root.addWidget(source_card)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.content_splitter.setChildrenCollapsible(False)
        self.preview_card = CardWidget(self.content_splitter)
        preview_layout = QVBoxLayout(self.preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(10)
        preview_layout.addWidget(StrongBodyLabel("图片预览"))
        self.image_navigation_layout = QHBoxLayout()
        self.image_navigation_layout.setSpacing(8)
        self.previous_image_button = ToolButton()
        self.previous_image_button.setText("‹")
        self.previous_image_button.setToolTip("上一张图片")
        self.previous_image_button.setAccessibleName("上一张图片")
        self.previous_image_button.setFixedSize(40, 56)
        self.previous_image_button.clicked.connect(self._select_previous_image)
        self.next_image_button = ToolButton()
        self.next_image_button.setText("›")
        self.next_image_button.setToolTip("下一张图片")
        self.next_image_button.setAccessibleName("下一张图片")
        self.next_image_button.setFixedSize(40, 56)
        self.next_image_button.clicked.connect(self._select_next_image)
        self.canvas = RoiCanvas()
        self.canvas.setText("请选择已录制对局并定位到一帧")
        self.canvas.setMinimumSize(400, 260)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.canvas.setStyleSheet(
            "border: 1px dashed rgba(120, 120, 120, 0.45); border-radius: 10px;"
        )
        self.canvas.roi_changed.connect(self._set_roi)
        self.image_navigation_layout.addWidget(self.previous_image_button)
        self.image_navigation_layout.addWidget(self.canvas, 1)
        self.image_navigation_layout.addWidget(self.next_image_button)
        preview_layout.addLayout(self.image_navigation_layout, 1)
        self.status = CaptionLabel("请选择已录制对局，播放或定位到单帧后开始标记。")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(32)
        preview_layout.addWidget(self.status)

        self.workbench_card = CardWidget(self.content_splitter)
        right = QVBoxLayout(self.workbench_card)
        right.setContentsMargins(20, 16, 20, 16)
        right.setSpacing(10)
        mode_row = QHBoxLayout()
        mode_row.addWidget(StrongBodyLabel("工作模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("区域配置", "region")
        self.mode_combo.addItem("模板裁剪", "template")
        self.mode_combo.setMinimumHeight(34)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        right.addLayout(mode_row)
        self.mode_detail_stack = QStackedWidget()

        self.region_detail_widget = QWidget()
        region_detail_layout = QVBoxLayout(self.region_detail_widget)
        region_detail_layout.setContentsMargins(0, 0, 0, 0)
        region_detail_layout.addWidget(CaptionLabel("选择区域后在图片中拖动框选；保存会更新标准化坐标。"))
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
        region_detail_layout.addWidget(StrongBodyLabel("已选区域坐标"))
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
        self.save_region_button = PrimaryPushButton("保存区域坐标")
        self.save_region_button.clicked.connect(self._save_region_annotation)
        region_detail_layout.addWidget(self.save_region_button)
        region_detail_layout.addWidget(CaptionLabel("提示：在“管理区域”中多选后，使用同一个“标注选中区域”按钮显示或隐藏区域框。"))
        self.region_name_combo.currentIndexChanged.connect(self._region_name_changed)

        self.template_detail_widget = QWidget()
        template_detail_layout = QVBoxLayout(self.template_detail_widget)
        template_detail_layout.setContentsMargins(0, 0, 0, 0)

        self.x_spin, self.y_spin, self.w_spin, self.h_spin = (SpinBox() for _ in range(4))
        for spin, maximum in (
            (self.x_spin, 1279),
            (self.y_spin, 719),
            (self.w_spin, 1280),
            (self.h_spin, 720),
        ):
            spin.setRange(0, maximum)
            spin.setMinimumHeight(34)

        grid = QGridLayout()
        for column, (label, spin) in enumerate(
            (("x", self.x_spin), ("y", self.y_spin), ("w", self.w_spin), ("h", self.h_spin))
        ):
            grid.addWidget(QLabel(label), 0, column)
            grid.addWidget(spin, 1, column)
        template_detail_layout.addWidget(StrongBodyLabel("模板裁剪坐标"))
        template_detail_layout.addLayout(grid)

        template_form = QFormLayout()
        self.template_kind_combo = ScrollSafeComboBox()
        for kind in sorted(TEMPLATE_KINDS):
            self.template_kind_combo.addItem(TEMPLATE_KIND_LABELS.get(kind, kind), kind)
        self.template_kind_combo.currentIndexChanged.connect(
            self._template_kind_changed
        )
        self.template_label_edit = EditableTemplateLabelComboBox()
        self.template_label_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.template_label_edit.setToolTip("点击输入框空白处即可展开标签；也可直接输入新标签")
        self.template_source_role_combo = ScrollSafeComboBox()
        for role in sorted(SOURCE_ROLES):
            self.template_source_role_combo.addItem(SOURCE_ROLE_LABELS.get(role, role), role)
        template_form.addRow("模板类型", self.template_kind_combo)
        template_form.addRow("模板标签", self.template_label_edit)
        template_form.addRow("来源角色", self.template_source_role_combo)
        template_detail_layout.addLayout(template_form)
        self.crop_template_button = PrimaryPushButton("裁剪并保存模板")
        self.delete_template_button = PushButton("删除选中模板")
        filter_row = QHBoxLayout()
        filter_row.addWidget(CaptionLabel("模板类型"))
        self.template_filter_combo = ScrollSafeComboBox()
        self.template_filter_combo.addItem("全部类型", "")
        for kind in ("anchor", "button", "effect", "rank", "suit", "status", "timer"):
            self.template_filter_combo.addItem(
                TEMPLATE_KIND_LABELS.get(kind, kind), kind
            )
        filter_row.addWidget(self.template_filter_combo)
        self.template_search_edit = SearchLineEdit()
        self.template_search_edit.setPlaceholderText(
            "搜索标签或文件名"
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
        for column, width in enumerate((90, 130, 110, 155, 155, 190)):
            self.template_table.setColumnWidth(column, width)
        self.template_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        # Keep the old attribute as a compatibility alias for integrations that
        # only use it to read or select saved templates.
        self.template_list = self.template_table
        for control in (
            self.template_kind_combo,
            self.template_label_edit,
            self.template_source_role_combo,
            self.template_filter_combo,
            self.template_search_edit,
            self.crop_template_button,
            self.delete_template_button,
        ):
            control.setMinimumHeight(34)
        template_detail_layout.addWidget(self.crop_template_button)
        template_detail_layout.addWidget(self.delete_template_button)
        template_detail_layout.addLayout(filter_row)
        template_detail_layout.addWidget(self.template_table)
        self.template_filter_combo.currentIndexChanged.connect(
            self._template_filter_changed
        )
        self.template_search_edit.textChanged.connect(
            self._refresh_template_status
        )

        self.mode_detail_stack.addWidget(self.region_detail_widget)
        self.mode_detail_stack.addWidget(self.template_detail_widget)
        right.addWidget(self.mode_detail_stack, 1)

        self.crop_template_button.clicked.connect(self._crop_template)
        self.delete_template_button.clicked.connect(self._delete_selected_templates)
        self.content_splitter.addWidget(self.preview_card)
        self.content_splitter.addWidget(self.workbench_card)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 1)
        root.addWidget(self.content_splitter, 1)
        self._mode_changed()
        self._template_kind_changed()
        self._update_image_folder_label()
        self._update_responsive_layout()

    def _refresh_sessions(self) -> None:
        selected = self.current_session
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        self.session_combo.addItem("请选择已录制对局", None)
        sessions = ()
        if self.sessions_root.is_dir():
            sessions = tuple(
                sorted(
                    (
                        path
                        for path in self.sessions_root.iterdir()
                        if path.is_dir()
                        and (path / "manifest.json").is_file()
                        and (path / "video" / "game.avi").is_file()
                        and (path / "video" / "frame_index.jsonl").is_file()
                    ),
                    reverse=True,
                )
            )
        for session in sessions:
            self.session_combo.addItem(session.name, str(session))
        self.session_combo.blockSignals(False)
        self.refresh_sessions_button.setEnabled(True)
        if selected is not None:
            index = self.session_combo.findData(str(selected))
            if index >= 0:
                self.session_combo.setCurrentIndex(index)
                self._select_session(selected)
                return
        self.current_session = None
        self._set_session_actions(False)
        self.session_status.setText(
            "请选择一个已录制对局；随后可播放、逐帧定位并直接框选模板。"
            if self.session_combo.count() > 1
            else "sessions 中没有可播放的对局录像。"
        )

    def _has_playable_session(self) -> bool:
        if not self.sessions_root.is_dir():
            return False
        return any(
            path.is_dir()
            and (path / "manifest.json").is_file()
            and (path / "video" / "game.avi").is_file()
            and (path / "video" / "frame_index.jsonl").is_file()
            for path in self.sessions_root.iterdir()
        )

    def _session_changed(self, _index: int) -> None:
        value = self.session_combo.currentData()
        if value:
            self._select_session(Path(str(value)))
        else:
            self._stop_session_decode()
            self.current_session = None
            self._session_record = None
            self._clear_session_frame()
            self._set_session_actions(False)

    def _select_session(self, session: Path) -> None:
        session = Path(session).resolve()
        self._stop_session_decode()
        self.current_session = session
        self._session_record = None
        self._session_start_frame = None
        self._clear_session_frame()
        try:
            manifest = json.loads((session / "manifest.json").read_text("utf-8"))
        except Exception as exc:
            self._set_session_actions(False)
            self.session_status.setText(f"对局信息读取失败：{exc}")
            return
        frame_count = max(0, int(manifest.get("frame_count", 0) or 0))
        self.session_playback_toolbar.set_frame_count(frame_count)
        self.session_frame_spin.setValue(0)
        self._set_session_actions(True)
        self.session_status.setText(
            f"{session.name}｜录像 {frame_count} 帧｜选择播放或单帧后即可在左侧框选。"
        )

    def _clear_session_frame(self) -> None:
        self.current_image_path = None
        self.current_image = None
        self.current_roi = None
        self._show_draft_roi = False
        self.single_image_test_button.setEnabled(False)
        self.canvas.clear()
        self.canvas.setText("请选择已录制对局并定位到一帧")
        self._update_image_navigation()

    def _set_session_actions(self, enabled: bool) -> None:
        for widget in (
            self.session_play_button,
            self.session_step_button,
            self.session_rewind_button,
            self.session_forward_button,
            self.session_frame_spin,
            self.session_jump_button,
            self.session_speed_combo,
        ):
            widget.setEnabled(enabled)
        self._update_session_play_button()

    def _ensure_session_decode(self) -> ReplayDecodeThread | None:
        if self.current_session is None:
            return None
        if self._session_decode_thread is None:
            thread = ReplayDecodeThread(
                self.current_session / "video" / "game.avi",
                self.current_session / "video" / "frame_index.jsonl",
                self,
                start_frame=self._session_start_frame,
            )
            self._session_start_frame = None
            thread.set_speed(float(self.session_speed_combo.currentData() or 1.0))
            thread.frame_ready.connect(self._show_session_frame)
            thread.failed.connect(self._show_session_error)
            thread.finished.connect(lambda: self._session_decode_finished(thread))
            self._session_decode_thread = thread
            thread.start()
        return self._session_decode_thread

    def _toggle_session_playback(self) -> None:
        if self._session_playing:
            self._pause_session_playback()
            return
        thread = self._ensure_session_decode()
        if thread is not None:
            thread.play()
            self._session_playing = True
            self._update_session_play_button()

    def _pause_session_playback(self) -> None:
        if self._session_decode_thread is not None:
            self._session_decode_thread.pause()
        self._session_playing = False
        self._update_session_play_button()

    def _step_session_frame(self) -> None:
        thread = self._ensure_session_decode()
        if thread is not None:
            thread.pause()
            thread.step()
            self._session_playing = False
            self._update_session_play_button()

    def _jump_to_session_frame(self, target: int | None = None) -> None:
        if self.current_session is None:
            return
        self._stop_session_decode()
        self._session_start_frame = (
            self.session_frame_spin.value() if target is None else int(target)
        )
        self._step_session_frame()

    def _session_index_records(self) -> tuple[FrameIndexRecord, ...]:
        if self.current_session is None:
            return ()
        try:
            return tuple(
                FrameIndexRecord.from_dict(raw)
                for raw in read_json_lines(
                    self.current_session / "video" / "frame_index.jsonl"
                )
            )
        except Exception:
            return ()

    def _seek_session_by_seconds(self, delta_seconds: float) -> None:
        if self.current_session is None:
            return
        records = self._session_index_records()
        if not records:
            self.session_status.setText("帧索引不可读，无法前进或后退")
            return
        current_ms = int(getattr(self._session_record, "monotonic_ms", 0))
        target_ms = max(0, current_ms + int(float(delta_seconds) * 1000))
        record = next(
            (item for item in records if item.monotonic_ms >= target_ms),
            records[-1],
        )
        self._stop_session_decode()
        self._session_start_frame = record.frame_index
        self._step_session_frame()

    def _session_speed_changed(self, speed: float | int = 1.0) -> None:
        if self._session_decode_thread is not None:
            self._session_decode_thread.set_speed(
                float(speed if isinstance(speed, float) else self.session_speed_combo.currentData() or 1.0)
            )

    def _show_session_frame(self, record: object, image: QImage) -> None:
        self._session_record = record
        self.current_image_path = None
        self.current_image = self._qimage_to_bgr(image)
        self.current_roi = None
        self._show_draft_roi = False
        self.canvas.set_source_size(self.current_image)
        self.single_image_test_button.setEnabled(True)
        frame_text = (
            f"帧 {getattr(record, 'frame_index', '—')}　"
            f"{getattr(record, 'monotonic_ms', 0)} ms　"
            f"此前丢帧 {getattr(record, 'dropped_before', 0)}"
        )
        self.session_playback_toolbar.set_current_frame(
            int(getattr(record, "frame_index", 0))
        )
        self.session_playback_toolbar.frame_status.setText(frame_text)
        self.session_status.setText(f"当前{frame_text}；可直接在左侧框选模板。")
        self._update_image_navigation()
        self._refresh_preview()
        if self.single_image_danzero_page is not None:
            self.single_image_danzero_page.set_frame_image(
                self.current_image.copy(), self._session_frame_source(),
            )
            self._start_template_recognition()

    def _show_session_error(self, message: str) -> None:
        self._pause_session_playback()
        self.session_status.setText(f"录像播放失败：{message}")

    def _session_decode_finished(self, thread: ReplayDecodeThread) -> None:
        if thread is not self._session_decode_thread:
            return
        self._session_decode_thread = None
        self._session_playing = False
        self._update_session_play_button()

    def _stop_session_decode(self) -> None:
        thread, self._session_decode_thread = self._session_decode_thread, None
        self._session_playing = False
        self._update_session_play_button()
        if thread is not None and thread.isRunning():
            thread.stop()
            thread.wait(5_000)

    def _update_session_play_button(self) -> None:
        if hasattr(self, "session_playback_toolbar"):
            self.session_playback_toolbar.set_playing(self._session_playing)

    def _session_frame_source(self) -> str:
        if self.current_session is None:
            return "session-frame"
        frame = getattr(self._session_record, "frame_index", None)
        suffix = f"#frame={frame}" if frame is not None else ""
        return f"sessions/{self.current_session.name}/video/game.avi{suffix}"

    @staticmethod
    def _qimage_to_bgr(image: QImage) -> np.ndarray:
        rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
        height, width = rgba.height(), rgba.width()
        bits = rgba.constBits()
        raw = bits.tobytes() if hasattr(bits, "tobytes") else bytes(bits)
        pixels = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)

    def _mode_changed(self, *_args) -> None:
        is_template = self.mode_combo.currentData() == "template"
        self.mode_detail_stack.setCurrentIndex(1 if is_template else 0)
        if is_template:
            self.mode_detail_stack.setMaximumHeight(16777215)
            self.mode_detail_stack.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Expanding,
            )
        else:
            self.mode_detail_stack.setMaximumHeight(290)
            self.mode_detail_stack.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Fixed,
            )
        self.show_selected_button.setVisible(not is_template)
        self._refresh_preview()

    def _update_responsive_layout(self) -> None:
        """Keep the canvas and controls comfortable in normal and full-screen windows."""
        # Six template columns remain readable only when both panes have room;
        # otherwise keep a generous vertical, scroll-free work flow.
        compact = self.width() < 1440
        orientation = (
            Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
        )
        orientation_changed = self.content_splitter.orientation() != orientation
        if orientation_changed:
            self.content_splitter.setOrientation(orientation)
        minimum_height = 250 if compact else 330
        maximum_height = 440 if compact else 16777215
        if self.canvas.minimumHeight() != minimum_height:
            self.canvas.setMinimumHeight(minimum_height)
        if self.canvas.maximumHeight() != maximum_height:
            self.canvas.setMaximumHeight(maximum_height)
        if orientation_changed:
            self.content_splitter.setSizes(
                (470, 500) if compact else (680, 680)
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "content_splitter"):
            self._update_responsive_layout()
            # A session frame can arrive before Qt finishes laying out the
            # split panes.  Always rescale from the source image on resize so
            # the preview never stays at its early, undersized pixmap.
            self._refresh_preview()

    def _update_image_folder_label(self) -> None:
        if self.image_folder is None:
            self.folder_path_label.setText("未选择图片文件夹")
            self.folder_path_label.setToolTip("")
            return
        folder = str(self.image_folder)
        self.folder_path_label.setText(folder)
        self.folder_path_label.setToolTip(folder)

    def _choose_image_folder(self) -> None:
        start = str(self.image_folder or self.service.profile_root)
        selected = QFileDialog.getExistingDirectory(self, "选择图片文件夹", start)
        if selected:
            self.set_image_folder(Path(selected))

    def set_image_folder(self, folder: Path | str | None) -> None:
        """Select a local image directory; intentionally no longer tied to recording output."""
        candidate = Path(folder).expanduser() if folder else None
        if candidate is not None and not candidate.is_dir():
            self.status.setText("所选图片文件夹不存在或不可访问")
            return
        self.image_folder = candidate.resolve() if candidate is not None else None
        self._update_image_folder_label()
        self._refresh_images()

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
        self.overlay_visible = False
        self._show_draft_roi = False
        self.current_roi = region.abs_box
        self._update_region_overlay_actions()
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
        self.overlay_visible = False
        self._show_draft_roi = False
        self.current_roi = region.abs_box
        self._update_region_overlay_actions()
        self._refresh_region_coordinate_table()
        self._refresh_preview()

    def _preview_selected_regions(self, regions: object) -> None:
        selected = tuple(regions) if regions else ()
        if not selected:
            self.preview_regions = ()
            self.overlay_visible = False
            self._show_draft_roi = False
            self._update_region_overlay_actions()
            self._refresh_region_coordinate_table()
            self._refresh_preview()
            self.status.setText("请先选择一个或多个区域")
            return
        selected_names = tuple(region.name for region in selected)
        active_names = tuple(region.name for region in self.preview_regions)
        self.overlay_visible = not (
            self.overlay_visible and selected_names == active_names
        )
        self.preview_regions = selected
        self.current_roi = selected[0].abs_box if selected else None
        self._show_draft_roi = False
        if selected:
            self._set_region_editor_values(selected[0])
        self._update_region_overlay_actions()
        self._refresh_region_coordinate_table()
        self._refresh_preview()
        if not self.overlay_visible:
            self.status.setText(f"已隐藏 {len(selected)} 个选中区域")

    def _update_region_overlay_actions(self) -> None:
        text = "隐藏选中区域" if self.overlay_visible else "标注选中区域"
        self.show_selected_button.setText(text)
        if self.region_config_page is not None:
            self.region_config_page.set_preview_visible(self.overlay_visible)

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
            self._show_draft_roi = True
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
        kind_filter: str,
        keyword: str,
    ) -> bool:
        if kind_filter and str(record.get("kind", "")) != kind_filter:
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
        kind_filter = str(self.template_filter_combo.currentData() or "")
        keyword = self.template_search_edit.text().strip()
        filtered = [
            record
            for record in records
            if self._template_matches(record, kind_filter, keyword)
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
                str(record.get("abs_box", "")),
                str(record.get("ratio_box", "")),
                Path(relative).name,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, sample_id)
                    item.setToolTip(sample_id)
                if column == 1:
                    item.setToolTip(f"{label}（{sample_id}）")
                if column == 5:
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
        kind = str(self.template_kind_combo.currentData() or "")
        if kind == "effect":
            generic_index = self.template_source_role_combo.findData("generic")
            if generic_index >= 0:
                self.template_source_role_combo.setCurrentIndex(generic_index)
        filter_index = self.template_filter_combo.findData(kind)
        if filter_index >= 0 and self.template_filter_combo.currentIndex() != filter_index:
            self.template_filter_combo.blockSignals(True)
            self.template_filter_combo.setCurrentIndex(filter_index)
            self.template_filter_combo.blockSignals(False)
        records = self.template_service.list_templates()
        self._refresh_template_labels(records)
        self._refresh_template_status()

    def _template_filter_changed(self, *_args) -> None:
        """Keep saved-template browsing and the crop kind in one context."""

        kind = str(self.template_filter_combo.currentData() or "")
        kind_index = self.template_kind_combo.findData(kind)
        if kind and kind_index >= 0 and self.template_kind_combo.currentIndex() != kind_index:
            self.template_kind_combo.blockSignals(True)
            self.template_kind_combo.setCurrentIndex(kind_index)
            self.template_kind_combo.blockSignals(False)
            self._refresh_template_labels(self.template_service.list_templates())
        self._refresh_template_status()

    def _template_label_value(self) -> str:
        data = self.template_label_edit.currentData()
        if data is not None and self.template_label_edit.currentIndex() >= 0:
            return str(data).strip()
        return self.template_label_edit.currentText().strip()

    def _refresh_images(self) -> None:
        current_path = self.current_image_path
        self.image_paths = (
            self.service.list_images_in_folder(self.image_folder)
            if self.image_folder is not None
            else ()
        )
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for path in self.image_paths:
            display_name = (
                path.relative_to(self.image_folder).as_posix()
                if self.image_folder is not None
                else path.name
            )
            self.image_combo.addItem(
                display_name,
                str(path),
            )
        self.image_combo.blockSignals(False)
        if self.image_paths:
            index = self.image_paths.index(current_path) if current_path in self.image_paths else 0
            self.image_combo.setCurrentIndex(index)
            self._image_changed(index)
        else:
            self._image_changed(-1)

    def _update_image_navigation(self) -> None:
        if self.current_image_path is None and self.current_session is not None:
            self.previous_image_button.setEnabled(False)
            self.next_image_button.setEnabled(False)
            frame = getattr(self._session_record, "frame_index", None)
            self.image_position_label.setText(
                f"帧 {frame}" if frame is not None else "帧 —"
            )
            return
        index = self.image_combo.currentIndex()
        count = self.image_combo.count()
        self.previous_image_button.setEnabled(index > 0)
        self.next_image_button.setEnabled(0 <= index < count - 1)
        self.image_position_label.setText(
            f"{index + 1 if index >= 0 else 0} / {count}"
        )

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
            self._show_draft_roi = False
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
        self._show_draft_roi = False
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
            self.canvas.setText("当前文件夹中没有可预览的图片")
            return

        regions = (
            self.preview_regions
            if self.mode_combo.currentData() == "region" and self.overlay_visible
            else ()
        )
        image = (
            self.service.overlay_regions(self.current_image, regions)
            if regions
            else self.current_image.copy()
        )
        should_draw_roi = (
            self.mode_combo.currentData() == "template"
            or self.overlay_visible
            or self._show_draft_roi
        )
        if (
            should_draw_roi
            and self.current_roi is not None
            and self.current_roi.fits_within((image.shape[1], image.shape[0]))
        ):
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
            self.status.setText("请先选择图片文件夹并打开一张图片")
            return
        regions = self.preview_regions
        if not regions:
            self.status.setText("请先选择一个或多个区域")
            return
        self._preview_selected_regions(regions)

    def _open_single_image_danzero(self) -> None:
        if self.current_image is None:
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
            if self.current_image_path is not None:
                self.single_image_danzero_page.set_image_path(self.current_image_path)
        if self.current_image_path is None:
            self.single_image_danzero_page.set_frame_image(
                self.current_image.copy(),
                self._session_frame_source(),
            )
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
        thread = OneShotThread(lambda: self._recognize_single_image(image), self)
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

    def _recognize_single_image(self, image) -> RecognitionResult:
        """Keep rank-only cards in the annotation view just like live play."""

        try:
            return self.recognition_service.recognize(
                image,
                allow_unknown_suit=True,
            )
        except TypeError as exc:
            # Older plug-ins and test doubles may not have the opt-in yet.
            if "allow_unknown_suit" not in str(exc):
                raise
            return self.recognition_service.recognize(image)

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
            self.session_combo,
            self.refresh_sessions_button,
            self.session_play_button,
            self.session_step_button,
            self.session_rewind_button,
            self.session_forward_button,
            self.session_frame_spin,
            self.session_jump_button,
            self.session_speed_combo,
        ):
            widget.setEnabled(not busy)

    def _crop_succeeded(self, sample) -> None:
        self.recognition_service.reload_templates()
        self._refresh_template_status()
        self._refresh_preview()
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
        if self.current_image is None:
            self.status.setText("请先选择图片或定位到对局录像帧")
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
        if self.current_image_path is None:
            source_image = self._session_frame_source()
        else:
            source_image = (
                self.current_image_path.relative_to(self.image_folder).as_posix()
                if self.image_folder is not None
                else self.current_image_path.name
            )
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
        self._stop_session_decode()
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
