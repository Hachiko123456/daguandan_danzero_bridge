from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..annotation_service import (
    REGION_DISPLAY_NAMES,
    ROLE_DISPLAY_NAMES,
    AnnotationService,
    RegionRecord,
    display_region_name,
    display_role,
)
from ..image_io import read_image_unicode
from ..models import Box


class AnnotationPage(QWidget):
    """Edit and overlay configured regions on recorded screenshots only."""

    HEADERS = ("名称", "角色", "绝对坐标", "比例坐标")

    def __init__(self, service: AnnotationService | None = None):
        super().__init__()
        self.service = service or AnnotationService()
        self.regions = list(self.service.list_regions())
        self.image_paths: tuple[Path, ...] = ()
        self.current_image_path: Path | None = None
        self.current_image = None
        self.setObjectName("annotationPage")
        self._build_ui()
        self._refresh_regions()
        self._refresh_images()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(QLabel("区域标注（仅可选择 screenshots 目录内图片）"))
        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self._image_changed)
        root.addWidget(self.image_combo)

        body = QHBoxLayout()
        self.region_table = QTableWidget(0, len(self.HEADERS))
        self.region_table.setHorizontalHeaderLabels(self.HEADERS)
        self.region_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.region_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.region_table.itemSelectionChanged.connect(self._load_selected_region)
        body.addWidget(self.region_table, 3)

        right = QVBoxLayout()
        self.canvas = QLabel("请选择录制截图")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumSize(520, 300)
        right.addWidget(self.canvas, 1)
        self.status = QLabel("区域：—")
        right.addWidget(self.status)

        form = QFormLayout()
        self.name_combo = QComboBox()
        for name, label in REGION_DISPLAY_NAMES.items():
            self.name_combo.addItem(label, name)
        self.role_combo = QComboBox()
        for role, label in ROLE_DISPLAY_NAMES.items():
            self.role_combo.addItem(label, role)

        self.x_spin, self.y_spin, self.w_spin, self.h_spin = (QSpinBox() for _ in range(4))
        for spin, maximum in (
            (self.x_spin, 1279),
            (self.y_spin, 719),
            (self.w_spin, 1280),
            (self.h_spin, 720),
        ):
            spin.setRange(0, maximum)

        form.addRow("名称", self.name_combo)
        form.addRow("角色", self.role_combo)
        grid = QGridLayout()
        for column, (label, spin) in enumerate(
            (("x", self.x_spin), ("y", self.y_spin), ("w", self.w_spin), ("h", self.h_spin))
        ):
            grid.addWidget(QLabel(label), 0, column)
            grid.addWidget(spin, 1, column)
        right.addLayout(form)
        right.addLayout(grid)

        actions = QHBoxLayout()
        self.show_selected_button = QPushButton("标注选中区域")
        self.save_button = QPushButton("保存修改")
        actions.addWidget(self.show_selected_button)
        actions.addWidget(self.save_button)
        right.addLayout(actions)
        self.show_selected_button.clicked.connect(self._show_selected_regions)
        self.save_button.clicked.connect(self._save_selected_region)
        body.addLayout(right, 2)
        root.addLayout(body, 1)

    def _refresh_regions(self) -> None:
        self.region_table.setRowCount(len(self.regions))
        for row, region in enumerate(self.regions):
            values = (
                display_region_name(region.name),
                display_role(region.role),
                str(region.abs_box.to_list()),
                str([round(value, 6) for value in region.ratio_box]),
            )
            for column, value in enumerate(values):
                self.region_table.setItem(row, column, QTableWidgetItem(value))
        self.region_table.resizeColumnsToContents()

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

    def _image_changed(self, index: int) -> None:
        if index < 0 or index >= len(self.image_paths):
            self.current_image_path = None
            self.current_image = None
            self._refresh_preview()
            return

        self.current_image_path = self.image_paths[index]
        try:
            self.current_image = read_image_unicode(self.current_image_path)
        except Exception as exc:
            self.current_image = None
            self.canvas.clear()
            self.canvas.setText(f"图片读取失败：{exc}")
            self.status.setText("图片读取失败")
            return
        self._refresh_preview()

    def _load_selected_region(self) -> None:
        rows = sorted({index.row() for index in self.region_table.selectedIndexes()})
        if not rows:
            self.status.setText("区域：—")
            self._refresh_preview()
            return
        region = self.regions[rows[0]]
        self.name_combo.setCurrentIndex(max(0, self.name_combo.findData(region.name)))
        self.role_combo.setCurrentIndex(max(0, self.role_combo.findData(region.role)))
        self.x_spin.setValue(region.abs_box.x)
        self.y_spin.setValue(region.abs_box.y)
        self.w_spin.setValue(region.abs_box.w)
        self.h_spin.setValue(region.abs_box.h)
        self.status.setText(
            f"已选择 {len(rows)} 个区域，当前编辑：{display_region_name(region.name)}"
        )
        self._refresh_preview()

    def _selected_regions(self) -> list[RegionRecord]:
        rows = sorted({index.row() for index in self.region_table.selectedIndexes()})
        return [self.regions[row] for row in rows if 0 <= row < len(self.regions)]

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

        regions = self._selected_regions()
        image = (
            self.service.overlay_regions(self.current_image, regions)
            if regions
            else self.current_image
        )
        self.canvas.setPixmap(self._pixmap_for_image(image))
        if regions:
            self.status.setText(f"已标注 {len(regions)} 个区域")
        elif self.current_image_path is not None:
            relative = self.current_image_path.relative_to(self.service.screenshots_root)
            self.status.setText(f"已预览：{relative}")

    def _show_selected_regions(self) -> None:
        if self.current_image is None:
            self.status.setText("请先从 screenshots 目录选择图片")
            return
        regions = self._selected_regions()
        if not regions:
            self._refresh_preview()
            self.status.setText("请先选择一个或多个区域")
            return
        self._refresh_preview()

    def _save_selected_region(self) -> None:
        rows = sorted({index.row() for index in self.region_table.selectedIndexes()})
        if len(rows) != 1:
            self.status.setText("编辑保存时请只选择一个区域")
            return

        row = rows[0]
        old = self.regions[row]
        selected_name = str(self.name_combo.currentData() or old.name)
        selected_role = str(self.role_combo.currentData() or old.role)
        try:
            updated = self.service.update_region(
                old.name,
                name=selected_name,
                role=selected_role,
                box=Box(
                    self.x_spin.value(),
                    self.y_spin.value(),
                    self.w_spin.value(),
                    self.h_spin.value(),
                ),
            )
        except Exception as exc:
            self.status.setText(f"保存失败：{exc}")
            return

        self.regions[row] = updated
        self._refresh_regions()
        self.region_table.selectRow(row)
        self.status.setText(f"已保存：{display_region_name(updated.name)}")

