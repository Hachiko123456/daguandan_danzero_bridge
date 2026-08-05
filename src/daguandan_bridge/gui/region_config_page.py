from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..annotation_service import (
    REGION_DISPLAY_NAMES,
    ROLE_DISPLAY_NAMES,
    AnnotationService,
    RegionRecord,
    display_region_name,
    display_role,
)
from ..models import Box


class RegionConfigPage(QDialog):
    """独立的区域元数据和坐标配置窗口。"""

    HEADERS = ("名称", "绝对坐标", "比例坐标")

    region_selected = Signal(object)
    preview_requested = Signal(object)
    region_updated = Signal(str, object)

    def __init__(
        self,
        service: AnnotationService,
        regions: Iterable[RegionRecord],
        parent=None,
    ):
        super().__init__(parent)
        self.service = service
        self.regions = list(regions)
        self.setObjectName("regionConfigPage")
        self.setWindowTitle("区域配置")
        self.resize(820, 560)
        self._build_ui()
        self._refresh_regions()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.addWidget(QLabel("区域名称和标准化坐标"))

        body = QHBoxLayout()
        self.region_table = QTableWidget(0, len(self.HEADERS))
        self.region_table.setHorizontalHeaderLabels(self.HEADERS)
        self.region_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.region_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.region_table.itemSelectionChanged.connect(self._load_selected_region)
        body.addWidget(self.region_table, 3)

        editor = QVBoxLayout()
        form = QFormLayout()
        self.name_combo = QComboBox()
        for name, label in REGION_DISPLAY_NAMES.items():
            self.name_combo.addItem(label, name)
        self.role_combo = QComboBox()
        for role, label in ROLE_DISPLAY_NAMES.items():
            self.role_combo.addItem(label, role)
        self.role_combo.setVisible(False)
        form.addRow("名称", self.name_combo)
        editor.addLayout(form)

        coordinates = QGridLayout()
        self.x_spin, self.y_spin, self.w_spin, self.h_spin = (QSpinBox() for _ in range(4))
        for spin, minimum, maximum in (
            (self.x_spin, 0, 1279),
            (self.y_spin, 0, 719),
            (self.w_spin, 1, 1280),
            (self.h_spin, 1, 720),
        ):
            spin.setRange(minimum, maximum)
        for column, (label, spin) in enumerate(
            (("x", self.x_spin), ("y", self.y_spin), ("w", self.w_spin), ("h", self.h_spin))
        ):
            coordinates.addWidget(QLabel(label), 0, column)
            coordinates.addWidget(spin, 1, column)
        editor.addLayout(coordinates)

        self.status = QLabel("请选择一个区域")
        self.status.setWordWrap(True)
        editor.addWidget(self.status)

        actions = QHBoxLayout()
        self.preview_selected_button = QPushButton("标注选中区域")
        self.save_button = QPushButton("保存修改")
        self.close_button = QPushButton("关闭")
        actions.addWidget(self.preview_selected_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.close_button)
        editor.addLayout(actions)
        editor.addStretch(1)

        self.preview_selected_button.clicked.connect(self._preview_selected_regions)
        self.save_button.clicked.connect(self._save_selected_region)
        self.close_button.clicked.connect(self.close)

        body.addLayout(editor, 2)
        root.addLayout(body, 1)

    def _refresh_regions(self) -> None:
        self.region_table.setRowCount(len(self.regions))
        for row, region in enumerate(self.regions):
            values = (
                display_region_name(region.name),
                str(region.abs_box.to_list()),
                str([round(value, 6) for value in region.ratio_box]),
            )
            for column, value in enumerate(values):
                self.region_table.setItem(row, column, QTableWidgetItem(value))
        self.region_table.resizeColumnsToContents()

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.region_table.selectedIndexes()})

    def _selected_regions(self) -> list[RegionRecord]:
        return [
            self.regions[row]
            for row in self._selected_rows()
            if 0 <= row < len(self.regions)
        ]

    def _load_selected_region(self) -> None:
        rows = self._selected_rows()
        if not rows:
            self.status.setText("请选择一个区域")
            return
        region = self.regions[rows[0]]
        self.name_combo.setCurrentIndex(max(0, self.name_combo.findData(region.name)))
        self.role_combo.setCurrentIndex(max(0, self.role_combo.findData(region.role)))
        self.set_box(region.abs_box, update_status=False)
        self.status.setText(
            f"已选择 {len(rows)} 个区域，当前编辑：{display_region_name(region.name)}"
        )
        self.region_selected.emit(region)

    def set_box(self, box: Box, *, update_status: bool = True) -> None:
        """把画布拖出的标准化坐标同步到编辑器。"""
        self.x_spin.setValue(box.x)
        self.y_spin.setValue(box.y)
        self.w_spin.setValue(box.w)
        self.h_spin.setValue(box.h)
        if update_status:
            self.status.setText(f"已更新坐标：x={box.x}, y={box.y}, w={box.w}, h={box.h}")

    def _preview_selected_regions(self) -> None:
        regions = self._selected_regions()
        if not regions:
            self.status.setText("请先选择一个或多个区域")
            return
        self.preview_requested.emit(tuple(regions))
        self.status.setText(f"已请求预览 {len(regions)} 个区域")

    def _save_selected_region(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            self.status.setText("编辑保存时请只选择一个区域")
            return

        row = rows[0]
        old = self.regions[row]
        try:
            updated = self.service.update_region(
                old.name,
                name=str(self.name_combo.currentData() or old.name),
                role=old.role,
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
        self.region_updated.emit(old.name, updated)
