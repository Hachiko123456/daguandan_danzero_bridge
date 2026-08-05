from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """基于标准化截图的绝对 ROI 坐标。"""

    x: int
    y: int
    w: int
    h: int

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @classmethod
    def from_value(cls, value: object) -> "Box":
        """从四元素 JSON 坐标创建 Box。"""
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("box 必须是包含 4 个整数的数组")
        x, y, w, h = (int(item) for item in value)
        return cls(x=x, y=y, w=w, h=h)

    @property
    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def fits_within(self, size: tuple[int, int]) -> bool:
        """判断坐标是否完整位于指定宽高内。"""
        width, height = size
        return (
            not self.is_empty
            and self.x >= 0
            and self.y >= 0
            and self.x + self.w <= width
            and self.y + self.h <= height
        )


@dataclass(frozen=True)
class TargetWindow:
    """目标窗口信息。"""

    hwnd: int
    title: str


@dataclass(frozen=True)
class ClientRect:
    """mss 截图所需的屏幕坐标。"""

    left: int
    top: int
    width: int
    height: int

    def to_mss_monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }
