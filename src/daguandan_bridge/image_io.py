from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .dependencies import import_required
from .models import Box
from .storage import atomic_write_json


@dataclass(frozen=True)
class StandardizationResult:
    """等比缩放到基准画面的图像和坐标变换信息。"""

    image: Any
    source_size: tuple[int, int]
    source_viewport: Box
    content_box: Box
    scale: float
    padding: tuple[int, int, int, int]
    aspect_error: float
    aspect_compatible: bool


def detect_content_viewport(
    image: Any,
    *,
    black_mean_threshold: float = 8.0,
    black_std_threshold: float = 3.0,
) -> Box:
    """只裁掉从图像边缘连续延伸的近黑、低方差边框。"""
    np = import_required("numpy", "numpy")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("图片尺寸必须大于 0")

    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        gray = image.astype(np.float32).mean(axis=2)
    row_content = (gray.mean(axis=1) > black_mean_threshold) | (
        gray.std(axis=1) > black_std_threshold
    )
    col_content = (gray.mean(axis=0) > black_mean_threshold) | (
        gray.std(axis=0) > black_std_threshold
    )
    row_indexes = np.flatnonzero(row_content)
    col_indexes = np.flatnonzero(col_content)
    if row_indexes.size == 0 or col_indexes.size == 0:
        return Box(0, 0, width, height)

    top = int(row_indexes[0])
    bottom = int(row_indexes[-1]) + 1
    left = int(col_indexes[0])
    right = int(col_indexes[-1]) + 1
    viewport = Box(left, top, right - left, bottom - top)
    if viewport.w < max(2, width // 4) or viewport.h < max(2, height // 4):
        return Box(0, 0, width, height)
    return viewport


def calculate_viewport_box(
    source_size: tuple[int, int],
    *,
    mode: str = "full",
    aspect_ratio: float = 16 / 9,
) -> Box:
    """Calculate a deterministic client-area viewport before standardization."""
    width, height = (int(source_size[0]), int(source_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("source_size must contain positive dimensions")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "full":
        return Box(0, 0, width, height)
    if normalized_mode != "bottom_aspect":
        raise ValueError("viewport mode must be full or bottom_aspect")
    ratio = float(aspect_ratio)
    if not 0 < ratio <= 10:
        raise ValueError("aspect_ratio must be positive")

    if width / height < ratio:
        viewport_height = max(1, min(height, int(round(width / ratio))))
        return Box(0, height - viewport_height, width, viewport_height)

    viewport_width = max(1, min(width, int(round(height * ratio))))
    return Box((width - viewport_width) // 2, 0, viewport_width, height)


def standardize_to_base(
    image: Any,
    base_size: tuple[int, int],
    aspect_tolerance: float = 0.03,
    detect_black_bars: bool = True,
    viewport_mode: str = "full",
    viewport_aspect_ratio: float = 16 / 9,
) -> StandardizationResult:
    """等比缩放内容并居中补边，不把任意宽高比强制拉伸。"""
    cv2 = import_required("cv2", "opencv-python")
    np = import_required("numpy", "numpy")
    base_width, base_height = (int(base_size[0]), int(base_size[1]))
    if base_width <= 0 or base_height <= 0:
        raise ValueError("base_size 必须是正整数尺寸")
    if aspect_tolerance < 0:
        raise ValueError("aspect_tolerance 不能为负数")

    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("图片尺寸必须大于 0")
    configured_viewport = calculate_viewport_box(
        (source_width, source_height),
        mode=viewport_mode,
        aspect_ratio=viewport_aspect_ratio,
    )
    configured_crop = image[
        configured_viewport.y : configured_viewport.y + configured_viewport.h,
        configured_viewport.x : configured_viewport.x + configured_viewport.w,
    ]
    if detect_black_bars:
        detected = detect_content_viewport(configured_crop)
        source_viewport = Box(
            configured_viewport.x + detected.x,
            configured_viewport.y + detected.y,
            detected.w,
            detected.h,
        )
    else:
        source_viewport = configured_viewport
    crop = image[
        source_viewport.y : source_viewport.y + source_viewport.h,
        source_viewport.x : source_viewport.x + source_viewport.w,
    ]

    target_ratio = base_width / base_height
    source_ratio = source_viewport.w / source_viewport.h
    aspect_error = abs(source_ratio - target_ratio) / target_ratio
    scale = min(base_width / source_viewport.w, base_height / source_viewport.h)
    scaled_width = max(1, min(base_width, int(round(source_viewport.w * scale))))
    scaled_height = max(1, min(base_height, int(round(source_viewport.h * scale))))
    interpolation = (
        cv2.INTER_AREA
        if source_viewport.w > scaled_width or source_viewport.h > scaled_height
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(
        crop,
        (scaled_width, scaled_height),
        interpolation=interpolation,
    )
    left = (base_width - scaled_width) // 2
    top = (base_height - scaled_height) // 2
    right = base_width - scaled_width - left
    bottom = base_height - scaled_height - top
    canvas_shape = (base_height, base_width, *resized.shape[2:])
    canvas = np.zeros(canvas_shape, dtype=resized.dtype)
    canvas[top : top + scaled_height, left : left + scaled_width] = resized
    return StandardizationResult(
        image=canvas,
        source_size=(source_width, source_height),
        source_viewport=source_viewport,
        content_box=Box(left, top, scaled_width, scaled_height),
        scale=float(scale),
        padding=(left, top, right, bottom),
        aspect_error=float(aspect_error),
        aspect_compatible=aspect_error <= aspect_tolerance,
    )


def resize_to_base(image: Any, base_size: tuple[int, int]) -> Any:
    """兼容旧调用：等比缩放并补边到 profile 基准分辨率。"""
    return standardize_to_base(
        image,
        base_size,
        aspect_tolerance=1.0,
        detect_black_bars=False,
    ).image


def read_image_unicode(path: Path) -> Any:
    """兼容中文路径读取图片。"""
    cv2 = import_required("cv2", "opencv-python")
    np = import_required("numpy", "numpy")

    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def save_image_unicode(path: Path, image: Any) -> None:
    """兼容中文路径保存图片。"""
    cv2 = import_required("cv2", "opencv-python")

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def save_capture_with_metadata(
    image_path: Path,
    result: StandardizationResult,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """保存标准化截图及同名 JSON 变换元数据。"""
    save_image_unicode(image_path, result.image)
    document: dict[str, Any] = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_size": list(result.source_size),
        "source_viewport": result.source_viewport.to_list(),
        "content_box": result.content_box.to_list(),
        "scale": result.scale,
        "padding": list(result.padding),
        "aspect_error": result.aspect_error,
        "aspect_compatible": result.aspect_compatible,
        "standardized_size": [result.image.shape[1], result.image.shape[0]],
    }
    if metadata:
        document.update(metadata)
    metadata_path = image_path.with_suffix(".json")
    atomic_write_json(metadata_path, document)
    return metadata_path
