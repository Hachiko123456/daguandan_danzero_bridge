"""OpenCV/NumPy adapter used by the video scan application use case."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

Frame = np.ndarray

CAP_PROP_FPS = cv2.CAP_PROP_FPS
CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES


def default_video_capture(path: str) -> Any:
    """Open a video with the production OpenCV implementation."""
    return cv2.VideoCapture(path)


def is_frame(value: object) -> bool:
    """Return whether ``value`` is a decoded NumPy image frame."""
    return isinstance(value, np.ndarray)


def has_pixels(value: object) -> bool:
    """Return whether a decoded image frame has at least one pixel."""
    return is_frame(value) and value.size > 0


def copy_frame(frame: Frame) -> Frame:
    """Copy a decoded frame without exposing NumPy operations upstream."""
    return frame.copy()


def crop_annotation_record(frame: Frame, record: Any) -> Frame | None:
    """Crop an annotation record's ratio box from a decoded image frame."""
    try:
        ratio = tuple(float(item) for item in getattr(record, "ratio_box"))
        if len(ratio) != 4:
            return None
        height, width = frame.shape[:2]
        x = round(ratio[0] * width)
        y = round(ratio[1] * height)
        crop_width = max(1, round(ratio[2] * width))
        crop_height = max(1, round(ratio[3] * height))
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + crop_width), min(height, y + crop_height)
        if left >= right or top >= bottom:
            return None
        return frame[top:bottom, left:right]
    except (AttributeError, TypeError, ValueError, IndexError):
        return None


def update_array_sha256(array: Frame, digest: Any) -> None:
    """Feed exact frame metadata and bytes into a caller-owned digest."""
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii", errors="replace"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(contiguous))
