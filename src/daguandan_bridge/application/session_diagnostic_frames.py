"""Persistent screenshots captured from the live listener for offline diagnosis.

The listener already owns the canonical, profile-standardized frame.  This
module stores that exact NumPy array as a lossless PNG together with a small,
provenance-rich sidecar.  It deliberately does not capture windows or perform
recognition; those responsibilities stay with the live pipeline and
``WindowDebugReportService`` respectively.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any, Mapping

import cv2
import numpy as np


FRAME_SCHEMA = "guandan.session-diagnostic-frame/v1"
SOURCE_KIND = "live_listener_frame"
MANUAL_SOURCE_KIND = "manual_window_capture"
_ALLOWED_SOURCES = frozenset({SOURCE_KIND, MANUAL_SOURCE_KIND})
_DIAGNOSTIC_DIRECTORY = "diagnostic_frames"
_FRAME_NAME = re.compile(r"^(?P<sequence>[0-9]{6,})\.(?P<suffix>png|json)$", re.IGNORECASE)
_REPARSE_POINT = 0x0400


class SessionDiagnosticFrameError(ValueError):
    """Raised when a saved diagnostic frame is unsafe, incomplete, or corrupt."""


@dataclass(frozen=True)
class SessionDiagnosticFrame:
    """One saved listener frame and its provenance sidecar."""

    sequence: int
    image_path: Path
    metadata_path: Path
    session_directory: Path
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": int(self.sequence),
            "image_path": str(self.image_path),
            "metadata_path": str(self.metadata_path),
            "session_directory": str(self.session_directory),
            "metadata": _json_safe(self.metadata),
        }


# A process-local lock prevents two store instances in the same GUI process
# from allocating the same next sequence.  Atomic replacement still protects
# readers and leaves incomplete pairs invisible after an interrupted write.
_DIRECTORY_LOCKS: dict[str, threading.RLock] = {}
_DIRECTORY_LOCKS_GUARD = threading.Lock()


def _directory_lock(directory: Path) -> threading.RLock:
    key = os.path.normcase(str(directory))
    with _DIRECTORY_LOCKS_GUARD:
        return _DIRECTORY_LOCKS.setdefault(key, threading.RLock())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _is_link_or_reparse(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        checker = getattr(path, "is_junction", None)
        if callable(checker) and checker():
            return True
        info = path.lstat()
    except FileNotFoundError:
        return False
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT) or stat.S_ISLNK(info.st_mode)


def _assert_no_reparse_components(path: Path) -> Path:
    """Reject links/reparse points in every existing component of ``path``."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = absolute
    while True:
        if _lexists(current) and _is_link_or_reparse(current):
            raise SessionDiagnosticFrameError(
                f"诊断截图路径不能经过符号链接或重解析点：{current}"
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    return absolute


def _ensure_directory(path: Path, *, field: str) -> Path:
    value = _assert_no_reparse_components(path)
    if _lexists(value) and not value.is_dir():
        raise SessionDiagnosticFrameError(f"{field} 不是目录：{value}")
    if not value.exists():
        value.mkdir(parents=True, exist_ok=True)
    value = _assert_no_reparse_components(value)
    if not value.is_dir():
        raise SessionDiagnosticFrameError(f"{field} 不是目录：{value}")
    return value


def _assert_regular_file(path: Path, *, field: str) -> Path:
    value = _assert_no_reparse_components(path)
    if not value.is_file():
        raise SessionDiagnosticFrameError(f"{field} 不是普通文件：{value}")
    try:
        mode = value.lstat().st_mode
    except FileNotFoundError as exc:
        raise SessionDiagnosticFrameError(f"{field} 不存在：{value}") from exc
    if not stat.S_ISREG(mode) or _is_link_or_reparse(value):
        raise SessionDiagnosticFrameError(f"{field} 不能是链接或特殊文件：{value}")
    return value


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_attribute(values: tuple[Any, ...], name: str, default: Any = None) -> Any:
    for value in values:
        candidate = _attribute(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _rect_payload(rect: Any) -> object:
    if rect is None:
        return None
    if isinstance(rect, Mapping):
        return _json_safe(dict(rect))
    if hasattr(rect, "to_dict") and callable(rect.to_dict):
        return _json_safe(rect.to_dict())
    names = ("left", "top", "width", "height")
    if all(getattr(rect, name, None) is not None for name in names):
        return [int(getattr(rect, name)) for name in names]
    if isinstance(rect, (list, tuple)):
        return _json_safe(list(rect))
    return _json_safe(rect)


def _as_iso_datetime(value: Any) -> str:
    if value is None:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    parent = _ensure_directory(path.parent, field="诊断截图父目录")
    if _lexists(path) and _is_link_or_reparse(path):
        raise SessionDiagnosticFrameError(f"拒绝覆盖链接或重解析点：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=os.fspath(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if _is_link_or_reparse(temporary):
            raise SessionDiagnosticFrameError(f"临时文件不能是链接或重解析点：{temporary}")
        os.replace(os.fspath(temporary), os.fspath(path))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, content)


def _decode_png(path: Path) -> np.ndarray:
    content = path.read_bytes()
    try:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise SessionDiagnosticFrameError(f"PNG 无法解码：{path}") from exc
    if image is None or image.size == 0:
        raise SessionDiagnosticFrameError(f"PNG 不是有效的非空图片：{path}")
    return image


def _validate_metadata(metadata: Mapping[str, Any], sequence: int) -> dict[str, Any]:
    value = dict(metadata)
    if value.get("schema") != FRAME_SCHEMA:
        raise SessionDiagnosticFrameError("诊断截图 metadata schema 不匹配")
    if value.get("source") not in _ALLOWED_SOURCES:
        raise SessionDiagnosticFrameError("诊断截图来源不受支持")
    source_phase = value.get("source_phase")
    if source_phase is not None and not isinstance(source_phase, str):
        raise SessionDiagnosticFrameError("诊断截图 metadata 的 source_phase 无效")
    if not str(value.get("session_id") or "").strip():
        raise SessionDiagnosticFrameError("诊断截图 metadata 缺少 session_id")
    try:
        metadata_sequence = int(value["sequence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionDiagnosticFrameError("诊断截图 metadata 缺少有效 sequence") from exc
    if metadata_sequence != int(sequence):
        raise SessionDiagnosticFrameError("诊断截图文件名与 metadata sequence 不一致")
    for field in ("width", "height"):
        try:
            if int(value[field]) <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionDiagnosticFrameError(f"诊断截图 metadata 缺少有效 {field}") from exc
    for field in ("raw_sha256", "png_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SessionDiagnosticFrameError(f"诊断截图 metadata 缺少有效 {field}")
    return value


class SessionDiagnosticFrameStore:
    """Store and validate exact standardized listener frames per session."""

    @staticmethod
    def _diagnostic_directory(session_directory: Path | str, *, create: bool) -> tuple[Path, Path]:
        session = _assert_no_reparse_components(Path(session_directory).expanduser())
        if _lexists(session) and not session.is_dir():
            raise SessionDiagnosticFrameError(f"对局目录不是目录：{session}")
        if create:
            session = _ensure_directory(session, field="对局目录")
        elif not session.is_dir():
            raise SessionDiagnosticFrameError(f"对局目录不存在：{session}")
        diagnostic = session / _DIAGNOSTIC_DIRECTORY
        if create:
            diagnostic = _ensure_directory(diagnostic, field="对局诊断截图目录")
        elif not diagnostic.is_dir():
            raise SessionDiagnosticFrameError(f"对局诊断截图目录不存在：{diagnostic}")
        _assert_no_reparse_components(diagnostic)
        return session, diagnostic

    @staticmethod
    def _next_sequence(directory: Path) -> int:
        highest = 0
        for entry in directory.iterdir():
            if _is_link_or_reparse(entry):
                raise SessionDiagnosticFrameError(f"诊断截图目录包含链接或重解析点：{entry}")
            match = _FRAME_NAME.match(entry.name)
            if match:
                highest = max(highest, int(match.group("sequence")))
        return highest + 1

    @classmethod
    def _record_from_paths(
        cls,
        image_path: Path | str,
        metadata_path: Path | str | None = None,
    ) -> SessionDiagnosticFrame:
        image = _assert_regular_file(Path(image_path).expanduser(), field="诊断截图 PNG")
        metadata = Path(metadata_path).expanduser() if metadata_path is not None else image.with_suffix(".json")
        metadata = _assert_regular_file(metadata, field="诊断截图 metadata")
        if image.parent != metadata.parent or image.stem != metadata.stem:
            raise SessionDiagnosticFrameError("PNG 与 metadata 必须是同编号同目录文件对")
        if image.parent.name != _DIAGNOSTIC_DIRECTORY:
            raise SessionDiagnosticFrameError("监听截图必须位于 session/diagnostic_frames 目录")
        match = _FRAME_NAME.match(image.name)
        if match is None or match.group("suffix").lower() != "png":
            raise SessionDiagnosticFrameError("诊断截图 PNG 文件名必须是六位以上数字序号")
        sequence = int(match.group("sequence"))
        try:
            raw_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SessionDiagnosticFrameError(f"诊断截图 metadata 无法读取：{metadata}") from exc
        if not isinstance(raw_metadata, Mapping):
            raise SessionDiagnosticFrameError("诊断截图 metadata 顶层必须是 JSON 对象")
        checked = _validate_metadata(raw_metadata, sequence)
        session = image.parent.parent
        return SessionDiagnosticFrame(sequence, image, metadata, session, checked)

    @classmethod
    def read_record(
        cls,
        image_path: Path | str,
        *,
        metadata_path: Path | str | None = None,
    ) -> SessionDiagnosticFrame:
        """Read and validate one saved frame pair without decoding image pixels."""

        return cls._record_from_paths(image_path, metadata_path)

    def save_snapshot(
        self,
        session_directory: Path | str,
        snapshot: Any,
        *,
        session_id: str,
        capture_generation: int,
        capture_seq: int,
        source: str = SOURCE_KIND,
        source_phase: str | None = None,
    ) -> SessionDiagnosticFrame:
        """Atomically save one exact standardized image and its provenance sidecar."""

        if not str(session_id).strip():
            raise SessionDiagnosticFrameError("session_id 不能为空")
        source = str(source or "").strip()
        if source not in _ALLOWED_SOURCES:
            raise SessionDiagnosticFrameError(f"不支持的诊断截图来源：{source or '<empty>'}")
        if source_phase is not None:
            source_phase = str(source_phase).strip() or None
        try:
            generation = int(capture_generation)
            capture_sequence = int(capture_seq)
        except (TypeError, ValueError) as exc:
            raise SessionDiagnosticFrameError("capture_generation/capture_seq 必须是整数") from exc
        if generation < 0 or capture_sequence < 0:
            raise SessionDiagnosticFrameError("capture_generation/capture_seq 不能为负数")

        session, directory = self._diagnostic_directory(session_directory, create=True)
        with _directory_lock(directory):
            sequence = self._next_sequence(directory)
            frame = _attribute(snapshot, "frame", snapshot)
            image_value = _attribute(snapshot, "image", None)
            if image_value is None:
                image_value = _attribute(frame, "image", None)
            image = np.asarray(image_value) if image_value is not None else None
            if image is None or image.size == 0:
                raise SessionDiagnosticFrameError("诊断截图不包含有效图片")
            if image.dtype != np.uint8 or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
                raise SessionDiagnosticFrameError("诊断截图必须是 uint8 的灰度、BGR 或 BGRA 图片")
            image = np.ascontiguousarray(image)
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise SessionDiagnosticFrameError("诊断截图无法编码为 PNG")
            png_bytes = encoded.tobytes()
            decoded = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None or not np.array_equal(np.ascontiguousarray(decoded), image):
                raise SessionDiagnosticFrameError("PNG 往返后像素与 listener snapshot 不一致")

            nested_metadata = _attribute(snapshot, "metadata", {})
            if not isinstance(nested_metadata, Mapping):
                nested_metadata = {}
            captured_at = _first_attribute((snapshot, frame), "captured_at", None)
            if captured_at is None:
                captured_at = _first_attribute((snapshot, frame), "wall_time", None)
            monotonic_ms = _first_attribute((snapshot, frame), "captured_monotonic_ms", 0)
            evidence_id = _first_attribute((snapshot, frame), "evidence_frame_id", "")
            rect = _first_attribute((snapshot, frame), "rect", None)
            dpi = _first_attribute((snapshot, frame), "dpi", None)
            backend = _first_attribute((snapshot, frame), "backend", None)
            if backend is None:
                backend = nested_metadata.get("backend", "unknown")
            if rect is None:
                rect = nested_metadata.get("rect")
            if dpi is None:
                dpi = nested_metadata.get("dpi")
            height, width = image.shape[:2]
            metadata: dict[str, Any] = {
                **_json_safe(dict(nested_metadata)),
                "schema": FRAME_SCHEMA,
                "session_id": str(session_id),
                "sequence": sequence,
                "capture_generation": generation,
                "capture_seq": capture_sequence,
                "evidence_frame_id": str(evidence_id or ""),
                "captured_at": _as_iso_datetime(captured_at),
                "captured_monotonic_ms": int(monotonic_ms or 0),
                "backend": str(backend or "unknown"),
                "rect": _rect_payload(rect),
                "dpi": int(dpi) if dpi is not None else None,
                "width": int(width),
                "height": int(height),
                "dtype": str(image.dtype),
                "channels": int(image.shape[2]) if image.ndim == 3 else 1,
                "raw_sha256": _sha256_bytes(image.tobytes(order="C")),
                "png_sha256": _sha256_bytes(png_bytes),
                "source": source,
            }
            if source_phase is not None:
                metadata["source_phase"] = source_phase
            checked_metadata = _validate_metadata(metadata, sequence)
            stem = f"{sequence:06d}"
            image_path = directory / f"{stem}.png"
            metadata_path = directory / f"{stem}.json"
            _atomic_bytes(image_path, png_bytes)
            try:
                _atomic_json(metadata_path, checked_metadata)
            except Exception:
                # An orphan PNG is intentionally left harmless: list_frames
                # only exposes complete PNG/JSON pairs.
                raise
            return SessionDiagnosticFrame(sequence, image_path, metadata_path, session, checked_metadata)

    def list_frames(self, session_directory: Path | str) -> tuple[SessionDiagnosticFrame, ...]:
        """Return complete, well-formed frame pairs in sequence order."""

        _, directory = self._diagnostic_directory(session_directory, create=False)
        entries = tuple(directory.iterdir())
        for entry in entries:
            if _is_link_or_reparse(entry):
                raise SessionDiagnosticFrameError(f"诊断截图目录包含链接或重解析点：{entry}")
        stems: set[str] = set()
        for entry in entries:
            match = _FRAME_NAME.match(entry.name)
            if match:
                stems.add(match.group("sequence"))
        records: list[SessionDiagnosticFrame] = []
        for sequence_text in sorted(stems, key=int):
            stem = f"{int(sequence_text):06d}"
            image_path = directory / f"{stem}.png"
            metadata_path = directory / f"{stem}.json"
            if not image_path.is_file() or not metadata_path.is_file():
                continue
            try:
                records.append(self._record_from_paths(image_path, metadata_path))
            except SessionDiagnosticFrameError:
                # Partial writes and malformed sidecars are not a usable pair.
                continue
        return tuple(sorted(records, key=lambda item: item.sequence))

    def load_image(self, record: SessionDiagnosticFrame) -> np.ndarray:
        """Decode one frame and verify PNG, dimensions, and raw-pixel hashes."""

        if not isinstance(record, SessionDiagnosticFrame):
            raise SessionDiagnosticFrameError("load_image 需要 SessionDiagnosticFrame")
        current = self._record_from_paths(record.image_path, record.metadata_path)
        image_path = _assert_regular_file(current.image_path, field="诊断截图 PNG")
        png_bytes = image_path.read_bytes()
        expected_png = str(current.metadata["png_sha256"])
        if _sha256_bytes(png_bytes) != expected_png:
            raise SessionDiagnosticFrameError("诊断截图 PNG 哈希不匹配，文件可能已被篡改")
        image = _decode_png(image_path)
        expected_shape = (int(current.metadata["height"]), int(current.metadata["width"]))
        if image.shape[:2] != expected_shape:
            raise SessionDiagnosticFrameError("诊断截图尺寸与 metadata 不匹配")
        expected_channels = int(current.metadata.get("channels", 1))
        actual_channels = int(image.shape[2]) if image.ndim == 3 else 1
        if actual_channels != expected_channels:
            raise SessionDiagnosticFrameError("诊断截图通道数与 metadata 不匹配")
        raw = np.ascontiguousarray(image)
        if _sha256_bytes(raw.tobytes(order="C")) != str(current.metadata["raw_sha256"]):
            raise SessionDiagnosticFrameError("诊断截图 raw_sha256 不匹配，像素可能已被篡改")
        return raw


__all__ = [
    "FRAME_SCHEMA",
    "SOURCE_KIND",
    "MANUAL_SOURCE_KIND",
    "SessionDiagnosticFrame",
    "SessionDiagnosticFrameError",
    "SessionDiagnosticFrameStore",
]
