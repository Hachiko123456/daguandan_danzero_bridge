from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from time import monotonic_ns
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from ..domain.recording import (
    IncidentMedia,
    IncidentMediaFailure,
    RecorderWarning,
    RecordingResult,
)
from ..image_io import save_image_unicode
from ..storage import append_json_line, atomic_write_json
from .pipeline_timing import PipelineTiming


RECORDING_INTEGRITY_SCHEMA = "guandan.recording-integrity/1"


def audit_recording_integrity(
    video_path: Path,
    index_path: Path,
    *,
    writer_frame_count: int | None = None,
    _include_last_decodable_frame: bool = False,
) -> dict[str, object]:
    """Actually decode a sealed video; container metadata alone is insufficient."""

    video = Path(video_path)
    index = Path(index_path)
    indexed_count = 0
    index_error = ""
    if index.is_file():
        try:
            with index.open("r", encoding="utf-8") as handle:
                indexed_count = sum(1 for line in handle if line.strip())
        except (OSError, UnicodeError) as exc:
            index_error = str(exc)
    capture = cv2.VideoCapture(str(video))
    decoded_count = 0
    opened = bool(capture.isOpened())
    decode_error = ""
    last_decodable_frame_sha256: str | None = None
    last_decodable_frame: np.ndarray | None = None
    try:
        if opened:
            while True:
                try:
                    ok, frame = capture.read()
                except cv2.error as exc:
                    decode_error = str(exc)
                    break
                if not ok or frame is None:
                    break
                decoded_count += 1
                last_decodable_frame = frame
    finally:
        capture.release()
    if last_decodable_frame is not None:
        last_decodable_frame_sha256 = _frame_sha256(last_decodable_frame)
    expected = int(writer_frame_count) if writer_frame_count is not None else indexed_count
    issues: list[str] = []
    if index_error:
        issues.append("frame_index_unreadable")
    if not video.is_file() or not opened:
        issues.append("video_unreadable")
    if indexed_count != expected:
        issues.append("writer_index_count_mismatch")
    if decoded_count != indexed_count:
        issues.append("indexed_decodable_count_mismatch")
    report: dict[str, object] = {
        "schema": RECORDING_INTEGRITY_SCHEMA,
        "status": "FAIL" if issues else "PASS",
        "video_path": str(video),
        "frame_index_path": str(index),
        "writer_frame_count": expected,
        "indexed_frame_count": indexed_count,
        "decodable_frame_count": decoded_count,
        "last_decodable_frame_index": decoded_count - 1 if decoded_count else None,
        "last_decodable_frame_sha256": last_decodable_frame_sha256,
        "issues": issues,
        "index_error": index_error,
        "decode_error": decode_error,
        # OpenCV exposes no packet offsets or corruption position.  Keep this
        # explicit so callers cannot infer that a short decode is a proven tail
        # loss merely from the frame counts.
        "tail_proof": {
            "status": "unavailable",
            "reason": "decoder_does_not_expose_authoritative_packet_offsets",
        },
    }
    if _include_last_decodable_frame:
        # This private value is consumed by SessionRecorder._recover_tail and
        # removed before the integrity document is persisted as JSON.
        report["_last_decodable_frame"] = last_decodable_frame
    return report


@dataclass
class _PendingIncidentMedia:
    directory: Path
    trigger_ms: int
    deadline_ms: int
    frames: list[tuple[int, np.ndarray]]


def _frame_sha256(frame: np.ndarray) -> str:
    """Hash the decoded pixel bytes, making recovery evidence auditable."""

    contiguous = np.ascontiguousarray(frame)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _frames_match(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare codec roundtrips with a tiny tolerance for AVI finalization."""

    if left.shape != right.shape:
        return False
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
    # A valid boundary frame differs by at most codec/finalization noise.  A
    # neighboring frame normally has sparse large differences even when the
    # scene is visually similar; the max guard prevents accepting that case.
    return bool(float(difference.mean()) <= 2.0 and int(difference.max()) <= 32)


def _has_authoritative_tail_proof(
    integrity: dict[str, object], *, decoded_count: int, indexed_count: int
) -> bool:
    """Accept RECOVERED only from an explicit packet/offset proof.

    The current OpenCV backend intentionally never emits this proof.  It is a
    narrow extension point for a future recorder that can persist authoritative
    packet boundaries; frame counts and ring-buffer membership alone are not
    sufficient because the same observation is possible after middle damage.
    """

    proof = integrity.get("tail_proof")
    if not isinstance(proof, dict):
        return False
    return bool(
        proof.get("status") == "verified"
        and proof.get("kind") == "authoritative_packet_sequence"
        and proof.get("all_prior_frames_verified") is True
        and int(proof.get("first_missing_frame_index", -1)) == decoded_count
        and int(proof.get("last_decodable_frame_index", -1)) == decoded_count - 1
        and int(proof.get("indexed_frame_count", -1)) == indexed_count
    )


class SessionRecorder:
    """Record standardized frames and their original capture timestamps."""

    def __init__(
        self,
        session_directory: Path,
        *,
        size: tuple[int, int],
        fps: float,
        buffer_seconds: float = 10.0,
        codec: str = "MJPG",
        max_video_bytes: int | None = None,
    ) -> None:
        if max_video_bytes is not None and (
            isinstance(max_video_bytes, bool)
            or not isinstance(max_video_bytes, int)
            or max_video_bytes < 0
        ):
            raise ValueError("录像容量必须为非负整数")
        self.session_directory = Path(session_directory)
        self.video_directory = self.session_directory / "video"
        self.video_directory.mkdir(parents=True, exist_ok=True)
        self.video_path = self.video_directory / "game.avi"
        self.index_path = self.video_directory / "frame_index.jsonl"
        self.index_path.touch(exist_ok=False)
        self.size = (int(size[0]), int(size[1]))
        self.fps = float(fps)
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError("录像尺寸必须为正数")
        if self.fps <= 0:
            raise ValueError("录像帧率必须为正数")
        if len(codec) != 4:
            raise ValueError("视频编码 FourCC 必须是四个字符")
        self.codec = codec
        self.max_video_bytes = int(max_video_bytes) if max_video_bytes is not None else None
        self._capacity_reached = False
        self._media_budget_lock = RLock()
        self._incident_budget_used = 0
        self._video_accounted_bytes = 65536
        # Do not create an AVI at all if the existing profile has used its quota.
        self._writer = (
            self._open_writer(self.video_path)
            if self.max_video_bytes is None or self.max_video_bytes >= 65536
            else None
        )
        buffer_frames = max(1, int(math.ceil(self.fps * buffer_seconds)))
        self._buffer: deque[tuple[int, int, np.ndarray]] = deque(maxlen=buffer_frames)
        self._frame_count = 0
        self._dropped_frames = 0
        self._pending_drops = 0
        self._closed = False
        self._lock = RLock()
        self._pending_incidents: list[_PendingIncidentMedia] = []
        self._incident_media_failures: list[IncidentMediaFailure] = []
        self._media_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="incident-media",
        )
        self._media_futures: list[
            tuple[Path, int, Future[IncidentMedia]]
        ] = []
        self._recording_integrity: dict[str, object] | None = None
        self.pipeline_timing = PipelineTiming()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def _open_writer(self, path: Path) -> cv2.VideoWriter:
        args = (str(path), cv2.VideoWriter_fourcc(*self.codec), self.fps, self.size)
        if self.max_video_bytes is not None and self.codec == "MJPG":
            # The built-in MJPEG backend reports encoded frame bytes. FFmpeg's
            # buffered file length alone can lag by hundreds of kilobytes.
            writer = cv2.VideoWriter(args[0], cv2.CAP_OPENCV_MJPEG, *args[1:])
        else:
            writer = cv2.VideoWriter(*args)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"无法打开录像编码器：{self.codec} -> {path}")
        return writer

    def write_frame(
        self,
        frame: np.ndarray,
        captured_monotonic_ms: int,
        wall_time: str,
    ) -> RecorderWarning | None:
        started_ns = monotonic_ns()
        with self._lock, self._media_budget_lock:
            self.pipeline_timing.elapsed("video_lock_wait", started_ns)
            self._ensure_open()
            expected_shape = (self.size[1], self.size[0], 3)
            if not isinstance(frame, np.ndarray) or frame.shape != expected_shape:
                return self._drop(
                    "frame_size_mismatch",
                    captured_monotonic_ms,
                    f"expected={expected_shape}, actual={getattr(frame, 'shape', None)}",
                )
            if frame.dtype != np.uint8:
                return self._drop(
                    "frame_dtype_mismatch",
                    captured_monotonic_ms,
                    f"expected=uint8, actual={frame.dtype}",
                )
            if self._capacity_reached:
                self._dropped_frames += 1
                return None
            if self.max_video_bytes is not None:
                current_bytes = max(
                    self._video_accounted_bytes,
                    self.video_path.stat().st_size if self.video_path.exists() else 0,
                )
                # Reserve an uncompressed-frame upper envelope plus AVI index
                # and codec buffer headroom before admitting another frame.
                reserve = frame.nbytes * 2 + 4096
                if (self._writer is None or current_bytes + reserve
                        + self._incident_budget_used > self.max_video_bytes):
                    self._capacity_reached = True
                    if self._writer is not None:
                        self._writer.release()
                        self._writer = None
                    self._buffer.clear()
                    self._pending_incidents.clear()
                    return self._drop(
                        "recording_capacity_reached", captured_monotonic_ms,
                        "录像容量已达上限，已停止录像；识别和推荐继续。"
                        f" allowance_bytes={self.max_video_bytes}",
                    )
            try:
                assert self._writer is not None
                encode_started_ns = monotonic_ns()
                try:
                    self._writer.write(frame)
                finally:
                    self.pipeline_timing.elapsed("video_encode", encode_started_ns)
                if self.max_video_bytes is not None:
                    try:
                        encoded = float(self._writer.get(cv2.VIDEOWRITER_PROP_FRAMEBYTES))
                    except (AttributeError, TypeError, ValueError, cv2.error):
                        encoded = 0.0
                    self._video_accounted_bytes += (
                        int(encoded) + 32 if math.isfinite(encoded) and encoded > 0
                        else frame.nbytes * 2 + 4096
                    )
            except cv2.error as exc:
                return self._drop("video_encode_failed", captured_monotonic_ms, str(exc))
            record = {
                "frame_index": self._frame_count,
                "monotonic_ms": int(captured_monotonic_ms),
                "wall_time": str(wall_time),
                "dropped_before": self._pending_drops,
            }
            index_started_ns = monotonic_ns()
            try:
                append_json_line(self.index_path, record)
            finally:
                self.pipeline_timing.elapsed("video_index_write", index_started_ns)
            buffered_frame = frame.copy()
            self._buffer.append(
                (self._frame_count, int(captured_monotonic_ms), buffered_frame)
            )
            self._advance_pending_incidents(
                int(captured_monotonic_ms),
                buffered_frame,
            )
            self._frame_count += 1
            self.pipeline_timing.increment("video_frames_written")
            self._pending_drops = 0
            return None

    def _drop(self, reason: str, monotonic_ms: int, details: str) -> RecorderWarning:
        self._dropped_frames += 1
        self._pending_drops += 1
        return RecorderWarning(reason, int(monotonic_ms), details)

    def save_evidence_frame(self, path: Path, frame: np.ndarray) -> Path:
        destination = Path(path)
        if destination.suffix.lower() != ".png":
            destination = destination.with_suffix(".png")
        if self.max_video_bytes is None:
            save_image_unicode(destination, frame)
            return destination
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("evidence PNG encoding failed")
        content = bytes(encoded)
        # Do not acquire _lock here: close() drains an incident worker while
        # holding that lock. Incident workers use their enclosing reservation.
        with self._media_budget_lock:
            self._ensure_open()
            video_bytes = max(self._video_accounted_bytes,
                self.video_path.stat().st_size if self.video_path.exists() else 0)
            if (self._capacity_reached or video_bytes + self._incident_budget_used
                    + len(content) > self.max_video_bytes):
                raise RuntimeError("recording_capacity_reached: evidence PNG refused")
            self._incident_budget_used += len(content)
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            finally:
                actual = destination.stat().st_size if destination.is_file() else 0
                self._incident_budget_used += actual - len(content)
        return destination

    def save_incident_media(
        self,
        incident_directory: Path,
        *,
        trigger_ms: int,
        before_ms: int = 5_000,
        after_ms: int = 5_000,
    ) -> IncidentMedia:
        with self._lock:
            self._ensure_open()
            if self._capacity_reached:
                raise RuntimeError("recording_capacity_reached: incident media disabled")
            incident_directory = Path(incident_directory)
            incident_directory.mkdir(parents=True, exist_ok=True)
            selected = [
                (timestamp, frame)
                for _frame_index, timestamp, frame in self._buffer
                if trigger_ms - before_ms <= timestamp <= trigger_ms + after_ms
            ]
            if not selected:
                raise RuntimeError("环形缓冲中没有事故时间范围内的帧")
            return self._write_incident_media(
                incident_directory,
                int(trigger_ms),
                selected,
            )

    def schedule_incident_media(
        self,
        incident_directory: Path,
        *,
        trigger_ms: int,
        before_ms: int = 5_000,
        after_ms: int = 5_000,
    ) -> None:
        """Keep pre-trigger frames now and finalize after future frames arrive."""

        with self._lock:
            self._ensure_open()
            if self._capacity_reached:
                return
            directory = Path(incident_directory)
            directory.mkdir(parents=True, exist_ok=True)
            selected = [
                (timestamp, frame)
                for _frame_index, timestamp, frame in self._buffer
                if trigger_ms - before_ms <= timestamp <= trigger_ms
            ]
            pending = _PendingIncidentMedia(
                directory=directory,
                trigger_ms=int(trigger_ms),
                deadline_ms=int(trigger_ms + max(0, after_ms)),
                frames=selected,
            )
            if after_ms <= 0:
                if selected:
                    self._write_incident_media(directory, int(trigger_ms), selected)
                return
            self._pending_incidents.append(pending)

    def _advance_pending_incidents(
        self,
        monotonic_ms: int,
        frame: np.ndarray,
    ) -> None:
        completed: list[_PendingIncidentMedia] = []
        for pending in self._pending_incidents:
            if pending.trigger_ms < monotonic_ms <= pending.deadline_ms:
                pending.frames.append((monotonic_ms, frame))
            if monotonic_ms >= pending.deadline_ms:
                completed.append(pending)
        for pending in completed:
            self._submit_pending_incident(pending)
            self._pending_incidents.remove(pending)

    def _submit_pending_incident(self, pending: _PendingIncidentMedia) -> None:
        if not pending.frames:
            return
        self._media_futures.append(
            (
                pending.directory,
                pending.trigger_ms,
                self._media_executor.submit(
                    self._write_incident_media,
                    pending.directory,
                    pending.trigger_ms,
                    pending.frames,
                ),
            )
        )

    def _collect_media_failures(self) -> None:
        for directory, trigger_ms, future in self._media_futures:
            try:
                future.result()
            except Exception as exc:
                failure = IncidentMediaFailure(
                    reason="incident_media_finalize_failed",
                    incident_directory=directory,
                    trigger_ms=trigger_ms,
                    details=str(exc),
                )
                self._incident_media_failures.append(failure)
                try:
                    atomic_write_json(
                        directory / "media_error.json",
                        {"schema_version": 1, **failure.to_dict()},
                    )
                except Exception:
                    # The in-memory failure is still returned to the manifest.
                    pass
        self._media_futures.clear()

    def _write_incident_media(
        self,
        incident_directory: Path,
        trigger_ms: int,
        selected: list[tuple[int, np.ndarray]],
    ) -> IncidentMedia:
        reserved = 0
        if self.max_video_bytes is not None:
            reserved = sum(frame.nbytes for _, frame in selected) * 6 + len(selected) * 4096 + 131072
            with self._media_budget_lock:
                video_bytes = max(self._video_accounted_bytes,
                    self.video_path.stat().st_size if self.video_path.exists() else 0)
                headroom = self.size[0] * self.size[1] * 6 + 4096
                if (self._capacity_reached or video_bytes + headroom + reserved
                        + self._incident_budget_used > self.max_video_bytes):
                    raise RuntimeError("recording_capacity_reached: incident media refused")
                self._incident_budget_used += reserved
        try:
            return self._write_incident_media_payload(incident_directory, trigger_ms, selected)
        finally:
            if reserved:
                with self._media_budget_lock:
                    paths = (incident_directory / "clip.avi",
                             incident_directory / "frames" / "trigger.png",
                             incident_directory / "contact_sheet.png")
                    actual = sum(path.stat().st_size for path in paths if path.is_file())
                    self._incident_budget_used += actual - reserved

    def _write_incident_media_payload(
        self, incident_directory: Path, trigger_ms: int,
        selected: list[tuple[int, np.ndarray]],
    ) -> IncidentMedia:
        clip_path = incident_directory / "clip.avi"
        clip_writer = self._open_writer(clip_path)
        try:
            for _, frame in selected:
                clip_writer.write(frame)
        finally:
            clip_writer.release()
        _, trigger_frame = min(
            selected, key=lambda item: abs(item[0] - int(trigger_ms))
        )
        frames_directory = incident_directory / "frames"
        # The caller has reserved the complete clip+PNG envelope already.
        trigger_path = frames_directory / "trigger.png"
        save_image_unicode(trigger_path, trigger_frame)
        contact_sheet_path = incident_directory / "contact_sheet.png"
        save_image_unicode(
            contact_sheet_path,
            self._make_contact_sheet([frame for _, frame in selected]),
        )
        atomic_write_json(
            incident_directory / "media.json",
            {
                "schema_version": 1,
                "trigger_ms": int(trigger_ms),
                "first_frame_ms": int(selected[0][0]),
                "last_frame_ms": int(selected[-1][0]),
                "frame_count": len(selected),
                "clip": clip_path.relative_to(incident_directory).as_posix(),
                "contact_sheet": contact_sheet_path.relative_to(
                    incident_directory
                ).as_posix(),
                "trigger_frame": trigger_path.relative_to(
                    incident_directory
                ).as_posix(),
            },
        )
        return IncidentMedia(
            clip_path=clip_path,
            contact_sheet_path=contact_sheet_path,
            trigger_frame_path=trigger_path,
            frame_count=len(selected),
        )

    def _make_contact_sheet(self, frames: list[np.ndarray]) -> np.ndarray:
        sample_count = min(9, len(frames))
        indexes = np.linspace(0, len(frames) - 1, sample_count, dtype=int)
        samples = [frames[int(index)] for index in indexes]
        columns = min(3, sample_count)
        rows = int(math.ceil(sample_count / columns))
        tile_width = min(self.size[0], 480)
        tile_height = max(1, int(round(tile_width * self.size[1] / self.size[0])))
        canvas = np.zeros(
            (rows * tile_height, columns * tile_width, 3), dtype=np.uint8
        )
        for index, frame in enumerate(samples):
            tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            row, column = divmod(index, columns)
            canvas[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = tile
        return canvas

    def close(self) -> RecordingResult:
        with self._lock:
            if not self._closed:
                try:
                    if self._writer is not None:
                        self._writer.release()
                except Exception as exc:
                    self._incident_media_failures.append(
                        IncidentMediaFailure(
                            reason="main_video_release_failed",
                            incident_directory=self.video_directory,
                            trigger_ms=-1,
                            details=str(exc),
                        )
                    )
                finally:
                    self._closed = True
                for pending in tuple(self._pending_incidents):
                    self._submit_pending_incident(pending)
                self._pending_incidents.clear()
                self._media_executor.shutdown(wait=True, cancel_futures=False)
                self._collect_media_failures()
            if self._recording_integrity is None:
                try:
                    try:
                        integrity = audit_recording_integrity(
                            self.video_path,
                            self.index_path,
                            writer_frame_count=self._frame_count,
                            _include_last_decodable_frame=True,
                        )
                    except TypeError:
                        # Keep compatibility with test doubles and older
                        # adapters that still expose the pre-recovery call
                        # signature; hash-only boundary checking remains safe.
                        integrity = audit_recording_integrity(
                            self.video_path,
                            self.index_path,
                            writer_frame_count=self._frame_count,
                        )
                except Exception as exc:
                    integrity = {
                        "schema": RECORDING_INTEGRITY_SCHEMA,
                        "status": "FAIL",
                        "recording_state": "partial",
                        "video_path": str(self.video_path),
                        "frame_index_path": str(self.index_path),
                        "writer_frame_count": self._frame_count,
                        "indexed_frame_count": None,
                        "decodable_frame_count": None,
                        "last_decodable_frame_index": None,
                        "last_decodable_frame_sha256": None,
                        "issues": ["integrity_audit_failed"],
                        "error": str(exc),
                    }
                self._recording_integrity = self._recover_tail(integrity)
                if self._capacity_reached:
                    self._recording_integrity.update({
                        "status": "PARTIAL", "recording_state": "partial",
                        "stop_reason": "recording_capacity_reached",
                        "max_video_bytes": self.max_video_bytes,
                        "omitted_capture_frames": self._dropped_frames,
                    })
                    self._recording_integrity["issues"] = list(dict.fromkeys([
                        *self._recording_integrity.get("issues", []),
                        "recording_capacity_reached",
                    ]))
            integrity = dict(self._recording_integrity)
            return RecordingResult(
                video_path=self.video_path,
                index_path=self.index_path,
                frame_count=self._frame_count,
                dropped_frames=self._dropped_frames,
                incident_media_failures=tuple(self._incident_media_failures),
                integrity=integrity,
            )

    def _recover_tail(self, integrity: dict[str, object]) -> dict[str, object]:
        if self.max_video_bytes is None:
            return self._recover_tail_payload(integrity)
        indexed = int(integrity.get("indexed_frame_count", 0) or 0)
        decoded = int(integrity.get("decodable_frame_count", 0) or 0)
        if indexed <= decoded:
            return self._recover_tail_payload(integrity)
        selected = [frame for index, _timestamp, frame in self._buffer if index >= decoded - 1]
        reserved = sum(frame.nbytes * 2 + 4096 for frame in selected) + 65536
        with self._media_budget_lock:
            video_bytes = max(self._video_accounted_bytes,
                self.video_path.stat().st_size if self.video_path.exists() else 0)
            if video_bytes + self._incident_budget_used + reserved > self.max_video_bytes:
                result = dict(integrity)
                result.pop("_last_decodable_frame", None)
                result.update(status="PARTIAL", recording_state="partial",
                    missing_frame_count=indexed - decoded,
                    tail_recovery={"status": "unavailable", "reason": "recording_capacity_reached"})
                return result
            self._incident_budget_used += reserved
        try:
            return self._recover_tail_payload(integrity)
        finally:
            recovery = self.video_directory / "tail_recovery"
            actual = sum(path.stat().st_size for path in recovery.glob("frames/*.png") if path.is_file())
            with self._media_budget_lock:
                self._incident_budget_used += actual - reserved

    def _recover_tail_payload(self, integrity: dict[str, object]) -> dict[str, object]:
        """Persist recoverable tail frames without pretending the AVI is repaired.

        OpenCV stops at the first unreadable MJPEG packet, so a short decode is
        only safely classifiable as a tail loss when an authoritative packet
        sequence proof *and* a same-codec boundary roundtrip agree.  The current
        OpenCV backend emits no such proof, so its candidate frames remain
        PARTIAL.  Any ambiguity, including a loss larger than the ring buffer,
        remains PARTIAL.
        """

        result = dict(integrity)
        decoded_frame = result.pop("_last_decodable_frame", None)
        indexed_count = int(result.get("indexed_frame_count", 0) or 0)
        decoded_count = int(result.get("decodable_frame_count", 0) or 0)
        missing_count = max(0, indexed_count - decoded_count)
        result["missing_frame_count"] = missing_count
        if missing_count <= 0:
            result["recording_state"] = "pass" if result.get("status") == "PASS" else "partial"
            return result

        result["tail_recovery"] = {
            "status": "not_attempted",
            "reason": "not_a_tail_mismatch",
        }
        if not self._buffer or decoded_count <= 0:
            result["status"] = "PARTIAL"
            result["recording_state"] = "partial"
            result["tail_recovery"] = {
                "status": "unavailable",
                "reason": "ring_buffer_empty_or_no_decodable_prefix",
            }
            return result

        buffered = {index: (timestamp, frame) for index, timestamp, frame in self._buffer}
        expected_missing = tuple(range(decoded_count, indexed_count))
        available = tuple(index for index in expected_missing if index in buffered)
        boundary = buffered.get(decoded_count - 1)
        last_hash = result.get("last_decodable_frame_sha256")
        boundary_matches = False
        if boundary is not None and isinstance(decoded_frame, np.ndarray):
            try:
                roundtrip = self._codec_roundtrip(boundary[1])
            except Exception:
                roundtrip = None
            boundary_matches = bool(
                isinstance(roundtrip, np.ndarray)
                and _frames_match(decoded_frame, roundtrip)
            )
        elif boundary is not None and last_hash:
            # Keeps deterministic monkeypatched/unit-test audits useful.  A
            # production OpenCV decode normally takes the stronger roundtrip
            # path above because MJPG is lossy at the pixel level.
            boundary_matches = _frame_sha256(boundary[1]) == last_hash
        authoritative_tail_proof = _has_authoritative_tail_proof(
            result,
            decoded_count=decoded_count,
            indexed_count=indexed_count,
        )
        is_safe_tail = (
            len(available) == len(expected_missing)
            and boundary_matches
            and authoritative_tail_proof
            and not result.get("index_error")
            and "video_unreadable" not in set(result.get("issues", ()) or ())
        )
        if not is_safe_tail:
            result["status"] = "PARTIAL"
            result["recording_state"] = "partial"
            if len(available) == len(expected_missing):
                records = self._read_index_records()
                try:
                    result["tail_recovery"] = self._write_tail_recovery(
                        tuple(
                            (
                                index,
                                records.get(index),
                                buffered[index][0],
                                buffered[index][1],
                            )
                            for index in expected_missing
                        ),
                        indexed_count=indexed_count,
                        decoded_count=decoded_count,
                        status="PARTIAL",
                        reason=(
                            "tail_only_not_proven"
                            if not authoritative_tail_proof
                            else "decode_boundary_not_proven_to_be_tail"
                        ),
                        candidate=True,
                    )
                except Exception as exc:
                    result["tail_recovery"] = {
                        "status": "failed",
                        "reason": "tail_recovery_write_failed",
                        "error": str(exc),
                        "missing_frame_count": missing_count,
                    }
            else:
                result["tail_recovery"] = {
                    "status": "unavailable",
                    "reason": "missing_frames_not_fully_in_ring_buffer",
                    "missing_frame_count": missing_count,
                    "available_frame_count": len(available),
                    "ring_buffer_frame_count": len(buffered),
                }
            return result

        records = self._read_index_records()
        try:
            recovery = self._write_tail_recovery(
                tuple(
                    (index, records.get(index), buffered[index][0], buffered[index][1])
                    for index in expected_missing
                ),
                indexed_count=indexed_count,
                decoded_count=decoded_count,
            )
        except Exception as exc:
            result["status"] = "PARTIAL"
            result["recording_state"] = "partial"
            result["tail_recovery"] = {
                "status": "failed",
                "reason": "tail_recovery_write_failed",
                "error": str(exc),
                "missing_frame_count": missing_count,
            }
            return result

        result["status"] = "RECOVERED"
        result["recording_state"] = "recovered"
        result["tail_recovery"] = recovery
        result["recovered_frame_count"] = missing_count
        return result

    def _codec_roundtrip(self, frame: np.ndarray) -> np.ndarray | None:
        """Decode one frame using the same AVI writer settings for comparison."""

        with tempfile.TemporaryDirectory(
            prefix=".tail-boundary-",
            dir=str(self.video_directory),
        ) as directory:
            path = Path(directory) / "frame.avi"
            writer = self._open_writer(path)
            try:
                writer.write(frame)
            finally:
                writer.release()
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    return None
                ok, decoded = capture.read()
                return decoded.copy() if ok and decoded is not None else None
            finally:
                capture.release()

    def _read_index_records(self) -> dict[int, dict[str, object]]:
        records: dict[int, dict[str, object]] = {}
        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if isinstance(raw, dict):
                        records[int(raw["frame_index"])] = raw
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            return {}
        return records

    def _write_tail_recovery(
        self,
        frames: tuple[
            tuple[int, dict[str, object] | None, int, np.ndarray], ...
        ],
        *,
        indexed_count: int,
        decoded_count: int,
        status: str = "RECOVERED",
        reason: str = "indexed_tail_not_decodable_from_sealed_video",
        candidate: bool = False,
    ) -> dict[str, object]:
        recovery = self.video_directory / "tail_recovery"
        staging = self.video_directory / f".tail_recovery.{uuid4().hex}.tmp"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            frame_directory = staging / "frames"
            frame_directory.mkdir()
            index_lines: list[dict[str, object]] = []
            hash_lines: list[dict[str, object]] = []
            for frame_index, record, timestamp, frame in frames:
                filename = f"frame_{frame_index:08d}.png"
                save_image_unicode(frame_directory / filename, frame)
                digest = _frame_sha256(frame)
                entry: dict[str, object] = {
                    "frame_index": frame_index,
                    "monotonic_ms": timestamp,
                    "source": "session_ring_buffer",
                    "frame": f"frames/{filename}",
                    "sha256": digest,
                }
                if record:
                    entry["wall_time"] = record.get("wall_time")
                    entry["dropped_before"] = record.get("dropped_before", 0)
                index_lines.append(entry)
                hash_lines.append(
                    {
                        "frame_index": frame_index,
                        "sha256": digest,
                        "encoding": "decoded_bgr_pixel_bytes",
                    }
                )
            (staging / "index.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in index_lines),
                encoding="utf-8",
            )
            (staging / "hashes.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in hash_lines),
                encoding="utf-8",
            )
            manifest = {
                "schema": "guandan.tail-recovery/1",
                "status": status,
                "reason": reason,
                "candidate": bool(candidate),
                "indexed_frame_count": indexed_count,
                "decodable_frame_count": decoded_count,
                "recovered_frame_count": len(frames),
                "first_recovered_frame_index": frames[0][0],
                "last_recovered_frame_index": frames[-1][0],
                "index": "index.jsonl",
                "hashes": "hashes.jsonl",
                "frames": "frames",
            }
            atomic_write_json(staging / "manifest.json", manifest)
            if recovery.exists():
                # close() is idempotent, but retain the first immutable recovery
                # package if a caller manually left one behind.
                shutil.rmtree(staging, ignore_errors=True)
                return {
                    **manifest,
                    "path": str(recovery),
                    "already_present": True,
                }
            os.replace(staging, recovery)
            return {**manifest, "path": str(recovery)}
        except Exception:
            # The staging directory is intentionally not exposed as a valid
            # recovery package when any member failed to write.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("录像器已经关闭")

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class InMemorySessionRecorder:
    """RecordingPort that keeps live processing metrics but never writes media."""

    def __init__(self, session_directory: Path) -> None:
        self.session_directory = Path(session_directory)
        self._frame_count = 0
        self._closed = False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return 0

    def write_frame(
        self,
        frame: np.ndarray,
        captured_monotonic_ms: int,
        wall_time: str,
    ) -> RecorderWarning | None:
        del frame, captured_monotonic_ms, wall_time
        if self._closed:
            raise RuntimeError("录像器已经关闭")
        self._frame_count += 1
        return None

    def save_evidence_frame(self, path: Path, frame: np.ndarray) -> Path:
        del frame
        return Path(path)

    def schedule_incident_media(
        self,
        incident_directory: Path,
        *,
        trigger_ms: int,
        before_ms: int = 5_000,
        after_ms: int = 5_000,
    ) -> None:
        del incident_directory, trigger_ms, before_ms, after_ms
        if self._closed:
            raise RuntimeError("录像器已经关闭")

    def close(self) -> RecordingResult:
        self._closed = True
        return RecordingResult(
            video_path=self.session_directory / "video" / "game.avi",
            index_path=self.session_directory / "video" / "frame_index.jsonl",
            frame_count=self._frame_count,
            dropped_frames=0,
            integrity={
                "schema": RECORDING_INTEGRITY_SCHEMA,
                "status": "NOT_RECORDED",
                "recording_state": "not_recorded",
                "writer_frame_count": self._frame_count,
                "indexed_frame_count": 0,
                "decodable_frame_count": 0,
                "issues": [],
            },
        )
