"""Parent-side RecordingPort proxy for one killable recorder child process."""
from __future__ import annotations

import multiprocessing
from pathlib import Path
from threading import RLock
from typing import Any

from ..application.recording_process_protocol import (
    RecordingCommand, RecordingOperation, RecordingProcessConfig,
    RecordingReady, RecordingResponse,
)
from ..domain.recording import RecorderWarning, RecordingResult
from ..live.pipeline_timing import PipelineTiming
from .recording_process_forensics import indexed_frame_count, stable_recording_files
from .recording_process import recording_process_bootstrap


class RecordingProcessError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ProcessSessionRecorder:
    """Own no media handles; every AVI/index mutation happens in its child."""

    def __init__(
        self,
        session_directory: Path,
        *,
        size: tuple[int, int],
        fps: float,
        codec: str = "MJPG",
        max_video_bytes: int | None = None,
        startup_timeout: float = 10.0,
        write_timeout: float = 1.5,
        close_timeout: float = 30.0,
        worker_module: str = "daguandan_bridge.infrastructure.recording_process_worker",
        worker_name: str = "run_session_recorder_worker",
        worker_args: tuple[object, ...] = (),
    ) -> None:
        self.session_directory = Path(session_directory)
        self.video_path = self.session_directory / "video" / "game.avi"
        self.index_path = self.session_directory / "video" / "frame_index.jsonl"
        self.size, self.fps, self.codec = tuple(size), float(fps), str(codec)
        self.max_video_bytes = max_video_bytes
        self.pipeline_timing = PipelineTiming()
        self._write_timeout = max(0.01, float(write_timeout))
        self._close_timeout = max(0.01, float(close_timeout))
        self._lock = RLock()
        self._sequence = self._submitted = self._frame_count = self._warning_drops = 0
        self._failure: tuple[str, str] | None = None
        self._result: RecordingResult | None = None
        self._worker_pid: int | None = None
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        config = RecordingProcessConfig(
            str(self.session_directory), self.size, self.fps, self.codec, max_video_bytes,
        )
        process = context.Process(
            target=recording_process_bootstrap,
            args=(config, child, worker_module, worker_name, worker_args),
            name=f"recording-{self.session_directory.name}",
            daemon=False,
        )
        self._connection, self._process = parent, process
        process.start()
        child.close()
        try:
            ready = self._receive(
                max(0.01, startup_timeout), timeout_code="recording_start_timeout"
            )
            if not isinstance(ready, RecordingReady):
                raise RecordingProcessError("recording_start_failed", _response_message(ready))
            self._worker_pid = ready.pid
        except BaseException:
            self._stop_process(grace=0.0)
            raise

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dropped_frames(self) -> int:
        return max(self._warning_drops, self._submitted - self._frame_count)

    @property
    def worker_pid(self) -> int | None:
        return self._worker_pid

    @property
    def worker_alive(self) -> bool:
        return bool(self._process and self._process.is_alive())

    def write_frame(self, frame: Any, monotonic_ms: int, wall_time: str) -> RecorderWarning | None:
        with self._lock:
            self._ensure_active()
            self._submitted += 1
            try:
                value = self._request(
                    RecordingOperation.WRITE, (frame, int(monotonic_ms), str(wall_time)),
                    timeout=self._write_timeout, timeout_code="recording_write_timeout",
                )
            except RecordingProcessError as exc:
                self._break(exc.code, str(exc))
                raise
            if isinstance(value, RecorderWarning):
                self._warning_drops += 1
                return value
            self._frame_count += 1
            return None

    def save_evidence_frame(self, path: Path, frame: Any) -> Path:
        with self._lock:
            try:
                return Path(self._request(
                    RecordingOperation.SAVE_EVIDENCE, (str(path), frame),
                    timeout=self._write_timeout,
                    timeout_code="recording_evidence_timeout",
                ))
            except RecordingProcessError as exc:
                self._break(exc.code, str(exc)); raise

    def schedule_incident_media(
        self, directory: Path, *, trigger_ms: int,
        before_ms: int = 5_000, after_ms: int = 5_000,
    ) -> None:
        with self._lock:
            try:
                self._request(
                    RecordingOperation.SCHEDULE_INCIDENT,
                    (str(directory), int(trigger_ms), int(before_ms), int(after_ms)),
                    timeout=self._write_timeout,
                    timeout_code="recording_incident_timeout",
                )
            except RecordingProcessError as exc:
                self._break(exc.code, str(exc)); raise

    def close(self) -> RecordingResult:
        with self._lock:
            if self._result is not None:
                return self._result
            if self._failure is None and self.worker_alive:
                try:
                    result = self._request(
                        RecordingOperation.CLOSE, None, timeout=self._close_timeout,
                        timeout_code="recording_close_timeout",
                    )
                    if not isinstance(result, RecordingResult):
                        raise RecordingProcessError("recording_close_invalid", "child returned no RecordingResult")
                    if not self._join_after_close():
                        raise RecordingProcessError("recording_close_timeout", "recorder child did not exit")
                    self._result = result
                    self._close_connection()
                    return result
                except RecordingProcessError as exc:
                    self._failure = exc.code, str(exc)
            self._stop_process(grace=0.0)
            self._result = self._aborted_result()
            return self._result

    def _request(
        self, operation: RecordingOperation, payload: object, *,
        timeout: float, timeout_code: str,
    ):
        self._sequence += 1
        request_id = self._sequence
        try:
            self._connection.send(RecordingCommand(request_id, operation, payload))
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RecordingProcessError("recording_worker_unavailable", str(exc)) from exc
        response = self._receive(timeout, timeout_code=timeout_code)
        if not isinstance(response, RecordingResponse) or response.request_id != request_id:
            raise RecordingProcessError("recording_protocol_error", _response_message(response))
        if response.status != "ok":
            raise RecordingProcessError(
                "recording_worker_error", f"{response.error_type}: {response.message}",
            )
        return response.payload

    def _receive(self, timeout: float, *, timeout_code: str):
        if not self._connection.poll(timeout):
            raise RecordingProcessError(
                timeout_code,
                "recorder child response timed out",
            )
        try:
            return self._connection.recv()
        except (EOFError, OSError) as exc:
            raise RecordingProcessError("recording_worker_crashed", str(exc)) from exc

    def _break(self, code: str, message: str) -> None:
        self._failure = code, message
        self._stop_process(grace=0.0)

    def _stop_process(self, *, grace: float) -> None:
        process = self._process
        if process is None:
            return
        process.join(max(0.0, grace))
        if process.is_alive():
            process.terminate(); process.join(2.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill(); process.join(2.0)
        if process.is_alive():
            raise RecordingProcessError("recording_worker_unstoppable", "recorder child is still alive")
        self._close_connection()

    def _join_after_close(self) -> bool:
        self._process.join(self._close_timeout)
        return not self._process.is_alive()

    def _close_connection(self) -> None:
        try:
            self._connection.close()
        except OSError:
            pass

    def _aborted_result(self) -> RecordingResult:
        files = stable_recording_files(self.video_path, self.index_path)
        try:
            indexed = indexed_frame_count(self.index_path)
        except (OSError, ValueError):
            indexed = 0
        code, message = self._failure or ("recording_worker_aborted", "recorder child aborted")
        dropped = max(0, self._submitted - indexed)
        return RecordingResult(
            self.video_path, self.index_path, indexed, dropped,
            integrity={
                "schema": "guandan.recording-integrity/1", "status": "FAIL",
                "recording_state": "aborted", "writer_frame_count": indexed,
                "indexed_frame_count": indexed, "decodable_frame_count": 0,
                "issues": [code], "worker_failure": {"code": code, "message": message},
                "worker_pid": self._worker_pid, "worker_confirmed_dead": not self.worker_alive,
                "final_files": files, "omitted_capture_frames": dropped,
            },
        )

    def _ensure_active(self) -> None:
        if self._result is not None:
            raise RecordingProcessError("recording_closed", "recorder is closed")
        if self._failure is not None or not self.worker_alive:
            raise RecordingProcessError("recording_worker_unavailable", "recorder child is unavailable")


def _response_message(value: object) -> str:
    if isinstance(value, RecordingResponse):
        return f"{value.error_type}: {value.message}"
    return f"unexpected recorder response: {type(value).__name__}"


__all__ = ["ProcessSessionRecorder", "RecordingProcessError"]
