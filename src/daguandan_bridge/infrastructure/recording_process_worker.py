"""Child-owned SessionRecorder command loop for Windows spawn/freeze."""
from __future__ import annotations

import os
from pathlib import Path
import traceback

from ..application.recording_process_protocol import (
    RecordingCommand, RecordingOperation, RecordingProcessConfig,
    RecordingReady, RecordingResponse,
)
from ..live.recorder import SessionRecorder


def run_session_recorder_worker(
    config: RecordingProcessConfig,
    connection,
) -> None:
    recorder = SessionRecorder(
        Path(config.session_directory), size=config.size, fps=config.fps,
        codec=config.codec, max_video_bytes=config.max_video_bytes,
    )
    connection.send(RecordingReady(os.getpid()))
    while True:
        command = connection.recv()
        if not isinstance(command, RecordingCommand):
            continue
        try:
            payload = _execute(recorder, command)
            connection.send(RecordingResponse(command.request_id, "ok", payload=payload))
            if command.operation is RecordingOperation.CLOSE:
                return
        except BaseException as exc:
            connection.send(RecordingResponse(
                command.request_id, "error", error_type=type(exc).__name__,
                message=f"{exc}\n{traceback.format_exc()}",
            ))


def _execute(recorder: SessionRecorder, command: RecordingCommand):
    value = command.payload
    if command.operation is RecordingOperation.WRITE:
        frame, monotonic_ms, wall_time = value
        return recorder.write_frame(frame, monotonic_ms, wall_time)
    if command.operation is RecordingOperation.SAVE_EVIDENCE:
        path, frame = value
        return recorder.save_evidence_frame(Path(path), frame)
    if command.operation is RecordingOperation.SCHEDULE_INCIDENT:
        path, trigger_ms, before_ms, after_ms = value
        recorder.schedule_incident_media(
            Path(path), trigger_ms=trigger_ms, before_ms=before_ms, after_ms=after_ms,
        )
        return None
    if command.operation is RecordingOperation.CLOSE:
        return recorder.close()
    raise ValueError(f"unsupported recording operation: {command.operation}")


__all__ = ["run_session_recorder_worker"]
