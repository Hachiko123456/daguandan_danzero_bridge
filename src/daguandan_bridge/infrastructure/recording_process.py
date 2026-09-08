"""Frozen/spawn-safe bootstrap for the per-session recorder child."""
from __future__ import annotations

import importlib

from ..application.recording_process_protocol import RecordingResponse


def recording_process_bootstrap(
    config, connection, worker_module: str, worker_name: str, worker_args: tuple = (),
) -> None:
    try:
        worker = getattr(importlib.import_module(worker_module), worker_name)
        worker(config, connection, *worker_args)
    except BaseException as exc:
        try:
            connection.send(RecordingResponse(
                -1, "startup_error", error_type=type(exc).__name__, message=str(exc)
            ))
        except BaseException:
            pass
    finally:
        connection.close()


__all__ = ["recording_process_bootstrap"]
