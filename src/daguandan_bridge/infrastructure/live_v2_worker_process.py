"""Spawn-safe child execution and process transport for the live-v2 host."""

from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
import os
import pickle
import threading
import time
import traceback
from typing import Callable

from ..application.live_v2_worker_protocol import (
    WorkerFailure,
    WorkerReady,
    WorkerReference,
    WorkerResult,
    WorkerResultStatus,
    _RunCommand,
    _WorkerTimingEvent,
    _StopCommand,
    _WorkerStartupFailure,
    _WorkerStopped,
)
from .live_v2_worker_dispatch import WorkerSendDispatcher


def processing_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _safe_send(connection: Connection, value: object) -> None:
    """Detect payload serialization failures before pipe delivery."""
    pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    connection.send(value)


def _failure(code: str, exc: BaseException) -> WorkerFailure:
    return WorkerFailure(code, type(exc).__name__, str(exc), traceback.format_exc())


def worker_process_main(
    reference: WorkerReference,
    generation: int,
    requests: Connection,
    results: Connection,
) -> None:
    """Long-lived module-level process entry point for Windows spawn/freeze."""
    pid = os.getpid()
    try:
        worker = reference.resolve()
    except BaseException as exc:
        try:
            message = _WorkerStartupFailure(
                generation, pid, _failure("worker_import_failed", exc)
            )
            _safe_send(results, message)
        finally:
            requests.close()
            results.close()
        return

    _safe_send(results, WorkerReady(generation, pid, processing_ms()))
    try:
        while True:
            command = requests.recv()
            if isinstance(command, _StopCommand):
                if command.worker_generation == generation:
                    _safe_send(results, _WorkerStopped(generation, pid))
                    return
                continue
            if not isinstance(command, _RunCommand):
                continue
            if command.worker_generation != generation:
                continue
            received_ms = processing_ms()
            _safe_send(results, _WorkerTimingEvent(
                generation, command.request.request_sequence,
                "child_received", received_ms,
            ))
            _execute_request(
                command, worker, generation, pid, results,
                received_ms=received_ms,
            )
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        requests.close()
        results.close()


def _execute_request(
    command: _RunCommand,
    worker: Callable,
    generation: int,
    pid: int,
    results: Connection,
    *,
    received_ms: int,
) -> None:
    request = command.request
    started_ms = processing_ms()
    _safe_send(results, _WorkerTimingEvent(
        generation, request.request_sequence, "child_started", started_ms,
    ))
    try:
        payload = worker(request)
    except BaseException as exc:
        finished_ms = processing_ms()
        _safe_send(results, _WorkerTimingEvent(
            generation, request.request_sequence, "child_finished", finished_ms,
        ))
        result = WorkerResult.terminal(
            request,
            status=WorkerResultStatus.ERROR,
            worker_generation=generation,
            worker_pid=pid,
            started_processing_ms=started_ms,
            finished_processing_ms=finished_ms,
            failure=_failure("worker_exception", exc),
        )
        _safe_send(results, result)
        return

    finished_ms = processing_ms()
    _safe_send(results, _WorkerTimingEvent(
        generation, request.request_sequence, "child_finished", finished_ms,
    ))
    result = WorkerResult.terminal(
        request,
        status=WorkerResultStatus.SUCCESS,
        worker_generation=generation,
        worker_pid=pid,
        started_processing_ms=started_ms,
        finished_processing_ms=finished_ms,
        payload=payload,
    )
    try:
        _safe_send(results, result)
    except BaseException as exc:
        fallback = WorkerResult.terminal(
            request,
            status=WorkerResultStatus.ERROR,
            worker_generation=generation,
            worker_pid=pid,
            started_processing_ms=started_ms,
            finished_processing_ms=processing_ms(),
            failure=_failure("result_serialization_failed", exc),
        )
        _safe_send(results, fallback)


class WorkerProcessEndpoint:
    """Own process handles and translate pipe activity into callbacks."""

    def __init__(
        self,
        worker: WorkerReference,
        generation: int,
        on_message: Callable[[int, object], None],
        on_exit: Callable[[int, int, int | None], None],
    ) -> None:
        context = multiprocessing.get_context("spawn")
        child_requests, parent_requests = context.Pipe(duplex=False)
        parent_results, child_results = context.Pipe(duplex=False)
        process = context.Process(
            target=worker_process_main,
            args=(worker, generation, child_requests, child_results),
            name=f"live-v2-worker-{generation}",
            daemon=False,
        )
        process.start()
        child_requests.close()
        child_results.close()
        self._generation = generation
        self._process = process
        self._requests = parent_requests
        self._results = parent_results
        self._sender = WorkerSendDispatcher(lambda value: self.send(value))
        self._on_message = on_message
        self._on_exit = on_exit
        self._receiver = threading.Thread(
            target=self._receive_loop,
            name=f"live-v2-results-{generation}",
            daemon=True,
        )
        self._watcher = threading.Thread(
            target=self._watch_loop,
            name=f"live-v2-watch-{generation}",
            daemon=True,
        )
        self._receiver.start()
        self._watcher.start()

    @property
    def pid(self) -> int:
        assert self._process.pid is not None
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def send(self, value: object) -> None:
        self._requests.send(value)

    def dispatch(
        self,
        value: object,
        *,
        on_start: Callable[[], None] = lambda: None,
        on_success: Callable[[], None] = lambda: None,
        on_failure: Callable[[str], None],
    ) -> None:
        self._sender.submit(
            value,
            on_start=on_start,
            on_success=on_success,
            on_failure=on_failure,
        )

    def terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()

    def shutdown(self, *, timeout: float, graceful: bool) -> None:
        if graceful and self._process.is_alive():
            self.dispatch(_StopCommand(self._generation), on_failure=lambda _message: None)
        self._process.join(timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
        if self._process.is_alive() and hasattr(self._process, "kill"):
            self._process.kill()
            self._process.join(2.0)
        self._sender.close()
        self.close_connections()

    def close_connections(self) -> None:
        for connection in self._requests, self._results:
            try:
                connection.close()
            except OSError:
                pass

    def _receive_loop(self) -> None:
        while True:
            try:
                message = self._results.recv()
            except (EOFError, OSError):
                return
            self._on_message(self._generation, message)
            if isinstance(message, (_WorkerStartupFailure, _WorkerStopped)):
                return

    def _watch_loop(self) -> None:
        self._process.join()
        # Drain any final result/startup-failure already written to the pipe
        # before reporting process exit to the supervisor.
        self._receiver.join()
        self._on_exit(
            self._generation,
            self.pid,
            self._process.exitcode,
        )


__all__ = ["WorkerProcessEndpoint", "processing_ms", "worker_process_main"]
