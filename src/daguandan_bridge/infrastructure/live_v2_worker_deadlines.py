"""Request deadlines measured from host acceptance, before pipe dispatch."""

from __future__ import annotations

import time

from ..application.live_v2_worker_protocol import WorkerRequest, WorkerResult


class RequestDeadlineLedger:
    def __init__(self) -> None:
        self._values: dict[tuple[str, int, int, int], float] = {}

    def record(
        self,
        request: WorkerRequest,
        discarded: tuple[WorkerResult, ...],
    ) -> None:
        for item in discarded:
            self._values.pop(_key(item), None)
        rejected = any(_key(item) == _key(request) for item in discarded)
        if not rejected and request.timeout_ms:
            self._values[_key(request)] = time.monotonic() + request.timeout_ms / 1000

    def take(self, request: WorkerRequest) -> float | None:
        return self._values.pop(_key(request), None)

    def clear(self) -> None:
        self._values.clear()


def _key(item: WorkerRequest | WorkerResult) -> tuple[str, int, int, int]:
    return (
        item.session_id,
        item.capture_generation,
        item.state_revision,
        item.request_sequence,
    )


__all__ = ["RequestDeadlineLedger"]
