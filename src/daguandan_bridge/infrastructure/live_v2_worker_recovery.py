"""Bounded restart policy for a persistent worker host."""

from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class WorkerRestartCircuit:
    limit: int = 3
    window_seconds: float = 30.0
    base_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 4.0
    _attempts: deque[float] = field(default_factory=deque)
    _failures: int = 0
    _next_allowed: float = 0.0

    def __post_init__(self) -> None:
        if self.limit < 1 or self.window_seconds <= 0:
            raise ValueError("restart limit and window must be positive")
        if self.base_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("restart backoff must be non-negative")

    def begin(self, now: float) -> str:
        while self._attempts and now - self._attempts[0] >= self.window_seconds:
            self._attempts.popleft()
        if now < self._next_allowed:
            return "restart_cooldown"
        if len(self._attempts) >= self.limit:
            self._next_allowed = max(
                self._next_allowed, self._attempts[0] + self.window_seconds
            )
            return "restart_circuit_open"
        self._attempts.append(now)
        return ""

    def succeeded(self, now: float) -> None:
        self._failures = 0
        self._next_allowed = now + self.base_backoff_seconds

    def failed(self, now: float) -> None:
        self._failures += 1
        delay = min(
            self.max_backoff_seconds,
            self.base_backoff_seconds * (2 ** (self._failures - 1)),
        )
        self._next_allowed = max(self._next_allowed, now + delay)


class WorkerAutoRecoveryMixin:
    """Restart a BROKEN host once for the next legal request."""

    def _recover_for_submit(self, request) -> None:
        now = self._monotonic_clock()
        with self._condition:
            if str(getattr(self._state, "value", self._state)) == "closed":
                raise RuntimeError("closed worker host cannot recover")
            problem = self._recovery.begin(now)
        if problem:
            raise RuntimeError(problem)
        try:
            self.restart(
                timeout=10.0,
                session_id=request.session_id,
                capture_generation=request.capture_generation,
                state_revision=request.state_revision,
            )
        except Exception as exc:
            with self._condition:
                if str(getattr(self._state, "value", self._state)) != "closed":
                    self._state = type(self._state).BROKEN
                self._recovery.failed(self._monotonic_clock())
            raise RuntimeError(f"worker_auto_restart_failed:{exc}") from exc
        with self._condition:
            self._recovery.succeeded(self._monotonic_clock())


__all__ = ["WorkerAutoRecoveryMixin", "WorkerRestartCircuit"]
