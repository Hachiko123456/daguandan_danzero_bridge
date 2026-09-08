"""Atomic cancellation and version-rebind behavior for worker hosts."""

from __future__ import annotations

from enum import Enum


class RebindPolicy(str, Enum):
    """Decide whether a state-only version change invalidates running work."""

    TERMINATE_IN_FLIGHT = "terminate_in_flight"
    PRESERVE_IN_FLIGHT_WITHIN_STREAM = "preserve_in_flight_within_stream"


class WorkerCancellationMixin:
    def _rebind_worker_version(self, version) -> None:
        endpoint = None
        with self._condition:
            if version != self._version:
                same_stream = version[:2] == self._version[:2]
                preserve = (
                    same_stream
                    and self._rebind_policy
                    is RebindPolicy.PRESERVE_IN_FLIGHT_WITHIN_STREAM
                )
                if preserve:
                    discarded = self._requests.rebind(
                        reason="version_rebound",
                        worker_generation=self._generation,
                        preserve_in_flight=True,
                    )
                else:
                    had_in_flight = self._requests.has_in_flight
                    discarded = self._requests.rebind(
                        reason="version_rebound",
                        worker_generation=self._generation,
                    )
                    if had_in_flight:
                        self._state = type(self._state).BROKEN
                        endpoint = self._endpoint
                self._publish_many_locked(discarded)
            self._session_id, self._capture_generation, self._state_revision = version
        if endpoint:
            endpoint.terminate()

    def cancel_all(self, *, reason: str):
        endpoint = None
        with self._condition:
            had_in_flight = self._requests.has_in_flight
            results = self._requests.discard_all(
                reason=reason,
                worker_generation=self._generation,
            )
            if had_in_flight:
                self._state = type(self._state).BROKEN
                endpoint = self._endpoint
        if endpoint:
            endpoint.terminate()
        return results


__all__ = ["RebindPolicy", "WorkerCancellationMixin"]
