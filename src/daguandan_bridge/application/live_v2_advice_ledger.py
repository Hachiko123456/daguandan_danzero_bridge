"""Thread-safe identity, phase and timer ledger for the advice pump."""

from __future__ import annotations

from threading import Condition, Lock, Timer

from ..live_v2.results import AdviceOpportunity, OpportunityStatus
from .live_v2_advice_protocol import AdviceRequestIdentity
from .live_v2_local_pass import LocalPassOpportunityPhase


class AdviceOpportunityLedger:
    def __init__(self) -> None:
        self._idle = Condition(Lock())
        self._submitted: set[tuple[str, int, int, int, str]] = set()
        self._pending: set[AdviceRequestIdentity] = set()
        self._requested_ms: dict[AdviceRequestIdentity, int] = {}
        self._deadlines_ms: dict[AdviceRequestIdentity, int] = {}
        self._opportunities: dict[AdviceRequestIdentity, AdviceOpportunity] = {}
        self._phases: dict[AdviceRequestIdentity, LocalPassOpportunityPhase] = {}
        self._timers: dict[AdviceRequestIdentity, Timer] = {}
        self._sequence = 0

    def register(
        self,
        opportunity: AdviceOpportunity,
        *,
        requested_ms: int,
        request_timeout_ms: int,
    ) -> AdviceRequestIdentity | None:
        version = opportunity.version
        key = (
            version.session_id, version.capture_generation,
            version.state_revision, version.turn_index, opportunity.opportunity_id,
        )
        with self._idle:
            if key in self._submitted:
                return None
            self._submitted.add(key)
            self._sequence += 1
            identity = AdviceRequestIdentity(
                version, self._sequence, opportunity.opportunity_id
            )
            self._pending.add(identity)
            self._requested_ms[identity] = requested_ms
            self._deadlines_ms[identity] = requested_ms + request_timeout_ms
            self._opportunities[identity] = opportunity
            self._phases[identity] = LocalPassOpportunityPhase.WAITING_HINT
            return identity

    def expire_due(
        self, now_ms: int
    ) -> tuple[
        tuple[AdviceRequestIdentity, LocalPassOpportunityPhase, int], ...
    ]:
        """Atomically terminalize every opportunity whose absolute deadline passed."""
        expired: list[
            tuple[AdviceRequestIdentity, LocalPassOpportunityPhase, int]
        ] = []
        with self._idle:
            due = sorted(
                (
                    identity for identity in self._pending
                    if self._deadlines_ms[identity] <= now_ms
                ),
                key=lambda identity: identity.request_sequence,
            )
            for identity in due:
                phase = self._phases.get(
                    identity, LocalPassOpportunityPhase.WAITING_HINT
                )
                deadline_ms = self._deadlines_ms[identity]
                self._complete_locked(identity, notify=False)
                expired.append((identity, phase, deadline_ms))
        return tuple(expired)

    def set_timer(self, identity: AdviceRequestIdentity, timer: Timer) -> None:
        with self._idle:
            self._timers[identity] = timer

    def begin_model(self, identity: AdviceRequestIdentity) -> bool:
        with self._idle:
            self._timers.pop(identity, None)
            if (
                identity not in self._pending
                or self._phases.get(identity)
                is not LocalPassOpportunityPhase.WAITING_HINT
            ):
                return False
            self._phases[identity] = LocalPassOpportunityPhase.MODEL_SUBMITTED
            return True

    def retire_superseded(
        self, incoming: AdviceOpportunity
    ) -> tuple[tuple[AdviceRequestIdentity, LocalPassOpportunityPhase], ...]:
        retired: list[tuple[AdviceRequestIdentity, LocalPassOpportunityPhase]] = []
        with self._idle:
            for identity in tuple(self._pending):
                same = (
                    incoming.status is OpportunityStatus.READY
                    and identity.opportunity_id == incoming.opportunity_id
                    and _same_version(identity, incoming)
                )
                if same:
                    continue
                phase = self._phases.get(
                    identity, LocalPassOpportunityPhase.WAITING_HINT
                )
                self._complete_locked(identity)
                retired.append((identity, phase))
        return tuple(retired)

    def take_local_pass(
        self,
        opportunity: AdviceOpportunity,
        *,
        now_ms: int,
        window_ms: int,
    ) -> AdviceRequestIdentity | None:
        with self._idle:
            identity = next(
                (
                    item for item, value in self._opportunities.items()
                    if value.opportunity_id == opportunity.opportunity_id
                    and _same_version(item, opportunity)
                    and item in self._pending
                ),
                None,
            )
            if identity is None:
                return None
            elapsed = max(0, now_ms - self._requested_ms.get(identity, now_ms))
            if (
                self._phases.get(identity)
                is not LocalPassOpportunityPhase.WAITING_HINT
                or elapsed > window_ms
            ):
                return None
            self._complete_locked(identity)
            return identity

    def accept_result(self, identity: AdviceRequestIdentity) -> bool:
        with self._idle:
            if (
                identity not in self._pending
                or self._phases.get(identity)
                is not LocalPassOpportunityPhase.MODEL_SUBMITTED
            ):
                return False
            self._complete_locked(identity)
            return True

    def complete(self, identity: AdviceRequestIdentity) -> bool:
        with self._idle:
            if identity not in self._pending:
                return False
            self._complete_locked(identity)
            return True

    def requested_ms(self, identity: AdviceRequestIdentity) -> int | None:
        with self._idle:
            return self._requested_ms.get(identity)

    def deadline_ms(self, identity: AdviceRequestIdentity) -> int | None:
        with self._idle:
            return self._deadlines_ms.get(identity)

    def pending(self) -> tuple[AdviceRequestIdentity, ...]:
        with self._idle:
            return tuple(self._pending)

    def cancel_timers(self) -> tuple[Timer, ...]:
        with self._idle:
            timers = tuple(self._timers.values())
            self._timers.clear()
            return timers

    def wait_idle(self, timeout: float) -> bool:
        with self._idle:
            return self._idle.wait_for(
                lambda: not self._pending, timeout=max(0.0, timeout)
            )

    def notify_idle(self) -> None:
        with self._idle:
            self._idle.notify_all()

    def _complete_locked(
        self, identity: AdviceRequestIdentity, *, notify: bool = True
    ) -> None:
        self._phases[identity] = LocalPassOpportunityPhase.TERMINAL
        self._pending.remove(identity)
        timer = self._timers.pop(identity, None)
        if timer is not None:
            timer.cancel()
        self._opportunities.pop(identity, None)
        if notify:
            self._idle.notify_all()


def _same_version(
    identity: AdviceRequestIdentity, opportunity: AdviceOpportunity
) -> bool:
    left, right = identity.version, opportunity.version
    return (
        left.session_id, left.capture_generation, left.state_revision, left.turn_index
    ) == (
        right.session_id, right.capture_generation, right.state_revision, right.turn_index
    )


__all__ = ["AdviceOpportunityLedger"]
