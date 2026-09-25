"""Read-only local control handling for a live-v2 advice opportunity."""

from typing import Any

from ..live_v2.identity import Seat
from ..live_v2.results import OpportunityStatus
from .live_v2_advice_pump import opportunity_is_current
from .live_v2_local_pass import local_pass_advice


class LiveV2ControlMixin:
    def preview_controls(
        self, fast: Any, *, captured_ms: int, capture_generation: int,
        frame_size: tuple[int, int],
    ):
        with self._lock:
            if capture_generation != self._generation or self._engine is None:
                return None

            # A visual PASS hint is presentation-only evidence.  Never retain
            # it across a recovery/resync/review-required boundary, because it
            # could otherwise be displayed alongside (or race with) the
            # authoritative withheld/recovery state.
            hint_allowed = (
                self.status == "running"
                and self._recovery_state == "RUNNING"
                and self._engine.state.snapshot.current_seat is Seat.SELF
            )
            if not hint_allowed:
                self._hint.reset()
                self._last_local_hint = None
                hint = None
            else:
                hint = self._hint.observe(
                    fast, session_id=self.store.session_id,
                    capture_generation=capture_generation, captured_ms=captured_ms,
                    now_ms=self._clock.processing_ms(), running=True,
                    frame_size=frame_size,
                )
                if hint is not None:
                    self._last_local_hint = hint
                elif (
                    getattr(fast, "active_player", None) in {"right", "opposite", "left"}
                    or getattr(fast, "effect_visible", False)
                    or getattr(fast, "super_double_visible", False)
                    or getattr(fast, "game_end_control", None)
                ):
                    # A known turn transition or animation invalidates the visual
                    # PASS display immediately; an unreadable single frame does not.
                    self._last_local_hint = None

            opportunity = self._engine.state.opportunity.current
            if (
                hint is not None
                and self._local_pass_context_is_current(opportunity)
                and self._advice_pump is not None
            ):
                self._advice_pump.confirm_local_pass(opportunity, hint)
            has_local_hint = hint is not None or self._last_local_hint is not None
            if not has_local_hint:
                # Do not make a no-op control sample project a new lifecycle
                # update; in particular, preserve a formal withheld/recovery
                # value instead of letting _plain_update clear it.
                return None
            update = self._plain_update(
                fast=fast, local_rule_hint=(hint or self._last_local_hint),
                local_rule_hint_pending=self._hint.pending,
            )
        if self._on_update:
            self._on_update(update)
        return update

    def _local_pass_context_is_current(self, opportunity) -> bool:
        """Return whether local PASS may replace the formal advice result."""

        if (
            self.status != "running"
            or self._recovery_state != "RUNNING"
            or self._engine is None
            or opportunity is None
            or opportunity.status is not OpportunityStatus.READY
        ):
            return False
        snapshot = self._engine.state.snapshot
        current = self._engine.state.version
        if snapshot.current_seat is not Seat.SELF or opportunity.seat is not Seat.SELF:
            return False
        if (
            current.session_id != self.store.session_id
            or current.capture_generation != self._generation
            or opportunity.version.session_id != self.store.session_id
            or opportunity.version.capture_generation != self._generation
        ):
            return False
        return opportunity_is_current(
            opportunity.version, opportunity.opportunity_id, current, opportunity
        )

    def _accept_local_pass(self, opportunity, hint: object) -> bool:
        with self._lock:
            if not self._local_pass_context_is_current(opportunity):
                return False
            if not getattr(hint, "is_current", lambda **_kwargs: False)(
                session_id=self.store.session_id,
                capture_generation=self._generation,
                now_ms=self._clock.processing_ms(),
            ):
                return False
            self.latest_advice = local_pass_advice(
                self._engine.state.snapshot, opportunity
            )
            return True


__all__ = ["LiveV2ControlMixin"]
