"""Read-only local control handling for a live-v2 advice opportunity."""

from typing import Any

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
            hint = self._hint.observe(
                fast, session_id=self.store.session_id,
                capture_generation=capture_generation, captured_ms=captured_ms,
                now_ms=self._clock.processing_ms(), running=self.status == "running",
                frame_size=frame_size,
            )
            if hint is None:
                return None
            opportunity = self._engine.state.opportunity.current
            accepted = False
            if opportunity is not None and self._advice_pump is not None:
                accepted = self._advice_pump.confirm_local_pass(opportunity, hint)
            update = self._plain_update(fast=fast, local_rule_hint=hint)
        if accepted and self._on_update:
            self._on_update(update)
        return update

    def _accept_local_pass(self, opportunity, hint: object) -> bool:
        with self._lock:
            if self.status != "running" or self._engine is None:
                return False
            current = self._engine.state.version
            active = self._engine.state.opportunity.current
            if not opportunity_is_current(
                opportunity.version, opportunity.opportunity_id, current, active
            ):
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
