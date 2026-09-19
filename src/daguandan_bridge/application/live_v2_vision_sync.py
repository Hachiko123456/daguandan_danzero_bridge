"""Synchronous, exact-frame helpers for the live-v2 vision runtime."""

from __future__ import annotations

import time

import numpy as np

from ..live_v2.identity import FrameIdentity, Seat, VersionIdentity
from .live_v2_vision_protocol import VisionRequestIdentity, VisionRuntimeStatus


class VisionSyncMixin:
    """Provide deterministic single-frame processing for recorded replay."""

    def process_frame_sync(
        self,
        image: np.ndarray,
        *,
        frame: FrameIdentity,
        version: VersionIdentity,
        expected_seat: Seat | str | None,
        opening_lead_seat: Seat | str | None = None,
        wild_rank: str = "",
        request_sequence: int,
        formal_action_boundary: FrameIdentity | None = None,
        repair_seats: tuple[Seat | str, ...] = (),
        timeout_ms: int = 120_000,
    ):
        """Wait for and return the result for the exact requested frame."""
        expected = VisionRequestIdentity(frame, version, request_sequence)
        values = list(
            self.submit(
                image,
                frame=frame,
                version=version,
                expected_seat=expected_seat,
                visual_self_opportunity=expected_seat is Seat.SELF,
                opening_lead_seat=opening_lead_seat,
                wild_rank=wild_rank,
                request_sequence=request_sequence,
                formal_action_boundary=formal_action_boundary,
                repair_seats=repair_seats,
                timeout_ms=timeout_ms,
            )
        )
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            for item in values:
                if item.identity != expected:
                    continue
                if item.status is not VisionRuntimeStatus.FRAME:
                    raise RuntimeError(
                        f"{item.failure_code or item.status.value}: "
                        f"{item.message or 'requested vision frame did not complete'}"
                    )
                if item.pipeline_result is None or item.pipeline_result.frame != frame:
                    raise RuntimeError(
                        "worker_payload_identity_mismatch: "
                        "exact frame result was not returned"
                    )
                return item.pipeline_result

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "vision_frame_timeout: exact requested frame result was not delivered"
                )
            try:
                values = [self.get_result(timeout=remaining)]
            except TimeoutError as exc:
                raise RuntimeError(
                    "vision_frame_timeout: exact requested frame result was not delivered"
                ) from exc


__all__ = ["VisionSyncMixin"]
