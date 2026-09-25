"""Small non-authoritative journal for the production live-v2 runtime."""

from __future__ import annotations

from typing import Any

from ..domain.recording import RecorderWarning
from ..live_v2.engine import EngineInput
from ..live_v2.results import EngineUpdate
from ..live_v2.identity import FrameIdentity
from .ports import SessionPersistencePort
from .live_v2_diagnostic_projection import project_engine_update


class LiveV2RuntimeJournal:
    """Persist diagnostics only; formal actions are owned by the event sink."""

    def __init__(self, store: SessionPersistencePort) -> None:
        self._store = store

    def append(self, update: EngineUpdate) -> None:
        self._store.append_observation(project_engine_update(update))

    def fault(self, kind: str, message: str, **context: Any) -> None:
        self._store.append_recognition_trace({
            "schema": "guandan.live-v2.runtime-fault/1",
            "kind": str(kind),
            "message": str(message),
            **{str(key): value for key, value in context.items()},
        })

    def lifecycle(self, kind: str, **context: Any) -> None:
        self._store.append_observation({
            "schema": "guandan.live-v2.lifecycle/1",
            "kind": str(kind),
            **{str(key): value for key, value in context.items()},
        })


class LiveV2LifecycleMixin:
    """LiveRuntimePort lifecycle methods shared by the session coordinator."""

    def record_frame(self, frame: Any, *, monotonic_ms: int,
                     wall_time: str) -> RecorderWarning | None:
        if self.status in {"finalizing", "sealed"}:
            return None
        try:
            warning = self.recorder.write_frame(frame, monotonic_ms, wall_time)
            if warning is not None:
                self._safe_fault(
                    "recording_warning", warning.details,
                    monotonic_ms=warning.monotonic_ms, reason=warning.reason,
                )
            return warning
        except Exception as exc:
            self._safe_fault("recording_failed", str(exc), monotonic_ms=monotonic_ms)
            return RecorderWarning("recording_failed", monotonic_ms, str(exc))

    def _capture_identity(
        self, trace_context: dict[str, object] | None, monotonic_ms: int,
    ) -> FrameIdentity | None:
        context = trace_context or {}
        supplied_generation = context.get("capture_generation")
        if supplied_generation is not None and (
            type(supplied_generation) is not int
            or supplied_generation != self._generation
        ):
            return None
        supplied_sequence = context.get("capture_seq")
        if supplied_sequence is None:
            sequence = self._frame_sequence + 1
        elif type(supplied_sequence) is not int or supplied_sequence <= self._frame_sequence:
            return None
        else:
            sequence = supplied_sequence
        supplied_ms = context.get("captured_ms", monotonic_ms)
        if type(supplied_ms) is not int or supplied_ms < 0:
            return None
        roi = context.get("roi_version", self._roi_version)
        source = context.get("source_id", self._source_id)
        if not isinstance(roi, str) or not roi.strip():
            roi = self._roi_version
        if not isinstance(source, str) or not source.strip():
            source = self._source_id
        self._frame_sequence = sequence
        self._last_ms = max(self._last_ms, supplied_ms)
        self._clock.advance(supplied_ms)
        return FrameIdentity(
            self.store.session_id, self._generation, sequence, supplied_ms,
            roi, source,
        )

    def pause(self):
        self._status_before_pause, self.status = self.status, "paused"
        self._hint.reset()
        self._last_local_hint = None
        return self._plain_update()

    def resume(self, *, monotonic_ms: int):
        if self.status != "paused":
            raise RuntimeError("session is not paused")
        self.status = self._status_before_pause or "running"
        self._clock.advance(monotonic_ms)
        self._status_before_pause = None
        if self._engine is None:
            return self._plain_update()
        return self._process(EngineInput(captured_watermark_ms=monotonic_ms))

    def begin_finalizing(self):
        if self.status != "sealed":
            self.status = "finalizing"
        self._hint.reset()
        self._last_local_hint = None
        return self._plain_update()

    def capture_interrupted(self, reason: str, *, monotonic_ms: int):
        self._hint.reset()
        self._last_local_hint = None
        self._clock.advance(monotonic_ms)
        self._safe_fault("capture_interrupted", reason, monotonic_ms=monotonic_ms)
        return self._plain_update(block_reason=f"capture_interrupted:{reason}")

    def analysis_failed(self, reason: str, *, monotonic_ms: int):
        self._clock.advance(monotonic_ms)
        self._safe_fault("analysis_failed", reason, monotonic_ms=monotonic_ms)
        return self._plain_update(block_reason=f"analysis_failed:{reason}")

    def poll_deadlines(self):
        with self._lock:
            if self._advice_pump:
                self._advice_pump.poll()
            return self._process(EngineInput()) if self._engine else self._plain_update()

    def wait_for_advice_idle(self, *, timeout: float = 60.0) -> bool:
        return not self._advice_pump or self._advice_pump.wait_idle(timeout)

    def finish(self):
        with self._lock:
            if self.status == "sealed":
                return self._plain_update()
            self.status = "finalizing"
            detached = self._detach_workers()
        self._close_detached(detached)
        integrity = None
        try:
            recording = self.recorder.close()
            frame_count = recording.frame_count
            dropped_frames = recording.dropped_frames
            media_failures = tuple(
                item.to_dict() for item in recording.incident_media_failures
            )
            integrity = recording.integrity or None
        except Exception as exc:
            self._safe_fault("recording_close_failed", str(exc))
            frame_count = int(getattr(self.recorder, "frame_count", 0))
            dropped_frames = 0
            media_failures = ()
            integrity = {
                "status": "FAIL", "writer_frame_count": frame_count,
                "issues": (f"recording_close_failed:{type(exc).__name__}",),
            }
        with self._lock:
            self.store.seal(
                frame_count=frame_count,
                dropped_frames=dropped_frames,
                metrics={
                    "runtime": "live_v2",
                    "confirmed_actions": len(self.rule_session.confirmed_actions),
                    **self._opportunity_metrics,
                },
                incident_media_failures=media_failures,
            )
            try:
                terminal_event = getattr(self, "_terminal_event", None)
                report = self.rule_session.health(
                    additional_events=(terminal_event,) if terminal_event is not None else (),
                    recording_integrity=integrity,
                )
            except Exception as exc:
                report = {
                    "schema": "guandan.live-v2.health/1", "status": "FAIL",
                    "issues": [{
                        "code": "HEALTH-AUDIT-FAILED", "severity": "FAIL",
                        "summary": f"{type(exc).__name__}: {exc}", "evidence": {},
                    }],
                }
                self._safe_fault("health_audit_failed", str(exc))
            try:
                self.store.append_post_seal_health_audit(
                    report, state=self.snapshot.semantic_dict(),
                    monotonic_ms=self._last_ms,
                )
            except Exception as exc:
                self._safe_fault("health_audit_failed", str(exc))
            if bool(getattr(self.store, "automatic_log_delivery_enabled", False)):
                self._deliver_automatic_log()
            self.status = "sealed"
            issues = tuple(report.get("issues", ()) or ())
            reason = ""
            if str(report.get("status", "")).upper() == "FAIL":
                reason = str(issues[0].get("code", "session_health_failed")) if issues else "session_health_failed"
            return self._plain_update(block_reason=reason)

    def _deliver_automatic_log(self) -> None:
        try:
            from ..automatic_log_delivery import export_automatic_session_log
            document = export_automatic_session_log(
                self.store.directory,
                include_media=bool(
                    getattr(self.store, "automatic_log_include_media", False)
                ),
                profiles_root=getattr(self.store, "profiles_root", None),
                profile_name=getattr(self.store, "profile_name", None),
            ).to_dict()
        except Exception as exc:
            document = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        self.automatic_log_delivery_result = document
        try:
            self.store.record_automatic_log_delivery(document)
        except Exception:
            pass

    def _safe_fault(self, kind: str, message: str, **context: Any) -> None:
        try:
            self._journal.fault(kind, message, **context)
        except Exception:
            pass

__all__ = ["LiveV2LifecycleMixin", "LiveV2RuntimeJournal"]
