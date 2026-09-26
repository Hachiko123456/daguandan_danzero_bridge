from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from daguandan_bridge.application.listener_evidence import (
    BoundedEvidenceWriter, ListenerEvidence, ListenerFrame,
)

pytestmark = pytest.mark.unit


def _frame(root, seq, generation=2):
    return ListenerFrame(root, "episode", SimpleNamespace(
        image=np.full((2, 2, 3), seq, dtype=np.uint8), evidence_frame_id=f"f{seq}"),
        generation, seq, "preopening_listener", details={"page": {"stage": "unknown"}})


def _persist(root, calls):
    def save(frame):
        calls.append(frame)
        directory = root / "diagnostic_frames"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{len(calls):06}.png"
        path.write_bytes(frame.snapshot.image.tobytes())
        path.with_suffix(".json").write_text("{}")
        return {"image_path": path, "metadata_path": path.with_suffix(".json")}
    return save


def test_ring_failure_dedup_and_single_recovery_frame(tmp_path):
    calls, results = [], []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda r, e: results.append((r, e)))
    for seq in range(1, 8):
        frame = evidence.remember(_frame(tmp_path, seq))
    assert evidence.incident("page_unknown", frame)
    assert not evidence.incident("page_unknown", frame)
    assert evidence.writer.wait_idle()
    assert [c.capture_seq for c in calls] == [7, 4, 5, 6]
    assert calls[0].source_phase == "failed_listener_frame"
    evidence.recover(evidence.remember(_frame(tmp_path, 8)))
    evidence.recover(evidence.remember(_frame(tmp_path, 9)))
    assert evidence.writer.wait_idle()
    assert [c.capture_seq for c in calls] == [7, 4, 5, 6, 8]
    manifests = list((tmp_path / "diagnostic_frames").glob("incident_*.json"))
    assert len(manifests) == 1
    doc = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert [f["role"] for f in doc["frames"]] == ["prior", "prior", "prior", "failure", "recovery"]
    assert doc["frames"][3]["capture_seq"] == 7
    assert doc["has_failure_frame"] is True
    assert doc["failure_role"] == "failure"
    assert all(error is None for _, error in results)
    assert evidence.writer.close()


def test_generation_change_does_not_mix_preincident_frames(tmp_path):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None)
    evidence.remember(_frame(tmp_path, 9, 1))
    frame = evidence.remember(_frame(tmp_path, 1, 2))
    assert evidence.incident("capture", frame)
    assert evidence.writer.wait_idle()
    assert [c.capture_generation for c in calls] == [2]
    assert evidence.writer.close()


def test_auto_count_bytes_and_write_failures_are_bounded(tmp_path):
    calls = []
    def fail(frame):
        calls.append(frame)
        raise OSError("disk full")
    results = []
    evidence = ListenerEvidence(fail, lambda r, e: results.append(e), max_incidents=2)
    frame = evidence.remember(_frame(tmp_path, 1))
    assert evidence.incident("capture", frame)
    assert not evidence.incident("capture", frame)
    assert evidence.incident("recognition", frame)
    assert not evidence.incident("worker", frame)
    assert evidence.writer.wait_idle()
    assert len(calls) == 2
    assert len(results) == 2
    assert all(isinstance(error, OSError) for error in results)
    assert evidence.incident_count == 2
    reserved = evidence.reserved_bytes
    assert reserved > 0
    assert evidence.directory_is_pinned(tmp_path)
    assert not evidence.incident("capture", frame)
    assert not evidence.incident("worker", frame)
    assert evidence.reserved_bytes == reserved
    assert evidence.writer.close()
    limited = ListenerEvidence(fail, lambda *_: None, max_bytes=1)
    assert not limited.incident("capture", frame)
    assert len(calls) == 2
    assert limited.writer.close()


def test_single_worker_queue_limit_and_shutdown_deadline():
    writer = BoundedEvidenceWriter(queue_limit=1)
    entered, release = Event(), Event()
    threads = []
    def blocked():
        threads.append(get_ident())
        entered.set()
        release.wait(5)
    first = writer.submit(blocked)
    assert entered.wait(2)
    second = writer.submit(lambda: threads.append(get_ident()))
    assert second is not None
    assert writer.submit(lambda: None) is None
    assert writer.close(timeout=0) is False
    assert writer.submit(lambda: None) is None
    release.set()
    assert first.wait(5000)
    assert second.wait(5000)
    assert writer.close()
    assert len(set(threads)) == 1
    assert threads[0] != get_ident()


def test_reporting_failure_does_not_kill_worker_or_recurse():
    writer = BoundedEvidenceWriter()
    def bad_report(*args):
        raise RuntimeError("UI is gone")
    first = writer.submit(lambda: 42, bad_report)
    assert first.wait()
    second = writer.submit(lambda: 43)
    assert second.wait()
    assert second.result == 43
    assert writer.close()


@pytest.mark.parametrize("changed", ["generation", "session", "phase"])
def test_recovery_never_attaches_foreign_capture_scope(tmp_path, changed):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None)
    failure = evidence.remember(_frame(tmp_path, 3))
    assert evidence.incident("page_unknown", failure)
    foreign = _frame(tmp_path, 4)
    if changed == "generation":
        foreign = replace(foreign, capture_generation=3)
    elif changed == "session":
        foreign = replace(foreign, session_id="another-session")
    else:
        foreign = replace(foreign, source_phase="live_session")
    evidence.recover(foreign)
    assert evidence.writer.wait_idle()
    assert [item.capture_seq for item in calls] == [3]
    assert not evidence.incident("page_unknown", failure)
    evidence.writer.close()


def test_missing_or_older_recovery_does_not_reset_episode_budget(tmp_path):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None)
    failure = evidence.remember(_frame(tmp_path, 3))
    assert evidence.incident("page_unknown", failure)
    evidence.recover(None)
    assert not evidence.incident("page_unknown", failure)
    newer_miss = evidence.remember(_frame(tmp_path, 5))
    assert not evidence.incident("page_unknown", newer_miss)
    evidence.recover(_frame(tmp_path, 4))
    assert not evidence.incident("page_unknown", newer_miss)
    evidence.recover(evidence.remember(_frame(tmp_path, 6)))
    assert evidence.writer.wait_idle()
    assert [item.capture_seq for item in calls] == [3, 6]
    assert evidence.incident_count == 1
    evidence.writer.close()


def test_first_capture_failure_without_frame_writes_no_phantom(tmp_path):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None)
    assert not evidence.incident("capture", None)
    assert evidence.incident_count == 0
    assert evidence.reserved_bytes == 0
    assert calls == []
    assert not list(tmp_path.iterdir())
    evidence.writer.close()


@contextmanager
def _paused_writer(writer, *, fill_queue=False):
    """Hold the active worker, optionally filling its sole pending slot."""
    entered, release = Event(), Event()

    def blocked():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release evidence writer")

    first = writer.submit(blocked)
    try:
        assert first is not None
        assert entered.wait(2)
        if fill_queue:
            assert writer.submit(lambda: None) is not None
        yield
    finally:
        release.set()
        assert writer.wait_idle()
        assert first.error is None


@pytest.mark.parametrize("retry_seq", [3, 4])
def test_queue_rejected_incident_preserves_quota_dedup_and_pins(tmp_path, retry_seq):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None,
                                max_incidents=1, queue_limit=1)
    failure = evidence.remember(_frame(tmp_path, 3))
    try:
        with _paused_writer(evidence.writer, fill_queue=True):
            for seq in (3, 4):
                assert not evidence.incident("capture", _frame(tmp_path, seq))
                assert evidence.incident_count == 0
                assert evidence.reserved_bytes == 0
                assert not evidence.directory_is_pinned(tmp_path)
                assert evidence._seen == {}
                assert evidence._pending_recovery == []
        retry = failure if retry_seq == 3 else evidence.remember(_frame(tmp_path, retry_seq))
        assert evidence.incident("capture", retry)
        assert evidence.writer.wait_idle()
        assert evidence.incident_count == 1
        assert evidence.reserved_bytes > 0
        assert evidence.directory_is_pinned(tmp_path)
        assert calls[0].capture_seq == retry_seq
        assert not evidence.incident("capture", retry)
    finally:
        assert evidence.writer.close()


@pytest.mark.parametrize("retry_seq", [4, 5])
def test_queue_rejected_recovery_can_retry_same_or_next_frame(tmp_path, retry_seq):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None, queue_limit=1)
    failure = evidence.remember(_frame(tmp_path, 3))
    try:
        assert evidence.incident("page_unknown", failure)
        assert evidence.writer.wait_idle()
        reserved = evidence.reserved_bytes
        pending = list(evidence._pending_recovery)
        seen = dict(evidence._seen)
        with _paused_writer(evidence.writer, fill_queue=True):
            evidence.recover(_frame(tmp_path, 4))
            evidence.recover(_frame(tmp_path, 4))
            assert evidence.reserved_bytes == reserved
            assert evidence._pending_recovery == pending
            assert evidence._seen == seen
            assert evidence._recovered == {}
            assert not evidence.incident("page_unknown", failure)
        recovery = _frame(tmp_path, retry_seq)
        evidence.recover(recovery)
        assert evidence.writer.wait_idle()
        assert evidence.reserved_bytes == reserved + evidence._frame_budget(recovery) + 64 * 1024
        reserved = evidence.reserved_bytes
        evidence.recover(recovery)
        evidence.recover(_frame(tmp_path, retry_seq + 1))
        assert evidence.writer.wait_idle()
        assert evidence.reserved_bytes == reserved
        assert [frame.capture_seq for frame in calls] == [3, retry_seq]
        assert evidence._pending_recovery == []
        assert evidence._seen == {}
        assert not evidence.incident("page_unknown", failure)
        manifest, = (tmp_path / "diagnostic_frames").glob("incident_*.json")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert [frame["role"] for frame in document["frames"]] == ["failure", "recovery"]
    finally:
        assert evidence.writer.close()


@pytest.mark.parametrize("changed", ["generation", "session", "phase", "scope_id"])
def test_queue_rejected_recovery_keeps_original_stream_binding(tmp_path, changed):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None, queue_limit=1)
    failure = evidence.remember(_frame(tmp_path, 3))
    try:
        assert evidence.incident("page_unknown", failure)
        assert evidence.writer.wait_idle()
        with _paused_writer(evidence.writer, fill_queue=True):
            evidence.recover(_frame(tmp_path, 4))
        foreign = replace(_frame(tmp_path, 4), **{
            "generation": {"capture_generation": 3},
            "session": {"session_id": "another-session"},
            "phase": {"source_phase": "live_session"},
            "scope_id": {"scope_id": "another-run"},
        }[changed])
        reserved = evidence.reserved_bytes
        evidence.recover(foreign)
        assert evidence.writer.wait_idle()
        assert evidence.reserved_bytes == reserved
        assert [frame.capture_seq for frame in calls] == [3]
        assert len(evidence._pending_recovery) == 1
        assert not evidence.incident("page_unknown", failure)
        evidence.recover(_frame(tmp_path, 4))
        assert evidence.writer.wait_idle()
        assert [frame.capture_seq for frame in calls] == [3, 4]
    finally:
        assert evidence.writer.close()


def test_partial_recovery_enqueue_only_commits_each_accepted_incident(tmp_path):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None, queue_limit=1)
    failure = evidence.remember(_frame(tmp_path, 3))
    try:
        for reason in ("capture", "recognition"):
            assert evidence.incident(reason, failure)
            assert evidence.writer.wait_idle()
        reserved = evidence.reserved_bytes
        recovery = _frame(tmp_path, 4)
        recovery_budget = evidence._frame_budget(recovery) + 64 * 1024
        with _paused_writer(evidence.writer):
            evidence.recover(recovery)
            assert evidence.reserved_bytes == reserved + recovery_budget
            assert [item.reason for item in evidence._pending_recovery] == ["recognition"]
            assert (failure.capture_scope, "recognition") in evidence._seen
            assert (failure.capture_scope, "capture") not in evidence._seen
            assert not evidence.incident("recognition", failure)
            evidence.recover(recovery)
            assert evidence.reserved_bytes == reserved + recovery_budget
        evidence.recover(recovery)
        assert evidence.writer.wait_idle()
        evidence.recover(_frame(tmp_path, 5))
        assert evidence.writer.wait_idle()
        assert evidence.reserved_bytes == reserved + 2 * recovery_budget
        assert evidence.incident_count == 2
        assert [frame.capture_seq for frame in calls] == [3, 3, 4, 4]
        manifests = list((tmp_path / "diagnostic_frames").glob("incident_*.json"))
        assert len(manifests) == 2
        for manifest in manifests:
            document = json.loads(manifest.read_text(encoding="utf-8"))
            assert [frame["role"] for frame in document["frames"]] == ["failure", "recovery"]
    finally:
        assert evidence.writer.close()


def test_accepted_recovery_storage_failure_consumes_reservation_once(tmp_path):
    calls, errors = [], []
    save = _persist(tmp_path, calls)

    def fail_recovery(frame):
        if frame.details["incident"]["role"] == "recovery":
            calls.append(frame)
            raise OSError("disk full during recovery")
        return save(frame)

    evidence = ListenerEvidence(fail_recovery, lambda _, error: errors.append(error), queue_limit=1)
    failure = evidence.remember(_frame(tmp_path, 3))
    try:
        assert evidence.incident("capture", failure)
        assert evidence.writer.wait_idle()
        reserved = evidence.reserved_bytes
        recovery = _frame(tmp_path, 4)
        evidence.recover(recovery)
        assert evidence.writer.wait_idle()
        assert isinstance(errors[-1], OSError)
        assert evidence.reserved_bytes == reserved + evidence._frame_budget(recovery) + 64 * 1024
        reserved = evidence.reserved_bytes
        evidence.recover(recovery)
        evidence.recover(_frame(tmp_path, 5))
        assert evidence.writer.wait_idle()
        assert evidence.reserved_bytes == reserved
        assert [frame.capture_seq for frame in calls] == [3, 4]
        assert evidence._pending_recovery == []
        assert not evidence.incident("capture", failure)
    finally:
        assert evidence.writer.close()


def test_recovery_never_exceeds_run_byte_budget(tmp_path):
    calls = []
    failure = _frame(tmp_path, 3)
    budget = ListenerEvidence._frame_budget(failure) + 64 * 1024
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None,
                                max_bytes=budget, queue_limit=1)
    try:
        assert evidence.incident("capture", evidence.remember(failure))
        assert evidence.writer.wait_idle()
        for seq in (4, 4, 5):
            evidence.recover(_frame(tmp_path, seq))
        assert evidence.writer.wait_idle()
        assert evidence.reserved_bytes == budget
        assert [frame.capture_seq for frame in calls] == [3]
        assert len(evidence._pending_recovery) == 1
        assert not evidence.incident("capture", failure)
    finally:
        assert evidence.writer.close()


@pytest.mark.parametrize("job", ["incident", "recovery"])
def test_begin_run_rejects_queued_jobs_without_resetting_state(tmp_path, job):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None,
                                max_incidents=1, queue_limit=1)
    failure = evidence.remember(_frame(tmp_path, 3))
    try:
        if job == "recovery":
            assert evidence.incident("capture", failure)
            assert evidence.writer.wait_idle()
        with _paused_writer(evidence.writer):
            if job == "incident":
                assert evidence.incident("capture", failure)
            else:
                evidence.recover(_frame(tmp_path, 4))
            reserved = evidence.reserved_bytes
            seen = dict(evidence._seen)
            recovered = dict(evidence._recovered)
            pending = list(evidence._pending_recovery)
            assert evidence.begin_run() is False
            assert evidence.incident_count == 1
            assert evidence.reserved_bytes == reserved
            assert evidence._seen == seen
            assert evidence._recovered == recovered
            assert evidence._pending_recovery == pending
            assert evidence.find(failure.snapshot, 2, "preopening_listener") is failure
            assert evidence.directory_is_pinned(tmp_path)
            assert not evidence.incident("recognition", failure)
        assert evidence.begin_run() is True
        assert evidence.incident_count == 0
        assert evidence.reserved_bytes == 0
        assert evidence._seen == {}
        assert evidence._recovered == {}
        assert evidence._pending_recovery == []
        assert evidence.find(failure.snapshot, 2, "preopening_listener") is None
        assert evidence.directory_is_pinned(tmp_path)
        # A run reset releases dedup/quota but retains old evidence origins.
        assert evidence.incident("capture", evidence.remember(failure))
        assert evidence.writer.wait_idle()
        assert evidence.incident_count == 1
    finally:
        assert evidence.writer.close()


@pytest.mark.parametrize("blocked_stage", ["persist", "completed"])
def test_begin_run_rejects_active_operation_and_completion_callback(tmp_path, blocked_stage):
    calls = []
    entered, release = Event(), Event()
    save = _persist(tmp_path, calls)

    def pause():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release evidence writer")

    def persist(frame):
        if blocked_stage == "persist":
            pause()
        return save(frame)

    def completed(*_):
        if blocked_stage == "completed":
            pause()

    evidence = ListenerEvidence(persist, completed, queue_limit=1)
    try:
        assert evidence.incident("capture", evidence.remember(_frame(tmp_path, 3)))
        assert entered.wait(2)
        reserved = evidence.reserved_bytes
        assert evidence.begin_run() is False
        assert evidence.incident_count == 1
        assert evidence.reserved_bytes == reserved
        release.set()
        assert evidence.writer.wait_idle()
        assert evidence.begin_run() is True
        assert evidence.incident_count == 0
        assert evidence.reserved_bytes == 0
    finally:
        release.set()
        assert evidence.writer.close()


def test_capture_error_uses_last_good_frame_as_context_not_failure(tmp_path):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None)
    try:
        evidence.remember(_frame(tmp_path, 2))
        context = evidence.remember(_frame(tmp_path, 3))
        assert evidence.incident("capture", context, failure_role="context",
                                 details={"error": "capture returned no frame"})
        assert evidence.writer.wait_idle()
        evidence.recover(_frame(tmp_path, 4))
        assert evidence.writer.wait_idle()
        assert [frame.capture_seq for frame in calls] == [3, 2, 4]
        assert [frame.details["incident"]["role"] for frame in calls] == ["context", "prior", "recovery"]
        assert calls[0].source_phase == "last_listener_frame"
        assert all(frame.source_phase == "preopening_listener" for frame in calls[1:])
        manifest, = (tmp_path / "diagnostic_frames").glob("incident_*.json")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        assert document["has_failure_frame"] is False
        assert document["failure_role"] == "context"
        assert document["details"] == {"error": "capture returned no frame"}
        assert [frame["role"] for frame in document["frames"]] == ["prior", "context", "recovery"]
        assert all(frame["source_phase"] != "failed_listener_frame" for frame in document["frames"])
    finally:
        assert evidence.writer.close()


def test_context_capture_error_without_any_frame_writes_nothing(tmp_path):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None)
    try:
        assert not evidence.incident("capture", None, failure_role="context")
        assert evidence.incident_count == evidence.reserved_bytes == 0
        assert not evidence.directory_is_pinned(tmp_path)
        assert calls == []
        assert list(tmp_path.iterdir()) == []
    finally:
        assert evidence.writer.close()


@pytest.mark.parametrize("role", ["prior", "recovery", ""])
def test_incident_rejects_invalid_failure_role_without_side_effects(tmp_path, role):
    calls = []
    evidence = ListenerEvidence(_persist(tmp_path, calls), lambda *_: None)
    try:
        with pytest.raises(ValueError, match="failure_role"):
            evidence.incident("capture", _frame(tmp_path, 3), failure_role=role)
        assert evidence.incident_count == evidence.reserved_bytes == 0
        assert not evidence.directory_is_pinned(tmp_path)
        assert evidence._pending_recovery == []
        assert evidence._seen == {}
        assert calls == []
    finally:
        assert evidence.writer.close()
