from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from daguandan_bridge.application.session_diagnostic_frames import (
    FRAME_SCHEMA,
    MANUAL_SOURCE_KIND,
    SOURCE_KIND,
    SessionDiagnosticFrameError,
    SessionDiagnosticFrameStore,
)
from daguandan_bridge.capture_service import FrameSnapshot
from daguandan_bridge.profiles import ProfileConfig
from daguandan_bridge.window_capture import CapturedStandardizedFrame
from daguandan_bridge.image_io import StandardizationResult


def _snapshot(image: np.ndarray, *, seq: int = 17) -> FrameSnapshot:
    standardization = StandardizationResult(
        image=image,
        source_size=(int(image.shape[1]), int(image.shape[0])),
        source_viewport=SimpleNamespace(to_list=lambda: [0, 0, image.shape[1], image.shape[0]]),
        content_box=SimpleNamespace(to_list=lambda: [0, 0, image.shape[1], image.shape[0]]),
        scale=1.0,
        padding=(0, 0, 0, 0),
        aspect_error=0.0,
        aspect_compatible=True,
    )
    frame = CapturedStandardizedFrame(
        standardization=standardization,
        rect=SimpleNamespace(left=10, top=20, width=image.shape[1], height=image.shape[0]),
        backend="screen",
        dpi=120,
        window_title="牌桌",
        raw_image=image.copy(),
    )
    return FrameSnapshot(
        frame=frame,
        captured_monotonic_ms=123456 + seq,
        evidence_frame_id=f"evidence-{seq}",
    )


def test_save_and_load_preserves_pixels_and_records_provenance(tmp_path):
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    store = SessionDiagnosticFrameStore()

    record = store.save_snapshot(
        tmp_path / "session-1",
        _snapshot(image),
        session_id="session-1",
        capture_generation=3,
        capture_seq=17,
    )

    assert record.image_path == tmp_path / "session-1" / "diagnostic_frames" / "000001.png"
    assert record.metadata_path == tmp_path / "session-1" / "diagnostic_frames" / "000001.json"
    assert store.load_image(record).shape == image.shape
    np.testing.assert_array_equal(store.load_image(record), image)
    assert record.metadata["schema"] == FRAME_SCHEMA
    assert record.metadata["source"] == SOURCE_KIND
    assert record.metadata["session_id"] == "session-1"
    assert record.metadata["capture_generation"] == 3
    assert record.metadata["capture_seq"] == 17
    assert record.metadata["evidence_frame_id"] == "evidence-17"
    assert record.metadata["raw_sha256"]
    assert record.metadata["png_sha256"]
    assert record.to_dict()["sequence"] == 1


def test_multiple_frames_and_restart_continue_sequence(tmp_path):
    image = np.full((3, 4, 3), 9, dtype=np.uint8)
    session = tmp_path / "session"
    first_store = SessionDiagnosticFrameStore()
    first = first_store.save_snapshot(session, _snapshot(image, seq=1), session_id="s", capture_generation=0, capture_seq=1)
    second = first_store.save_snapshot(session, _snapshot(image + 1, seq=2), session_id="s", capture_generation=0, capture_seq=2)

    restarted = SessionDiagnosticFrameStore()
    third = restarted.save_snapshot(session, _snapshot(image + 2, seq=3), session_id="s", capture_generation=1, capture_seq=3)

    assert (first.sequence, second.sequence, third.sequence) == (1, 2, 3)
    assert [item.sequence for item in restarted.list_frames(session)] == [1, 2, 3]


def test_incomplete_pairs_are_ignored_and_sequence_is_not_reused(tmp_path):
    image = np.full((2, 3, 3), 11, dtype=np.uint8)
    store = SessionDiagnosticFrameStore()
    record = store.save_snapshot(tmp_path / "session", _snapshot(image), session_id="s", capture_generation=0, capture_seq=1)
    record.metadata_path.unlink()

    assert store.list_frames(tmp_path / "session") == ()
    next_record = store.save_snapshot(tmp_path / "session", _snapshot(image), session_id="s", capture_generation=0, capture_seq=2)
    assert next_record.sequence == 2


def test_directory_and_symlink_safety(tmp_path):
    store = SessionDiagnosticFrameStore()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-session"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前 Windows 环境不允许创建测试符号链接")
    with pytest.raises(SessionDiagnosticFrameError):
        store.save_snapshot(link, _snapshot(image), session_id="s", capture_generation=0, capture_seq=1)

    session = tmp_path / "session"
    record = store.save_snapshot(session, _snapshot(image), session_id="s", capture_generation=0, capture_seq=1)
    link_file = record.image_path.with_name("000002.png")
    try:
        link_file.symlink_to(record.image_path)
    except (OSError, NotImplementedError):
        pytest.skip("当前 Windows 环境不允许创建测试符号链接")
    with pytest.raises(SessionDiagnosticFrameError):
        store.list_frames(session)


def test_hash_or_dimension_tampering_is_rejected(tmp_path):
    image = np.full((4, 5, 3), 7, dtype=np.uint8)
    store = SessionDiagnosticFrameStore()
    record = store.save_snapshot(tmp_path / "session", _snapshot(image), session_id="s", capture_generation=0, capture_seq=1)

    tampered = image.copy()
    tampered[0, 0, 0] = 8
    ok, encoded = cv2.imencode(".png", tampered)
    assert ok
    record.image_path.write_bytes(encoded.tobytes())
    with pytest.raises(SessionDiagnosticFrameError, match="哈希"):
        store.load_image(record)

    record.image_path.write_bytes(cv2.imencode(".png", image)[1].tobytes())
    payload = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    payload["width"] = 999
    record.metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionDiagnosticFrameError, match="尺寸"):
        store.load_image(record)



def test_manual_source_metadata_is_lossless_and_selectable(tmp_path):
    image = np.full((3, 4, 3), 19, dtype=np.uint8)
    store = SessionDiagnosticFrameStore()
    record = store.save_snapshot(
        tmp_path / "manual-diagnostic" ,
        _snapshot(image, seq=9),
        session_id="diagnostic_1",
        capture_generation=0,
        capture_seq=1,
        source=MANUAL_SOURCE_KIND,
        source_phase="manual_window_capture",
    )

    assert record.metadata["source"] == MANUAL_SOURCE_KIND
    assert record.metadata["source_phase"] == "manual_window_capture"
    np.testing.assert_array_equal(store.load_image(record), image)
    listed = store.list_frames(tmp_path / "manual-diagnostic")
    assert len(listed) == 1
    assert listed[0].metadata["source"] == MANUAL_SOURCE_KIND
