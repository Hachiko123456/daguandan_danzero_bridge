import json

import numpy as np

from daguandan_bridge.capture_service import CaptureService, FrameSnapshot
from daguandan_bridge.image_io import standardize_to_base
from daguandan_bridge.models import ClientRect
from daguandan_bridge.profiles import ProfileConfig, create_profile
from daguandan_bridge.window_capture import CapturedStandardizedFrame


def _frame_snapshot() -> FrameSnapshot:
    image = np.full((720, 1280, 3), 127, dtype=np.uint8)
    standardization = standardize_to_base(image, (1280, 720))
    frame = CapturedStandardizedFrame(
        standardization=standardization,
        rect=ClientRect(10, 20, 1280, 720),
        backend="test",
        dpi=96,
        window_title="Test Window",
    )
    return FrameSnapshot(frame)


def test_session_saves_sequential_frame_metadata_and_finish(tmp_path):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",)),
    )
    session = service.start_screenshot_session("test_game", 1500)

    saved = service.save_session_frame(session, _frame_snapshot())
    completed = service.finish_screenshot_session(session)
    document = json.loads(completed.metadata_path.read_text(encoding="utf-8"))

    assert saved.name == "000001.png"
    assert saved.is_file()
    assert document["capture_interval_ms"] == 1500
    assert document["frame_count"] == 1
    assert document["finished_at"] is not None
    assert document["first_capture"]["capture_backend"] == "test"


def test_session_rejects_non_positive_interval(tmp_path):
    service = CaptureService(tmp_path / "profiles")
    create_profile(
        service.profiles_root,
        ProfileConfig("test_game", "Test", ("Test",)),
    )

    try:
        service.start_screenshot_session("test_game", 0)
    except ValueError as exc:
        assert "录制间隔" in str(exc)
    else:
        raise AssertionError("expected an invalid interval to be rejected")
