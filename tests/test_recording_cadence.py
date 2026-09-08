from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from daguandan_bridge.gui.recording_cadence import RecordingCadenceGate


def test_virtual_153_second_high_rate_capture_is_bounded_to_ten_fps() -> None:
    gate = RecordingCadenceGate(10)
    timestamps = tuple(round(index * 153_000 / 4_749) for index in range(4_750))
    admitted = [stamp for stamp in timestamps if gate.admit(stamp)]

    assert 1_529 <= len(admitted) <= 1_531
    assert admitted[-1] - admitted[0] >= 152_900
    stats = gate.stats
    assert stats.observed == 4_750
    assert stats.admitted == len(admitted)
    assert stats.sampled_out == 4_750 - len(admitted)
    assert stats.to_dict()["policy"] == "capture_timestamp_absolute_cadence"


def test_cadence_uses_capture_timestamps_not_call_rate_or_source_fields() -> None:
    gate = RecordingCadenceGate(10)
    assert [gate.admit(value) for value in (1_000, 1_010, 1_099, 1_100, 1_450)] == [
        True, False, False, True, True
    ]
    assert gate.stats.admitted == 3
    assert gate.stats.sampled_out == 2


def test_repository_game_material_first_153_seconds_stays_within_target_rate() -> None:
    video = Path(
        "data/profiles/tencent_daguandan/sessions/"
        "game_20260814_004447_aab3dc/video/game.avi"
    )
    if not video.is_file():
        pytest.skip("repository validation AVI is unavailable")
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert fps > 0 and frames > 0
    sample_count = min(frames, int(153 * fps))
    gate = RecordingCadenceGate(10)
    admitted = sum(gate.admit(round(index * 1000 / fps)) for index in range(sample_count))

    assert admitted <= 1_532
    assert gate.stats.sampled_out == sample_count - admitted
