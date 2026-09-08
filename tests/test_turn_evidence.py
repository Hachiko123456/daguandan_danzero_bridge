import numpy as np

from daguandan_bridge.live.turn_evidence import TurnEvidenceCache


def test_cache_is_memory_bounded_and_copies_input():
    frame = np.zeros((8, 8, 3), np.uint8)
    cache = TurnEvidenceCache(max_frames=3, max_bytes=frame.nbytes * 2)
    for stamp in (100, 200, 300, 400):
        cache.observe(frame, captured_ms=stamp, generation=2)
    assert cache.retained_bytes == frame.nbytes * 2
    before = cache.before(400, generation=2)
    assert before is not None and before.captured_ms == 300
    frame[:] = 255
    assert not before.frame.any()
    assert not before.frame.flags.writeable


def test_cache_rejects_future_duplicate_expired_and_wrong_generation():
    frame = np.zeros((8, 8, 3), np.uint8)
    cache = TurnEvidenceCache(max_age_ms=500)
    cache.observe(frame, captured_ms=100, generation=2)
    cache.observe(frame + 1, captured_ms=100, generation=2)
    assert cache.before(100, generation=2) is None
    assert not cache.before(200, generation=2).frame.any()
    assert cache.before(601, generation=2) is None
    assert cache.before(200, generation=3) is None
    cache.observe(frame, captured_ms=200, generation=3)
    assert cache.before(200, generation=3) is None


def test_cache_drops_oversized_frame_without_exceeding_quota():
    cache = TurnEvidenceCache(max_bytes=100)
    cache.observe(np.zeros((8, 8, 3), np.uint8), captured_ms=100, generation=1)
    assert cache.retained_bytes == 0
    assert cache.before(200, generation=1) is None


def test_consumed_seat_never_uses_pre_action_pixels_as_next_cycle_baseline():
    cache = TurnEvidenceCache()
    frame = np.zeros((8, 8, 3), np.uint8)
    cache.observe(frame, captured_ms=100, generation=2)
    cache.observe(frame + 255, captured_ms=200, generation=2)
    cache.mark_consumed("right", 220)
    assert cache.before(300, generation=2, seat="right") is None
    assert cache.before(300, generation=2, seat="left").captured_ms == 100
    cache.observe(frame + 255, captured_ms=250, generation=2)
    assert cache.before(300, generation=2, seat="right").captured_ms == 250


def test_seat_roi_history_survives_full_size_frame_ring_pressure():
    cache = TurnEvidenceCache(max_frames=4, max_bytes=16 * 1024 * 1024)
    frame = np.zeros((720, 1280, 3), np.uint8)
    roi = np.zeros((32, 64, 3), np.uint8)
    for index in range(20):
        stamp = 1_000 + index * 50
        frame[0, 0, 0] = index
        roi[:, :, 0] = index
        cache.observe(
            frame,
            captured_ms=stamp,
            generation=3,
            seat_rois={"opposite": roi, "left": roi + 1},
        )

    assert cache.retained_bytes <= cache.max_bytes
    full_anchor = cache.before(1_950, generation=3, seat="opposite")
    roi_anchor = cache.before_roi(1_950, generation=3, seat="opposite")
    assert full_anchor is not None and full_anchor.captured_ms > 1_000
    assert roi_anchor is not None and roi_anchor.captured_ms == 1_000
    assert not roi_anchor.frame.flags.writeable


def test_seat_roi_history_respects_consumed_and_generation_boundaries():
    cache = TurnEvidenceCache(max_frames=1, max_bytes=1_000_000)
    frame = np.zeros((64, 64, 3), np.uint8)
    roi = np.zeros((8, 8, 3), np.uint8)
    cache.observe(frame, captured_ms=100, generation=2, seat_rois={"right": roi})
    cache.observe(frame, captured_ms=200, generation=2, seat_rois={"right": roi + 1})
    cache.mark_consumed("right", 150)
    assert cache.before_roi(300, generation=2, seat="right").captured_ms == 200
    assert cache.before_roi(300, generation=3, seat="right") is None
    cache.observe(frame, captured_ms=400, generation=3, seat_rois={"right": roi + 2})
    assert cache.before_roi(400, generation=3, seat="right") is None
    assert cache.before_roi(450, generation=3, seat="right").captured_ms == 400
