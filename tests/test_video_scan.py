import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from daguandan_bridge.application.action_trace_projection import ActionTraceProjector
from daguandan_bridge.application.video_scan import VideoActionScanner, VideoScanRequest


class _Result:
    def __init__(self, cards=(), is_pass=False, confidence=0.9, source="fake"):
        self.cards = tuple(cards)
        self.is_pass = is_pass
        self.confidence = confidence
        self.source = source
        self.diagnostics = ()
        self.suit_options = ()


class _Recognition:
    def __init__(self):
        self.calls = []

    def recognize(self, frame, *, allow_unknown_suit=False):
        self.calls.append(("frame", int(frame[0, 0, 0]), allow_unknown_suit))
        return type("Opening", (), {
            "round_level": "6", "my_hand": ("6S",),
            "lead_player": None, "current_player": None,
            "unresolved_fields": (), "diagnostics": (), "buttons": (),
        })()

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_unknown_suit, allow_pass):
        value = int(frame[0, 0, 0])
        self.calls.append((seat, value, wild_rank, allow_unknown_suit, allow_pass))
        if seat == "left" and value in {1, 2}:
            return _Result(("2?",), confidence=0.7)
        if seat == "left" and value == 4:
            return _Result(("2H",), confidence=0.95)
        if seat == "self" and value == 6:
            return _Result(is_pass=True, confidence=0.9)
        return _Result()


class _ArrayCapture:
    """Small lossless in-memory capture used to prove exact cache semantics."""

    def __init__(self, frames):
        self._frames = [frame.copy() for frame in frames]
        self._ordinal = 0

    def isOpened(self):
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FPS:
            return 10.0
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        return 0

    def read(self):
        if self._ordinal >= len(self._frames):
            return False, None
        frame = self._frames[self._ordinal].copy()
        self._ordinal += 1
        return True, frame

    def release(self):
        return None


class _RoiRecognition:
    """Recognizer with independent play crops and a stable context crop."""

    def __init__(self, *, fail_left_once=False):
        self.calls = []
        self.fail_left_once = fail_left_once
        records = (
            type("Region", (), {"name": "my_hand", "role": "hand", "ratio_box": (0.0, 0.0, 0.25, 1.0)})(),
            type("Region", (), {"name": "level_rank", "role": "generic", "ratio_box": (0.25, 0.0, 0.25, 1.0)})(),
            type("Region", (), {"name": "left_play", "role": "play", "ratio_box": (0.75, 0.0, 0.25, 1.0)})(),
        )
        self.annotation_service = type("Annotations", (), {
            "list_regions": lambda _self: records,
        })()

    def recognize(self, frame, *, allow_unknown_suit=False):
        self.calls.append(("frame", int(frame[0, 0, 0])))
        return type("Opening", (), {
            "round_level": "6", "my_hand": ("6S",), "lead_player": None,
            "current_player": None, "unresolved_fields": (), "diagnostics": (),
            "buttons": (),
        })()

    def play_roi(self, frame, seat):
        index = ("self", "right", "opposite", "left").index(seat)
        return frame[:, index * 2:(index + 1) * 2]

    def recognize_play_region(self, frame, seat, *, wild_rank, allow_unknown_suit, allow_pass):
        marker = int(self.play_roi(frame, seat)[0, 0, 0])
        self.calls.append((seat, marker))
        if seat == "left" and self.fail_left_once:
            self.fail_left_once = False
            raise RuntimeError("temporary left-region failure")
        return _Result((f"{marker}H",) if marker else ())


def _video(path: Path, count: int = 7) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (8, 8))
    assert writer.isOpened()
    for value in range(count):
        writer.write(np.full((8, 8, 3), value, dtype=np.uint8))
    writer.release()


def test_scan_without_index_or_session_logs_writes_every_decoded_frame(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    video = source / "game.avi"
    _video(video)
    (source / "timeline.jsonl").write_text("not input\n", encoding="utf-8")
    (source / "truth_log.json").write_text(json.dumps({"lead_player": "wrong"}), encoding="utf-8")
    recognition = _Recognition()
    result = VideoActionScanner(recognition).scan(
        VideoScanRequest(video, output, session_id="video-only")
    )
    assert result.status == "complete"
    assert result.decoded_frames == 7
    assert result.observation_count == 7
    assert result.failed_frames == 0
    assert result.needs_review_count >= 1
    with gzip.open(result.observations_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 7
    assert [row["frame_index"] for row in rows] == list(range(7))
    assert {row["timestamp_source"] for row in rows} == {"video_fps"}
    assert json.loads(result.manifest_path.read_text(encoding="utf-8"))["input_policy"]["video_only"]
    assert not (source / "frame_observations.jsonl.gz").exists()


def test_scan_rejects_writing_generated_output_into_source_tree(tmp_path):
    source = tmp_path / "session"
    video_directory = source / "video"
    video_directory.mkdir(parents=True)
    video = video_directory / "game.avi"
    _video(video, count=1)
    with np.testing.assert_raises_regex(ValueError, "源视频目录之外"):
        VideoActionScanner(_Recognition()).scan(VideoScanRequest(video, source))


def test_scan_uses_index_for_timestamp_only_and_keeps_short_video_failures(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    video = source / "game.avi"
    _video(video, count=2)
    index = source / "frame_index.jsonl"
    index.write_text("\n".join(json.dumps({"frame_index": 10 + i, "monotonic_ms": 500 + i * 33}) for i in range(4)) + "\n", encoding="utf-8")
    result = VideoActionScanner(_Recognition()).scan(VideoScanRequest(video, output, index))
    with gzip.open(result.observations_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 4
    assert rows[0]["frame_index"] == 10
    assert rows[0]["timestamp_ms"] == 500
    assert rows[2]["decode_ok"] is False
    assert "decode_error:video_ended_before_frame_index" in rows[2]["recognition_errors"]
    assert result.failed_frames == 2
    assert "video_shorter_than_frame_index" in json.loads(result.summary_path.read_text(encoding="utf-8"))["warnings"]


def test_projector_preserves_variants_repairs_suit_and_splits_after_empty_gap():
    def row(frame, left=(), self_pass=False):
        return {"frame_index": frame, "timestamp_ms": frame * 100, "decode_ok": True,
                "regions": {
                    "left": {"cards": list(left), "is_pass": False, "confidence": .8, "source": "fake"},
                    "self": {"cards": [], "is_pass": self_pass, "confidence": .8, "source": "fake"},
                }}

    result = ActionTraceProjector().project([
        row(0), row(1, ("2?",)), row(2, ("2H",)), row(3), row(4, ("2H",)), row(5, self_pass=True),
    ])
    left = [item for item in result["actions"] if item["actor"] == "left"]
    assert len(left) == 2
    assert left[0]["cards"] == ["2H"]
    assert left[0]["evidence_frames"] == [1, 2]
    assert left[0]["cards_before_repair"] == ["2?"]
    assert left[0]["repair_status"] == "resolved"
    assert left[0]["repair_frame"] == 2
    assert left[0]["repair_reason"] == "later_frame_resolved_unknown_suit"
    assert left[0]["uncertainty"] == []
    assert left[1]["frame_start"] == 4
    self_actions = [item for item in result["actions"] if item["actor"] == "self"]
    assert self_actions[0]["is_pass"] is True


def test_projector_identifies_video_opening_as_unverified_candidate():
    observations = []
    for frame in range(3):
        observations.append({"frame_index": frame, "timestamp_ms": frame, "decode_ok": True,
                             "regions": {"left": {"cards": ["2H" if frame else ""], "is_pass": False},
                                         "self": {}, "right": {}, "opposite": {}}})
    opening = ActionTraceProjector().project(observations)["opening"]
    assert opening["lead_player"] == "left"
    assert opening["status"] == "needs_review"


def test_projector_drops_preopening_pass_noise_and_merges_card_animation():
    def row(frame, *, opposite_pass=False, self_cards=()):
        return {"frame_index": frame, "timestamp_ms": frame * 100, "decode_ok": True,
                "regions": {
                    "opposite": {"cards": [], "is_pass": opposite_pass, "confidence": .68, "source": "passed-template"},
                    "self": {"cards": list(self_cards), "is_pass": False, "confidence": .8, "source": "cards-template"},
                }}

    observations = [
        *[row(frame, opposite_pass=True) for frame in range(72, 80)],
        row(112, self_cards=("2D", "2?", "3?", "3D", "4H", "4D")),
        row(113, self_cards=("2D", "2D", "3?", "4H", "4D")),
        row(114, self_cards=("2D", "2C", "3?", "3?", "4H", "4D")),
        row(115, self_cards=("2D", "2C", "3?", "3D", "4H")),
        *[row(frame, self_cards=("2D", "2C", "3H", "3D", "4H", "4D")) for frame in range(116, 165)],
    ]

    result = ActionTraceProjector().project(observations)

    assert result["actions"]
    first = result["actions"][0]
    assert first["actor"] == "self"
    assert first["is_pass"] is False
    assert first["cards"] == ["2C", "2D", "3D", "3H", "4D", "4H"]
    assert first["frame_start"] == 112
    assert first["frame_end"] == 164
    assert first["best_frame"] == 116
    assert first["evidence_frames"] == list(range(112, 165))
    assert all(item["actor"] != "opposite" or not item["is_pass"] for item in result["actions"][:1])
    assert len([item for item in result["actions"] if item["actor"] == "self"]) == 1


def test_projector_splits_direct_replacement_of_complete_display():
    def row(frame, cards):
        return {"frame_index": frame, "timestamp_ms": frame, "decode_ok": True,
                "regions": {"left": {"cards": list(cards), "is_pass": False, "confidence": .9}}}

    result = ActionTraceProjector().project([
        row(0, ("2H",)), row(1, ("2D",)),
    ])
    left = [item for item in result["actions"] if item["actor"] == "left"]
    assert len(left) == 2
    assert [item["cards"] for item in left] == [["2H"], ["2D"]]


def test_scan_reports_progress_actions_and_honors_stop_request(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    video = source / "game.avi"
    _video(video, count=7)
    progress = []
    actions = []
    result = VideoActionScanner(_Recognition()).scan(
        VideoScanRequest(video, output),
        stop_requested=lambda: len(progress) >= 4,
        on_progress=lambda done, total, frame: progress.append((done, total, frame)),
        on_action=actions.append,
    )
    assert result.status == "stopped"
    assert result.decoded_frames == 4
    assert [row[0] for row in progress] == [1, 2, 3, 4]
    assert actions


def test_scan_exact_frame_and_play_roi_cache_keeps_all_observation_rows(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    video = source / "game.avi"
    video.touch()
    first = np.zeros((4, 8, 3), dtype=np.uint8)
    same_as_first = first.copy()
    right_changed = first.copy()
    right_changed[:, 2:4] = 9
    recognition = _RoiRecognition()
    result = VideoActionScanner(
        recognition,
        video_capture_factory=lambda _path: _ArrayCapture((first, same_as_first, right_changed)),
    ).scan(VideoScanRequest(video, output))

    with gzip.open(result.observations_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 3
    assert rows[0]["recognition_mode"] == "fresh"
    assert rows[1]["recognition_mode"] == "reused"
    assert rows[1]["reused_from_frame"] == 0
    assert rows[2]["recognition_mode"] == "fresh"
    assert rows[2]["regions"]["self"]["recognition_mode"] == "reused"
    assert rows[2]["regions"]["right"]["recognition_mode"] == "fresh"
    assert rows[2]["regions"]["left"]["recognition_mode"] == "reused"
    assert rows[0]["frame_sha256"] == rows[1]["frame_sha256"]
    assert rows[0]["regions"]["left"]["roi_sha256"] == rows[2]["regions"]["left"]["roi_sha256"]
    assert rows[0]["regions"]["right"]["roi_sha256"] != rows[2]["regions"]["right"]["roi_sha256"]
    assert [item for item in recognition.calls if item[0] == "frame"] == [("frame", 0), ("frame", 0)]
    assert len([item for item in recognition.calls if item[0] != "frame"]) == 5
    assert result.frame_cache_hits == 1
    assert result.roi_cache_hits == 3
    assert result.fresh_recognitions == 7
    assert result.reused_recognitions == 8
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["recognition_cache"] == {
        "fresh_recognitions": 7, "reused_recognitions": 8,
        "frame_cache_hits": 1, "roi_cache_hits": 3,
    }


def test_scan_does_not_cache_recognition_exceptions_as_empty_results(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    video = source / "game.avi"
    video.touch()
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    recognition = _RoiRecognition(fail_left_once=True)
    result = VideoActionScanner(
        recognition,
        video_capture_factory=lambda _path: _ArrayCapture((frame, frame, frame)),
    ).scan(VideoScanRequest(video, output))

    with gzip.open(result.observations_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert "region_recognition_error:left:RuntimeError:temporary left-region failure" in rows[0]["recognition_errors"]
    assert rows[1]["recognition_errors"] == []
    assert rows[1]["regions"]["left"]["recognition_mode"] == "fresh"
    assert rows[2]["recognition_mode"] == "reused"
    assert len([item for item in recognition.calls if item[0] == "left"]) == 2
    assert result.frame_cache_hits == 1


def test_scan_reuses_non_play_context_when_only_play_roi_changes(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    video = source / "game.avi"
    video.touch()
    first = np.zeros((4, 8, 3), dtype=np.uint8)
    changed_play = first.copy()
    changed_play[:, 6:8] = 9
    recognition = _RoiRecognition()
    result = VideoActionScanner(
        recognition,
        video_capture_factory=lambda _path: _ArrayCapture((first, changed_play)),
    ).scan(VideoScanRequest(video, output))

    with gzip.open(result.observations_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows[0]["recognition_cache"]["context"]["hash_source"] == "non_play_rois"
    assert rows[1]["recognition_cache"]["context"]["mode"] == "reused"
    assert rows[1]["recognition_cache"]["context"]["reused_from_frame"] == 0
    assert rows[1]["regions"]["left"]["recognition_mode"] == "fresh"
    assert result.fresh_recognitions == 6
    assert result.reused_recognitions == 4  # context + three unchanged play ROIs
    assert [item for item in recognition.calls if item[0] == "frame"] == [("frame", 0)]


def test_pending_suit_frame_retention_is_hard_capped_and_released():
    """The rereread window must stay bounded: it used to hold the whole run.

    An unbounded window measured ~0.65 GB per session (245 full 1280x720 BGR
    frames), doubled by the two parallel batch workers, which paged the desktop
    to death.  Evicted frames are served by the targeted seek reread instead.
    """
    from daguandan_bridge.application import video_scan

    store = video_scan._PendingSuitFrameStore()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hidden = {"regions": {"left": {"cards": ["A?"], "is_pass": False}}}
    for index in range(video_scan._MAX_RETAINED_SUIT_FRAMES + 10):
        store.observe(index, frame, hidden)

    assert len(store.frames) == video_scan._MAX_RETAINED_SUIT_FRAMES
    assert store.evicted == 10
    assert min(store.frames) == 10  # the oldest frames are the ones evicted

    store.release()
    assert store.frames == {}
