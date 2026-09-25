from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import daguandan_bridge.application.opening_lead_diagnosis as diagnosis
from daguandan_bridge.application.opening_lead_diagnosis import diagnose_session


def _write_fixture(root: Path) -> tuple[Path, dict[str, str]]:
    session = root / "session"
    video = session / "video"
    video.mkdir(parents=True)
    trace = [
        {
            "phase": "waiting_for_initial_state",
            "round_level": "3",
            "hand_count": 26,
            "diagnostics": ["opening lead unresolved"],
        },
        {
            "phase": "waiting_for_initial_state",
            "round_level": "3",
            "hand_count": 27,
            "diagnostics": ["opening lead unresolved"],
        },
        {
            "phase": "waiting_for_initial_state",
            "round_level": "3",
            "hand_count": 27,
            "diagnostics": ["opening lead unresolved"],
        },
    ]
    (session / "recognition_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8"
    )
    (video / "frame_index.jsonl").write_text(
        json.dumps({"frame_index": 0, "wall_time": "2026-09-23T00:00:00+08:00"}) + "\n",
        encoding="utf-8",
    )
    (video / "game.avi").write_bytes(b"fixture video is not decoded without --frame")
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (session / "recognition_trace.jsonl", video / "frame_index.jsonl", video / "game.avi")
    }
    return session, hashes


def test_fixture_report_counts_hand27_and_opening_misses_without_writing(tmp_path):
    session, before = _write_fixture(tmp_path)

    report = diagnose_session(session)

    assert report["read_only"] is True
    assert report["summary"]["trace_rows"] == 3
    assert report["summary"]["hand27_count"] == 2
    assert report["summary"]["opening_unrecognized_count"] == 3
    assert report["frame"] is None
    after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            session / "recognition_trace.jsonl",
            session / "video" / "frame_index.jsonl",
            session / "video" / "game.avi",
        )
    }
    assert after == before


def test_fixture_frame_report_keeps_card_evidence_separate_from_weak_marker(tmp_path, monkeypatch):
    session, _ = _write_fixture(tmp_path)
    fake_result = SimpleNamespace(
        round_level="3",
        my_hand=tuple(f"{rank}S" for rank in ("A", "K")),
        current_player="left",
        lead_player=None,
        field_confidences={"events": 0.91},
        unresolved_fields=("lead_player",),
        diagnostics=("opening lead unresolved",),
        events=(SimpleNamespace(
            player="opposite",
            cards=("2H",),
            is_pass=False,
            confidence=0.91,
            source="template:cards",
        ),),
    )
    fake_trace = {
        "schema": "guandan.recognition-trace/1",
        "candidates": [
            {
                "field": "first_play_opposite",
                "score": 0.64,
                "threshold": 0.80,
                "accepted": False,
                "rejection_reason": "below_threshold",
            },
            {
                "field": "timer_left",
                "score": 0.79,
                "threshold": 0.62,
                "accepted": True,
            },
            {
                "field": "opposite_play",
                "score": 0.91,
                "threshold": 0.60,
                "accepted": True,
            },
        ],
    }
    monkeypatch.setattr(
        diagnosis,
        "_analyze_frame",
        lambda _session, _frame: diagnosis._FrameEvidence(
            frame_index=0,
            frame_index_record={"frame_index": 0},
            recognition=diagnosis._recognition_dict(fake_result, fake_trace),
            trace=fake_trace,
        ),
    )

    report = diagnose_session(session, frame=0)
    recognition = report["frame"]["recognition"]

    assert recognition["events"][0]["cards"] == ["2H"]
    assert recognition["marker"]["candidate_player"] == "opposite"
    assert recognition["marker"]["below_threshold"] is True
    assert recognition["marker"]["score"] == 0.64
    assert recognition["timer"]["candidate_player"] == "left"
    assert recognition["timer"]["accepted"] is True
    assert report["summary"]["first_reliable_play_action"]["cards"] == ["2H"]


def test_cli_help_is_available():
    script = Path(__file__).parents[1] / "scripts" / "diagnose_opening_lead.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "read-only diagnosis" in completed.stdout.lower()
    assert "--frame" in completed.stdout
