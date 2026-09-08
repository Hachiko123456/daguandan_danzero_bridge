from __future__ import annotations

import hashlib
import json
from pathlib import Path

from daguandan_bridge.application.session_dataset_catalog import (
    FIXED_SPLITS,
    SessionDatasetCatalogBuilder,
)
from daguandan_bridge.domain.truth import TruthEvidence
from daguandan_bridge.domain.truth import LabelProvenance
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthTurn,
    save_truth_log,
)


HAND = tuple(f"{rank}{suit}" for rank in ("2", "3", "4", "5", "6", "7") for suit in "SHCD") + ("8S", "8H", "8C")


def _make_session(root: Path, session_id: str, *, truth: bool, timeline: bool, video: bool) -> Path:
    session = root / session_id
    (session / "video").mkdir(parents=True)
    (session / "manifest.json").write_text("{\"session_id\": \"x\"}\n", encoding="utf-8")
    if video:
        (session / "video" / "game.avi").write_bytes(b"video")
    (session / "video" / "frame_index.jsonl").write_text(
        "{\"frame_index\": 0}\n{\"frame_index\": 10}\n", encoding="utf-8"
    )
    if timeline:
        (session / "timeline.jsonl").write_text("{\"event_type\": \"initial_state_confirmed\"}\n", encoding="utf-8")
    if truth:
        save_truth_log(
            session / "truth_log.json",
            TruthLog(
                source_session_id=session_id,
                initial_state=TruthInitialState("6", "left", HAND),
                turns=(
                    TruthTurn(
                        1,
                        "left",
                        False,
                        ("2S", "2H"),
                        evidence=TruthEvidence((3, 4), 100),
                    ),
                    TruthTurn(2, "self", True, (), evidence=TruthEvidence((9,), 200)),
                ),
                label_status="verified",
                provenance=LabelProvenance(source="human_review", annotator="test"),
            ),
        )
    return session


def test_catalog_is_read_only_deterministic_and_truthlog_has_priority(tmp_path: Path):
    source = tmp_path / "sessions"
    source.mkdir()
    for split, session_ids in FIXED_SPLITS.items():
        for session_id in session_ids:
            _make_session(source, session_id, truth=True, timeline=True, video=True)
    _make_session(source, "game_silver", truth=False, timeline=True, video=True)
    _make_session(source, "manual_raw", truth=False, timeline=False, video=True)
    before = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }

    first = SessionDatasetCatalogBuilder().build(source, tmp_path / "out-1")
    second = SessionDatasetCatalogBuilder().build(source, tmp_path / "out-2")
    assert first.session_count == 11
    assert first.truth_session_count == 9
    assert first.verified_truth_session_count == 9
    assert first.draft_truth_session_count == 0
    assert first.case_count == 18
    assert first.catalog_path.read_bytes() == second.catalog_path.read_bytes()
    assert first.cases_path.read_bytes() == second.cases_path.read_bytes()
    assert first.gaps_path.read_bytes() == second.gaps_path.read_bytes()
    after = {
        path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert before == after

    catalog = json.loads(first.catalog_path.read_text(encoding="utf-8"))
    by_id = {item["session_id"]: item for item in catalog["sessions"]}
    assert by_id["game_silver"]["classification"] == "silver"
    assert by_id["manual_raw"]["classification"] == "raw"
    assert by_id["game_20260814_004447_aab3dc"]["split"] == "development"
    assert by_id["game_20260822_002135_9c2328"]["split"] == "release_gate"
    case = json.loads(first.cases_path.read_text(encoding="utf-8").splitlines()[0])
    assert case["truth_status"] == "human_confirmed"
    assert case["frame_range"] == {"start": 3, "end": 4}
    assert "opening" in case["scenario_tags"]
    assert case["video_path"].endswith("/video/game.avi")


def test_coverage_report_exposes_missing_fixed_sessions_and_tags(tmp_path: Path):
    source = tmp_path / "sessions"
    source.mkdir()
    _make_session(source, FIXED_SPLITS["development"][0], truth=True, timeline=True, video=True)
    result = SessionDatasetCatalogBuilder().build(source, tmp_path / "out")
    gaps = json.loads(result.gaps_path.read_text(encoding="utf-8"))
    assert gaps["gold_session_count"] == 1
    assert FIXED_SPLITS["regression"][0] in gaps["missing_fixed_split_sessions"]
    assert "triple" in gaps["missing_scenario_tags"]


def test_missing_truth_is_indexed_without_being_promoted(tmp_path: Path):
    source = tmp_path / "sessions"
    source.mkdir()
    _make_session(source, "game_unreviewed", truth=False, timeline=True, video=True)
    result = SessionDatasetCatalogBuilder().build(source, tmp_path / "out")
    catalog = json.loads(result.catalog_path.read_text(encoding="utf-8"))
    item = catalog["sessions"][0]
    assert item["classification"] == "silver"
    assert item["split"] == "unassigned"
    assert result.case_count == 0


def test_draft_truth_is_indexed_as_silver_and_conflicts_are_reported(tmp_path: Path):
    source = tmp_path / "sessions"
    source.mkdir()
    session = _make_session(source, "game_draft", truth=True, timeline=True, video=True)
    draft = json.loads((session / "truth_log.json").read_text(encoding="utf-8"))
    draft["label_status"] = "draft"
    draft["provenance"] = {"source": "legacy_migration"}
    (session / "truth_log.json").write_text(
        json.dumps(draft, ensure_ascii=False), encoding="utf-8"
    )
    (session / "timeline.jsonl").write_text(
        json.dumps({
            "event_type": "player_played",
            "event_id": "timeline-1",
            "actor": "left",
            "turn_id": 1,
            "payload": {"cards": ["9S"], "is_pass": False},
        }) + "\n",
        encoding="utf-8",
    )

    result = SessionDatasetCatalogBuilder().build(source, tmp_path / "out")
    catalog = json.loads(result.catalog_path.read_text(encoding="utf-8"))
    item = catalog["sessions"][0]
    case = json.loads(result.cases_path.read_text(encoding="utf-8").splitlines()[0])

    assert item["classification"] == "silver"
    assert item["truth_qualification"] == "draft_truth"
    assert item["truth_vs_timeline"]["status"] == "conflict"
    assert item["truth_vs_timeline"]["first_mismatch_turn"] == 1
    assert case["truth_status"] == "draft_truth"
