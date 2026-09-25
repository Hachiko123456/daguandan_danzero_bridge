from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.application.session_qualification import qualify_session, qualify_sessions


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _session(root: Path, name: str, *, truth: str = "verified", initial: str = "confirmed", resource: bool = True, evidence: bool = False) -> Path:
    session = root / name
    (session / "video").mkdir(parents=True)
    (session / "video" / "game.avi").write_bytes(b"not-a-real-video")
    (session / "video" / "frame_index.jsonl").write_text('{"frame_index": 0}\n', encoding="utf-8")
    manifest = {"session_id": name, "profile": "test", "status": "sealed", "frame_count": 1, "initial_state_status": initial, "recording_phase": "live"}
    if resource:
        manifest["configuration_hash"] = "x"
        manifest["template_manifest_hash"] = "y"
    _write_json(session / "manifest.json", manifest)
    _write_json(session / "truth_log.json", {"schema": "guandan.truth/4", "source_session_id": name, "label_status": truth, "initial_state": {"round_level": "2", "lead_player": "self", "my_hand": ["2S"]}, "turns": []})
    if evidence:
        _write_json(session / "opening_evidence.json", {"status": "confirmed", "observable": True, "hand_count": 27, "frame_indices": [0, 1]})
    return session


def test_fake_video_never_becomes_strict(tmp_path: Path):
    sessions = tmp_path / "sessions"
    session = _session(sessions, "game_fake", evidence=True)
    record = qualify_session(session, tmp_path / "profile")
    assert record["classification"] == "invalid"
    assert record["strict_eligible"] is False
    assert "video_not_decodable" in record["reasons"]


def test_manifest_and_verified_truth_are_not_opening_proof(tmp_path: Path):
    sessions = tmp_path / "sessions"
    session = _session(sessions, "game_unproven")
    record = qualify_session(session, tmp_path / "profile")
    assert record["strict_eligible"] is False
    assert record["opening_observability"]["status"] == "unproven"


def test_unconfirmed_source_is_not_observable(tmp_path: Path):
    sessions = tmp_path / "sessions"
    session = _session(sessions, "game_missing", initial="unconfirmed")
    record = qualify_session(session, tmp_path / "profile")
    assert record["classification"] == "invalid"
    assert record["strict_eligible"] is False


def test_deterministic_order_and_diagnostic_classification(tmp_path: Path):
    sessions = tmp_path / "sessions"
    _session(sessions, "z_manual", truth="missing", resource=False)
    _session(sessions, "a_manual", truth="draft", resource=False)
    rows = qualify_sessions(sessions, tmp_path / "profile")
    assert [row["session_id"] for row in rows] == ["a_manual", "z_manual"]
    assert all(row["classification"] in {"diagnostic_only", "invalid"} for row in rows)


def test_resource_identity_is_required_for_strict(tmp_path: Path):
    sessions = tmp_path / "sessions"
    session = _session(sessions, "game_resource", evidence=True, resource=False)
    # Make the bytes decodable status irrelevant to this assertion; missing
    # identity alone must still forbid strict qualification.
    record = qualify_session(session, tmp_path / "profile")
    assert record["strict_eligible"] is False
    assert record["classification"] in {"resource_mismatch", "invalid"}


