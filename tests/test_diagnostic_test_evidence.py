from __future__ import annotations

import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from daguandan_bridge import automatic_log_delivery as delivery
from daguandan_bridge.diagnostic_test_evidence import build_test_evidence


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _session(tmp_path: Path, *, with_truth: bool = True) -> Path:
    session = tmp_path / "sessions" / "game_test_001"
    session.mkdir(parents=True)
    _write_json(
        session / "manifest.json",
        {"schema": "test/1", "status": "sealed", "frame_count": 10},
    )
    events = [
        {
            "event_id": "e1",
            "event_type": "initial_state_confirmed",
            "actor": "left",
            "turn_id": 1,
            "monotonic_ms": 100,
            "evidence_refs": [],
            "payload": {"lead_player": "left", "round_level": "6"},
        },
        {
            "event_id": "e1b",
            "event_type": "lead_player_confirmed",
            "actor": "left",
            "turn_id": 1,
            "monotonic_ms": 120,
            "evidence_refs": [],
            "payload": {"lead_player": "left"},
        },
        {
            "event_id": "e2",
            "event_type": "player_played",
            "actor": "left",
            "turn_id": 1,
            "monotonic_ms": 200,
            "evidence_refs": ["o1"],
            "payload": {"cards": ["2S"], "frame_indices": [10, 11]},
        },
        {
            "event_id": "e3",
            "event_type": "player_played",
            "actor": "self",
            "turn_id": 2,
            "monotonic_ms": 300,
            "evidence_refs": ["o2"],
            "payload": {"cards": ["3S"], "frame_indices": [11, 12]},
        },
        {
            "event_id": "e4",
            "event_type": "rule_rejection",
            "actor": "right",
            "turn_id": 3,
            "monotonic_ms": 400,
            "evidence_refs": [],
            "payload": {"reason": "foreign_candidate"},
        },
        {
            "event_id": "e5",
            "event_type": "recovery_budget_exceeded",
            "actor": "opposite",
            "turn_id": 4,
            "monotonic_ms": 500,
            "evidence_refs": [],
            "payload": {},
        },
        {
            "event_id": "e6",
            "event_type": "advice_withheld",
            "actor": "self",
            "turn_id": 5,
            "monotonic_ms": 600,
            "evidence_refs": [],
            "payload": {"reason": "untrusted_history"},
        },
    ]
    _write_jsonl(session / "timeline.jsonl", events)
    _write_jsonl(
        session / "advice.jsonl",
        [
            {"request_id": "a1", "status": "requested", "turn_id": 2},
            {"request_id": "a1", "status": "ready", "turn_id": 2},
        ],
    )
    _write_jsonl(
        session / "decisions.jsonl",
        [
            {
                "request_id": "a1",
                "status": "ready",
                "turn_id": 2,
                "model_advice": {"cards": ["3S"], "is_pass": False},
            }
        ],
    )
    _write_jsonl(
        session / "observations.jsonl",
        [
            {
                "id": "o1",
                "player": "left",
                "cards": ["2S"],
                "frame_index": 10,
                "monotonic_ms": 200,
                "confidence": 0.9,
                "source": "test",
            },
            {
                "id": "o2",
                "player": "self",
                "cards": ["3S"],
                "frame_index": 11,
                "monotonic_ms": 300,
                "confidence": 0.9,
                "source": "test",
            },
            {
                "id": "foreign",
                "player": "right",
                "cards": ["4S"],
                "frame_index": 9,
                "monotonic_ms": 150,
                "confidence": 0.8,
                "source": "test",
            },
        ],
    )
    raw = (session / "observations.jsonl").read_bytes()
    with gzip.open(session / "observations.jsonl.gz", "wb") as handle:
        handle.write(raw)
    (session / "observations.jsonl").unlink()
    _write_json(session / "health_audit.json", {"status": "FAIL", "issues": []})
    (session / "video").mkdir()
    _write_jsonl(session / "video" / "frame_index.jsonl", [{"frame_index": 10}])
    if with_truth:
        _write_json(
            session / "truth_log.json",
            {
                "label_status": "human_confirmed",
                "turns": [{"actor": "left", "cards": ["2S"], "is_pass": False}],
            },
        )
    return session


def _read_jsonl_bytes(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]


def test_evidence_is_bounded_and_marks_program_output_unverified(tmp_path: Path) -> None:
    bundle = build_test_evidence(_session(tmp_path), max_bytes=16 * 1024)

    assert set(bundle.files) == {
        "test_evidence/session_facts.json",
        "test_evidence/opening_evidence.jsonl",
        "test_evidence/proposed_actions.jsonl",
        "test_evidence/action_evidence.jsonl",
        "test_evidence/anomaly_windows.jsonl",
        "test_evidence/advice_input_output.jsonl",
    }
    assert sum(len(value) for value in bundle.files.values()) <= 16 * 1024
    assert bundle.truncated is False
    facts = json.loads(bundle.files["test_evidence/session_facts.json"])
    assert facts["truth_status"] == "unverified"
    assert facts["truth_log_present"] is True
    anomalies = _read_jsonl_bytes(bundle.files["test_evidence/anomaly_windows.jsonl"])
    types = {str(row["type"]) for row in anomalies}
    assert {"preopening_foreign_candidate", "same_frame_multi_seat", "rule_rejection_streak", "recovery_budget_exceeded", "untrusted_advice"} <= types


def test_small_budget_truncates_only_at_jsonl_line_boundaries(tmp_path: Path) -> None:
    bundle = build_test_evidence(_session(tmp_path), max_bytes=16 * 1024)
    for name, data in bundle.files.items():
        assert len(data) <= 16 * 1024
        if name.endswith(".jsonl"):
            for line in data.splitlines():
                json.loads(line)


def test_automatic_delivery_publishes_evidence_without_media(tmp_path: Path) -> None:
    session = _session(tmp_path)
    before = {p.relative_to(session).as_posix(): p.read_bytes() for p in session.rglob("*") if p.is_file()}
    result = delivery.AutomaticLogDeliveryService(documents_root=tmp_path / "docs").export(session)

    assert result.status == "PASS"
    evidence_dir = result.output_directory / "test_evidence"
    assert (evidence_dir / "session_facts.json").is_file()
    with ZipFile(result.diagnostic_zip_path) as archive:
        names = set(archive.namelist())
        assert "test_evidence/session_facts.json" in names
        assert "session/video/game.avi" not in names
    after = {p.relative_to(session).as_posix(): p.read_bytes() for p in session.rglob("*") if p.is_file()}
    assert before == after


def test_evidence_failure_does_not_break_canonical_delivery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path, with_truth=False)

    def fail(_session: Path):
        raise RuntimeError("synthetic evidence failure")

    monkeypatch.setattr(delivery, "build_test_evidence", fail)
    result = delivery.AutomaticLogDeliveryService(documents_root=tmp_path / "docs").export(session)
    summary = json.loads(result.machine_summary_path.read_text(encoding="utf-8"))

    assert result.status == "PASS"
    assert result.diagnostic_zip_path.is_file()
    assert summary["test_evidence"]["status"] == "error"
    assert summary["test_evidence"]["error_type"] == "RuntimeError"


def test_observation_part_file_is_used_when_gzip_is_not_sealed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    (session / "observations.jsonl.gz").unlink()
    (session / "observations.jsonl.part").write_text(
        json.dumps({"id": "part-1", "player": "left", "cards": ["2S"]}) + "\n",
        encoding="utf-8",
    )
    bundle = build_test_evidence(session)
    facts = json.loads(bundle.files["test_evidence/session_facts.json"])
    assert facts["counts"]["observations"] == 1
