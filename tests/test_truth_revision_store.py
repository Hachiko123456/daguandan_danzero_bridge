from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from daguandan_bridge.application.truth_revision_store import (
    RevisionConflictError,
    TruthRevisionStore,
    save_truth_log_versioned,
    truth_log_semantic_sha256,
)
from daguandan_bridge.domain.truth import TruthEvidence
from daguandan_bridge.live.truth_log import (
    TruthInitialState,
    TruthLog,
    TruthLogCardInventoryError,
    TruthTurn,
    load_truth_log,
    save_truth_log,
)


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


def _log(session_id: str, *, card: str = "5H") -> TruthLog:
    return TruthLog(
        source_session_id=session_id,
        initial_state=TruthInitialState("2", "self", HAND),
        turns=(
            TruthTurn(
                1, "self", False, (card,),
                evidence=TruthEvidence((12,), 100, "my_play"),
            ),
            TruthTurn(2, "right", True, (), evidence=TruthEvidence((13,), 200)),
        ),
    )


def _immutable_source(session: Path) -> dict[str, str]:
    video = session / "video"
    video.mkdir(parents=True)
    (video / "game.avi").write_bytes(b"immutable video")
    (video / "frame_index.jsonl").write_text('{"frame_index": 0}\n', encoding="utf-8")
    (session / "timeline.jsonl").write_text('{"event_id": "E-1"}\n', encoding="utf-8")
    return {
        path.relative_to(session).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (video / "game.avi", video / "frame_index.jsonl", session / "timeline.jsonl")
    }


def test_versioned_save_is_path_independent_non_mutating_and_deduplicated(tmp_path: Path):
    session = tmp_path / "copied sessions" / "renamed export"
    source_hashes = _immutable_source(session)
    log = _log("game_original_id")
    store = TruthRevisionStore(session)

    first = store.save_draft(log, author="reviewer")
    second = store.save_draft(log, author="reviewer")

    assert first.revision_id == second.revision_id == "revision-000001"
    assert first.label_status == "draft"
    assert (session / "truth_revisions" / "revision-000001.json").is_file()
    assert store.manifest().session_id == "game_original_id"
    assert load_truth_log(session / "truth_log.json").to_dict() == log.to_dict()
    assert source_hashes == {
        path.relative_to(session).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (session / "video" / "game.avi", session / "video" / "frame_index.jsonl", session / "timeline.jsonl")
    }


def test_semantic_revision_stales_derived_result_and_preserves_history(tmp_path: Path):
    session = tmp_path / "portable-copy"
    source_hashes = _immutable_source(session)
    store = TruthRevisionStore(session)
    first_log = _log("game_original_id", card="5H")
    first = store.save_draft(first_log)
    store.register_derived_result("semantic-replay", session / "external-validation" / "replay.json")

    changed_log = _log("game_original_id", card="6H")
    second = store.save_draft(changed_log, expected_parent_revision_id=first.revision_id)
    results = store.derived_results()

    assert second.revision_id == "revision-000002"
    assert second.parent_revision_id == first.revision_id
    assert "turns[0].cards[0]" in second.changed_fields
    assert results[0].status == "stale"
    assert results[0].stale_reason == "truth_semantics_changed"
    assert store.load_revision(first.revision_id).to_dict() == first_log.to_dict()
    assert store.load_revision(second.revision_id).to_dict() == changed_log.to_dict()
    assert source_hashes == {
        path.relative_to(session).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (session / "video" / "game.avi", session / "video" / "frame_index.jsonl", session / "timeline.jsonl")
    }


def test_evidence_only_revision_keeps_semantic_result_fresh(tmp_path: Path):
    session = tmp_path / "session"
    _immutable_source(session)
    store = TruthRevisionStore(session)
    first_log = _log("game_original_id")
    first = store.save_draft(first_log)
    store.register_derived_result("semantic-replay", "C:/validation/replay.json")
    revised_turn = replace(first_log.turns[0], evidence=TruthEvidence((12, 13), 100, "my_play"))
    evidence_only = replace(first_log, turns=(revised_turn, first_log.turns[1]))

    second = store.save_draft(evidence_only)

    assert second.revision_id == "revision-000002"
    assert truth_log_semantic_sha256(first_log) == truth_log_semantic_sha256(evidence_only)
    assert store.derived_results()[0].status == "fresh"


def test_publish_rejects_logs_that_live_reducer_cannot_replay(tmp_path: Path):
    session = tmp_path / "invalid-publish"
    _immutable_source(session)
    invalid = TruthLog(
        source_session_id="invalid-publish",
        initial_state=TruthInitialState("2", "self", HAND),
        turns=(TruthTurn(1, "self", True, ()),),
    )

    with pytest.raises(ValueError, match="LiveReducer 全量回放失败"):
        TruthRevisionStore(session).publish(invalid, author="test")

    assert not (session / "truth_log.json").exists()


def test_explicit_publish_bootstraps_legacy_draft_and_does_not_promote_other_sessions(tmp_path: Path):
    confirmed = tmp_path / "game_20260816_125402_1687ea"
    other = tmp_path / "game_other_draft"
    _immutable_source(confirmed)
    _immutable_source(other)
    confirmed_log = _log("game_20260816_125402_1687ea")
    other_log = _log("game_other_draft")
    save_truth_log(confirmed / "truth_log.json", confirmed_log)
    save_truth_log(other / "truth_log.json", other_log)

    revision = TruthRevisionStore(confirmed).publish(confirmed_log, author="human-review")

    assert revision.revision_id == "revision-000002"
    assert revision.label_status == "verified"
    assert load_truth_log(confirmed / "truth_log.json").label_status == "verified"
    assert load_truth_log(other / "truth_log.json").label_status == "draft"
    history = TruthRevisionStore(confirmed).manifest().revisions
    assert [item.label_status for item in history] == ["draft", "verified"]
    assert TruthRevisionStore(other).manifest().current_revision_id == "revision-000001"
    assert not (other / "truth_revision_manifest.json").exists()


def test_legacy_manifest_created_by_result_registration_keeps_baseline_loadable(tmp_path: Path):
    session = tmp_path / "legacy"
    _immutable_source(session)
    log = _log("legacy-source")
    save_truth_log(session / "truth_log.json", log)
    store = TruthRevisionStore(session)

    store.register_derived_result("old-replay", "C:/validation/old.json")
    second = store.save_draft(_log("legacy-source", card="6H"))

    assert second.parent_revision_id == "revision-000001"
    assert store.load_revision("revision-000001").to_dict() == log.to_dict()


def test_legacy_adapter_is_deduplicated_and_honors_optimistic_parent(tmp_path: Path):
    session = tmp_path / "arbitrary-directory-name"
    _immutable_source(session)
    log = _log("stable-source-id")
    first = save_truth_log_versioned(session / "truth_log.json", log)
    same = save_truth_log_versioned(session / "truth_log.json", log)

    assert same.revision_id == first.revision_id
    with pytest.raises(RevisionConflictError):
        save_truth_log_versioned(
            session / "truth_log.json",
            _log("stable-source-id", card="6H"),
            expected_parent_revision_id="revision-000000",
        )



def test_revision_store_rejects_impossible_inventory_even_for_draft(tmp_path: Path):
    session = tmp_path / "invalid-inventory"
    _immutable_source(session)
    invalid = TruthLog(
        source_session_id="invalid-inventory",
        initial_state=TruthInitialState("2", "right", HAND),
        turns=(
            TruthTurn(1, "right", False, ("3D",)),
            TruthTurn(2, "opposite", False, ("3D",)),
        ),
    )

    with pytest.raises(TruthLogCardInventoryError, match="方块3（3D）共 3 张"):
        TruthRevisionStore(session).save_draft(invalid, author="test")

    assert not (session / "truth_log.json").exists()
    assert not (session / "truth_revisions").exists()



def test_publish_normalizes_stale_trick_ids_before_versioning(tmp_path: Path):
    session = tmp_path / "normalize-tricks"
    _immutable_source(session)
    log = TruthLog(
        source_session_id="normalize-tricks",
        initial_state=TruthInitialState("2", "self", HAND),
        turns=(
            TruthTurn(1, "self", False, ("2S",), trick_id=8),
            TruthTurn(2, "right", True, (), trick_id=8),
            TruthTurn(3, "opposite", True, (), trick_id=8),
            TruthTurn(4, "left", True, (), trick_id=8),
            TruthTurn(5, "self", False, ("2H",), trick_id=9),
        ),
    )

    revision = TruthRevisionStore(session).publish(log, author="test")
    saved = load_truth_log(session / "truth_log.json", session_id="normalize-tricks")

    assert revision.label_status == "verified"
    assert [turn.trick_id for turn in saved.turns] == [1, 1, 1, 1, 2]
