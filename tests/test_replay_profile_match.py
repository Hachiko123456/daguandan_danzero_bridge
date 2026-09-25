from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from daguandan_bridge.application.session_replay_audit import prepared_replay_input


def _write_session(root: Path, *, profile_snapshot: Path | None = None) -> Path:
    session = root / "session"
    video = session / "video"
    video.mkdir(parents=True)
    (video / "game.avi").write_bytes(b"synthetic-avi")
    (video / "frame_index.jsonl").write_text(
        json.dumps({"frame_index": 0, "monotonic_ms": 0}) + "\n",
        encoding="utf-8",
    )
    profile = profile_snapshot or (root / "fallback-profile")
    profile.mkdir(parents=True, exist_ok=True)
    profile_json = profile / "profile.json"
    profile_json.write_text('{"name":"test-profile"}\n', encoding="utf-8")
    manifest = {
        "session_id": "test-session",
        "configuration_hash": hashlib.sha256(profile_json.read_bytes()).hexdigest(),
    }
    (session / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return session


def _zip_tree(source_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source_root).as_posix())


def test_prepared_replay_input_uses_embedded_profile_for_directory(tmp_path: Path) -> None:
    source = tmp_path / "input"
    snapshot = source / "session" / "profile_snapshot" / "tencent_daguandan"
    _write_session(source, profile_snapshot=snapshot)

    with prepared_replay_input(source / "session", fallback_profile=tmp_path / "wrong") as replay:
        assert replay.input_kind == "session"
        assert replay.embedded_snapshot is True
        assert replay.profile_path == snapshot.resolve()
        assert replay.profile_resource_match["status"] == "embedded_match"
        assert replay.source_health["required_files"] == {
            "video": True,
            "frame_index": True,
        }


def test_prepared_replay_input_uses_same_contract_for_zip(tmp_path: Path) -> None:
    source = tmp_path / "input"
    snapshot = source / "profile_snapshot" / "tencent_daguandan"
    session = _write_session(source, profile_snapshot=snapshot)
    # The ZIP contract stores session/* and profile_snapshot/* at the archive root.
    snapshot.parent.parent.mkdir(parents=True, exist_ok=True)
    # _write_session created the profile at the desired archive-root location;
    # only the session directory is copied into the archive source below.
    archive_source = tmp_path / "archive_source"
    (archive_source / "session").mkdir(parents=True)
    for path in session.rglob("*"):
        if path.is_file():
            target = archive_source / "session" / path.relative_to(session)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    for path in snapshot.rglob("*"):
        if path.is_file():
            target = archive_source / "profile_snapshot" / path.relative_to(snapshot)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    archive = tmp_path / "diagnostic.zip"
    _zip_tree(source, archive)

    with prepared_replay_input(archive, fallback_profile=tmp_path / "wrong") as replay:
        assert replay.input_kind == "diagnostic_zip"
        assert replay.embedded_snapshot is True
        assert replay.profile_path.name == "tencent_daguandan"
        assert replay.profile_resource_match["status"] == "embedded_match"
        assert replay.session.joinpath("video", "game.avi").is_file()


def test_prepared_replay_input_keeps_missing_video_compatibility_explicit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "session"
    video = source / "video"
    video.mkdir(parents=True)
    (video / "frame_index.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="session|ZIP"):
        with prepared_replay_input(source, fallback_profile=tmp_path / "profile"):
            pass

    with prepared_replay_input(
        source,
        fallback_profile=tmp_path / "profile",
        allow_missing_video=True,
    ) as replay:
        assert replay.session == source.resolve()
        assert replay.source_health["required_files"]["video"] is False


def test_prepared_replay_input_accepts_directory_wrapper(tmp_path: Path) -> None:
    source = tmp_path / "wrapper"
    snapshot = source / "profile_snapshot" / "tencent_daguandan"
    _write_session(source, profile_snapshot=snapshot)

    with prepared_replay_input(source, fallback_profile=tmp_path / "wrong") as replay:
        assert replay.input_kind == "session"
        assert replay.session == (source / "session").resolve()
        assert replay.embedded_snapshot is True
        assert replay.profile_resource_match["status"] == "embedded_match"
