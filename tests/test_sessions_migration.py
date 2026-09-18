from pathlib import Path
import json
import pytest

from daguandan_bridge.session_paths import (
    resolve_sessions_root, sessions_root_info,
)
from daguandan_bridge.sessions_migration import (
    migrate_sessions, rollback_sessions, sessions_migration_status,
)


def test_sessions_root_env_override_and_default(tmp_path: Path):
    profiles = tmp_path / "data" / "profiles"
    default = profiles / "tencent_daguandan" / "sessions"
    assert resolve_sessions_root(profiles, "tencent_daguandan") == default.resolve()
    target = tmp_path / "D-drive" / "sessions"
    assert resolve_sessions_root(
        profiles, "tencent_daguandan", environ={"DAGUANDAN_SESSIONS_ROOT": str(target)}
    ) == target.resolve()
    assert sessions_root_info(
        profiles, "tencent_daguandan", environ={"DAGUANDAN_SESSIONS_ROOT": str(target)}
    )["source"] == "environment"


def test_sessions_migration_copies_verifies_and_rolls_back(tmp_path: Path):
    profiles = tmp_path / "data" / "profiles"
    source = profiles / "tencent_daguandan" / "sessions"
    target = tmp_path / "D-drive" / "sessions"
    session = source / "game_1" / "video"
    session.mkdir(parents=True)
    (session / "game.avi").write_bytes(b"video")
    (source / "game_1" / "manifest.json").write_text(
        json.dumps({"status": "sealed"}), encoding="utf-8"
    )

    result = migrate_sessions(
        profiles_root=profiles, profile_name="tencent_daguandan", target_root=target
    )
    assert result.status == "active"
    assert (target / "game_1" / "video" / "game.avi").read_bytes() == b"video"
    assert (source / "game_1" / "manifest.json").is_file()
    assert resolve_sessions_root(profiles, "tencent_daguandan") == target.resolve()
    receipt = json.loads((target / ".sessions-migration.json").read_text(encoding="utf-8"))
    assert receipt["source_preserved"] is True
    assert sessions_migration_status(
        profiles_root=profiles, profile_name="tencent_daguandan"
    )["active_root"] == str(target.resolve())

    rollback = rollback_sessions(profiles_root=profiles, profile_name="tencent_daguandan")
    assert rollback.status == "rolled_back"
    assert resolve_sessions_root(profiles, "tencent_daguandan") == source.resolve()
    assert target.is_dir()


def test_sessions_migration_refuses_partial_recording(tmp_path: Path):
    profiles = tmp_path / "data" / "profiles"
    source = profiles / "tencent_daguandan" / "sessions"
    source.mkdir(parents=True)
    (source / "game_1" ).mkdir()
    (source / "game_1" / "observations.jsonl.part").write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match=".part"):
        migrate_sessions(
            profiles_root=profiles, profile_name="tencent_daguandan",
            target_root=tmp_path / "D-drive" / "sessions",
        )
