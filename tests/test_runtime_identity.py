from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.runtime_identity import (
    BUILD_MANIFEST_FILENAME,
    build_runtime_identity,
    default_build_manifest_path,
    get_runtime_identity,
)


def test_runtime_identity_is_process_stable_and_returns_a_defensive_copy():
    first = get_runtime_identity()
    second = get_runtime_identity()

    assert first["schema"] == "guandan.runtime-identity/1"
    assert first["run_id"] == second["run_id"]
    assert first["implementation_fingerprint"]
    assert Path(str(first["executable_path"])).name == first["executable_path"]

    first["run_id"] = "mutated"
    assert get_runtime_identity()["run_id"] == second["run_id"]


def test_missing_and_invalid_build_manifests_are_explicit(tmp_path):
    missing = build_runtime_identity(
        build_manifest_path=tmp_path / BUILD_MANIFEST_FILENAME,
        executable_path=tmp_path / "DaguandanAssistant.exe",
        frozen=True,
        run_id="RUN-MISSING",
        environ={"LOCALAPPDATA": str(tmp_path / "local")},
    )
    assert missing["build_status"] == "unidentified"
    assert missing["build_id"] == "unidentified"

    manifest = tmp_path / BUILD_MANIFEST_FILENAME
    manifest.write_text("{broken", encoding="utf-8")
    invalid = build_runtime_identity(
        build_manifest_path=manifest,
        executable_path=tmp_path / "DaguandanAssistant.exe",
        frozen=True,
        run_id="RUN-INVALID",
        environ={"LOCALAPPDATA": str(tmp_path / "local")},
    )
    assert invalid["build_status"] == "invalid"
    assert invalid["build_id"] == "invalid"

    manifest.write_text(json.dumps({"schema": "wrong", "build_id": "BUILD"}), encoding="utf-8")
    wrong_schema = build_runtime_identity(
        build_manifest_path=manifest,
        executable_path=tmp_path / "DaguandanAssistant.exe",
        frozen=True,
        run_id="RUN-WRONG-SCHEMA",
        environ={"LOCALAPPDATA": str(tmp_path / "local")},
    )
    assert wrong_schema["build_status"] == "invalid"

    manifest.write_text(
        json.dumps(
            {"schema": "guandan.build-manifest/1", "build_id": "BUILD"}
        ),
        encoding="utf-8",
    )
    missing_source = build_runtime_identity(
        build_manifest_path=manifest,
        executable_path=tmp_path / "DaguandanAssistant.exe",
        frozen=True,
        run_id="RUN-MISSING-SOURCE",
        environ={"LOCALAPPDATA": str(tmp_path / "local")},
    )
    assert missing_source["build_status"] == "invalid"


def test_build_manifest_identity_is_sanitized_and_supports_nested_source(tmp_path):
    username = "identity-secret-user"
    manifest = tmp_path / BUILD_MANIFEST_FILENAME
    manifest.write_text(
        json.dumps(
            {
                "schema": "guandan.build-manifest/1",
                "build_id": "BUILD-123",
                "source": {"commit": "abc123", "dirty": False},
                "implementation_fingerprint": "fingerprint-123",
                "private_path": f"C:/Users/{username}/source",
            }
        ),
        encoding="utf-8",
    )

    identity = build_runtime_identity(
        build_manifest_path=manifest,
        executable_path=f"C:/Users/{username}/app/DaguandanAssistant.exe",
        frozen=True,
        run_id="RUN-VALID",
        environ={"LOCALAPPDATA": f"C:/Users/{username}/AppData/Local"},
    )
    encoded = json.dumps(identity, ensure_ascii=False)

    assert identity["build_status"] == "identified"
    assert identity["build_id"] == "BUILD-123"
    assert identity["implementation_fingerprint"] == "fingerprint-123"
    assert identity["executable_path"] == "DaguandanAssistant.exe"
    assert username not in encoded
    assert "private_path" not in encoded


def test_environment_override_selects_build_manifest_path(tmp_path):
    expected = tmp_path / "custom-build.json"

    actual = default_build_manifest_path(
        executable_path=tmp_path / "assistant.exe",
        frozen=True,
        environ={"DAGUANDAN_BUILD_MANIFEST": str(expected)},
    )

    assert actual == expected
