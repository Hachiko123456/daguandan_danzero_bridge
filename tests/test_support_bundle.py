from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from daguandan_bridge import support_bundle as support
from daguandan_bridge.support_bundle import (
    RedactionContext,
    SupportBundleError,
    SupportBundleSources,
    UnsafeSupportSourceError,
    export_support_bundle,
)


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _read_archive(path: Path) -> tuple[dict[str, bytes], dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        entries = {name: archive.read(name) for name in archive.namelist()}
    return entries, json.loads(entries["support_manifest.json"].decode("utf-8"))


def _sources(root: Path) -> SupportBundleSources:
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "startup.log").write_text(
        "user=Alice host=DESKTOP-SECRET "
        'path="C:\\Users\\Alice\\Secret Project\\startup.log" '
        "email=alice@example.com Bearer abc.def-123 token=plain-secret\n",
        encoding="utf-8",
    )
    _write_json(
        root / "runtime" / "runtime_identity.json",
        {
            "executable_path": r"C:\Users\Alice\Private App\assistant.exe",
            "username": "Alice",
            "computer_name": "DESKTOP-SECRET",
            "api_key": "runtime-api-key",
            "environment_variables": {
                "PATH": r"C:\Users\Alice\bin",
                "TOKEN": "must-not-leak",
                "TEMP": r"C:\Users\Alice\Temp",
                "USERNAME": "Alice",
            },
        },
    )
    _write_json(
        root / "doctor" / "doctor.json",
        {
            "status": "warn",
            "window": "not_found",
            "log_source": r"\\private-server\support-share\doctor.json",
            "authorization": "Bearer doctor-secret",
            "contact": "alice@example.com",
        },
    )
    _write_json(
        root / "build" / "build_manifest.json",
        {
            "schema": "guandan.build-manifest/1",
            "build_id": "BUILD-test-123",
            "files": [{"path": "app.exe", "sha256": "a" * 64}],
        },
    )
    _write_json(
        root / "incident" / "incident.json",
        {"code": "LEVEL-BELOW-THRESHOLD", "user": "Alice", "key": "secret"},
    )
    (root / "trace").mkdir()
    (root / "trace" / "recognition_trace.jsonl").write_text(
        json.dumps(
            {
                "result": "blocked",
                "path": r"C:\Users\Alice\session\recognition.jsonl",
                "token": "trace-secret",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "frames").mkdir()
    (root / "frames" / "full.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
    (root / "roi").mkdir()
    (root / "roi" / "level.png").write_bytes(b"\x89PNG\r\n\x1a\nroi")
    return SupportBundleSources(
        root=root,
        startup_log=Path("logs/startup.log"),
        runtime_identity=Path("runtime/runtime_identity.json"),
        doctor=Path("doctor/doctor.json"),
        build_manifest=Path("build/build_manifest.json"),
        incident=Path("incident/incident.json"),
        recognition_trace=Path("trace/recognition_trace.jsonl"),
        frames=(Path("frames/full.png"),),
        roi=(Path("roi/level.png"),),
    )


def _redaction() -> RedactionContext:
    return RedactionContext(
        usernames=("Alice",),
        computer_names=("DESKTOP-SECRET",),
    )


def test_default_bundle_uses_fixed_allowlist_and_disables_sensitive_capabilities(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path / "source")
    destination = tmp_path / "support.zip"

    result = export_support_bundle(destination, sources, redaction=_redaction())
    entries, manifest = _read_archive(destination)

    assert result.destination == destination
    assert manifest["schema"] == "guandan.support-bundle/1"
    assert manifest["build_id"] == "BUILD-test-123"
    assert set(entries) == {
        "startup/startup.log",
        "runtime/runtime_identity.json",
        "doctor/doctor.json",
        "build/build_manifest.json",
        "incident/incident.json",
        "support_manifest.json",
    }
    assert manifest["capabilities"] == {
        "startup_log": True,
        "runtime_identity": True,
        "doctor": True,
        "build_manifest": True,
        "incident": True,
        "frames": False,
        "roi": False,
        "recognition_trace": False,
    }
    assert {
        (item["capability"], item["reason"]) for item in manifest["missing"]
    } == {
        ("frames", "disabled"),
        ("roi", "disabled"),
        ("recognition_trace", "disabled"),
    }
    assert manifest["privacy"]["contains_sensitive_images"] is False

    payload_text = b"\n".join(
        content for name, content in entries.items() if name != "support_manifest.json"
    ).decode("utf-8")
    for forbidden in (
        "Alice",
        "DESKTOP-SECRET",
        "alice@example.com",
        "plain-secret",
        "runtime-api-key",
        "must-not-leak",
        "doctor-secret",
        "C:\\Users",
        "private-server",
    ):
        assert forbidden.casefold() not in payload_text.casefold()
    assert "<OMITTED_ENVIRONMENT_DUMP>" in payload_text
    assert "<REDACTED>" in payload_text

    records = {record["path"]: record for record in manifest["files"]}
    assert set(records) == set(entries) - {"support_manifest.json"}
    for name, record in records.items():
        assert record["size"] == len(entries[name])
        assert record["sha256"] == hashlib.sha256(entries[name]).hexdigest()
        assert record["classification"].startswith("sanitized-")


def test_frames_roi_and_trace_require_opt_in_and_use_generated_names(tmp_path: Path) -> None:
    sources = _sources(tmp_path / "source")
    destination = tmp_path / "support-with-images.zip"

    export_support_bundle(
        destination,
        sources,
        include_frames=True,
        include_roi=True,
        include_recognition_trace=True,
        redaction=_redaction(),
    )
    entries, manifest = _read_archive(destination)

    assert "frames/frames_0001.png" in entries
    assert "roi/roi_0001.png" in entries
    assert "trace/recognition_trace.jsonl" in entries
    assert manifest["capabilities"]["frames"] is True
    assert manifest["capabilities"]["roi"] is True
    assert manifest["capabilities"]["recognition_trace"] is True
    assert manifest["privacy"]["contains_sensitive_images"] is True
    assert entries["frames/frames_0001.png"] == b"\x89PNG\r\n\x1a\nframe"
    trace = entries["trace/recognition_trace.jsonl"].decode("utf-8")
    assert "trace-secret" not in trace
    assert "Alice" not in trace
    assert "<PATH>" in trace


def test_missing_inputs_are_reported_without_inventing_capabilities(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()

    export_support_bundle(
        tmp_path / "support.zip",
        SupportBundleSources(
            root=root,
            startup_log=Path("missing/startup.log"),
        ),
    )
    entries, manifest = _read_archive(tmp_path / "support.zip")

    assert set(entries) == {"support_manifest.json"}
    assert not any(manifest["capabilities"].values())
    reasons = {
        item["capability"]: item["reason"] for item in manifest["missing"]
    }
    assert reasons["startup_log"] == "not_found"
    assert reasons["runtime_identity"] == "not_provided"
    assert reasons["doctor"] == "not_provided"
    assert reasons["build_manifest"] == "not_provided"
    assert reasons["frames"] == "disabled"
    assert reasons["roi"] == "disabled"
    assert reasons["recognition_trace"] == "disabled"


@pytest.mark.parametrize(
    "unsafe_path",
    [
        Path("../outside.log"),
        Path("logs/../../outside.log"),
    ],
)
def test_parent_traversal_sources_are_rejected(
    tmp_path: Path,
    unsafe_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()

    with pytest.raises(UnsafeSupportSourceError):
        export_support_bundle(
            tmp_path / "support.zip",
            SupportBundleSources(root=root, startup_log=unsafe_path),
        )


def test_absolute_sources_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    absolute = tmp_path / "outside.log"
    absolute.write_text("outside", encoding="utf-8")

    with pytest.raises(UnsafeSupportSourceError):
        export_support_bundle(
            tmp_path / "support.zip",
            SupportBundleSources(root=root, startup_log=absolute.resolve()),
        )


def test_unsafe_disabled_capability_declarations_are_still_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\nprivate")

    with pytest.raises(UnsafeSupportSourceError):
        export_support_bundle(
            tmp_path / "support.zip",
            SupportBundleSources(root=root, frames=(outside.resolve(),)),
            include_frames=False,
        )


@pytest.mark.parametrize(
    "name",
    [
        "model.npz",
        "weights.ckpt",
        ".env",
        ".env.local",
        "environment.json",
        "process_environment_dump.txt",
    ],
)
def test_models_and_environment_dumps_are_rejected(tmp_path: Path, name: str) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / name).write_bytes(b"must never be packaged")

    with pytest.raises(UnsafeSupportSourceError):
        export_support_bundle(
            tmp_path / "support.zip",
            SupportBundleSources(root=root, startup_log=Path(name)),
        )


def test_reparse_or_symlink_sources_are_rejected(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "source"
    root.mkdir()
    source = root / "linked.log"
    source.write_text("evidence", encoding="utf-8")
    original = support._is_link_or_reparse
    monkeypatch.setattr(
        support,
        "_is_link_or_reparse",
        lambda path: path.name == "linked.log" or original(path),
    )

    with pytest.raises(UnsafeSupportSourceError):
        export_support_bundle(
            tmp_path / "support.zip",
            SupportBundleSources(root=root, startup_log=Path("linked.log")),
        )


@pytest.mark.parametrize("name", ["/absolute.txt", "../escape.txt", "C:/drive.txt", "a\\b.txt"])
def test_archive_entry_validator_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(SupportBundleError):
        support._validated_archive_path(name)


def test_invalid_image_content_is_rejected_even_with_an_allowed_extension(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "fake.png").write_bytes(b"not an image")

    with pytest.raises(UnsafeSupportSourceError):
        export_support_bundle(
            tmp_path / "support.zip",
            SupportBundleSources(root=root, frames=(Path("fake.png"),)),
            include_frames=True,
        )


def test_environment_dump_content_is_omitted_even_under_an_allowed_log_name(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "startup.log").write_text(
        "PATH=C:\\private\\bin\n"
        "USERNAME=Alice\n"
        "COMPUTERNAME=SECRET-PC\n"
        "TEMP=C:\\private\\temp\n"
        "API_TOKEN=secret\n",
        encoding="utf-8",
    )

    export_support_bundle(
        tmp_path / "support.zip",
        SupportBundleSources(root=root, startup_log=Path("startup.log")),
    )
    entries, _manifest = _read_archive(tmp_path / "support.zip")

    assert entries["startup/startup.log"] == b"<OMITTED_ENVIRONMENT_DUMP>\n"


def test_unquoted_paths_with_spaces_and_short_unicode_identities_are_redacted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "startup.log").write_text(
        "drive=C:\\Users\\Alice\\Secret Project\\private\\frame.png trailing=remove-me\n"
        "unc=\\\\server-name\\Secret Share\\private\\frame.png trailing=remove-me\n"
        "identity=Li chinese=李雷 host=PC cn_host=研发\n"
        "boundary=Limit PCLab 李雷峰 研发部\n",
        encoding="utf-8",
    )

    export_support_bundle(
        tmp_path / "support.zip",
        SupportBundleSources(root=root, startup_log=Path("startup.log")),
        redaction=RedactionContext(
            usernames=("Alice", "Li", "李雷"),
            computer_names=("PC", "研发"),
        ),
    )
    entries, _manifest = _read_archive(tmp_path / "support.zip")
    text = entries["startup/startup.log"].decode("utf-8")

    for leaked in (
        "Alice",
        "Secret Project",
        "private",
        "frame.png",
        "server-name",
        "Secret Share",
        "remove-me",
    ):
        assert leaked not in text
    assert "drive=<PATH>" in text
    assert "unc=<PATH>" in text
    assert (
        "identity=<REDACTED_IDENTITY> chinese=<REDACTED_IDENTITY> "
        "host=<REDACTED_IDENTITY> cn_host=<REDACTED_IDENTITY>"
    ) in text
    assert "boundary=Limit PCLab 李雷峰 研发部" in text


def test_automatic_short_and_chinese_identity_values_are_redacted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "startup.log").write_text(
        "user=Li host=研发 boundary=Limit 研发部\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("USERNAME", "Li")
    monkeypatch.setenv("COMPUTERNAME", "研发")
    monkeypatch.setattr(support.getpass, "getuser", lambda: "Li")
    monkeypatch.setattr(support.platform, "node", lambda: "研发")

    export_support_bundle(
        tmp_path / "support.zip",
        SupportBundleSources(root=root, startup_log=Path("startup.log")),
    )
    entries, _manifest = _read_archive(tmp_path / "support.zip")
    text = entries["startup/startup.log"].decode("utf-8")

    assert "user=<REDACTED_IDENTITY> host=<REDACTED_IDENTITY>" in text
    assert "boundary=Limit 研发部" in text


def test_limited_reader_stats_before_open_and_reads_only_limit_plus_one() -> None:
    class FakeHandle:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.requested: int | None = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size: int) -> bytes:
            self.requested = size
            return self.payload[:size]

    class FakePath:
        def __init__(self, declared_size: int, payload: bytes) -> None:
            self.declared_size = declared_size
            self.handle = FakeHandle(payload)
            self.open_calls = 0

        def stat(self):
            return type("Stat", (), {"st_size": self.declared_size})()

        def open(self, mode: str):
            assert mode == "rb"
            self.open_calls += 1
            return self.handle

    oversized = FakePath(5, b"12345")
    with pytest.raises(SupportBundleError, match="exceeds 4 bytes"):
        support._read_limited(
            oversized,
            per_file_limit=4,
            budget=support._PayloadBudget(100),
            relative=Path("oversized.log"),
            kind="text",
        )
    assert oversized.open_calls == 0

    bounded = FakePath(4, b"1234")
    assert support._read_limited(
        bounded,
        per_file_limit=4,
        budget=support._PayloadBudget(100),
        relative=Path("bounded.log"),
        kind="text",
    ) == b"1234"
    assert bounded.open_calls == 1
    assert bounded.handle.requested == 5


def test_cumulative_payload_limit_fails_before_publication_or_unbounded_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    first = root / "first.png"
    second = root / "second.png"
    first.write_bytes(b"\x89PNG\r\n\x1a\n")
    second.write_bytes(b"\x89PNG\r\n\x1a\n")
    destination = tmp_path / "support.zip"
    destination.write_bytes(b"previous-good-bundle")
    monkeypatch.setattr(support, "_MAX_PAYLOAD_BYTES", 12)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("support exporter must not call Path.read_bytes")
        ),
    )

    with pytest.raises(SupportBundleError, match="before reading: second.png"):
        export_support_bundle(
            destination,
            SupportBundleSources(
                root=root,
                frames=(Path("first.png"), Path("second.png")),
            ),
            include_frames=True,
        )

    with destination.open("rb") as handle:
        assert handle.read() == b"previous-good-bundle"
    assert list(tmp_path.glob(".support.zip.*.tmp")) == []


def test_publication_is_atomic_and_preserves_an_existing_bundle_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "startup.log").write_text("safe", encoding="utf-8")
    destination = tmp_path / "support.zip"
    destination.write_bytes(b"previous-good-bundle")
    original = support._write_archive_entry

    def fail_on_manifest(archive, name, content):
        if name == "support_manifest.json":
            raise OSError("simulated archive failure")
        return original(archive, name, content)

    monkeypatch.setattr(support, "_write_archive_entry", fail_on_manifest)

    with pytest.raises(OSError, match="simulated archive failure"):
        export_support_bundle(
            destination,
            SupportBundleSources(root=root, startup_log=Path("startup.log")),
        )

    assert destination.read_bytes() == b"previous-good-bundle"
    assert list(tmp_path.glob(".support.zip.*.tmp")) == []


def test_destination_must_be_a_zip(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()

    with pytest.raises(SupportBundleError, match="end with .zip"):
        export_support_bundle(
            tmp_path / "support.tar",
            SupportBundleSources(root=root),
        )
