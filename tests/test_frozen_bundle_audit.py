from pathlib import Path

from daguandan_bridge.frozen_bundle_audit import (
    audit_frozen_bundle,
    collect_pyinstaller_provenance,
)


def _fake_pe(path: Path, *, upx: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MZ" + (b"UPX!" if upx else b"") + b"not-a-real-pe")


def test_audit_rejects_unknown_java_icu_and_upx_sources(tmp_path):
    bundle = tmp_path / "bundle"
    java = tmp_path / "jdk" / "bin" / "icuuc.dll"
    java.parent.mkdir(parents=True)
    java.write_bytes(b"source")
    target = bundle / "_internal" / "icuuc.dll"
    _fake_pe(target, upx=True)

    result = audit_frozen_bundle(
        bundle,
        provenance={"_internal/icuuc.dll": str(java)},
        allowed_venv_root=tmp_path / "venv",
        allowed_python_root=tmp_path / "python",
        project_root=tmp_path / "project",
        windows_root=tmp_path / "windows",
    )
    codes = set(result.document["summary"]["error_codes"])

    assert result.ok is False
    assert "NATIVE-BANNED-SOURCE" in codes
    assert "NATIVE-CONFLICTING-ICU" in codes
    assert "NATIVE-UPX-DETECTED" in codes
    assert "NATIVE-UNKNOWN-PROVENANCE" in codes


def test_generated_executable_has_explicit_provenance_class(tmp_path):
    bundle = tmp_path / "bundle"
    _fake_pe(bundle / "DaguandanAssistant.exe")

    result = audit_frozen_bundle(bundle)

    record = result.document["files"][0]
    assert record["source_class"] == "pyinstaller-output"
    # Fake bytes are intentionally rejected as invalid PE, but provenance is
    # neither guessed nor classified as an arbitrary PATH dependency.
    assert "NATIVE-UNKNOWN-PROVENANCE" not in result.document["summary"]["error_codes"]


def test_collect_toc_provenance_detects_conflicting_sources(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    source = tmp_path / "venv" / "Lib" / "site-packages" / "PySide6" / "Qt6Core.dll"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"dll")
    (work / "Analysis-00.toc").write_text(
        repr([("_internal/PySide6/Qt6Core.dll", str(source), "BINARY")]),
        encoding="utf-8",
    )

    provenance = collect_pyinstaller_provenance(work)

    assert provenance["_internal/pyside6/qt6core.dll"] == str(source.resolve())
