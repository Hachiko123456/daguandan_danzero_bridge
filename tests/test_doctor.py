from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from daguandan_bridge import doctor
from daguandan_bridge.build_manifest import write_build_manifest


def _portable_root(tmp_path: Path, *, missing_template: bool = False) -> Path:
    root = tmp_path / "portable"
    profile = root / "data" / "profiles" / "tencent_daguandan"
    template = profile / "templates" / "rank" / "7_level.png"
    fabledan = profile / "models" / "best.npz"
    danzero = profile / "models" / "danzero" / "q_network.ckpt"
    template.parent.mkdir(parents=True)
    fabledan.parent.mkdir(parents=True)
    danzero.parent.mkdir(parents=True)
    if not missing_template:
        template.write_bytes(b"png")
    fabledan.write_bytes(b"npz")
    danzero.write_bytes(b"checkpoint")
    (profile / "profile.json").write_text(
        json.dumps({"name": "tencent_daguandan"}), encoding="utf-8"
    )
    (profile / "regions_config.json").write_text(
        json.dumps({"schema_version": 2, "regions": []}), encoding="utf-8"
    )
    (profile / "templates_config.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "templates": [{"file": "templates/rank/7_level.png"}],
            }
        ),
        encoding="utf-8",
    )
    return root


def _startup_state(tmp_path: Path):
    run_directory = tmp_path / "diagnostics" / "runs" / "RUN-TEST"
    run_directory.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        enabled=True,
        run_directory=run_directory,
        root_source="test",
        error=None,
    )


def _passing_probe(spec: doctor.DependencySpec) -> dict[str, object]:
    return {
        "status": "PASS",
        "summary": "Dependency imported successfully",
        "evidence": {
            "module": spec.module,
            "distribution": spec.distribution,
            "version": "test",
            "probe_exit_code": 0,
        },
    }


def _write_valid_build_manifest(root: Path) -> Path:
    (root / "DaguandanAssistant.exe").write_bytes(b"portable-executable")
    write_build_manifest(
        root,
        root,
        source_identity={
            "commit": "c" * 40,
            "tree": "d" * 40,
            "branch": "test",
            "dirty": False,
            "status_sha256": None,
        },
        python_identity={
            "version": "3.12.0",
            "implementation": "CPython",
            "architecture": "AMD64",
        },
        dependency_versions={"PyInstaller": "test"},
    )
    return root / "build_manifest.json"


def test_doctor_report_has_stable_schema_checks_and_no_window_probe(tmp_path):
    report = doctor.collect_doctor_report(
        root=_portable_root(tmp_path),
        dependencies=(doctor.DEPENDENCIES[0],),
        dependency_probe=_passing_probe,
        startup_state=_startup_state(tmp_path),
    )
    checks = {item["id"]: item for item in report["checks"]}

    assert report["schema"] == "guandan.doctor/1"
    assert report["overall_status"] == "PASS"
    assert report["capabilities"]["window_probe"] is False
    assert report["capabilities"]["capture_probe"] is False
    assert report["capabilities"]["recognition_probe"] is False
    assert report["capabilities"]["frames"] is False
    assert checks["RESOURCE-TEMPLATE-FILES"]["status"] == "PASS"
    assert checks["DEPENDENCY-NUMPY"]["status"] == "PASS"
    assert checks["BUILD-INTEGRITY"]["status"] == "WARN"
    assert all(
        set(item) == {"id", "status", "summary", "evidence", "duration_ms"}
        for item in report["checks"]
    )


def test_doctor_reports_a_missing_referenced_template_as_fail(tmp_path):
    report = doctor.collect_doctor_report(
        root=_portable_root(tmp_path, missing_template=True),
        dependencies=(),
        startup_state=_startup_state(tmp_path),
    )
    checks = {item["id"]: item for item in report["checks"]}

    assert report["overall_status"] == "FAIL"
    assert checks["RESOURCE-TEMPLATE-FILES"]["status"] == "FAIL"
    assert checks["RESOURCE-TEMPLATE-FILES"]["evidence"]["missing_count"] == 1


def test_run_doctor_exit_codes_are_zero_two_and_three(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DAGUANDAN_DIAGNOSTICS_ROOT", str(tmp_path / "startup-diagnostics")
    )
    output = tmp_path / "doctor.json"

    monkeypatch.setattr(
        doctor,
        "collect_doctor_report",
        lambda: {"schema": doctor.DOCTOR_SCHEMA, "overall_status": "PASS", "checks": []},
    )
    assert doctor.run_doctor(output) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == doctor.DOCTOR_SCHEMA

    monkeypatch.setattr(
        doctor,
        "collect_doctor_report",
        lambda: {"schema": doctor.DOCTOR_SCHEMA, "overall_status": "FAIL", "checks": []},
    )
    assert doctor.run_doctor(output) == 2

    def crash():
        raise RuntimeError("internal")

    monkeypatch.setattr(doctor, "collect_doctor_report", crash)
    assert doctor.run_doctor(output) == 3
    assert json.loads(output.read_text(encoding="utf-8"))["overall_status"] == "ERROR"


def test_run_entrypoint_routes_doctor_before_dpi_and_gui_imports():
    source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")

    assert source.index("if args.doctor:") < source.index(
        "from daguandan_bridge.dpi import enable_windows_dpi_awareness"
    )
    assert "daguandan_bridge.gui" not in source.split("if args.doctor:", 1)[0]


def _complete_frozen_doctor(build_id: str = "BUILD-test") -> dict[str, object]:
    return {
        "schema": doctor.DOCTOR_SCHEMA,
        "overall_status": "PASS",
        "identity": {
            "schema": "guandan.runtime-identity/1",
            "frozen": True,
            "build_status": "identified",
            "build_id": build_id,
        },
        "checks": [
            {
                "id": check_id,
                "status": "PASS",
                "summary": "ok",
                "evidence": {},
                "duration_ms": 0.0,
            }
            for check_id in doctor.DOCTOR_REQUIRED_CHECK_IDS
        ],
    }


def test_frozen_doctor_acceptance_requires_complete_unique_checks_and_identity():
    report = _complete_frozen_doctor()

    passed = doctor.validate_frozen_doctor_report(
        report,
        expected_build_id="BUILD-test",
    )

    assert passed["status"] == "PASS"
    assert passed["required_check_count"] == len(doctor.DOCTOR_REQUIRED_CHECK_IDS)

    report["checks"].append(dict(report["checks"][0]))
    failed = doctor.validate_frozen_doctor_report(
        report,
        expected_build_id="BUILD-other",
    )
    assert failed["status"] == "FAIL"
    assert "duplicate_check_ids" in failed["failures"]
    assert "identity_build_id_mismatch" in failed["failures"]


def test_frozen_doctor_acceptance_rejects_empty_checks_despite_pass_status():
    report = _complete_frozen_doctor()
    report["checks"] = []

    result = doctor.validate_frozen_doctor_report(
        report,
        expected_build_id="BUILD-test",
    )

    assert result["status"] == "FAIL"
    assert "required_checks_missing" in result["failures"]


def test_frozen_doctor_fails_when_build_manifest_is_missing_or_invalid(tmp_path):
    root = _portable_root(tmp_path)
    missing = doctor.collect_doctor_report(
        root=root,
        frozen=True,
        data_root=tmp_path / "missing-user-data",
        dependencies=(),
        startup_state=_startup_state(tmp_path),
    )
    missing_check = next(
        item for item in missing["checks"] if item["id"] == "BUILD-INTEGRITY"
    )
    assert missing_check["status"] == "FAIL"

    (root / "build_manifest.json").write_text("{broken", encoding="utf-8")
    invalid = doctor.collect_doctor_report(
        root=root,
        frozen=True,
        data_root=tmp_path / "invalid-user-data",
        dependencies=(),
        startup_state=_startup_state(tmp_path),
    )
    invalid_check = next(
        item for item in invalid["checks"] if item["id"] == "BUILD-INTEGRITY"
    )
    assert invalid_check["status"] == "FAIL"
    assert invalid_check["evidence"]["checked_files"] == 0


def test_build_integrity_rejects_model_and_executable_tampering(
    tmp_path,
):
    root = _portable_root(tmp_path)
    _write_valid_build_manifest(root)
    valid = doctor.collect_doctor_report(
        root=root,
        frozen=True,
        data_root=tmp_path / "user-data",
        dependencies=(),
        startup_state=_startup_state(tmp_path),
    )
    valid_check = next(
        item for item in valid["checks"] if item["id"] == "BUILD-INTEGRITY"
    )
    assert valid_check["status"] == "PASS"

    model = root / "data" / "profiles" / "tencent_daguandan" / "models" / "best.npz"
    assert len(model.read_bytes()) == len(b"bad")
    model.write_bytes(b"bad")
    tampered = doctor.collect_doctor_report(
        root=root,
        frozen=True,
        data_root=tmp_path / "user-data",
        dependencies=(),
        startup_state=_startup_state(tmp_path),
    )
    tampered_check = next(
        item for item in tampered["checks"] if item["id"] == "BUILD-INTEGRITY"
    )
    assert tampered_check["status"] == "FAIL"
    assert any(
        "best.npz" in error
        for error in tampered_check["evidence"]["errors"]
    )
    assert tampered_check["evidence"]["mutable_differences"] == []

    executable = root / "DaguandanAssistant.exe"
    original = executable.read_bytes()
    executable.write_bytes(b"X" + original[1:])
    immutable_tamper = doctor.collect_doctor_report(
        root=root,
        frozen=True,
        data_root=tmp_path / "user-data",
        dependencies=(),
        startup_state=_startup_state(tmp_path),
    )
    immutable_check = next(
        item
        for item in immutable_tamper["checks"]
        if item["id"] == "BUILD-INTEGRITY"
    )
    assert immutable_check["status"] == "FAIL"
    assert any(
        "DaguandanAssistant.exe" in error
        for error in immutable_check["evidence"]["errors"]
    )


def test_frozen_doctor_seeds_external_data_and_never_write_probes_bundle(tmp_path):
    root = _portable_root(tmp_path)
    _write_valid_build_manifest(root)

    def snapshot() -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    report = doctor.collect_doctor_report(
        root=root,
        frozen=True,
        data_root=tmp_path / "external-user-data",
        dependencies=(),
        startup_state=_startup_state(tmp_path),
    )
    checks = {item["id"]: item for item in report["checks"]}

    assert report["overall_status"] == "PASS"
    assert checks["STORAGE-BUNDLE-DATA"]["status"] == "PASS"
    assert checks["STORAGE-BUNDLE-DATA"]["evidence"]["write_probe"] is False
    assert checks["STORAGE-RUNTIME-LAYOUT"]["status"] == "PASS"
    assert checks["STORAGE-DATA"]["status"] == "PASS"
    assert snapshot() == before
    assert not tuple(root.rglob(".doctor-write-*.tmp"))


def test_critical_dependency_probes_include_concrete_qtgui_and_blackjack_modules(
    tmp_path,
):
    by_id = {item.check_id: item for item in doctor.DEPENDENCIES}
    assert by_id["DEPENDENCY-PYSIDE6-QTGUI"].module == "PySide6.QtGui"
    assert by_id["DEPENDENCY-RLCARD-BLACKJACK"].module == "rlcard.envs.blackjack"

    def fail_concrete_submodules(spec: doctor.DependencySpec) -> dict[str, object]:
        return {
            "status": "FAIL" if spec.module in {"PySide6.QtGui", "rlcard.envs.blackjack"} else "PASS",
            "summary": "simulated probe",
            "evidence": {"module": spec.module},
        }

    report = doctor.collect_doctor_report(
        root=_portable_root(tmp_path),
        dependencies=(
            by_id["DEPENDENCY-PYSIDE6-QTGUI"],
            by_id["DEPENDENCY-RLCARD-BLACKJACK"],
        ),
        dependency_probe=fail_concrete_submodules,
        startup_state=_startup_state(tmp_path),
    )
    assert report["overall_status"] == "FAIL"
    failed_ids = {
        item["id"] for item in report["checks"] if item["status"] == "FAIL"
    }
    assert {
        "DEPENDENCY-PYSIDE6-QTGUI",
        "DEPENDENCY-RLCARD-BLACKJACK",
    } <= failed_ids


def test_import_probe_preserves_sanitized_qtgui_dll_failure_details(
    tmp_path,
    monkeypatch,
):
    class QtGuiLoadError(OSError):
        winerror = 126
        errno = 193

    monkeypatch.setattr(doctor, "_distribution_version", lambda _name: "6.test")

    def fail_import(_module):
        raise QtGuiLoadError(
            r"DLL load failed: C:\Users\Alice\Private\Qt6Gui.dll token=super-secret"
        )

    monkeypatch.setattr(doctor.importlib, "import_module", fail_import)
    output = tmp_path / "qtgui-probe.json"

    assert doctor.run_import_probe("DEPENDENCY-PYSIDE6-QTGUI", output) == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    evidence = payload["evidence"]
    assert payload["status"] == "FAIL"
    assert evidence["module"] == "PySide6.QtGui"
    assert evidence["distribution"] == "PySide6"
    assert evidence["version"] == "6.test"
    assert evidence["winerror"] == 126
    assert evidence["errno"] == 193
    assert "DLL load failed" in evidence["message"]
    assert "Alice" not in evidence["message"]
    assert "super-secret" not in evidence["message"]
    assert "C:\\Users" not in evidence["message"]
    assert len(evidence["message"]) <= 500


def test_successful_frozen_import_passes_when_distribution_metadata_is_absent(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        doctor,
        "_distribution_version",
        lambda _name: "not-installed",
    )
    monkeypatch.setattr(doctor.importlib, "import_module", lambda _module: object())
    output = tmp_path / "qtgui-no-metadata.json"

    assert doctor.run_import_probe("DEPENDENCY-PYSIDE6-QTGUI", output) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["summary"] == "Dependency imported successfully"
    assert payload["evidence"]["version"] == "not-installed"


def test_parent_rejects_pass_payload_when_probe_process_exit_is_nonzero(
    tmp_path,
    monkeypatch,
):
    dependency = next(
        item
        for item in doctor.DEPENDENCIES
        if item.check_id == "DEPENDENCY-PYSIDE6-QTGUI"
    )
    monkeypatch.setattr(
        doctor,
        "current_startup_diagnostics",
        lambda: SimpleNamespace(run_directory=tmp_path),
    )

    def fake_run(command, **_kwargs):
        output = Path(command[command.index("--doctor-output") + 1])
        output.write_text(
            json.dumps(
                {
                    "schema": doctor.IMPORT_PROBE_SCHEMA,
                    "status": "PASS",
                    "summary": "forged pass",
                    "evidence": {"module": dependency.module},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=3221225781,
            stdout=b"",
            stderr=(
                rb"native crash C:\Users\Alice\Qt6Gui.dll password=leak-me"
            ),
        )

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    result = doctor._subprocess_dependency_probe(dependency)

    assert result["status"] == "FAIL"
    assert result["summary"] == "Dependency import probe process exited with a non-zero status"
    assert result["evidence"]["probe_exit_code"] == 3221225781
    assert result["evidence"]["reported_status"] == "PASS"
    assert result["evidence"]["probe_id"] == dependency.check_id
    assert result["evidence"]["scratch_file"].endswith(".json")
    encoded = json.dumps(result, ensure_ascii=False)
    assert "Alice" not in encoded
    assert "leak-me" not in encoded
    assert "C:\\\\Users" not in encoded


def test_native_probe_crash_without_payload_has_sanitized_stderr_and_logical_id(
    tmp_path,
    monkeypatch,
):
    dependency = next(
        item
        for item in doctor.DEPENDENCIES
        if item.check_id == "DEPENDENCY-RLCARD-BLACKJACK"
    )
    monkeypatch.setattr(
        doctor,
        "current_startup_diagnostics",
        lambda: SimpleNamespace(run_directory=tmp_path),
    )
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=3221225781,
            stdout=b"",
            stderr=rb"load C:\Users\Alice\bad.dll bearer abc.def.secret",
        ),
    )

    result = doctor._subprocess_dependency_probe(dependency)

    assert result["status"] == "FAIL"
    assert result["evidence"]["probe_id"] == dependency.check_id
    assert result["evidence"]["scratch_file"].endswith(".json")
    assert result["evidence"]["probe_exit_code"] == 3221225781
    assert "load" in result["evidence"]["stderr_summary"]
    encoded = json.dumps(result, ensure_ascii=False)
    assert "Alice" not in encoded
    assert "abc.def.secret" not in encoded


def test_frozen_doctor_warns_for_hash_verified_dirty_development_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    marker = root / "DEVELOPMENT_BUILD_NOT_FORMALLY_QUALIFIED.txt"
    marker.write_text(
        "This bundle was built from an explicitly allowed dirty working tree.\r\n"
        "It is for isolated development validation only and is not a formally qualified release.\r\n",
        encoding="utf-8",
        newline="",
    )
    manifest_path = _write_valid_build_manifest(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["dirty"] = True
    manifest["source"]["status_sha256"] = "f" * 64
    from daguandan_bridge.build_manifest import compute_build_id

    manifest["build_id"] = compute_build_id(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check = doctor._build_integrity_check(root, frozen=True)

    assert check["status"] == "WARN"
    assert check["evidence"]["errors"] == []
    assert any(
        "development build" in item
        for item in check["evidence"]["warnings"]
    )
