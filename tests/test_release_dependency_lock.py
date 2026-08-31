from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import daguandan_bridge.release_lock as release_lock
from daguandan_bridge.release_lock import verify_release_inputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = Path(
    r"C:\Users\yhx\AppData\Local\DaguandanAssistant\release-inputs\wheelhouse-cp312-win_amd64"
)


def test_runtime_requirements_are_exactly_pinned():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "torch==2.13.0" in pyproject
    assert "torch==2.13.0" in requirements
    assert "torch>=" not in pyproject
    assert "torch>=" not in requirements


def test_release_lock_has_hash_for_every_wheel():
    wheel_lock = json.loads((PROJECT_ROOT / "wheelhouse.lock.json").read_text(encoding="utf-8"))
    requirements = (PROJECT_ROOT / "requirements-release.lock").read_text(encoding="utf-8")

    assert wheel_lock["schema"] == "guandan.wheelhouse-lock/1"
    assert wheel_lock["files"]
    for record in wheel_lock["files"]:
        assert f"--hash=sha256:{record['sha256']}" in requirements


def test_toolchain_locks_python_dll_and_critical_runtime_inventory():
    toolchain = json.loads(
        (PROJECT_ROOT / "release_toolchain.lock.json").read_text(encoding="utf-8")
    )
    runtime = toolchain["python_runtime"]
    inventory_path = PROJECT_ROOT / runtime["inventory_lock"]["path"]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = inventory["files"]
    python_dll = runtime["python_dll"]

    assert toolchain["schema"] == "guandan.release-toolchain-lock/3"
    assert inventory["schema"] == "guandan.python-runtime-lock/1"
    assert runtime["inventory_lock"]["sha256"] == hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    assert toolchain["locks"]["python_runtime_lock_sha256"] == runtime[
        "inventory_lock"
    ]["sha256"]
    assert python_dll["path"] == "python312.dll"
    assert python_dll in records
    assert any(item["path"] == "Lib/encodings/__init__.py" for item in records)
    assert any(item["path"] == "Lib/encodings/aliases.py" for item in records)
    assert any(item["path"] == "Lib/venv/__init__.py" for item in records)
    assert any(
        item["path"].startswith("Lib/ensurepip/_bundled/")
        and item["path"].endswith(".whl")
        for item in records
    )
    assert runtime["aggregate_sha256"] == inventory["aggregate_sha256"]
    assert inventory["aggregate_sha256"] == release_lock.runtime_inventory_sha256(records)
    assert inventory["file_count"] == len(records) >= 2000
    assert toolchain["acquisition"]["provider"] == "python.org"
    assert len(toolchain["acquisition"]["sha256"]) == 64


def test_runtime_lock_detects_derived_python_dll_mismatch(tmp_path):
    base = tmp_path / "python"
    base.mkdir()
    dll = base / "python312.dll"
    dll.write_bytes(b"runtime")
    record = {
        "path": "python312.dll",
        "bytes": len(b"runtime"),
        "sha256": hashlib.sha256(b"runtime").hexdigest(),
    }
    toolchain = {
        "platform": {"python_version": "3.12.0", "python_cache_tag": "cpython-312"},
        "python_runtime": {
            "files": [record],
            "python_dll": {**record, "path": "python311.dll"},
            "aggregate_sha256": release_lock.runtime_inventory_sha256([record]),
        },
    }
    errors = []

    release_lock._verify_python_runtime_lock(toolchain, base, errors)

    codes = {item["code"] for item in errors}
    assert "LOCK-PYTHON-DLL-DERIVED" in codes


def _fake_complete_python_runtime(tmp_path: Path) -> Path:
    base = tmp_path / "python"
    contents = {
        "python.exe": b"python",
        "pythonw.exe": b"pythonw",
        "python3.dll": b"python3",
        "python312.dll": b"python312",
        "vcruntime140.dll": b"vcruntime",
        "vcruntime140_1.dll": b"vcruntime1",
        "LICENSE.txt": b"license",
        "DLLs/_socket.pyd": b"socket",
        "Lib/encodings/__init__.py": b"encodings",
        "Lib/encodings/aliases.py": b"aliases-before",
        "Lib/venv/__init__.py": b"venv",
        "Lib/ensurepip/_bundled/pip-test.whl": b"pip-wheel",
        "Lib/os.py": b"os-module",
        "libs/python312.lib": b"import-lib",
    }
    for relative, payload in contents.items():
        path = base.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return base


@pytest.mark.parametrize("mutation", ["aliases", "missing", "unexpected"])
def test_complete_runtime_inventory_rejects_mutated_missing_or_unexpected_input(
    tmp_path,
    mutation,
):
    base = _fake_complete_python_runtime(tmp_path)
    inventory = release_lock.create_python_runtime_lock(base)
    records = inventory["files"]
    python_dll = next(item for item in records if item["path"] == "python312.dll")
    toolchain = {
        "platform": {
            "python_version": "3.12.0",
            "python_cache_tag": "cpython-312",
        },
        "python_runtime": {"python_dll": python_dll},
    }
    if mutation == "aliases":
        (base / "Lib" / "encodings" / "aliases.py").write_bytes(b"changed")
    elif mutation == "missing":
        (base / "Lib" / "os.py").unlink()
    else:
        (base / "Lib" / "unexpected.py").write_bytes(b"new input")
    errors: list[dict[str, object]] = []

    release_lock._verify_python_runtime_lock(
        toolchain,
        base,
        errors,
        runtime_inventory=inventory,
    )

    codes = {item["code"] for item in errors}
    if mutation == "aliases":
        assert "LOCK-PYTHON-RUNTIME-INTEGRITY" in codes
    else:
        assert "LOCK-PYTHON-RUNTIME-FILESET" in codes


def test_prepared_release_wheelhouse_matches_all_committed_locks():
    if not WHEELHOUSE.is_dir():
        pytest.skip("external release wheelhouse has not been prepared")

    report = verify_release_inputs(
        project_root=PROJECT_ROOT,
        wheelhouse_root=WHEELHOUSE,
        python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    )

    assert report["status"] == "PASS", report["errors"]


def test_installed_inventory_gate_rejects_any_extra_distribution(monkeypatch):
    if not WHEELHOUSE.is_dir():
        pytest.skip("external release wheelhouse has not been prepared")
    wheel_lock = json.loads((PROJECT_ROOT / "wheelhouse.lock.json").read_text(encoding="utf-8"))
    toolchain = json.loads(
        (PROJECT_ROOT / "release_toolchain.lock.json").read_text(encoding="utf-8")
    )
    expected = {
        release_lock._canonical_name(record["distribution"]): record["version"]
        for record in wheel_lock["files"]
    }
    expected["pip"] = toolchain["tools"]["pip"]
    monkeypatch.setattr(
        release_lock,
        "collect_installed_distributions",
        lambda: dict(expected),
    )
    passed = verify_release_inputs(
        project_root=PROJECT_ROOT,
        wheelhouse_root=WHEELHOUSE,
        python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        verify_installed=True,
    )
    assert passed["status"] == "PASS", passed["errors"]
    assert passed["installed_distributions"] == expected

    monkeypatch.setattr(
        release_lock,
        "collect_installed_distributions",
        lambda: {**expected, "unlocked-package": "1.0"},
    )
    failed = verify_release_inputs(
        project_root=PROJECT_ROOT,
        wheelhouse_root=WHEELHOUSE,
        python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        verify_installed=True,
    )
    assert failed["status"] == "FAIL"
    issue = next(
        item for item in failed["errors"] if item["code"] == "LOCK-INSTALLED-DISTRIBUTIONS"
    )
    assert issue["evidence"]["unexpected"] == ["unlocked-package"]


def test_fresh_venv_uses_the_hash_locked_python_launcher_and_pip(tmp_path):
    bootstrap = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    build_env = tmp_path / "build-env"
    created = subprocess.run(
        [str(bootstrap), "-I", "-S", "-m", "venv", str(build_env)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    build_python = build_env / "Scripts" / "python.exe"
    toolchain = json.loads(
        (PROJECT_ROOT / "release_toolchain.lock.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256(build_python.read_bytes()).hexdigest() == (
        toolchain["python_executable"]["sha256"]
    )
    queried = subprocess.run(
        [
            str(build_python),
            "-I",
            "-c",
            "import importlib.metadata; print(importlib.metadata.version('pip'))",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert queried.returncode == 0, queried.stderr
    assert queried.stdout.strip() == toolchain["tools"]["pip"]


def _install_distribution_metadata(
    site_packages: Path,
    name: str,
    version: str,
    *,
    egg_info: bool = False,
) -> None:
    normalized = name.replace("-", "_").replace(".", "_")
    suffix = ".egg-info" if egg_info else ".dist-info"
    metadata_root = site_packages / f"{normalized}-{version}{suffix}"
    metadata_root.mkdir(parents=True, exist_ok=True)
    filename = "PKG-INFO" if egg_info else "METADATA"
    (metadata_root / filename).write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )


def test_fresh_inventory_ignores_repo_egg_info_even_when_repo_is_on_sys_path(
    tmp_path,
    monkeypatch,
):
    fresh_site = tmp_path / "fresh" / "Lib" / "site-packages"
    repo_src = tmp_path / "repo" / "src"
    _install_distribution_metadata(fresh_site, "pip", "23.2.1")
    _install_distribution_metadata(
        repo_src,
        "daguandan-danzero-bridge",
        "0.1.0",
        egg_info=True,
    )
    monkeypatch.syspath_prepend(str(repo_src))
    monkeypatch.chdir(repo_src.parent)

    inventory = release_lock.collect_installed_distributions([fresh_site])

    assert inventory == {"pip": "23.2.1"}
    assert "daguandan-danzero-bridge" not in inventory


def test_extra_distribution_really_in_fresh_site_packages_still_fails_gate(
    tmp_path,
):
    if not WHEELHOUSE.is_dir():
        pytest.skip("external release wheelhouse has not been prepared")
    wheel_lock = json.loads(
        (PROJECT_ROOT / "wheelhouse.lock.json").read_text(encoding="utf-8")
    )
    toolchain = json.loads(
        (PROJECT_ROOT / "release_toolchain.lock.json").read_text(encoding="utf-8")
    )
    fresh_site = tmp_path / "fresh" / "Lib" / "site-packages"
    for record in wheel_lock["files"]:
        _install_distribution_metadata(
            fresh_site,
            record["distribution"],
            record["version"],
        )
    _install_distribution_metadata(fresh_site, "pip", toolchain["tools"]["pip"])

    passed = verify_release_inputs(
        project_root=PROJECT_ROOT,
        wheelhouse_root=WHEELHOUSE,
        python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        verify_installed=True,
        installed_distribution_paths=[fresh_site],
    )
    assert passed["status"] == "PASS", passed["errors"]

    _install_distribution_metadata(fresh_site, "unlocked-package", "1.0")
    failed = verify_release_inputs(
        project_root=PROJECT_ROOT,
        wheelhouse_root=WHEELHOUSE,
        python_executable=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        verify_installed=True,
        installed_distribution_paths=[fresh_site],
    )
    assert failed["status"] == "FAIL"
    issue = next(
        item
        for item in failed["errors"]
        if item["code"] == "LOCK-INSTALLED-DISTRIBUTIONS"
    )
    assert issue["evidence"]["unexpected"] == ["unlocked-package"]
