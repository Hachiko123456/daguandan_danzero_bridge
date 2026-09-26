from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_build_cache.py"
SPEC = importlib.util.spec_from_file_location("release_build_cache_tested", SCRIPT)
assert SPEC and SPEC.loader
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)
pytestmark = pytest.mark.unit
KEY = "a" * 64
OTHER_KEY = "b" * 64


def write(path: Path, data: bytes = b"payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    python = tmp_path / "python"
    root.mkdir()
    python.mkdir()
    for name in (*cache.ENVIRONMENT_FILES, *cache.WORK_FILES, *cache.RESOURCE_FILES):
        write(root / name, name.encode("utf-8"))
    for name in cache.RESOURCE_TREES:
        write(root / name / "resource.bin", name.encode("utf-8"))
    write(root / "src" / "demo" / "__init__.py", b"VALUE = 1\n")
    # A key/inspection must never attempt to run this dummy interpreter.
    write(python / "python.exe", b"not an executable")
    return root, python


@pytest.fixture
def directory(tmp_path):
    root = tmp_path / "cache"
    write(root / "Scripts" / "python.exe", b"never execute cache bytes")
    write(root / "Lib" / "module.py", b"raise RuntimeError('never import cache')\n")
    write(root / "Lib" / "__pycache__" / "module.cpython-312.pyc", b"bytecode")
    (root / "empty").mkdir()
    return root


def receipt_path(directory):
    return directory.with_name(directory.name + ".receipt.json")


def receipt_value(directory):
    return json.loads(receipt_path(directory).read_text(encoding="utf-8"))


def change_receipt(directory, mutate):
    value = receipt_value(directory)
    mutate(value)
    receipt_path(directory).write_text(json.dumps(value), encoding="utf-8")


def cli(*arguments, cwd=None, env=None):
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-S", str(SCRIPT), *map(str, arguments)],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.stderr == "", result.stderr
    return result.returncode, json.loads(result.stdout)


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("kind", ["environment", "work"])
def test_seal_then_inspect_hashes_every_file_and_empty_directory(directory, kind):
    before = snapshot(directory)
    result = cache.seal_cache(directory, kind, KEY)
    assert result["sealed"] is True
    assert result["file_count"] == 3
    assert result["receipt_path"] == str(receipt_path(directory))
    value = receipt_value(directory)
    assert "empty" in value["directories"]
    assert any(item["path"].endswith(".pyc") for item in value["files"])
    assert cache.inspect_cache(directory, kind, KEY)["hit"] is True
    assert snapshot(directory) == before


def test_missing_directory_and_missing_or_partial_receipt_are_misses(directory):
    assert cache.inspect_cache(directory / "absent", "environment", KEY)["reason"] == "directory_missing"
    assert cache.inspect_cache(directory, "environment", KEY)["reason"] == "receipt_missing"
    receipt_path(directory).write_bytes(b'{"schema":')
    assert cache.inspect_cache(directory, "environment", KEY)["reason"] == "receipt_invalid"


@pytest.mark.parametrize("kind,key,reason", [
    ("work", KEY, "kind_mismatch"),
    ("environment", OTHER_KEY, "key_mismatch"),
])
def test_kind_and_key_must_match(directory, kind, key, reason):
    cache.seal_cache(directory, "environment", KEY)
    assert cache.inspect_cache(directory, kind, key)["reason"] == reason


@pytest.mark.parametrize("mutation", ["delete", "extra", "same_size_mtime", "pyc", "extra_directory", "missing_directory"])
def test_corruption_and_full_tree_identity(directory, mutation):
    cache.seal_cache(directory, "work", KEY)
    victim = directory / "Lib" / "module.py"
    if mutation == "delete":
        victim.unlink()
    elif mutation == "extra":
        write(directory / "injected.py", b"untrusted")
    elif mutation == "same_size_mtime":
        info = victim.stat()
        victim.write_bytes(b"X" * info.st_size)
        os.utime(victim, ns=(info.st_atime_ns, info.st_mtime_ns))
        assert victim.stat().st_size == info.st_size
        assert victim.stat().st_mtime_ns == info.st_mtime_ns
    elif mutation == "pyc":
        (directory / "Lib" / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"tampered")
    elif mutation == "extra_directory":
        (directory / "unexpected-empty").mkdir()
    else:
        (directory / "empty").rmdir()
    result = cache.inspect_cache(directory, "work", KEY)
    assert result["hit"] is False
    assert result["reason"] in {"file_set_mismatch", "file_hash_mismatch", "file_size_mismatch", "directory_set_mismatch"}


def test_mtime_only_does_not_invalidate_content(directory):
    cache.seal_cache(directory, "work", KEY)
    path = directory / "Lib" / "module.py"
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
    assert cache.inspect_cache(directory, "work", KEY)["hit"] is True


def test_inspect_is_read_only(directory, monkeypatch):
    cache.seal_cache(directory, "environment", KEY)
    before = snapshot(directory.parent)

    def forbidden(*args, **kwargs):
        pytest.fail("inspect attempted a write")

    monkeypatch.setattr(cache, "_atomic_json", forbidden)
    monkeypatch.setattr(cache.tempfile, "mkstemp", forbidden)
    monkeypatch.setattr(cache.os, "replace", forbidden)
    assert cache.inspect_cache(directory, "environment", KEY)["hit"] is True
    assert snapshot(directory.parent) == before


@pytest.mark.parametrize("name", cache.ENVIRONMENT_FILES)
def test_every_lock_byte_changes_both_keys_even_with_same_metadata(project, name):
    root, python = project
    first = cache.build_keys(root, python)
    path = root / name
    info = path.stat()
    path.write_bytes(b"x" * info.st_size)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    second = cache.build_keys(root, python)
    assert first["environment_key"] != second["environment_key"]
    assert first["work_key"] != second["work_key"]


@pytest.mark.parametrize("mutation", ["change", "add", "delete"])
def test_source_changes_only_work_key(project, mutation):
    root, python = project
    victim = write(root / "src" / "demo" / "obsolete.py", b"A = 1\n")
    first = cache.build_keys(root, python)
    if mutation == "change":
        info = victim.stat()
        victim.write_bytes(b"B = 2\n")
        os.utime(victim, ns=(info.st_atime_ns, info.st_mtime_ns))
    elif mutation == "add":
        write(root / "src" / "demo" / "new.py", b"B = 2\n")
    else:
        victim.unlink()
    second = cache.build_keys(root, python)
    assert first["environment_key"] == second["environment_key"]
    assert first["work_key"] != second["work_key"]


@pytest.mark.parametrize("name", [*cache.WORK_FILES, *cache.RESOURCE_FILES,
                                   "release_assets/resource.bin", "data/profiles/tencent_daguandan/templates/resource.bin",
                                   "src/demo/packaged.dat"])
def test_packaging_inputs_and_resource_bytes_change_only_work_key(project, name):
    root, python = project
    path = root / name
    if not path.exists():
        write(path)
    first = cache.build_keys(root, python)
    path.write_bytes(b"new packaged resource bytes")
    second = cache.build_keys(root, python)
    assert first["environment_key"] == second["environment_key"]
    assert first["work_key"] != second["work_key"]


def test_generated_source_files_and_unrelated_trees_do_not_change_keys(project):
    root, python = project
    first = cache.build_keys(root, python)
    for name in ("tests/test_new.py", "reports/report.json", ".aoci/state.json",
                 "src/demo/__pycache__/__init__.pyc", "src/demo.egg-info/PKG-INFO"):
        write(root / name, b"irrelevant generated data")
    second = cache.build_keys(root, python)
    assert first == second
    assert len(first["environment_key"]) == len(first["work_key"]) == 64
    assert int(first["work_key"], 16) >= 0


def test_keys_are_deterministic_and_path_normalized_without_resolve(project):
    root, python = project
    first = cache.build_keys(root, python)
    assert cache.build_keys(str(root) + os.sep + ".", str(python) + os.sep) == first
    another = root.parent / "other-python"
    another.mkdir()
    second = cache.build_keys(root, another)
    assert first["environment_key"] != second["environment_key"]
    if os.name == "nt":
        assert cache.build_keys(str(root).upper(), str(python).upper()) == first


def test_interpreter_and_helper_identity_change_keys(project, monkeypatch):
    root, python = project
    first = cache.build_keys(root, python)
    monkeypatch.setattr(cache, "HELPER_VERSION", "future")
    second = cache.build_keys(root, python)
    assert first["environment_key"] != second["environment_key"]
    monkeypatch.setattr(cache.platform, "machine", lambda: "different-architecture")
    third = cache.build_keys(root, python)
    assert second["environment_key"] != third["environment_key"]


@pytest.mark.parametrize("name", [*cache.ENVIRONMENT_FILES, *cache.WORK_FILES, *cache.RESOURCE_FILES])
def test_missing_required_input_is_error(project, name):
    root, python = project
    (root / name).unlink()
    with pytest.raises((cache.CacheError, OSError)):
        cache.build_keys(root, python)


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "C:/absolute", "file:stream", "dir\\file",
                                    "dir/../file", "./file", "file.", "NUL", "COM1.txt", "dir//file"])
def test_unsafe_receipt_paths_are_refused_not_followed(directory, unsafe):
    cache.seal_cache(directory, "work", KEY)
    change_receipt(directory, lambda value: value["files"][0].update(path=unsafe))
    with pytest.raises(cache.UnsafePath):
        cache.inspect_cache(directory, "work", KEY)


@pytest.mark.parametrize("content", [b"", b"{", b"[]", b"null", b"\xff", b'{"schema":"x","schema":"y"}',
                                    b"[" * 2000 + b"]" * 2000])
def test_bad_receipts_are_normal_misses(directory, content):
    receipt_path(directory).write_bytes(content)
    assert cache.inspect_cache(directory, "work", KEY)["reason"] == "receipt_invalid"


@pytest.mark.parametrize("change", [
    lambda value: value.update(file_count=99),
    lambda value: value.update(files=[]),
    lambda value: value.update(files=[1]),
    lambda value: value.update(directories=[1]),
    lambda value: value["files"][0].update(size=True),
    lambda value: value["files"][0].update(sha256="invalid"),
    lambda value: value["files"].append(dict(value["files"][0])),
    lambda value: value.update(directory="not-the-cache"),
])
def test_invalid_receipt_structure_never_hits(directory, change):
    cache.seal_cache(directory, "work", KEY)
    change_receipt(directory, change)
    assert cache.inspect_cache(directory, "work", KEY)["hit"] is False


def test_receipt_byte_and_entry_limits(directory, monkeypatch):
    cache.seal_cache(directory, "work", KEY)
    monkeypatch.setattr(cache, "MAX_RECEIPT_BYTES", 10)
    assert cache.inspect_cache(directory, "work", KEY)["reason"] == "receipt_invalid"
    monkeypatch.setattr(cache, "MAX_RECEIPT_BYTES", 32 * 1024 * 1024)
    value = receipt_value(directory)
    value["directories"] += [f"extra-{i}" for i in range(12)]
    receipt_path(directory).write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(cache, "MAX_ENTRIES", 10)
    assert cache.inspect_cache(directory, "work", KEY)["reason"] == "receipt_invalid"


def make_symlink(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


@pytest.mark.parametrize("location", ["root", "ancestor", "receipt", "file", "subdirectory"])
def test_symlinks_in_all_path_positions_are_refused(directory, tmp_path, location):
    cache.seal_cache(directory, "work", KEY)
    inspected = directory
    link = None
    if location == "root":
        link = tmp_path / "linked-cache"
        make_symlink(link, directory, directory=True)
        inspected = link
    elif location == "ancestor":
        link = tmp_path / "linked-parent"
        make_symlink(link, tmp_path, directory=True)
        inspected = link / "cache"
    elif location == "receipt":
        receipt = receipt_path(directory)
        target = write(tmp_path / "saved-receipt.json", receipt.read_bytes())
        receipt.unlink()
        link = receipt
        make_symlink(link, target)
    elif location == "file":
        link = directory / "linked-file"
        make_symlink(link, directory / "Lib" / "module.py")
    else:
        link = directory / "linked-directory"
        make_symlink(link, directory / "Lib", directory=True)
    try:
        with pytest.raises(cache.UnsafePath):
            cache.inspect_cache(inspected, "work", KEY)
        with pytest.raises(cache.UnsafePath):
            cache.seal_cache(inspected, "work", KEY)
    finally:
        link.unlink()


def test_missing_directory_does_not_hide_unsafe_receipt(directory):
    missing = directory.parent / "missing"
    link = receipt_path(missing)
    make_symlink(link, directory / "nonexistent")
    try:
        with pytest.raises(cache.UnsafePath):
            cache.inspect_cache(missing, "work", KEY)
    finally:
        link.unlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction integration")
@pytest.mark.parametrize("location", ["root", "ancestor", "child", "receipt"])
def test_real_windows_junctions_refused(directory, tmp_path, location):
    cache.seal_cache(directory, "work", KEY)
    link = {"root": tmp_path / "junction", "ancestor": tmp_path / "junction",
            "child": directory / "junction", "receipt": receipt_path(directory)}[location]
    if location == "receipt":
        link.unlink()
    target = tmp_path if location == "ancestor" else directory / "Lib"
    completed = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                               capture_output=True, text=True, errors="replace", timeout=10)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    inspected = link / "cache" if location == "ancestor" else link if location == "root" else directory
    try:
        with pytest.raises(cache.UnsafePath):
            cache.inspect_cache(inspected, "work", KEY)
        with pytest.raises(cache.UnsafePath):
            cache.seal_cache(inspected, "work", KEY)
    finally:
        os.rmdir(link)  # Remove only the junction, never recurse into its target.


@pytest.mark.parametrize("which", ["project", "python"])
def test_key_rejects_linked_root_before_normalization(project, tmp_path, which):
    root, python = project
    link = tmp_path / "linked-root"
    make_symlink(link, root if which == "project" else python, directory=True)
    try:
        with pytest.raises(cache.UnsafePath):
            cache.build_keys(link if which == "project" else root, link if which == "python" else python)
    finally:
        link.unlink()


def test_non_symlink_reparse_attribute_is_refused():
    assert cache._reparse(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400))


@pytest.mark.parametrize("path", ["../cache", "cache/../elsewhere", "cache:stream", "NUL", "C:relative"])
def test_unsafe_cli_paths_are_refused(path):
    code, value = cli("inspect", "--directory", path, "--kind", "work", "--key", KEY, "--output", "-")
    assert code != 0
    assert value["reason"] == "unsafe_path"


def test_filesystem_root_refused():
    with pytest.raises(cache.UnsafePath):
        cache.inspect_cache(Path.cwd().anchor, "work", KEY)


@pytest.mark.parametrize("failure", ["replace", "fsync", "inventory"])
@pytest.mark.parametrize("existing", [False, True])
def test_seal_failure_never_publishes_partial_receipt_or_deletes_cache(directory, monkeypatch, failure, existing):
    if existing:
        cache.seal_cache(directory, "work", KEY)
    before = snapshot(directory.parent)

    def fail(*args, **kwargs):
        raise OSError("injected atomic failure")

    if failure == "inventory":
        monkeypatch.setattr(cache, "_record", fail)
    else:
        monkeypatch.setattr(cache.os, failure, fail)
    with pytest.raises(OSError, match="injected atomic failure"):
        cache.seal_cache(directory, "work", OTHER_KEY)
    assert snapshot(directory.parent) == before
    assert not list(directory.parent.glob(".*.tmp"))
    assert directory.is_dir()


def test_seal_refuses_changes_during_inventory(directory, monkeypatch):
    original = cache._record
    mutated = False

    def record(*args, **kwargs):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            mutated = True
            write(directory / "late-file", b"late")
        return result

    monkeypatch.setattr(cache, "_record", record)
    with pytest.raises(cache.CacheError, match="tree changed"):
        cache.seal_cache(directory, "work", KEY)
    assert not receipt_path(directory).exists()


def test_empty_cache_cannot_be_sealed(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(cache.CacheError, match="empty"):
        cache.seal_cache(empty, "environment", KEY)


@pytest.mark.parametrize("command", [[], ["key"], ["inspect"], ["seal"]])
def test_all_help_is_json_and_works_without_site(command):
    code, value = cli(*command, "--help")
    assert code == 0
    assert "usage:" in value["help"]


def test_cli_key_output_is_json_and_environment_does_not_leak(project, tmp_path):
    root, python = project
    output = tmp_path / "key.json"
    env = {**os.environ, "AWS_SECRET_ACCESS_KEY": "sentinel-private-secret", "PYTHONPATH": "invalid"}
    code, value = cli("key", "--project-root", root, "--python-root", python, "--output", output,
                      cwd=python, env=env)
    assert code == 0
    assert json.loads(output.read_text()) == value
    assert "sentinel-private-secret" not in output.read_text()
    assert value["environment_key"] == cache.build_keys(root, python)["environment_key"]


def test_cli_seal_inspect_and_miss_exit_zero(directory, tmp_path):
    output = tmp_path / "result.json"
    args = ("--directory", directory, "--kind", "environment", "--key", KEY, "--output", output)
    code, value = cli("seal", *args)
    assert code == 0 and value["sealed"]
    code, value = cli("inspect", *args)
    assert code == 0 and value["hit"]
    assert json.loads(output.read_text()) == value
    receipt_path(directory).write_bytes(b"partial")
    code, value = cli("inspect", *args)
    assert code == 0 and not value["hit"]
    assert json.loads(output.read_text()) == value


@pytest.mark.parametrize("output", ["internal", "receipt"])
@pytest.mark.parametrize("command", ["inspect", "seal"])
def test_output_cannot_mutate_cache_or_receipt(directory, tmp_path, output, command):
    cache.seal_cache(directory, "work", KEY)
    before = snapshot(directory.parent)
    path = directory / "result.json" if output == "internal" else receipt_path(directory)
    code, value = cli(command, "--directory", directory, "--kind", "work", "--key", KEY, "--output", path)
    assert code != 0 and value["reason"] == "unsafe_path"
    assert snapshot(directory.parent) == before


def test_output_symlink_is_refused_before_sealing(directory, tmp_path):
    target = write(tmp_path / "output-target", b"do not overwrite")
    link = tmp_path / "output.json"
    make_symlink(link, target)
    try:
        code, value = cli("seal", "--directory", directory, "--kind", "work", "--key", KEY, "--output", link)
        assert code != 0 and value["reason"] == "unsafe_path"
        assert target.read_bytes() == b"do not overwrite"
        assert not receipt_path(directory).exists()
    finally:
        link.unlink()


def test_output_cannot_overwrite_source(project):
    root, python = project
    path = root / "run.py"
    before = path.read_bytes()
    code, value = cli("key", "--project-root", root, "--python-root", python, "--output", path)
    assert code != 0 and value["reason"] == "unsafe_path"
    assert path.read_bytes() == before


def test_invalid_cli_arguments_are_json_errors():
    code, value = cli("inspect", "--not-an-argument")
    assert code != 0 and "error" in value


@pytest.mark.parametrize("parent", ["__pycache__", "generated.egg-info"])
def test_all_source_python_files_count_even_in_generated_named_directories(project, parent):
    root, python = project
    first = cache.build_keys(root, python)
    write(root / "src" / parent / "unexpected.py", b"code = True\n")
    second = cache.build_keys(root, python)
    assert first["environment_key"] == second["environment_key"]
    assert first["work_key"] != second["work_key"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction integration")
@pytest.mark.parametrize("which", ["project", "python"])
def test_key_rejects_real_junction_roots(project, tmp_path, which):
    root, python = project
    link = tmp_path / "root-junction"
    target = root if which == "project" else python
    result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                            capture_output=True, text=True, errors="replace", timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        with pytest.raises(cache.UnsafePath):
            cache.build_keys(link if which == "project" else root, link if which == "python" else python)
    finally:
        os.rmdir(link)


def test_inspect_refuses_tree_mutation_while_hashing(directory, monkeypatch):
    cache.seal_cache(directory, "work", KEY)
    original = cache._record
    mutated = False

    def record(*args, **kwargs):
        nonlocal mutated
        result = original(*args, **kwargs)
        if not mutated:
            mutated = True
            write(directory / "late-file", b"late")
        return result

    monkeypatch.setattr(cache, "_record", record)
    assert cache.inspect_cache(directory, "work", KEY)["reason"] == "tree_changed"


def test_snapshot_detects_same_size_mtime_changes_by_hash_without_trusting_file_stats(directory):
    cache.seal_cache(directory, "work", KEY)
    value = receipt_value(directory)
    for item in value["files"]:
        assert set(item) == {"path", "size", "sha256"}
    assert "mtime" not in json.dumps(value)


@pytest.mark.parametrize("command", ["inspect", "seal"])
def test_invalid_hash_cannot_be_used(directory, command):
    code, value = cli(command, "--directory", directory, "--kind", "work", "--key", "bad", "--output", "-")
    assert code != 0 and "64 hexadecimal" in value["error"]


def test_unsafe_output_ancestor_refused(directory, tmp_path):
    # Exercise ancestor rejection even when the output leaf does not exist.
    if os.name != "nt":
        pytest.skip("junction creation is Windows-specific")
    link = tmp_path / "output-junction"
    target = tmp_path / "output-directory"
    target.mkdir()
    result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                            capture_output=True, text=True, errors="replace", timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        code, value = cli("seal", "--directory", directory, "--kind", "work", "--key", KEY,
                          "--output", link / "result.json")
        assert code != 0 and value["reason"] == "unsafe_path"
        assert not receipt_path(directory).exists()
        assert not list(target.iterdir())
    finally:
        os.rmdir(link)


@pytest.mark.parametrize("unsafe_root", ["../elsewhere", "cache:stream", "dir/../cache"])
def test_unsafe_receipt_root_spelling_refused_without_following_it(directory, unsafe_root):
    cache.seal_cache(directory, "work", KEY)
    change_receipt(directory, lambda value: value.update(directory=unsafe_root))
    with pytest.raises(cache.UnsafePath):
        cache.inspect_cache(directory, "work", KEY)
