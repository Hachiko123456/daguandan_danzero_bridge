from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "qualify_release.py"
SPEC = importlib.util.spec_from_file_location("qualify_release", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["qualify_release"] = MODULE
SPEC.loader.exec_module(MODULE)


def test_qualification_parser_requires_external_roots():
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "--release-root",
            r"C:\outside\candidate",
            "--wheelhouse",
            r"C:\outside\wheelhouse",
        ]
    )
    assert args.release_root.name == "candidate"
    assert args.wheelhouse.name == "wheelhouse"


def test_tree_hash_is_order_independent(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = MODULE._tree_hash(tmp_path)
    (tmp_path / "a.txt").touch()
    second = MODULE._tree_hash(tmp_path)
    assert first == second


def test_clean_runtime_environment_removes_host_pollution(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "private")
    monkeypatch.setenv("JAVA_HOME", "jdk")
    environment = MODULE._clean_runtime_environment(tmp_path / "data")
    assert environment["DAGUANDAN_DATA_ROOT"].endswith("data")
    assert "PYTHONPATH" not in environment
    assert "JAVA_HOME" not in environment


def test_stage_failure_is_machine_readable():
    failure = MODULE._StageFailure("boom", 7, ["cmd"], Path("x.log"), {"foo": "bar"})
    assert str(failure) == "boom"
    assert failure.exit_code == 7
    assert failure.command == ["cmd"]
    assert failure.evidence == {"foo": "bar"}
