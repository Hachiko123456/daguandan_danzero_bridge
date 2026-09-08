import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from daguandan_bridge.bounded_log import RotatingTextLog, append_bounded_event


def test_rotating_text_logs_keep_only_two_bounded_utf8_segments(tmp_path):
    path = tmp_path / "startup.log"
    with RotatingTextLog(path, max_bytes=128) as log:
        for index in range(100):
            log.write(f"{index}: 测试日志，保护磁盘容量\n")
        log.flush()
    files = list(tmp_path.iterdir())
    assert len(files) == 2
    assert all(file.stat().st_size <= 128 for file in files)
    assert "99:" in path.read_text("utf-8")
    for file in files:
        file.read_text("utf-8")


def test_huge_unicode_entry_is_truncated_without_exceeding_budget(tmp_path):
    path = tmp_path / "exceptions.log"
    with RotatingTextLog(path, max_bytes=128) as log:
        assert log.write("重复异常" * 1000) == 4000
    assert path.stat().st_size <= 128
    assert "truncated" in path.read_text("utf-8")


def test_bounded_event_truncates_evidence_but_keeps_json_lines_valid(tmp_path):
    path = tmp_path / "startup.jsonl"
    for index in range(50):
        append_bounded_event(path, {"event": "error", "index": index,
                                  "evidence": {"large": "x" * 10000}}, max_bytes=256)
    assert len(list(tmp_path.iterdir())) == 2
    for file in tmp_path.iterdir():
        assert file.stat().st_size <= 256
        for line in file.read_text("utf-8").splitlines():
            assert json.loads(line)["evidence"]["truncated"] is True


def test_event_name_itself_cannot_break_json_budget(tmp_path):
    path = tmp_path / "events.jsonl"
    append_bounded_event(path, {"event": "x" * 10000, "evidence": {}}, max_bytes=128)
    assert json.loads(path.read_text("utf-8"))["event"] == "entry_omitted"
    assert path.stat().st_size <= 128


def test_rotation_sharing_violation_switches_to_bounded_discard_and_print_survives(tmp_path, monkeypatch):
    path = tmp_path / "startup.log"
    log = RotatingTextLog(path, max_bytes=128)
    log.write("a" * 100)

    def fail_rename(*_args, **_kwargs):
        raise PermissionError("sharing violation")

    monkeypatch.setattr(Path, "replace", fail_rename)
    assert log.write("b" * 100) == 100
    assert log.dropped_writes == 1
    assert log.dropped_characters == 100
    assert log.io_errors == 1
    assert "PermissionError" in log.last_error
    assert path.read_text("utf-8") == "a" * 100

    class NoEncodeString(str):
        def encode(self, *_args, **_kwargs):
            raise AssertionError("discard mode must not allocate an encoded copy")

        def __getitem__(self, _item):
            raise AssertionError("discard mode must not slice input")

    # A failed stream must not stat or reopen/append on future writes.
    def no_filesystem_retry(*_args, **_kwargs):
        raise AssertionError("discard mode must not retry filesystem operations")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "stat", no_filesystem_retry)
        scoped.setattr(Path, "open", no_filesystem_retry)
        assert log.write(NoEncodeString("x" * 10000)) == 10000
        with redirect_stdout(log), redirect_stderr(log):
            print("normal business print", flush=True)
            print("normal error print", file=__import__("sys").stderr, flush=True)
        log.flush()
        log.close()
        log.close()
    assert log.closed
    assert log.dropped_characters >= 10100
    assert path.stat().st_size == 100
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match="closed"):
        log.write("explicit close is still a normal closed stream")


def test_rotation_reopen_failure_retains_backup_and_never_grows_more_files(tmp_path, monkeypatch):
    path = tmp_path / "exceptions.log"
    log = RotatingTextLog(path, max_bytes=128)
    log.write("a" * 100)
    original_open = Path.open

    def fail_reopen(current, mode="r", *args, **kwargs):
        if current == path and mode == "w":
            raise OSError("disk full during reopen")
        return original_open(current, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_reopen)
    assert log.write("b" * 100) == 100
    for _ in range(100):
        assert log.write("further output") == len("further output")
    log.flush()
    log.close()
    assert log.dropped_writes == 101
    assert log.io_errors == 1
    assert not path.exists()
    assert path.with_name(path.name + ".1").read_text("utf-8") == "a" * 100
    assert sum(file.stat().st_size for file in tmp_path.iterdir()) == 100


@pytest.mark.parametrize("failure", ["write", "flush", "close"])
def test_existing_stream_io_failure_never_escapes_flush_close_or_redirected_print(tmp_path, failure):
    log = RotatingTextLog(tmp_path / "startup.log", max_bytes=128)
    original = log._file

    class FailingHandle:
        def write(self, text):
            if failure == "write":
                raise OSError("write error")
            return original.write(text)

        def flush(self):
            if failure == "flush":
                raise OSError("flush error")
            return original.flush()

        def close(self):
            original.close()
            if failure == "close":
                raise OSError("close error")

    log._file = FailingHandle()
    with redirect_stdout(log):
        print("business still running", flush=True)
    log.flush()
    log.close()
    log.close()
    assert log.closed
    assert log.io_errors >= 1
    assert original.closed


@pytest.mark.parametrize("target", ["current", "backup", "parent"])
@pytest.mark.parametrize("reparse", [False, True])
def test_initial_log_rejects_existing_or_dangling_link_by_lstat_without_exists(tmp_path, monkeypatch, target, reparse):
    path = tmp_path / "startup.log"
    suspicious = {"current": path, "backup": path.with_name("startup.log.1"), "parent": tmp_path}[target]
    original_lstat = Path.lstat

    def fake_lstat(current, *args, **kwargs):
        if current == suspicious:
            return SimpleNamespace(
                st_mode=stat.S_IFREG if reparse else stat.S_IFLNK,
                st_file_attributes=0x400 if reparse else 0,
            )
        return original_lstat(current, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    # Filesystem exists() is false for a dangling link; bypassing it is crucial.
    monkeypatch.setattr(Path, "exists", lambda _self: False)
    with pytest.raises(OSError, match="traverses a link"):
        RotatingTextLog(path, max_bytes=128)
    assert not list(tmp_path.iterdir())


def test_initial_path_permission_error_is_not_mistaken_for_absent_path(tmp_path, monkeypatch):
    path = tmp_path / "startup.log"
    original = Path.lstat

    def denied(current, *args, **kwargs):
        if current == path:
            raise PermissionError("denied path metadata")
        return original(current, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError, match="denied path metadata"):
        RotatingTextLog(path, max_bytes=128)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("target", ["current", "backup", "parent"])
def test_rotation_rechecks_all_paths_and_discards_if_swapped_to_link(tmp_path, monkeypatch, target):
    path = tmp_path / "startup.log"
    log = RotatingTextLog(path, max_bytes=128)
    log.write("a" * 100)
    suspicious = {"current": path, "backup": path.with_name("startup.log.1"), "parent": tmp_path}[target]
    original_lstat = Path.lstat

    def fake_lstat(current, *args, **kwargs):
        if current == suspicious:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original_lstat(current, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    assert log.write("b" * 100) == 100
    assert log.dropped_writes == 1
    assert "traverses a link" in log.last_error
    log.close()
    assert path.read_text("utf-8") == "a" * 100
    assert not path.with_name("startup.log.1").exists()


def test_initial_open_failure_still_raises_for_startup_fallback(tmp_path, monkeypatch):
    def denied(*_args, **_kwargs):
        raise PermissionError("initial open denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError, match="initial open denied"):
        RotatingTextLog(tmp_path / "startup.log", max_bytes=128)


def test_fileno_works_for_active_file_and_counters_remain_bounded(tmp_path, monkeypatch):
    log = RotatingTextLog(tmp_path / "startup.log", max_bytes=128)
    assert log.fileno() == log._file.fileno()
    log._discarding = True
    log.dropped_writes = log.dropped_characters = 2**63 - 1
    assert log.write("discarded") == 9
    assert log.dropped_writes == log.dropped_characters == 2**63 - 1
    log.close()
    with pytest.raises(ValueError, match="closed"):
        log.fileno()
