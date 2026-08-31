from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest

import daguandan_bridge.release_manager as release_manager
from daguandan_bridge.runtime_layout import (
    RuntimeLayoutError,
    runtime_storage_lock,
    runtime_storage_lock_owner_path,
)


def _hold_runtime_lock(runtime_text: str, ready) -> None:
    with runtime_storage_lock(
        Path(runtime_text), operation="multiprocess-holder", timeout_seconds=5.0
    ):
        ready.set()
        time.sleep(1.0)


def _crash_with_runtime_lock(runtime_text: str, ready) -> None:
    with runtime_storage_lock(
        Path(runtime_text), operation="crashing-holder", timeout_seconds=5.0
    ):
        ready.set()
        os._exit(23)


def test_runtime_storage_lock_times_out_with_owner_metadata_across_processes(tmp_path):
    runtime = tmp_path / "runtime"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_runtime_lock, args=(str(runtime), ready))
    process.start()
    assert ready.wait(10)
    owner = json.loads(
        runtime_storage_lock_owner_path(runtime).read_text(encoding="utf-8")
    )
    assert owner["state"] == "held"
    assert owner["operation"] == "multiprocess-holder"
    assert owner["pid"] == process.pid

    with pytest.raises(RuntimeLayoutError, match="multiprocess-holder"):
        with runtime_storage_lock(
            runtime, operation="contender", timeout_seconds=0.1
        ):
            pytest.fail("contender entered a held cross-process transaction")

    process.join(10)
    assert process.exitcode == 0


def test_runtime_storage_lock_is_recoverable_after_process_crash(tmp_path):
    runtime = tmp_path / "runtime"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_crash_with_runtime_lock, args=(str(runtime), ready))
    process.start()
    assert ready.wait(10)
    process.join(10)
    assert process.exitcode == 23

    with runtime_storage_lock(
        runtime, operation="crash-recovery", timeout_seconds=2.0
    ) as owner:
        assert owner["operation"] == "crash-recovery"


def test_pointer_transaction_cas_never_restores_over_newer_foreign_values(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "runtime"
    install = release_manager.ensure_install_root(runtime)
    release_path = install / "active.json"
    data_path = runtime / "data" / "v1" / "active.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    release_before = {"schema": "before-release", "value": 1}
    data_before = {"schema": "before-data", "value": 1}
    release_manager.atomic_write_json(release_path, release_before)
    release_manager.atomic_write_json(data_path, data_before)
    generation = runtime / "data" / "v1" / "generations" / "build-a"
    generation.mkdir(parents=True)
    marker = {
        "schema": release_manager.GENERATION_SCHEMA,
        "data_schema": release_manager.DATA_SCHEMA_VERSION,
        "build_id": "build-a",
        "generation_id": "build-a",
    }
    marker_path = generation / release_manager.GENERATION_MARKER
    release_manager.atomic_write_json(marker_path, marker)
    marker_sha = release_manager.sha256_file(marker_path)
    foreign_release = {"schema": "foreign-release", "value": 3}
    foreign_data = {"schema": "foreign-data", "value": 3}
    real_write = release_manager.atomic_write_json

    def inject_newer_transaction(path, value):
        target = Path(path)
        if target == release_path and value.get("schema") == "candidate-release":
            real_write(data_path, foreign_data)
            real_write(release_path, foreign_release)
            raise OSError("simulated newer transaction")
        return real_write(path, value)

    monkeypatch.setattr(release_manager, "atomic_write_json", inject_newer_transaction)
    with pytest.raises(OSError, match="newer transaction"):
        release_manager._transactional_pointer_switch(
            install,
            transition_id="activate-cas-test",
            release_document={"schema": "candidate-release", "value": 2},
            data_document={
                "schema": release_manager.ACTIVE_GENERATION_SCHEMA,
                "data_schema": release_manager.DATA_SCHEMA_VERSION,
                "build_id": "build-a",
                "generation_id": "build-a",
            },
            expected_marker_sha256=marker_sha,
        )

    assert json.loads(release_path.read_text(encoding="utf-8")) == foreign_release
    assert json.loads(data_path.read_text(encoding="utf-8")) == foreign_data
    journal = json.loads(
        (install / "transactions" / "activate-cas-test.json").read_text(
            encoding="utf-8"
        )
    )
    assert journal["phase"] == "CONFLICTED"


def test_incomplete_data_publish_is_recovered_only_when_cas_still_matches(tmp_path):
    runtime = tmp_path / "runtime"
    install = release_manager.ensure_install_root(runtime)
    release_path = install / "active.json"
    data_path = runtime / "data" / "v1" / "active.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    release_before = {"schema": "before-release", "value": 1}
    data_before = {"schema": "before-data", "value": 1}
    release_after = {
        "schema": "after-release",
        "value": 2,
        "transaction_id": "crash-test",
    }
    data_after = {
        "schema": "after-data",
        "value": 2,
        "transaction_id": "crash-test",
    }
    release_manager.atomic_write_json(release_path, release_before)
    release_manager.atomic_write_json(data_path, data_after)
    journal_path = install / "transactions" / "crash-test.json"
    release_manager.atomic_write_json(
        journal_path,
        {
            "schema": "guandan.pointer-transaction/2",
            "transition_id": "crash-test",
            "phase": "DATA_PUBLISHED",
            "release_before": release_before,
            "data_before": data_before,
            "release_after": release_after,
            "data_after": data_after,
        },
    )

    release_manager._recover_pointer_transactions(install)

    assert json.loads(release_path.read_text(encoding="utf-8")) == release_before
    assert json.loads(data_path.read_text(encoding="utf-8")) == data_before
    assert json.loads(journal_path.read_text(encoding="utf-8"))["phase"] == "RECOVERED"
