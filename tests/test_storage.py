from __future__ import annotations

import json
from pathlib import Path

from daguandan_bridge.storage import atomic_write_json


def test_atomic_write_json_retries_transient_windows_replace_denial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "status.json"
    target.write_text('{"old": true}\n', encoding="utf-8")
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(path: Path, destination: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(5, "destination is briefly open")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    atomic_write_json(target, {"ready": True})

    assert attempts == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"ready": True}
    assert list(tmp_path.glob(".status.json.*.tmp")) == []
