"""Stable post-mortem facts collected only after the recorder child exits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_facts(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False, "bytes": 0, "mtime_ns": None, "sha256": None}
    stat = path.stat()
    return {
        "path": str(path), "exists": True, "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def indexed_frame_count(path: Path) -> int:
    if not Path(path).is_file():
        return 0
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count


def stable_recording_files(video: Path, index: Path) -> dict[str, object]:
    return {"video": file_facts(video), "index": file_facts(index)}


__all__ = ["file_facts", "indexed_frame_count", "stable_recording_files"]
