from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retain a sealed live session only after one source frame index."
    )
    parser.add_argument("session_directory", type=Path)
    parser.add_argument("--after-frame-index", required=True, type=int)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_root))
    from daguandan_bridge.live.session_trim import trim_session_after_frame

    result = trim_session_after_frame(
        args.session_directory,
        after_frame_index=args.after_frame_index,
    )
    print(
        "trimmed "
        f"source_frames={result.source_frame_count} "
        f"retained_frames={result.retained_frame_count} "
        f"first_source_frame={result.first_retained_source_frame_index}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
