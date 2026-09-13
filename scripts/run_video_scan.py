"""对指定会话的视频重跑一次纯视频扫描，输出到 reports/video-scans/。

用法：
    python scripts/run_video_scan.py <session_dir>
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from daguandan_bridge.annotation_service import AnnotationService
from daguandan_bridge.application.session_workbench import default_profile_context
from daguandan_bridge.application.video_scan import VideoActionScanner, VideoScanRequest
from daguandan_bridge.recognition_service import ScreenshotRecognitionService
from daguandan_bridge.template_service import TemplateService


def main() -> int:
    session = Path(sys.argv[1]).resolve()
    video = session / "video" / "game.avi"
    frame_index = session / "video" / "frame_index.jsonl"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        Path("reports/video-scans") / f"{session.name}_{stamp}"
    ).resolve()
    profile = default_profile_context().profile_path
    recognition = ScreenshotRecognitionService(
        AnnotationService(profile.parent, profile.name),
        TemplateService(profile.parent, profile.name),
    )
    scanner = VideoActionScanner(recognition)
    result = scanner.scan(
        VideoScanRequest(
            video_path=video,
            frame_index_path=frame_index if frame_index.is_file() else None,
            output_directory=output,
            session_id=session.name,
        )
    )
    print(f"status={result.status} output={result.output_directory}")
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
