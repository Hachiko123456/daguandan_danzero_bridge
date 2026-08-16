from __future__ import annotations

from pathlib import Path
import sys


# 开发时使用仓库根目录；PyInstaller 打包后则始终使用启动 exe 所在目录。
# 运行数据需要可写，因此不能把它放在 PyInstaller 的 _internal 临时运行目录。
PROJECT_ROOT: Path = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[2]
)
DATA_DIR: Path = PROJECT_ROOT / "data"
PROFILES_ROOT: Path = DATA_DIR / "profiles"

DEFAULT_BASE_SIZE: tuple[int, int] = (1280, 720)
DEFAULT_AUTO_CAPTURE_INTERVAL_SEC: float = 1.0

SCREENSHOTS_DIR_NAME = "screenshots"
PICS_DIR_NAME = "pics"
PROFILE_CONFIG_FILE_NAME = "profile.json"
TEMPLATES_CONFIG_FILE_NAME = "templates_config.json"
REGIONS_CONFIG_FILE_NAME = "regions_config.json"

CAPTURE_WINDOW_NAME = "Poke Vision Capture - SPACE Start/Pause, C Save, Q/ESC Quit"
ROI_WINDOW_NAME = "Poke Vision ROI - SPACE/ENTER Confirm, ESC Finish"
