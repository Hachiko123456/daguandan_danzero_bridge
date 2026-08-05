from __future__ import annotations

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
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
