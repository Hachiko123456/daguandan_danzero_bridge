from __future__ import annotations

from pathlib import Path

from .runtime_layout import resolve_runtime_layout


# Source mode keeps the historical repository paths.  Frozen mode resolves an
# immutable bundle resource root plus a versioned, writable user generation.
# ``run.py`` atomically seeds that generation before constructing services.
RUNTIME_LAYOUT = resolve_runtime_layout()
PROJECT_ROOT: Path = RUNTIME_LAYOUT.bundle_root
RESOURCE_DATA_DIR: Path = RUNTIME_LAYOUT.resource_data_dir
RUNTIME_ROOT: Path = RUNTIME_LAYOUT.runtime_root
DATA_DIR: Path = RUNTIME_LAYOUT.data_dir
PROFILES_ROOT: Path = RUNTIME_LAYOUT.profiles_root
LOGS_ROOT: Path = RUNTIME_LAYOUT.logs_root
DIAGNOSTICS_ROOT: Path = RUNTIME_LAYOUT.diagnostics_root
PREFERENCES_ROOT: Path = RUNTIME_LAYOUT.preferences_root
CACHE_ROOT: Path = RUNTIME_LAYOUT.cache_root

DEFAULT_BASE_SIZE: tuple[int, int] = (1280, 720)
DEFAULT_AUTO_CAPTURE_INTERVAL_SEC: float = 1.0

SCREENSHOTS_DIR_NAME = "screenshots"
PICS_DIR_NAME = "pics"
PROFILE_CONFIG_FILE_NAME = "profile.json"
TEMPLATES_CONFIG_FILE_NAME = "templates_config.json"
REGIONS_CONFIG_FILE_NAME = "regions_config.json"

CAPTURE_WINDOW_NAME = "Poke Vision Capture - SPACE Start/Pause, C Save, Q/ESC Quit"
ROI_WINDOW_NAME = "Poke Vision ROI - SPACE/ENTER Confirm, ESC Finish"
