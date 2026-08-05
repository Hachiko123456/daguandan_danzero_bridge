# 大掼蛋 DanZero 桥接器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated Windows desktop package that records Tencent DaGuandan screenshots, ships the existing template assets, and exposes a manual-state DanZero advice API.

**Architecture:** Keep capture, GUI, and DanZero as separate packages. The capture service owns profile configuration, frame standardization, sessions, and metadata. The GUI only drives a `CaptureController`; the DanZero API only accepts explicit `GuanDanState` values and has no dependency on screenshot files or Qt.

**Tech Stack:** Python 3.12, PySide6, PySide6-Fluent-Widgets, OpenCV, MSS, pywin32, NumPy, PyTorch, pytest.

## Global Constraints

- Create the project only at `C:\project\python_project\daguandan_danzero_bridge`.
- Never modify, delete, or move files in `C:\project\python_project\poke_vision_toolkit`; copy from it.
- Ship the 93 existing files from `data/profiles/tencent_daguandan/templates` under the same relative path in the new project.
- Do not add image recognition, state inference, client automation, annotation UI, or automatic strategy requests.
- Support Python `>=3.12,<3.13` on Windows 11.
- Do not initialize Git or create commits unless the user later requests a repository.

---

## File Structure

```text
daguandan_danzero_bridge/
  pyproject.toml
  requirements.txt
  README.md
  run.py
  start_gui.bat
  data/profiles/tencent_daguandan/
    profile.json
    templates/                         # copied assets
  src/daguandan_bridge/
    __init__.py
    config.py
    dependencies.py
    dpi.py
    models.py
    storage.py
    image_io.py
    profiles.py
    window_capture.py
    capture_service.py
    danzero/
      __init__.py
      state.py
      rules.py
      advisor.py
      _vendor/guandan_rlcard/          # copied engine and q_network.ckpt
    gui/
      __init__.py
      app.py
      controller.py
      workers.py
      main_window.py
      capture_page.py
      widgets.py
  tests/
    conftest.py
    test_capture_service.py
    test_capture_page.py
    test_template_assets.py
    test_danzero_api.py
```

### Task 1: Create the isolated package and default profile

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `README.md`, `run.py`, `start_gui.bat`
- Create: `src/daguandan_bridge/__init__.py`, `config.py`, `dependencies.py`, `dpi.py`, `models.py`, `storage.py`, `image_io.py`, `profiles.py`, `window_capture.py`
- Create: `data/profiles/tencent_daguandan/profile.json`
- Test: `tests/conftest.py`, `tests/test_template_assets.py`

**Consumes:** Selected project name and source modules from `poke_vision_toolkit`.

**Produces:** An installable `daguandan_bridge` package with a fixed profile root and a valid Tencent DaGuandan profile.

- [ ] **Step 1: Write the profile-and-asset test**

```python
from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.profiles import load_profile_config, get_profile_paths

def test_tencent_daguandan_profile_and_templates_are_packaged():
    paths = get_profile_paths(PROFILES_ROOT, "tencent_daguandan")
    config = load_profile_config(paths)
    assert config.name == "tencent_daguandan"
    assert config.recording_interval_sec == 1.5
    assert len(list(paths.templates_dir.rglob("*"))) >= 93
```

- [ ] **Step 2: Run the test to verify it fails before setup**

Run: `set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_template_assets.py`

Expected: collection fails because `daguandan_bridge` does not yet exist.

- [ ] **Step 3: Copy and rename the capture foundations**

Copy these source files without functional changes, replacing only relative package references from `poke_vision` to `daguandan_bridge`:

```text
config.py, dependencies.py, dpi.py, models.py, storage.py,
image_io.py, profiles.py, window_capture.py
```

Set `PROJECT_ROOT = Path(__file__).resolve().parents[2]` and `PROFILES_ROOT = PROJECT_ROOT / "data" / "profiles"` in `config.py`. Create `profile.json` with the current `tencent_daguandan` capture settings and `recording_interval_sec: 1.5`.

- [ ] **Step 4: Add packaging and launch files**

Use this project metadata:

```toml
[project]
name = "daguandan-danzero-bridge"
version = "0.1.0"
description = "Tencent DaGuandan screenshot recorder and manual-state DanZero bridge."
requires-python = ">=3.12,<3.13"
```

Declare only `opencv-python`, `mss`, `pywin32`, `numpy`, `PySide6`, `PySide6-Fluent-Widgets`, and `torch`. Make `run.py` add `src` to `sys.path`, enable DPI awareness, and call `daguandan_bridge.gui.app.main()`.

- [ ] **Step 5: Copy template assets and validate packaging**

Copy every file from the original template directory into `data/profiles/tencent_daguandan/templates/`, preserving paths and bytes. Then run:

Run: `set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_template_assets.py`

Expected: PASS with the profile loaded and at least 93 packaged template files found.

### Task 2: Implement the recording service and capture session format

**Files:**
- Create: `src/daguandan_bridge/capture_service.py`
- Test: `tests/test_capture_service.py`

**Consumes:** `ProfileConfig`, `CapturedStandardizedFrame`, image I/O, storage, and profile paths from Task 1.

**Produces:** `CaptureService`, `FrameSnapshot`, and `ScreenshotSession`, independent of Qt.

- [ ] **Step 1: Write the session lifecycle test**

```python
def test_session_saves_sequential_frame_metadata_and_finish(tmp_path, frame_snapshot):
    service = CaptureService(tmp_path / "profiles")
    create_profile(service.profiles_root, ProfileConfig("test_game", "Test", ("Test",)))
    session = service.start_screenshot_session("test_game", 1500)
    saved = service.save_session_frame(session, frame_snapshot)
    completed = service.finish_screenshot_session(session)
    document = json.loads(completed.metadata_path.read_text(encoding="utf-8"))
    assert saved.name == "000001.png"
    assert document["capture_interval_ms"] == 1500
    assert document["frame_count"] == 1
    assert document["finished_at"] is not None
    assert document["first_capture"]["capture_backend"] == "test"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_capture_service.py::test_session_saves_sequential_frame_metadata_and_finish`

Expected: FAIL because `CaptureService` is not defined.

- [ ] **Step 3: Implement the service boundary**

Define these public interfaces:

```python
@dataclass(frozen=True)
class FrameSnapshot:
    frame: CapturedStandardizedFrame
    captured_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

@dataclass
class ScreenshotSession:
    profile_name: str
    directory: Path
    started_at: datetime
    interval_ms: int
    frame_count: int = 0
    finished_at: datetime | None = None
    first_capture: dict[str, Any] | None = None

class CaptureService:
    def capture_frame(self, profile_name: str) -> FrameSnapshot: ...
    def start_screenshot_session(self, profile_name: str, interval_ms: int) -> ScreenshotSession: ...
    def save_session_frame(self, session: ScreenshotSession, snapshot: FrameSnapshot) -> Path: ...
    def finish_screenshot_session(self, session: ScreenshotSession) -> ScreenshotSession: ...
    def screenshot_folder(self, profile_name: str) -> Path: ...
```

Reuse the existing frame standardization and metadata format. Reject non-positive intervals, paths outside the active profile's screenshot root, and writes after `finished_at` is set.

- [ ] **Step 4: Run all capture-service tests**

Run: `set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_capture_service.py`

Expected: PASS, including session creation, frame save, metadata, completion, and invalid-interval rejection.

### Task 3: Build the dedicated recording GUI

**Files:**
- Create: `src/daguandan_bridge/gui/__init__.py`, `app.py`, `workers.py`, `controller.py`, `widgets.py`, `capture_page.py`, `main_window.py`
- Test: `tests/test_capture_page.py`

**Consumes:** `CaptureService` from Task 2 and PySide6.

**Produces:** A one-page application containing only the screenshot recorder; it has no annotation page or recognition page.

- [ ] **Step 1: Write the offscreen GUI test**

```python
def test_capture_window_initializes_with_recording_controls(qapp):
    controller = CaptureController(CaptureService())
    window = DaguandanBridgeWindow(controller)
    assert window.capture_page.start_button.text() == "开始预览"
    assert window.capture_page.start_session_button.text() == "开始本局录制"
    assert window.capture_page.end_session_button.text() == "结束本局"
    controller.shutdown()
```

- [ ] **Step 2: Run the GUI test to verify it fails**

Run: `set QT_QPA_PLATFORM=offscreen && set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_capture_page.py`

Expected: FAIL because the GUI package is absent.

- [ ] **Step 3: Implement the controller and worker**

Copy the safe capture worker loop and adapt the controller to these operations only:

```python
class CaptureController(QObject):
    def start_capture(self) -> None: ...
    def stop_capture(self) -> None: ...
    def save_current_frame(self) -> Path | None: ...
    def start_screenshot_session(self, interval_ms: int) -> ScreenshotSession | None: ...
    def finish_screenshot_session(self) -> ScreenshotSession | None: ...
    def open_screenshot_folder(self) -> None: ...
    def shutdown(self) -> bool: ...
```

Emit `frame_ready`, `frame_cleared`, `state_changed`, `error`, `screenshot_session_updated`, and `screenshot_session_finished`. A capture error must clear the current frame and let the page stop its recording timer.

- [ ] **Step 4: Implement the one-page window**

Adapt the existing capture page controls and timer logic. `DaguandanBridgeWindow` must contain only `capture_page`; do not import annotation, template-management, recognition, auto-tracking, or game-advice GUI modules. Set the app title to `大掼蛋 DanZero 桥接器`.

- [ ] **Step 5: Run the GUI test and launch smoke test**

Run: `set QT_QPA_PLATFORM=offscreen && set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_capture_page.py`

Expected: PASS.

Run: `set QT_QPA_PLATFORM=offscreen && .venv\Scripts\python.exe run.py --help`

Expected: exit code 0 and standard argparse help output.

### Task 4: Package the manual-state DanZero API

**Files:**
- Create: `src/daguandan_bridge/danzero/__init__.py`, `state.py`, `rules.py`, `advisor.py`
- Create: `src/daguandan_bridge/danzero/_vendor/guandan_rlcard/**`
- Test: `tests/test_danzero_api.py`

**Consumes:** Existing `game_state.py`, `guandan_rules.py`, `local_strategy.py`, the vendored engine, and `q_network.ckpt`.

**Produces:** A documented API at `daguandan_bridge.danzero` with no Qt or screenshot dependency.

- [ ] **Step 1: Write API import and validation tests**

```python
from daguandan_bridge.danzero import DanzeroAdvisor, GuanDanState, GameStateError

def test_danzero_public_api_can_be_imported_without_qt():
    assert DanzeroAdvisor.__name__ == "DanzeroAdvisor"
    assert GuanDanState.__name__ == "GuanDanState"

def test_incomplete_manual_state_is_rejected_before_model_execution():
    state = GuanDanState(round_level="2", wild_rank="2")
    with pytest.raises(GameStateError):
        DanzeroAdvisor().recommend(state)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_danzero_api.py`

Expected: FAIL because `daguandan_bridge.danzero` is absent.

- [ ] **Step 3: Copy the engine and create the narrow wrapper**

Copy the vendored `poke_vision/_vendor/guandan_rlcard` directory including `baselines/danzero/q_network.ckpt`; replace imports starting with `poke_vision._vendor` with `daguandan_bridge.danzero._vendor`.

Copy and rename `GuanDanState`, `GameStateError`, `PlayEvent`, and `LocalStrategySnapshot` into `state.py`; copy card conversion helpers into `rules.py`. Adapt `LocalGuandanAdvisor` into this wrapper:

```python
class DanzeroAdvisor:
    def __init__(self) -> None:
        self._advisor = LocalGuandanAdvisor("danzero")

    def recommend(self, state: GuanDanState, *, request_id: str = "") -> LocalAdvice:
        return self._advisor.recommend(state.local_snapshot(), request_id=request_id)
```

Re-export `DanzeroAdvisor`, `GuanDanState`, `GameStateError`, and `LocalAdvice` from `danzero/__init__.py`. Keep the model SHA-256 validation before inference.

- [ ] **Step 4: Run DanZero API tests**

Run: `set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_danzero_api.py`

Expected: PASS without starting Qt or capturing a window.

### Task 5: Document, install, and verify the complete project

**Files:**
- Modify: `README.md`, `requirements.txt`, `pyproject.toml`, `start_gui.bat`
- Test: all tests under `tests/`

**Consumes:** Tasks 1 through 4.

**Produces:** A self-contained project a user can open in PyCharm, install, launch, and call from Python.

- [ ] **Step 1: Add exact README usage examples**

Include these sections: environment setup, `python run.py` launch, recording session output layout, template asset location, and the following manual API example:

```python
from daguandan_bridge.danzero import DanzeroAdvisor, GuanDanState

state = GuanDanState(round_level="2", wild_rank="2", current_player="self", lead_player="self")
state.set_hand(("AS", "KH", "3D"))
advice = DanzeroAdvisor().recommend(state)
print(advice.cards)
```

State explicitly that screenshots are never automatically interpreted or sent to DanZero.

- [ ] **Step 2: Install the project into its own virtual environment**

Run: `py -3.12 -m venv .venv`

Run: `.venv\Scripts\python.exe -m pip install -r requirements.txt`

Expected: all declared runtime dependencies install successfully.

- [ ] **Step 3: Run the complete test suite**

Run: `set QT_QPA_PLATFORM=offscreen && set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q`

Expected: zero failures across capture, GUI, template asset, and DanZero API tests.

- [ ] **Step 4: Perform final evidence checks**

Run: `.venv\Scripts\python.exe -c "from daguandan_bridge.danzero import DanzeroAdvisor; print(DanzeroAdvisor.__name__)"`

Expected: `DanzeroAdvisor`.

Run: `Get-ChildItem data\profiles\tencent_daguandan\templates -Recurse -File | Measure-Object`

Expected: `Count` is at least 93.

Run: `set QT_QPA_PLATFORM=offscreen && .venv\Scripts\python.exe run.py --help`

Expected: exit code 0.
