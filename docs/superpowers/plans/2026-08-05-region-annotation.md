# 区域配置迁移与标注模块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Tencent DaGuandan region/template metadata and add an in-app annotation page that edits regions and draws selected colored boxes only on recorded screenshots.

**Architecture:** Keep region persistence and image overlay logic in a Qt-free `annotation_service.py`. Add an `AnnotationPage` beside the existing recorder page; it recursively lists images under the profile screenshots root, shows all region metadata in a multi-select table, and edits one selected region through a form. Overlay coordinates use each region's normalized ratio box so standard 1280×720 coordinates render correctly on recorded images.

**Tech Stack:** Python 3.12, PySide6, OpenCV/NumPy, existing `CaptureService`, pytest.

## Global Constraints

- Only modify `C:\project\python_project\daguandan_danzero_bridge`.
- Copy `regions_config.json` and `templates_config.json` from the existing Tencent DaGuandan profile without changing their records during migration.
- Annotation image choices are restricted to files recursively below `data/profiles/tencent_daguandan/screenshots`.
- Multi-selection is for overlay; editing fields applies to one selected region at a time.
- Persist all edits in schema-version-2 `regions_config.json`, recalculating `ratio_box` from the configured 1280×720 base size.
- Do not add automatic image recognition or DanZero state inference.

---

### Task 1: Migrate region/template metadata and add persistence service

**Files:**
- Create: `data/profiles/tencent_daguandan/regions_config.json`, `templates_config.json`
- Create: `src/daguandan_bridge/annotation_service.py`
- Test: `tests/test_annotation_service.py`

**Interfaces:**
- `AnnotationService.list_regions() -> tuple[RegionRecord, ...]`
- `AnnotationService.list_recorded_images() -> tuple[Path, ...]`
- `AnnotationService.update_region(name, *, role, source_image, box) -> RegionRecord`
- `AnnotationService.overlay_regions(image, regions) -> np.ndarray`

- [ ] Write tests for 20 migrated regions, recursive screenshot filtering, edit persistence, and distinct-color overlays.
- [ ] Run `set PYTHONPATH=src && .venv\Scripts\python.exe -m pytest -q tests\test_annotation_service.py` and confirm it fails because the service/config files are absent.
- [ ] Copy both JSON files from the original profile and implement `RegionRecord`, JSON validation, coordinate normalization, screenshot discovery, and deterministic color assignment.
- [ ] Run the service tests and confirm all pass.

### Task 2: Add screenshot-backed annotation page

**Files:**
- Create: `src/daguandan_bridge/gui/annotation_page.py`
- Modify: `src/daguandan_bridge/gui/main_window.py`
- Test: `tests/test_annotation_page.py`

**Interfaces:**
- The page exposes `region_table`, `image_combo`, `show_selected_button`, `save_button`, and `canvas` for UI tests.
- `AnnotationPage` consumes `AnnotationService` and never opens a path outside the screenshot root.

- [ ] Write an offscreen test that finds all 20 rows, accepts extended selection, and has the overlay/save controls.
- [ ] Run the page test and confirm it fails before the page exists.
- [ ] Implement a table with columns name, role, source image, absolute box, ratio box; add a recursive screenshot combo; add editable name/role/source/x/y/w/h fields; add “标注选中区域” and “保存修改”.
- [ ] Draw selected regions in different colors on the selected recorded image and show labels; enable multi-select only for overlay.
- [ ] Add the page as a second tab beside the recorder page without importing the old annotation page or recognition UI.
- [ ] Run the page test and confirm it passes.

### Task 3: Integrate editing and verify recorder regression

**Files:**
- Modify: `src/daguandan_bridge/gui/main_window.py`, `README.md`
- Test: `tests/test_capture_page.py`, `tests/test_template_assets.py`, all annotation tests

- [ ] Verify editing one region changes both `abs_box` and `ratio_box` on disk and reloads correctly.
- [ ] Verify selecting several rows and pressing overlay produces separate non-background color pixels for each box.
- [ ] Run `set QT_QPA_PLATFORM=offscreen; $env:PYTHONPATH='src'; .venv\Scripts\python.exe -m pytest -q` and require zero failures.
- [ ] Update README with the “截图录制 / 区域标注” workflow and the screenshot-root restriction.
- [ ] Run `run.py --help` and verify the application package still imports without legacy `poke_vision` references.
