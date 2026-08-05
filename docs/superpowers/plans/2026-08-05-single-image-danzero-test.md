# 单图状态标注与 DanZero 测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复模板标注反馈和标签选择，并新增可人工确认单张截图状态、构建 DanZero 参数和异步测试返回值的窗口。

**Architecture:** `SingleImageDanzeroPage` 只负责单图表单和结果展示，通过 `build_state()` 生成 UI 无关的 `GuanDanState`；`AnnotationPage` 负责传入当前图片并持有工作线程。DanZero 仍走现有 `DanzeroAdvisor`，调用通过 `OneShotWorker` 脱离 GUI 线程。

**Tech Stack:** Python 3、PySide6、OpenCV、现有 `GuanDanState`/`DanzeroAdvisor`、pytest、QTest。

## Global Constraints

- 当前首版是人工确认标注，不声称自动 OCR/牌面识别；仓库没有识别模型依赖。
- DanZero 输入必须通过 `GuanDanState` 的现有校验，不能绕过 `readiness_errors`。
- 所有耗时策略调用必须运行在非 GUI 线程，结果和错误通过 Qt 信号回主线程。
- UI 可见牌局角色使用中文，传给状态模型的座位保持 `self/left/opposite/right`。
- 保留当前工作区已有模板裁剪、区域配置和相关未提交文件，不使用破坏性 Git 操作。

---

### Task 1: 修复预览页重复信息、模板标签和 ROI 反馈

**Files:**
- Modify: `src/daguandan_bridge/gui/annotation_page.py`
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- Consumes: existing `RoiCanvas.roi_changed`, `TemplateService.list_templates()`。
- Produces: editable `template_label_edit` combo, `_refresh_template_labels()`, persistent template ROI status。

- [ ] **Step 1: Write failing tests**

测试断言普通预览状态不再写入“已预览：…”，模板标签控件是可编辑下拉框且包含历史标签，拖框后 `x/y/w/h` 和状态文本同步更新。

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "template_label or preview_status or template_roi"`

Expected: FAIL because当前状态仍写入“已预览”，模板标签是 `QLineEdit`，且 ROI 状态会被刷新覆盖。

- [ ] **Step 3: Implement minimal fixes**

使用可编辑 `QComboBox` 保存历史标签和新输入；读取标签使用 `currentText()`；刷新模板列表后同步历史标签。`_refresh_preview()` 在存在 `current_roi` 时保留“已选模板框”状态，否则清空重复的预览路径状态。

- [ ] **Step 4: Run targeted tests**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "template_label or preview_status or template_roi"`

Expected: PASS。

- [ ] **Step 5: Commit**

```powershell
git add src/daguandan_bridge/gui/annotation_page.py tests/test_annotation_page.py
git commit -m "fix: improve template annotation feedback"
```

### Task 2: 添加单图状态标注模型与表单

**Files:**
- Create: `src/daguandan_bridge/gui/single_image_danzero_page.py`
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- Consumes: `GuanDanState`, `GameStateError`。
- Produces: `SingleImageDanzeroPage(QDialog)`, `build_state() -> GuanDanState`, `play_event_table`, `build_button`, `test_button`, `state_built` and `test_requested` signals。

- [ ] **Step 1: Write failing tests**

新增测试创建窗口，设置级牌/玩家/手牌，添加一条“对家出牌”事件和一条“右家不出”事件，断言 `build_state()` 的 `GuanDanState` 字段和事件顺序；无效牌码必须在状态区显示错误。

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "single_image_state"`

Expected: FAIL because `single_image_danzero_page.py` 不存在。

- [ ] **Step 3: Implement the minimal form**

创建中文座位/动作映射、上下文下拉框、手牌输入、事件表格和添加/删除按钮。`build_state()` 调用 `set_context()`、`confirm_hand()`、`record_play()`/`record_pass()`，把 `GameStateError` 转为状态标签并重新抛出供按钮处理。构建参数区显示 `state.summary()`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "single_image_state"`

Expected: PASS。

- [ ] **Step 5: Commit**

```powershell
git add src/daguandan_bridge/gui/single_image_danzero_page.py tests/test_annotation_page.py
git commit -m "feat: add single image game state annotation form"
```

### Task 3: 接入异步 DanZero 测试和结果展示

**Files:**
- Modify: `src/daguandan_bridge/gui/single_image_danzero_page.py`
- Modify: `src/daguandan_bridge/gui/annotation_page.py`
- Modify: `src/daguandan_bridge/gui/workers.py` only if the existing worker interface needs a typed result path
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- Consumes: `SingleImageDanzeroPage.test_requested`, `DanzeroAdvisor.recommend`, existing `OneShotWorker`。
- Produces: `AnnotationPage.single_image_test_button`, `_open_single_image_danzero()`, `_run_danzero_test()` and success/error result rendering。

- [ ] **Step 1: Write failing tests**

测试按钮能打开单图窗口；注入一个假的 advisor 后点击“测试 DanZero”，断言收到 `GuanDanState`、结果显示推荐牌和 engine input，策略异常显示错误且按钮恢复。

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "danzero_test"`

Expected: FAIL because主页面没有单图测试按钮和异步结果链。

- [ ] **Step 3: Implement the worker-backed integration**

在 `AnnotationPage` 增加按钮并传入当前图片路径；页面构建状态后禁用测试按钮，创建 `OneShotWorker(lambda: advisor.recommend(state, request_id=...))`，在主线程显示 `LocalAdvice` 的 `cards/play_type/is_pass/elapsed_ms`、`format_engine_input_summary(advice.engine_input)` 和 JSON。任何异常写入错误框并在 finished 中恢复按钮。

- [ ] **Step 4: Run targeted and full tests**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "danzero_test"`，随后 `.venv\\Scripts\\python.exe -m pytest -q`。

Expected: targeted tests and full suite PASS。

- [ ] **Step 5: Commit**

```powershell
git add src/daguandan_bridge/gui/annotation_page.py src/daguandan_bridge/gui/single_image_danzero_page.py tests/test_annotation_page.py
git commit -m "feat: test DanZero from a single annotated image"
```

### Task 4: 文档与最终验证

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update workflow documentation**

说明模板标签是可编辑下拉框，单图按钮采用人工确认状态，不是自动 OCR；列出测试 DanZero 的输入字段和结果内容。

- [ ] **Step 2: Run compile, tests and Qt audit**

Run: `.venv\\Scripts\\python.exe -m compileall -q src tests`、`.venv\\Scripts\\python.exe -m pytest -q`、`git diff --check` 和 `python C:\\project\\python_project\\pyside6-qfluentwidgets-ui\\scripts\\audit_qt_stack.py .`。

- [ ] **Step 3: Run GUI smoke test**

用 `QT_QPA_PLATFORM=offscreen` 构造主窗口，确认单图按钮能打开窗口、配置状态可构建、测试按钮完成后结果可见且主窗口仍可关闭。
