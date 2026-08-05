# 区域配置与预览交互优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将区域配置拆为独立窗口，改善图片导航和区域选择/拖框的即时视觉反馈。

**Architecture:** 新建 `RegionConfigPage(QDialog)` 管理区域表格、中文下拉框、坐标编辑和保存；`AnnotationPage` 只管理图片预览、模板裁剪和配置窗口生命周期。两者通过 `region_selected`、`preview_requested`、`region_updated` 信号传递记录，避免重复持久化逻辑。

**Tech Stack:** Python 3、PySide6、OpenCV、pytest、pytest-qt 风格的 QApplication/QTest 离屏测试。

## Global Constraints

- 区域内部名称和角色保持英文稳定键，所有用户可见区域名称、角色和状态文本使用中文。
- 所有坐标仍以 1280×720 标准化画面为基准，预览缩放不能改变 ROI 映射。
- 不混用 PyQt 和 PySide；耗时模板保存继续在 `QThread` 中执行，GUI 控件只由主线程更新。
- 保留当前工作区中已有的模板裁剪、模板配置和文档改动，不使用破坏性 Git 操作。

---

### Task 1: 新增独立区域配置窗口

**Files:**
- Create: `src/daguandan_bridge/gui/region_config_page.py`
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- Consumes: `AnnotationService`, `RegionRecord`, `REGION_DISPLAY_NAMES`, `ROLE_DISPLAY_NAMES`, `Box`。
- Produces: `RegionConfigPage(QDialog)`，信号 `region_selected(object)`、`preview_requested(object)`、`region_updated(object, object)`。

- [ ] **Step 1: Write the failing tests**

新增测试验证 `RegionConfigPage` 的四列中文表格、名称/角色下拉框、选中行发出区域记录、点击“标注选中区域”发出当前区域集合、保存后发出旧名称和新记录。

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "region_config_page"`

Expected: FAIL because `region_config_page.py` and `RegionConfigPage` do not exist。

- [ ] **Step 3: Implement the minimal dialog**

创建 `RegionConfigPage`，构建四列 `QTableWidget`、中文 `QComboBox`、四个 `QSpinBox` 和“标注选中区域/保存修改”按钮。表格选择时加载第一行并发送 `region_selected`；预览按钮发送当前选中记录；保存调用 `AnnotationService.update_region`，刷新表格并发送 `region_updated(old_name, updated)`，异常写入状态标签。

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "region_config_page"`

Expected: PASS。

- [ ] **Step 5: Commit the isolated dialog**

```powershell
git add src/daguandan_bridge/gui/region_config_page.py tests/test_annotation_page.py
git commit -m "feat: add standalone region configuration page"
```

### Task 2: 重构预览页布局并接入配置窗口

**Files:**
- Modify: `src/daguandan_bridge/gui/annotation_page.py`
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- Consumes: `RegionConfigPage` signals and the existing `RoiCanvas`/`OneShotWorker`。
- Produces: `region_config_button`、左右两侧图片导航、`_open_region_config`、`_region_selected`、`_region_updated`。

- [ ] **Step 1: Write the failing tests**

新增测试断言区域表格不再位于预览页、`region_config_button` 可以打开独立窗口、箭头是画布同一横向布局的左右兄弟控件、配置窗口选中区域后预览像素发生矩形叠加。

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "standalone or side_navigation or selected_region_preview"`

Expected: FAIL because the current preview page still直接包含区域表格，箭头在图片选择行上方，且没有配置窗口信号接入。

- [ ] **Step 3: Implement the layout and signal bridge**

从预览页移除区域表格和区域编辑表单，增加“查看区域配置”按钮；用 `QHBoxLayout` 放置左箭头、`RoiCanvas`、右箭头，图片下拉框移动到画布下方。创建/复用 `RegionConfigPage`，连接选中、预览和保存信号，维护 `preview_regions` 并刷新预览。保留模板模式的模板类型、标签、来源角色和模板 ROI 坐标编辑器。

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "standalone or side_navigation or selected_region_preview"`

Expected: PASS。

- [ ] **Step 5: Commit the page integration**

```powershell
git add src/daguandan_bridge/gui/annotation_page.py tests/test_annotation_page.py
git commit -m "feat: integrate region configuration with preview page"
```

### Task 3: 修复区域模式拖框视觉反馈

**Files:**
- Modify: `src/daguandan_bridge/gui/annotation_page.py`
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- Consumes: existing `RoiCanvas.roi_changed` and `RegionConfigPage.set_box`。
- Produces: region/template modes both render `current_roi` after release, while keeping the existing light `QRubberBand` drag path。

- [ ] **Step 1: Write the failing regression test**

新增测试在“区域配置”模式调用 ROI 更新后，断言画布上 ROI 边框对应像素为橙色；并验证拖动后坐标仍是原图坐标。

- [ ] **Step 2: Run the regression test to verify it fails**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "region_mode_roi"`

Expected: FAIL because `_refresh_preview` 当前只在模板模式绘制 `current_roi`。

- [ ] **Step 3: Implement the minimal rendering fix**

让 `_refresh_preview` 在两种模式都先复制/叠加基础图片，再对有效 `current_roi` 画橙色边框。区域模式把拖框坐标同步给配置窗口，模板模式继续同步模板坐标控件；切换图片时清除旧 ROI。

- [ ] **Step 4: Run targeted and full tests**

Run: `.venv\\Scripts\\python.exe -m pytest -q tests/test_annotation_page.py -k "region_mode_roi"`，随后运行 `.venv\\Scripts\\python.exe -m pytest -q`。

Expected: targeted test and complete suite all PASS。

- [ ] **Step 5: Commit the ROI fix**

```powershell
git add src/daguandan_bridge/gui/annotation_page.py tests/test_annotation_page.py
git commit -m "fix: show region ROI feedback in preview"
```

### Task 4: 完成桌面 UI 验证和交付检查

**Files:**
- Modify: `README.md` only if the button/window workflow needs updated documentation。

- [ ] **Step 1: Run compile and whitespace checks**

Run: `.venv\\Scripts\\python.exe -m compileall -q src tests` and `git diff --check`。

- [ ] **Step 2: Run Qt stack audit**

Run: `python C:\\project\\python_project\\pyside6-qfluentwidgets-ui\\scripts\\audit_qt_stack.py .`

Expected: PySide6 only, no PyQt imports。

- [ ] **Step 3: Run a fresh application smoke test**

Construct `DaguandanBridgeWindow` with `QT_QPA_PLATFORM=offscreen`，打开区域配置按钮，验证窗口可显示、箭头边界状态正确、关闭主窗口和配置窗口不阻塞。

- [ ] **Step 4: Review final diff and status**

Run: `git status --short --branch` and `git diff --stat`，确认只包含本次 UI 变更以及工作区已有的模板功能文件。
