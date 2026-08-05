# 区域标注界面优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让区域标注页面在选择图片后立即预览，并用中文下拉框选择区域名称和角色，同时彻底移除区域标注中的来源图片字段而不改变内部英文标识。

**Architecture:** 保持 `AnnotationService` 为不依赖 Qt 的配置与图像服务，集中维护内部 key 到中文 label 的映射、旧配置兼容和坐标持久化。 `AnnotationPage` 只负责下拉框、表格、图片预览和状态反馈；图片切换、表格选择和手动标注按钮统一调用同一个预览渲染入口。

**Tech Stack:** Python 3.12、PySide6、OpenCV、NumPy、pytest、现有 `AnnotationService` 与 `AnnotationPage`。

## Global Constraints

- 内部区域名称继续使用现有 20 个英文 key，内部角色继续使用 `hand`、`play`、`anchor`、`generic`。
- 用户可见的区域名称和角色必须使用中文；覆盖框标签也使用中文。
- `regions_config.json` 保持 schema version 2，保存时保留 `abs_box` 与重新计算后的 `ratio_box`。
- 服务层读取旧 `source_image` 键时必须兼容，但保存后的区域配置不再写入该键。
- 区域图片只能来自 `data/profiles/tencent_daguandan/screenshots` 及其子目录。
- 不添加自动识图、自动追踪、拖拽绘制或 DanZero 状态推断。
- 保留现有多选叠加和单选编辑规则；编辑保存必须只选中一个区域。
- 只修改当前项目；当前工作区已有的未提交区域标注基础代码属于本次功能上下文，不得丢弃或重置。

---

### Task 1: 收敛区域元数据和服务持久化契约

**Files:**
- Modify: `src/daguandan_bridge/annotation_service.py`
- Modify: `tests/test_annotation_service.py`

**Interfaces:**
- `REGION_DISPLAY_NAMES: dict[str, str]`：20 个内部名称到中文名称的固定映射。
- `ROLE_DISPLAY_NAMES: dict[str, str]`：4 个内部角色到中文名称的固定映射。
- `display_region_name(name: str) -> str`：把内部名称转换为用户可见名称。
- `display_role(role: str) -> str`：把内部角色转换为用户可见名称。
- `RegionRecord` 字段为 `name`、`role`、`abs_box`、`ratio_box`，不再暴露 `source_image`。
- `AnnotationService.update_region(old_name, *, name, role, box) -> RegionRecord`。

- [ ] **Step 1: 写中文映射、旧字段兼容和更新契约的失败测试**

在 `tests/test_annotation_service.py` 增加以下行为测试：

~~~python
def test_region_and_role_labels_are_chinese_but_keep_internal_keys():
    from daguandan_bridge.annotation_service import (
        REGION_DISPLAY_NAMES,
        ROLE_DISPLAY_NAMES,
    )

    assert REGION_DISPLAY_NAMES["my_hand"] == "我的手牌"
    assert REGION_DISPLAY_NAMES["table_anchor_1"] == "牌桌锚点一"
    assert ROLE_DISPLAY_NAMES == {
        "hand": "手牌",
        "play": "出牌",
        "anchor": "锚点",
        "generic": "通用区域",
    }


def test_legacy_source_image_is_ignored_and_not_written(tmp_path):
    service = _temp_service(tmp_path)
    original = service.list_regions()[0]

    updated = service.update_region(
        original.name,
        name=original.name,
        role="play",
        box=Box(10, 20, 300, 120),
    )
    payload = json.loads(service.regions_path.read_text(encoding="utf-8"))

    assert not hasattr(updated, "source_image")
    assert all("source_image" not in item for item in payload["regions"])


def test_update_region_can_rename_and_reject_duplicate_names(tmp_path):
    service = _temp_service(tmp_path)
    original = service.list_regions()[0]
    target = service.list_regions()[1]

    renamed = service.update_region(
        original.name,
        name="my_hand",
        role="play",
        box=original.abs_box,
    )

    assert renamed.name == "my_hand"
    with pytest.raises(ValueError, match="重复"):
        service.update_region(
            target.name,
            name="my_hand",
            role=target.role,
            box=target.abs_box,
        )
~~~

保留已有坐标、图片递归发现和多色覆盖测试；把原来传入 `source_image` 的更新测试改成新接口，并增加 `import pytest`。

- [ ] **Step 2: 运行服务测试，确认按预期失败**

运行：

~~~powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q tests/test_annotation_service.py
~~~

预期：失败原因是 `RegionRecord` 仍有 `source_image`、更新接口仍要求 `source_image`，以及中文映射和重复名称校验尚未实现；不能接受导入错误或测试代码错误。

- [ ] **Step 3: 实现最小服务层改动**

在 `annotation_service.py` 中：

1. 添加完整的 `REGION_DISPLAY_NAMES`、`ROLE_DISPLAY_NAMES`，并让缺失的非法内部名称继续触发校验错误。
2. 从 `RegionRecord` 移除 `source_image`；`_record_from_json` 不读取该字段，因而可以兼容旧 JSON。
3. 将 `update_region` 改为接收 `name`，校验新名称在固定映射中且不存在于其他区域；保留旧名称用于定位原记录。
4. 写 JSON 时只写 `name`、`role`、`abs_box`、`ratio_box`。
5. `overlay_regions` 使用中文显示名绘制框上标签，内部 `region.name` 仍保持英文 key。

- [ ] **Step 4: 运行服务测试，确认通过**

运行同一条服务测试命令，预期所有服务测试通过，且旧配置中的 `source_image` 不会阻止读取。

- [ ] **Step 5: 提交服务层任务**

~~~powershell
git add src/daguandan_bridge/annotation_service.py tests/test_annotation_service.py
git commit -m "feat: localize region metadata and remove source image persistence"
~~~

---

### Task 2: 用中文下拉框替换编辑字段

**Files:**
- Modify: `src/daguandan_bridge/gui/annotation_page.py`
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- 页面暴露 `region_table`、`image_combo`、`name_combo`、`role_combo`、`show_selected_button`、`save_button`、`canvas`。
- 页面不再创建 `source_edit`。
- 表格列为“名称、角色、绝对坐标、比例坐标”。

- [ ] **Step 1: 写界面行为失败测试**

在 `tests/test_annotation_page.py` 增加或调整测试：

~~~python
def test_annotation_page_uses_chinese_dropdowns_and_removes_source_field():
    app = QApplication.instance() or QApplication([])
    page = AnnotationPage(AnnotationService())

    assert not hasattr(page, "source_edit")
    assert [page.region_table.horizontalHeaderItem(i).text() for i in range(4)] == [
        "名称", "角色", "绝对坐标", "比例坐标"
    ]
    assert page.name_combo.itemText(page.name_combo.findData("my_hand")) == "我的手牌"
    assert page.role_combo.itemText(page.role_combo.findData("hand")) == "手牌"
    assert page.region_table.item(0, 0).text() == "左侧首出牌提示"
    assert page.region_table.item(0, 1).text() == "通用区域"

    page.close()
    app.processEvents()
~~~

把图片测试扩展为：创建一个 1280×720 黑色截图，实例化页面后检查图片下拉框有一项、`current_image` 非空且 `canvas.pixmap()` 非空；此时不选择任何区域，证明图片切换会显示原图。

- [ ] **Step 2: 运行页面测试，确认按预期失败**

运行：

~~~powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q tests/test_annotation_page.py
~~~

预期：失败于 `name_combo` 不存在、来源图片列仍存在或图片无选区时没有 pixmap；若出现 Qt 初始化错误，先修正测试环境而不是改生产代码。

- [ ] **Step 3: 实现下拉框和表格中文展示**

在 `annotation_page.py` 中：

1. 从服务层导入映射和显示函数。
2. 用 `name_combo = QComboBox()` 添加固定名称列表，显示中文、`itemData` 保存英文 key；用同样方式初始化角色下拉框。
3. 删除 `QLineEdit` 导入、`name_edit`、`source_edit` 及其表单行。
4. 移除“来源图片”表格列，表格名称和角色单元格使用中文显示函数。
5. `_load_selected_region` 用 `findData` 设置两个下拉框，并把状态栏和保存成功提示改为中文显示名。
6. 保存时将 `name_combo.currentData()`、`role_combo.currentData()` 和坐标传给服务层，并在名称改变后仍能定位刷新后的表格行。

- [ ] **Step 4: 运行页面测试，确认下拉框行为通过**

运行同一条页面测试命令，预期中文下拉框、表格列和单选编辑测试通过。

- [ ] **Step 5: 提交界面任务**

~~~powershell
git add src/daguandan_bridge/gui/annotation_page.py tests/test_annotation_page.py
git commit -m "feat: use Chinese region dropdowns"
~~~

---

### Task 3: 统一图片预览刷新路径

**Files:**
- Modify: `src/daguandan_bridge/gui/annotation_page.py`
- Modify: `tests/test_annotation_page.py`

**Interfaces:**
- `_refresh_preview()`：根据当前图片和当前选区显示原图或带框图片。
- `_image_changed(index)`：只负责加载图片并调用 `_refresh_preview()`。
- `_load_selected_region()`：加载编辑值后调用 `_refresh_preview()`。
- `_show_selected_regions()`：保留手动操作入口，但复用 `_refresh_preview()`。

- [ ] **Step 1: 写预览刷新失败测试**

在页面测试中加入：

~~~python
def test_selecting_image_shows_preview_without_selected_regions(tmp_path):
    page = _page_with_one_image(tmp_path)

    assert page.region_table.selectedItems() == []
    assert page.current_image is not None
    assert page.canvas.pixmap() is not None
    assert not page.canvas.pixmap().isNull()

    page.close()
~~~

再验证选中第一行后，预览仍有 pixmap 且状态栏包含中文区域名。

- [ ] **Step 2: 运行测试确认失败**

运行页面测试，预期当前 `_show_selected_regions()` 因没有选区直接返回，导致 `canvas.pixmap()` 为空。

- [ ] **Step 3: 实现统一预览渲染**

将当前图片的 QImage/QPixmap 转换提取到一个小方法：

1. `current_image is None` 时清空 pixmap 并显示“暂无可预览的录制图片”。
2. 没有选区时直接使用当前原图。
3. 有选区时调用 `AnnotationService.overlay_regions` 后再转换为 QPixmap。
4. 所有路径都用 `KeepAspectRatio` 和 `SmoothTransformation` 缩放到 canvas。
5. 图片切换、表格选择和手动标注按钮全部调用同一方法，避免逻辑分叉。

- [ ] **Step 4: 运行页面和全量测试**

~~~powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q tests/test_annotation_page.py tests/test_annotation_service.py
~~~

预期所有新增预览测试和原有多选叠加测试通过。

- [ ] **Step 5: 提交预览任务**

~~~powershell
git add src/daguandan_bridge/gui/annotation_page.py tests/test_annotation_page.py
git commit -m "feat: preview selected annotation images immediately"
~~~

---

### Task 4: 清理配置和用户文档

**Files:**
- Modify: `data/profiles/tencent_daguandan/regions_config.json`
- Modify: `README.md`
- Test: `tests/test_annotation_service.py`

**Interfaces:**
- 现有区域配置保留 20 条记录、schema version 2、坐标和比例坐标；每条区域记录不包含 `source_image`。
- `templates_config.json` 不属于本次区域编辑字段，不修改其模板来源信息。

- [ ] **Step 1: 写配置迁移失败测试**

在服务测试中增加：

~~~python
def test_profile_region_config_has_no_source_image_field():
    service = AnnotationService()
    payload = json.loads(service.regions_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert len(payload["regions"]) == 20
    assert all("source_image" not in item for item in payload["regions"])
~~~

- [ ] **Step 2: 运行测试确认失败**

运行服务测试，预期只因当前迁移配置仍含 `source_image` 而失败。

- [ ] **Step 3: 移除区域配置中的旧字段并更新 README**

从 `regions_config.json` 的 20 条区域记录移除 `source_image`，不改变其他字段值。更新 README 的区域标注说明：图片选择后会直接预览，名称和角色使用中文下拉框，保存时只修改名称、角色和坐标；删除“来源图片会写回配置”的描述。

- [ ] **Step 4: 运行配置测试确认通过**

运行服务测试，确认配置迁移和兼容读取测试通过。

- [ ] **Step 5: 提交配置和文档任务**

~~~powershell
git add data/profiles/tencent_daguandan/regions_config.json README.md tests/test_annotation_service.py
git commit -m "docs: update region annotation workflow"
~~~

---

### Task 5: 全量验证和交付检查

**Files:**
- Verify: all modified files from Tasks 1–4

- [ ] **Step 1: 运行完整测试**

~~~powershell
$env:QT_QPA_PLATFORM = "offscreen"
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q
~~~

预期：全部测试通过，包含原有录制、模板、DanZero API 和新增区域标注测试。

- [ ] **Step 2: 运行编译和 Qt 栈检查**

~~~powershell
.venv\Scripts\python.exe -m compileall -q src tests
python C:\project\python_project\pyside6-qfluentwidgets-ui\scripts\audit_qt_stack.py .
~~~

预期：Python 源码编译成功，项目仍只使用 PySide6，不出现 PyQt 混用；不在本任务中强制迁移到 QFluentWidgets。

- [ ] **Step 3: 核验工作区内容**

~~~powershell
git diff --check
git status --short --branch
git log --oneline -6
~~~

确认没有未预期的生成文件、来源图片编辑控件残留、英文用户可见 label 或未提交的必要代码。

- [ ] **Step 4: 完成最终提交**

~~~powershell
git status --short
git diff --stat HEAD~4..HEAD
~~~

若所有验证通过，向用户报告修改文件、测试结果和当前 Git 状态，不宣称完成任何未实际验证的 Windows DPI 或打包质量。

