# 当前行动玩家动作窗口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除不必要的 `WAIT_CLEAR` 和人工确认阻断，让实时与复测只围绕当前行动玩家执行同一套动作识别与自动重试流程。

**Architecture:** 将 `ZoneLifecycle` 简化为当前区域的等待、稳定、采样窗口；`LiveOrchestrator` 在超时或共识失败时记录重试并保持运行；实时与状态机复测共享相同的非逐帧绕过配置。事后纠错 API 不删除，但不会参与自动流程。

**Tech Stack:** Python 3.11+, PySide6, OpenCV, pytest, 现有 `LiveOrchestrator` / `ZoneLifecycle` / `BurstConsensus` / `LiveReducer`。

## Global Constraints

- 当前回合只识别 `snapshot.current_player` 的出牌区域。
- 不提交无法通过共识和规则校验的猜测动作。
- 识别超时不进入人工确认，不暂停状态机；仅记录重试并继续等待。
- 复测状态机管线不能用 `sample_every_frame` 绕过动作窗口。
- 保留事后纠错接口，不新增按钮。
- 不覆盖工作区中用户已有的其他未提交修改。

---

### Task 1: 简化区域生命周期

**Files:**
- Modify: `src/daguandan_bridge/live/zone_lifecycle.py`
- Test: `tests/test_zone_lifecycle.py`

**Interfaces:**
- Consumes: `ZoneFrameMetrics`。
- Produces: `ZoneDecision`，仍提供 `WAIT_ACTION`、`SETTLING`、`BURST_READ` 阶段供编排器使用；不再产生 `WAIT_CLEAR`。

- [ ] **Step 1: 写失败测试**

增加测试，验证 `started_with_clear_zone=False` 激活后第一次静态旧牌不会产生采样，只有内容变化后稳定才进入 `BURST_READ`；增加测试验证 `pass_visible=True` 不需要清空旧牌即可开始稳定采样。

- [ ] **Step 2: 运行测试确认失败**

运行：

```powershell
& '.venv\\Scripts\\python.exe' -m pytest tests/test_zone_lifecycle.py -q
```

预期：新增测试因当前仍从 `WAIT_CLEAR` 开始而失败。

- [ ] **Step 3: 最小实现**

删除 `WAIT_CLEAR` 分支及其构造入口，初始化始终进入 `WAIT_ACTION`；`WAIT_ACTION` 只依据 `content_changed`、动作显著变化或稳定的 `pass_visible` 进入动作窗口。保留动画稳定与 burst 失效逻辑，删除不再需要的 `started_with_clear_zone` 构造参数和 `VALIDATE` 依赖。

- [ ] **Step 4: 运行测试确认通过**

运行同一 pytest 命令，确认生命周期测试全部通过，并确认原有动画稳定、burst 失效、超时测试按新的非阻断契约更新。

- [ ] **Step 5: 提交**

```powershell
git add tests/test_zone_lifecycle.py src/daguandan_bridge/live/zone_lifecycle.py
git commit -m "refactor: remove wait-clear action gate"
```

### Task 2: 让实时编排器自动重试而不人工确认阻断

**Files:**
- Modify: `src/daguandan_bridge/live/orchestrator.py`
- Modify: `src/daguandan_bridge/live/display_text.py`
- Test: `tests/test_live_orchestrator.py`

**Interfaces:**
- Consumes: 简化后的 `ZoneLifecycle`。
- Produces: 超时/共识不确定时仍为 `running` 的 `LiveUpdate`，并追加可审计的 `recognition_retry` 事件；`correct_latest` 保持可用。

- [ ] **Step 1: 写失败测试**

增加测试，验证动作超时后 `update.status == "running"`、当前玩家不变、事件类型为 `recognition_retry`；增加测试验证当前玩家提交后下一次调用只使用下一玩家的识别结果；增加测试验证明确“不出”可以提交。

- [ ] **Step 2: 运行测试确认失败**

运行：

```powershell
& '.venv\\Scripts\\python.exe' -m pytest tests/test_live_orchestrator.py -q
```

预期：超时测试仍返回 `review_required`，新测试失败。

- [ ] **Step 3: 最小实现**

移除编排器对 `WAIT_CLEAR` 参数的传递；删除 `sample_every_frame` 对区域生命周期的旁路；将 `_require_review` 的自动路径统一为记录 `recognition_retry`、清空 burst、保持 `running` 并重新激活当前玩家动作窗口。保留 incident 和事后纠错方法；更新显示文案，使重试不再显示“需要人工确认”。

- [ ] **Step 4: 运行测试确认通过**

运行：

```powershell
& '.venv\\Scripts\\python.exe' -m pytest tests/test_live_orchestrator.py tests/test_zone_lifecycle.py -q
```

预期：相关测试全部通过，既有事后纠错测试仍通过。

- [ ] **Step 5: 提交**

```powershell
git add tests/test_live_orchestrator.py src/daguandan_bridge/live/orchestrator.py src/daguandan_bridge/live/display_text.py
git commit -m "fix: retry live recognition without manual review"
```

### Task 3: 让状态机复测使用实时同构动作窗口

**Files:**
- Modify: `src/daguandan_bridge/live/replay.py`
- Modify: `src/daguandan_bridge/gui/replay_page.py`
- Test: `tests/test_live_replay.py`
- Test: `tests/test_replay_page.py`

**Interfaces:**
- Consumes: `replay_video_through_live_pipeline(..., use_live_pipeline=True)`。
- Produces: 视频输入经过真实区域生命周期、自动重试、共识和 Reducer；不请求 DanZero。

- [ ] **Step 1: 写失败测试**

增加回放测试，断言状态机复测创建的编排器不使用 `sample_every_frame=True`，并在静态旧牌、动画帧和超时情况下继续走动作窗口；保留可信日志复测的 DanZero 调用次数断言。

- [ ] **Step 2: 运行测试确认失败**

运行：

```powershell
& '.venv\\Scripts\\python.exe' -m pytest tests/test_live_replay.py tests/test_replay_page.py -q
```

预期：新增配置断言因当前 pipeline 开启逐帧旁路而失败。

- [ ] **Step 3: 最小实现**

把状态机管线调用改为 `sample_every_frame=False`，使用统一的动作窗口和自动重试配置；删除只为旁路服务的 `pass_min_stable_frames` 分支配置，不改变可信日志驱动路径。

- [ ] **Step 4: 运行测试确认通过**

运行同一 pytest 命令，确认状态机复测和可信日志测试通过。

- [ ] **Step 5: 提交**

```powershell
git add tests/test_live_replay.py tests/test_replay_page.py src/daguandan_bridge/live/replay.py src/daguandan_bridge/gui/replay_page.py
git commit -m "fix: replay live action window faithfully"
```

### Task 4: 清理实时页面的人工确认阻断提示并保留事后纠错

**Files:**
- Modify: `src/daguandan_bridge/gui/live_assistant_page.py`
- Modify: `src/daguandan_bridge/gui/live_controller.py`
- Test: `tests/test_live_assistant_page.py`

**Interfaces:**
- Consumes: `LiveUpdate.status == "running"` 与 `recognition_retry` 事件。
- Produces: 页面不显示人工确认面板；既有事后纠错按钮仍可用。

- [ ] **Step 1: 写失败测试**

增加页面测试，验证 `recognition_retry` 更新时人工确认栏保持隐藏、状态显示为自动重试；保留现有纠错控件测试。

- [ ] **Step 2: 运行测试确认失败**

运行：

```powershell
& '.venv\\Scripts\\python.exe' -m pytest tests/test_live_assistant_page.py -q
```

预期：新页面测试失败，因为当前只识别 `review_required` 文案。

- [ ] **Step 3: 最小实现**

更新实时页面状态和日志显示，隐藏人工确认栏；不删除 `correct_latest` 对应的事后纠错控件和接口。

- [ ] **Step 4: 运行测试确认通过**

运行：

```powershell
& '.venv\\Scripts\\python.exe' -m pytest tests/test_live_assistant_page.py tests/test_live_end_to_end.py -q
```

预期：页面与实时端到端测试通过。

- [ ] **Step 5: 提交**

```powershell
git add tests/test_live_assistant_page.py src/daguandan_bridge/gui/live_assistant_page.py src/daguandan_bridge/gui/live_controller.py
git commit -m "ui: keep live flow automatic with post-hoc correction"
```

### Task 5: 全量回归与交付检查

**Files:**
- Verify: `tests/`
- Verify: `git diff --check`

- [ ] **Step 1: 运行功能回归**

```powershell
& '.venv\\Scripts\\python.exe' -m pytest tests/test_zone_lifecycle.py tests/test_live_orchestrator.py tests/test_live_replay.py tests/test_replay_page.py tests/test_live_assistant_page.py tests/test_live_end_to_end.py -q
```

- [ ] **Step 2: 运行完整测试**

```powershell
& '.venv\\Scripts\\python.exe' -m pytest -q
```

- [ ] **Step 3: 检查差异**

```powershell
git diff --check
git status --short
```

- [ ] **Step 4: 交付报告**

报告实际修改文件、测试结果、仍存在的与本次无关的既有失败，不把未验证的行为称为完成。
