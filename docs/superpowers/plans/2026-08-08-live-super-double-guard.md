# 实时对局超级加倍保护 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在实时 `running` 管线中优先处理现有 `super_double` 模板信号，显示“正在决定是否加倍”，暂停出牌识别和建议推进，并在按钮消失后自动恢复。

**Architecture:** 保持 `LiveStatus` 和复测协议不变，把加倍保护作为 `analyze_frame()` 中快速信号之后的短路分支。通过 `LiveUpdate.fast_signals` 让 UI 根据当前帧显示临时状态；不写入玩家动作事件、不创建 review、不调用 `recognize_play_region()`。加倍期间清空未提交的 burst，恢复后从新帧开始识别。

**Tech Stack:** Python 3、PySide6/QFluentWidgets、pytest、现有 `LiveOrchestrator` 和 `FastSignalResult`。

## Global Constraints

- 复用现有 `FastSignalResult.super_double_visible` 和 `super_double.png` 模板，不新增按钮或用户操作。
- `waiting_lead` 已有的加倍等待逻辑保持不变。
- 可信日志复测的输入、状态机驱动方式和 DanZero 请求策略不改变。
- 加倍按钮存在时不得提交出牌/不出牌、推进回合、创建 review 或触发 DanZero 建议。

---

### Task 1: 为 running 状态增加失败回归测试

**Files:**
- Modify: `tests/test_live_orchestrator.py`
- Test helper: `tests/test_live_orchestrator.py` 中现有 `FakeRecognitionService` 和 `_orchestrator()`

**Interfaces:**
- `FakeRecognitionService(super_double_visible: bool = False)` 返回带有 `super_double_visible` 的 `FastSignalResult`，并继续统计 `targeted_calls`。
- `_orchestrator(..., recognition=None)` 在传入识别服务时使用它，否则保持现有默认行为。

- [ ] **Step 1: 写失败测试**

新增测试 `test_running_super_double_pauses_action_pipeline_without_review`：

```python
def test_running_super_double_pauses_action_pipeline_without_review(tmp_path):
    recognition = FakeRecognitionService(
        [_play("7S") for _ in range(3)],
        super_double_visible=True,
    )
    orchestrator = _orchestrator(
        tmp_path,
        [_play("7S") for _ in range(3)],
        recognition=recognition,
    )
    before = orchestrator.snapshot

    update = orchestrator.ingest_frame(
        np.zeros((32, 64, 3), np.uint8),
        monotonic_ms=100,
        wall_time="super-double",
        metrics=ZoneFrameMetrics(
            monotonic_ms=100,
            occupied=True,
            motion_score=0.9,
            pass_visible=True,
            effect_visible=True,
        ),
    )

    assert update.status == "running"
    assert update.fast_signals is not None
    assert update.fast_signals.super_double_visible is True
    assert update.review is None
    assert orchestrator.snapshot == before
    assert recognition.targeted_calls == 0
    assert [event.event_type for event in orchestrator.events] == [
        "initial_state_confirmed",
        "turn_started",
    ]
    orchestrator.finish()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
pytest tests/test_live_orchestrator.py::test_running_super_double_pauses_action_pipeline_without_review -q
```

Expected: FAIL because the current `running` branch continues into the zone pipeline and calls `recognize_play_region()`/may accumulate a review instead of short-circuiting.

- [ ] **Step 3: 保留恢复行为的测试**

新增测试 `test_running_super_double_resumes_targeted_recognition_after_clear`，使用 `recognition.super_double_visible = False` 后连续送入 3 帧正常动作帧，并断言 `recognize_play_region()` 至少被调用一次；该断言确保保护只覆盖按钮存在的帧。

- [ ] **Step 4: 运行测试确认仍是红灯**

Run:

```powershell
pytest tests/test_live_orchestrator.py::test_running_super_double_pauses_action_pipeline_without_review -q
```

Expected: FAIL only because生产代码尚未增加保护分支。

### Task 2: 为页面临时状态增加失败回归测试

**Files:**
- Modify: `tests/test_live_assistant_page.py`
- Modify: `src/daguandan_bridge/gui/live_assistant_page.py`

**Interfaces:**
- `LiveAssistantPage.apply_update(update)` 继续接收现有 `LiveUpdate`；当 `update.fast_signals.super_double_visible` 为真时显示精确文案“正在决定是否加倍”。
- 测试使用 `SimpleNamespace(current_player="left", trick_id=1, turn_id=1)` 作为 UI 所需的最小 snapshot，不新增测试基础设施。
- 测试文件补充导入 `FastSignalResult` 和 `LiveUpdate`，沿用现有 `FakeRuntime` 与 `_app()`。

- [ ] **Step 1: 写失败测试**

新增一个不依赖实时线程的页面测试，构造已有 `LiveUpdate`，并断言：

```python
def test_live_page_shows_super_double_decision_state():
    from types import SimpleNamespace

    page = LiveAssistantPage(FakeRuntime())
    update = LiveUpdate(
        status="running",
        snapshot=SimpleNamespace(current_player="left", trick_id=1, turn_id=1),
        fast_signals=FastSignalResult(
            expected_player="left",
            active_player=None,
            pass_visible=True,
            self_action_buttons_visible=False,
            effect_visible=True,
            super_double_visible=True,
        ),
    )

    page.apply_update(update)

    assert page.live_status.text() == "状态：正在决定是否加倍"
    assert "加倍按钮显示期间不进行出牌识别" in page.turn_status.text()
    page.close()
```

测试使用该文件现有的 Qt 应用夹具、`FakeRuntime` 和 snapshot 构造方式，避免新增 UI 测试基础设施。

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
pytest tests/test_live_assistant_page.py::test_live_page_shows_super_double_decision_state -q
```

Expected: FAIL because `apply_update()` 当前只按 `update.status` 显示“运行中”。

### Task 3: 实现实时管线和 UI 的最小修改

**Files:**
- Modify: `src/daguandan_bridge/live/orchestrator.py`，`LiveOrchestrator.analyze_frame()` 的 `running` 快速信号分支
- Modify: `src/daguandan_bridge/gui/live_assistant_page.py`，`LiveAssistantPage.apply_update()`

**Interfaces:**
- 不增加 `LiveStatus` 枚举值，不改变 `LiveUpdate` 结构，不改变 `LiveSessionStore` 日志协议。

- [ ] **Step 1: 实现 orchestrator 短路保护**

在 `analyze_frame()` 已取得 `fast` 并获得状态锁后、创建/观察 `_zone` 之前加入：

```python
self._apply_fast_signal(fast)
if fast.super_double_visible:
    self._clear_burst()
    self._pass_stable[expected] = 0
    return self._update(fast_signals=fast)
```

保留现有 self lead 等后续逻辑，确保只有按钮存在时短路。

- [ ] **Step 2: 实现页面状态显示**

在 `apply_update()` 的普通状态分支中先判断：

```python
if update.fast_signals is not None and update.fast_signals.super_double_visible:
    self.live_status.setText("状态：正在决定是否加倍")
    self.turn_status.setText("超级加倍按钮显示期间不进行出牌识别")
else:
    # 现有 running/review_required 等文案逻辑
```

`waiting_lead` 分支保持优先，避免改变原有“等待首发标志”文案。

- [ ] **Step 3: 运行两个新增测试确认通过**

Run:

```powershell
pytest tests/test_live_orchestrator.py::test_running_super_double_pauses_action_pipeline_without_review tests/test_live_assistant_page.py::test_live_page_shows_super_double_decision_state -q
```

Expected: PASS。

### Task 4: 回归验证

**Files:**
- No additional production files.
- Verify: `tests/test_live_orchestrator.py`, `tests/test_live_assistant_page.py`, `tests/test_live_replay.py`, `tests/test_live_end_to_end.py`。

- [ ] **Step 1: 运行实时核心和 UI 回归测试**

```powershell
pytest tests/test_live_orchestrator.py tests/test_live_assistant_page.py tests/test_live_end_to_end.py -q
```

- [ ] **Step 2: 运行复测相关测试**

```powershell
pytest tests/test_live_replay.py tests/test_replay_page.py -q
```

- [ ] **Step 3: 运行全量测试**

```powershell
pytest -q
```

- [ ] **Step 4: 检查变更范围**

```powershell
git diff --check
git status --short
```

确认只有本功能相关的测试和生产代码变更；不覆盖用户已有的未提交改动。
