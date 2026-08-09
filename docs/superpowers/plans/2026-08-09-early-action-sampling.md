# 首次动作采样延迟 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让默认实时动作识别策略在区域变化后立即采样，同时保留显式策略等待。

**Architecture:** `RecognitionStrategySpec` 继续作为策略等待时间的唯一默认来源。`LiveOrchestrator` 的通用延迟仅作为调用方显式覆盖值，默认不再向每种策略强加 1000ms。

**Tech Stack:** Python 3、pytest、LiveOrchestrator、ZoneLifecycle。

## Global Constraints

- 不修改历史 session、录像或日志。
- 不改变不出模板判定逻辑。
- 不改变任一策略在 `RecognitionStrategySpec` 中声明的等待时间。

---

### Task 1: 默认策略首次采样延迟

**Files:**
- Modify: `src/daguandan_bridge/live/orchestrator.py:162`
- Modify: `tests/test_live_orchestrator.py`

**Interfaces:**
- Consumes: `LiveOrchestrator(..., settle_ms=0, recognition_strategy=...)`
- Produces: 默认 `two_valid_streak` 在动作变化的首帧进入出牌识别；`reference_single_shot` 保持 1000ms 后才识别。

- [ ] **Step 1: Write the failing tests**

```python
def test_default_two_valid_streak_samples_on_the_first_changed_frame(tmp_path):
    orchestrator = _default_orchestrator(tmp_path, [_play("7S")] * 2)
    _feed(orchestrator, 100, motion=0.2)
    assert orchestrator.recognition_service.targeted_calls == 1

def test_reference_single_shot_keeps_its_1000ms_settle_delay(tmp_path):
    orchestrator = _default_orchestrator(
        tmp_path, [_play("7S")], recognition_strategy="reference_single_shot"
    )
    _feed(orchestrator, 100, motion=0.2)
    _feed(orchestrator, 1_000)
    assert orchestrator.recognition_service.targeted_calls == 0
    _feed(orchestrator, 1_100)
    assert orchestrator.recognition_service.targeted_calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run --no-capture-output -n yhx python -m pytest -q tests/test_live_orchestrator.py -k "default_two_valid_streak_samples_on_the_first_changed_frame or reference_single_shot_keeps_its_1000ms_settle_delay"`

Expected: the default two-valid-streak test fails because its first call is delayed for 1000ms.

- [ ] **Step 3: Write minimal implementation**

```python
class LiveOrchestrator:
    def __init__(self, ..., settle_ms: int = 0, ...):
        ...
```

Keep the existing zone construction:

```python
settle_ms=max(self.settle_ms, spec.settle_ms)
```

- [ ] **Step 4: Run focused regression tests**

Run: `conda run --no-capture-output -n yhx python -m pytest -q tests/test_live_orchestrator.py tests/test_zone_lifecycle.py`

Expected: all selected tests pass.

- [ ] **Step 5: Inspect the diff**

Run: `git diff --check -- src/daguandan_bridge/live/orchestrator.py tests/test_live_orchestrator.py`

Expected: no whitespace errors.
