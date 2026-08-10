# 遮挡出牌、简明对局日志与自动封存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让遮挡花色的出牌不断局且不把猜测固化为事实，同时以单行日志展示事件并可靠检测结算页自动封存。

**Architecture:** 识别层输出“点数 + 花色候选”，Reducer 持久化 `rank?` 与候选元数据并按张数推进。策略层在临时副本上枚举库存可行的花色分配并汇总 DanZero 建议；终局识别使用独立底部按钮区域。

**Tech Stack:** Python 3.12、PySide6、OpenCV、pytest、现有 DanZero/rlcard 适配器。

## Global Constraints

- 不操作游戏客户端；只读取截图、记录 session 和显示建议。
- 原始 timeline 事件不可变；花色推断只能作为 payload 元数据和策略副本存在。
- 遮挡牌必须保留张数；不得因缺少花色直接作为空牌或少牌提交。
- 所有 Qt 更新保持主线程信号路径；业务逻辑放在非 UI 模块。
- 保留现有未提交的用户修改，不重置、覆盖或批量格式化无关文件。

---

### Task 1: 表示遮挡牌及其库存可行分配

**Files:**
- Create: `src/daguandan_bridge/live/card_uncertainty.py`
- Modify: `src/daguandan_bridge/recognition_service.py`
- Modify: `src/daguandan_bridge/live/consensus.py`
- Test: `tests/test_card_uncertainty.py`
- Test: `tests/test_recognition_service.py`

**Interfaces:**
- Produces `SuitOptions = tuple[str, ...]` and `enumerate_feasible_suit_assignments(cards, suit_options, known_cards, level_rank, table_cards) -> tuple[tuple[str, ...], ...]`.
- `PlayRegionResult` and `RecognitionSample` carry `suit_options: tuple[SuitOptions, ...]` aligned to `cards`.

- [ ] **Step 1: Write failing tests** for a red `5?` retaining its slot, black `8?` being able to resolve to clubs when two spade eights are known, and an impossible third exact card being rejected.
- [ ] **Step 2: Run** `pytest tests/test_card_uncertainty.py tests/test_recognition_service.py -q` and verify the new assertions fail because no uncertainty metadata or allocator exists.
- [ ] **Step 3: Implement** color-aware fallback candidates in `_recognize_cards`, preserve `rank?`, and add the pure, bounded feasible-assignment helper. Include every candidate’s rank, double-deck limit, current level-heart wild semantics, exact action legality and table-beating requirement.
- [ ] **Step 4: Update consensus** to vote on cards plus suit options and accept an uncertain action only when the allocator returns at least one legal assignment.
- [ ] **Step 5: Re-run** `pytest tests/test_card_uncertainty.py tests/test_recognition_service.py tests/test_live_consensus.py -q`.

### Task 2: 让 Reducer、生命周期与策略隔离不确定花色

**Files:**
- Modify: `src/daguandan_bridge/live/models.py`
- Modify: `src/daguandan_bridge/live/orchestrator.py`
- Modify: `src/daguandan_bridge/live/reducer.py`
- Modify: `src/daguandan_bridge/danzero/advisor.py`
- Test: `tests/test_live_orchestrator.py`
- Test: `tests/test_live_reducer.py`
- Test: `tests/test_live_advice.py`

**Interfaces:**
- `LiveEvent.payload["unknown_suits"]` is a list of `{"index": int, "candidates": list[str]}` records.
- `LiveAdvice` exposes `suit_resolution: Literal["exact", "consistent", "ambiguous", "bounded"]` and optional `alternate_advice` summary.

- [ ] **Step 1: Write failing tests** that commit `("3H", "3S", "4D", "4S", "5?", "7H")`, assert left loses six cards, and assert a later exact spade eight does not cause a false third-spade rejection.
- [ ] **Step 2: Run** the focused reducer/orchestrator/advice tests and verify failures identify missing metadata propagation and unknown-card strategy handling.
- [ ] **Step 3: Implement** metadata propagation from recognition sample through consensus, event creation and observation persistence. Keep reducer event cards as `rank?`, use their count for turn and finish transitions, and never replace them in saved history.
- [ ] **Step 4: Implement** an advice-only state resolver that enumerates at most 16 feasible concrete histories, runs the existing advisor per unique state, aggregates matching advice, and records uncertainty details in `advice.jsonl`. A stale or failed variant must not invalidate a feasible variant.
- [ ] **Step 5: Re-run** `pytest tests/test_live_reducer.py tests/test_live_orchestrator.py tests/test_live_advice.py -q`.

### Task 3: 单行对局动态

**Files:**
- Modify: `src/daguandan_bridge/gui/live_assistant_page.py`
- Modify: `src/daguandan_bridge/live/display_text.py`
- Test: `tests/test_live_assistant_page.py`

**Interfaces:**
- `compact_cards_text(cards)` renders ranks contiguously and maps jokers to Chinese labels.
- `_append_event_to_timeline(event)` emits exactly one text line for each event.

- [ ] **Step 1: Write failing UI tests** expecting `左家出牌：33445?7` on one `toPlainText()` line, and one-line head/wind/end entries.
- [ ] **Step 2: Run** `pytest tests/test_live_assistant_page.py -q` and verify it fails because cards are rendered in a separate HTML table and prefix/action use `<br>`.
- [ ] **Step 3: Implement** compact text rendering; remove event/card table line breaks while retaining text selection, colors and automatic scrolling. Surface `ambiguous` DanZero advice as “花色推断，低置信”.
- [ ] **Step 4: Re-run** `pytest tests/test_live_assistant_page.py tests/test_live_advice.py -q`.

### Task 4: 终局按钮区域与自动封存

**Files:**
- Modify: `data/profiles/tencent_daguandan/regions_config.json`
- Modify: `src/daguandan_bridge/recognition_service.py`
- Test: `tests/test_recognition_service.py`
- Test: `tests/test_live_orchestrator.py`
- Test: `tests/test_live_controller.py`

**Interfaces:**
- `game_end_controls` is a profile region separate from `button_actions`.
- Fast and opening signals merge ordinary and terminal button matches before assigning `game_end_control`.

- [ ] **Step 1: Write failing tests** that paste real terminal-button templates into the bottom region and expect `continue_game`/`change_table`; add a controller test where an `events` tuple contains the terminal event but `event` is absent.
- [ ] **Step 2: Run** `pytest tests/test_recognition_service.py tests/test_live_orchestrator.py tests/test_live_controller.py -q` and verify bottom buttons are currently missed.
- [ ] **Step 3: Add** the bottom profile ROI and a shared terminal-button recognizer. Merge annotations without duplicating labels, emit only one `game_end_detected`, and make the controller inspect both `update.event` and `update.events` before requesting one asynchronous finish.
- [ ] **Step 4: Re-run** the focused end-control tests.

### Task 5: 真实录像回归与交付验证

**Files:**
- Modify: `docs/live_assistant.md`
- Modify: `README.md`
- Test: `tests/test_live_end_to_end.py`
- Test: `tests/test_live_replay.py`

- [ ] **Step 1: Add** a regression fixture sourced from the latest session’s first action and end frame without committing user recordings; tests must crop/synthesize only the needed ROIs.
- [ ] **Step 2: Run** focused real-session replay for `game_20260810_000418_3ccf81`, inspect generated timeline/advice output, and verify six first-action cards, one-line logs, lifecycle continuity and `sealed` terminal status.
- [ ] **Step 3: Update** user documentation to describe `?` cards, inference confidence and auto-seal conditions.
- [ ] **Step 4: Run** `python -m compileall -q src`, targeted tests, then `pytest -q` with `QT_QPA_PLATFORM=offscreen`.
