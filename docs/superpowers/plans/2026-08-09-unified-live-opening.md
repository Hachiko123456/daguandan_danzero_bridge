# Unified Live Opening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify opening detection across live play and state-machine replay, reset only the live UI timeline for a new session, and make the timeline selectable and copyable.

**Architecture:** Add one lightweight opening-signal API to the recognition service and let the orchestrator own all lead confirmation. The live page becomes a presenter that resets transient UI state and renders the canonical event stream as selectable rich text. State-machine replay starts without a persisted lead so it exercises the same opening stage.

**Tech Stack:** Python 3.11, PySide6, QFluentWidgets, OpenCV, pytest.

## Global Constraints

- Do not delete or overwrite historical files under `data/profiles/tencent_daguandan/sessions`.
- Keep DanZero advice in the same timeline and distinguish it with color, without adding a copy button.
- Do not execute tests in this delivery; author the tests and provide commands only.
- Preserve the existing trusted-log replay behavior, which intentionally bypasses visual recognition.

---

### Task 1: Unified opening signal

**Files:**
- Modify: `src/daguandan_bridge/recognition_service.py`
- Modify: `src/daguandan_bridge/live/orchestrator.py`
- Test: `tests/test_recognition_service.py`
- Test: `tests/test_live_orchestrator.py`

**Interfaces:**
- Produces `OpeningSignal(super_double_visible, marker_player, active_player, self_action_buttons_visible)`.
- `LiveOrchestrator` consumes this signal while `status == "waiting_lead"`.

- [x] Write tests showing super-double suppresses candidates, matching marker/timer confirms, and conflicting candidates wait.
- [x] Add the `OpeningSignal` result and `recognize_opening_signal(image)` to the recognition service.
- [x] Change waiting-lead analysis to consume this one signal and use the existing stability counter only after the signal is unblocked.

### Task 2: Live page state lifecycle and copyable timeline

**Files:**
- Modify: `src/daguandan_bridge/gui/live_assistant_page.py`
- Test: `tests/test_live_assistant_page.py`

**Interfaces:**
- `LiveAssistantPage._reset_transient_session_ui()` clears only widgets and in-memory keys.
- Timeline entries are appended to a read-only selectable rich-text document.

- [x] Write tests showing a second start clears prior visible events, resets lead to automatic, retains no prior DanZero line, and exposes selectable text.
- [x] Start live sessions with `lead_player=None`; show any initial candidate as information only.
- [x] Replace the custom timeline widget list with a selectable `QTextBrowser`; render advice with a distinct teal style and retain auto-scroll.

### Task 3: Replay opening parity

**Files:**
- Modify: `src/daguandan_bridge/live/replay.py`
- Test: `tests/test_live_replay.py`

**Interfaces:**
- `replay_video_through_live_pipeline(..., use_live_pipeline=True)` starts the orchestrator with `lead_player=None`.

- [x] Write a test whose persisted timeline has a wrong lead and whose first opening signal resolves a different lead; assert the live-pipeline replay follows the signal.
- [x] Start video state-machine replay with no lead only for the live-pipeline path; retain current non-pipeline and trusted-log behavior.

### Task 4: Delivery review

**Files:**
- Review: `tests/test_recognition_service.py`
- Review: `tests/test_live_orchestrator.py`
- Review: `tests/test_live_assistant_page.py`
- Review: `tests/test_live_replay.py`

- [x] Confirm each test names a production break: stale UI, super-double false lead, candidate conflict, replay baseline bypass, or noncopyable timeline.
- [x] Do not run pytest, compileall, or any replay in this delivery; provide the exact test commands for the user to run later.
