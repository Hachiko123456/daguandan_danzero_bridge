# Trusted Log Live DanZero Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use trusted `truth_log.json` files as a deterministic event source that exercises the production live state transition and DanZero request pipeline without starting a new game or decoding video.

**Architecture:** Keep video recognition replay and deterministic reducer replay unchanged. Add a trusted-action entry point to `LiveOrchestrator`, then add a replay runner that starts a real orchestrator with an injected advisor, submits trusted actions, waits for each advice result, and copies diagnostics into a separate `replay_runs` directory. Reuse the existing replay page, session selector, mode combo, start button, and diagnostics editor.

**Tech Stack:** Python 3, PySide6/QFluentWidgets, pytest, existing `LiveOrchestrator`, `LiveReducer`, `LiveSessionStore`, `LatestOnlyWorker`, and `DanzeroAdvisor`.

## Global Constraints

- Preserve all existing user changes and do not overwrite the source session directory's `truth_log.json`, video, timeline, or advice files.
- The trusted replay must call the production `LiveOrchestrator.start()`, trusted action transition, advisor worker, advice completion, and visibility path.
- The existing deterministic state replay must remain DanZero-free.
- The existing video scan and video pipeline replay modes must retain their current behavior.
- No new replay page or required extra main button; add one mode to the existing mode selector and reuse `开始复测`.
- Every production behavior change gets a failing test before implementation.

---

### Task 1: Add the trusted action transition and advice wait contract

**Files:**
- Modify: `src/daguandan_bridge/live/orchestrator.py:200-230, 645-715, 788-810, 950-1067`
- Test: `tests/test_live_advice.py`
- Test: `tests/test_live_orchestrator.py`

**Interfaces:**
- Produce `LiveOrchestrator.commit_trusted_action(*, actor: Seat, cards: tuple[str, ...] = (), is_pass: bool, monotonic_ms: int, evidence_refs: tuple[str, ...] = ()) -> LiveUpdate`.
- Produce `LiveOrchestrator.wait_for_advice(key: AdviceRequestKey, timeout: float = 60.0) -> LiveAdvice | None`.
- The trusted transition must publish a normal action event, rotate the reducer, append the next `turn_started` event, and invoke `_request_advice_if_needed()` exactly like the visual consensus commit path.

- [ ] **Step 1: Write the failing tests**

  Add tests showing that a trusted action advances the same state path as a committed visual action and that a requested advice key can be waited on until the fake advisor returns `ready`.

- [ ] **Step 2: Run the focused tests and verify the expected failure**

  Run:

  ```powershell
  $env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_live_advice.py tests/test_live_orchestrator.py
  ```

  Expected failure: `LiveOrchestrator` has no `commit_trusted_action` or `wait_for_advice` method.

- [ ] **Step 3: Implement the minimal production behavior**

  Add a trusted transition that validates the actor against `snapshot.current_player`, records with confidence `1.0` and source `trusted_log_replay`, publishes the event, resets the burst, starts the next turn, and requests advice. Add per-request completion events so the replay worker can wait without polling or sleeping in the GUI thread. Signal completion for ready, failed, stale, and ignored results.

- [ ] **Step 4: Run the focused tests and verify they pass**

  Re-run the command from Step 2. Expected: all focused tests pass with no new warnings.

---

### Task 2: Implement trusted-log DanZero replay and durable run artifacts

**Files:**
- Modify: `src/daguandan_bridge/live/replay.py:1-80, 372-573`
- Test: `tests/test_live_replay.py`

**Interfaces:**
- Produce `TrustedAdviceReplayResult` with output and summary paths plus processed turn and advice status counts.
- Produce `replay_truth_through_live_advisor(session: Path, advisor: Any, *, truth_log: TruthLog | Path | None = None, output_root: Path | None = None, stop_requested: Callable[[], bool] | None = None, on_advice: Callable[[dict[str, object]], None] | None = None, advice_timeout_sec: float = 60.0) -> TrustedAdviceReplayResult`.

- [ ] **Step 1: Write the failing tests**

  Add a small trusted log with one non-self action followed by one self action. Assert the runner starts the production orchestrator, calls the fake advisor once at the self turn, writes a separate run directory, and leaves the source `truth_log.json` unchanged. Add a failure test for an actor that does not match the reducer's expected player.

- [ ] **Step 2: Run the focused test and verify it fails for the missing runner**

  Run:

  ```powershell
  $env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_live_replay.py -k trusted
  ```

  Expected failure: the trusted replay function and result type do not exist.

- [ ] **Step 3: Implement the minimal runner**

  Load the trusted log, create a temporary `LiveSessionStore` and `SessionRecorder`, instantiate `LiveOrchestrator` with the supplied advisor, call `start()`, use `ingest_fast_signal(active_player="self")` to exercise the production advice visibility path, submit each `TruthTurn` through `commit_trusted_action()`, wait at each requested self turn, then finish the orchestrator. Copy only generated diagnostics into `session/replay_runs/<run_id>/` and write `summary.json`; never delete or overwrite source artifacts.

- [ ] **Step 4: Run the focused test and verify it passes**

  Re-run the command from Step 2. Expected: trusted runner tests pass and show the fake advisor call was made by the real orchestrator path.

---

### Task 3: Reuse the replay page for the trusted live test mode

**Files:**
- Modify: `src/daguandan_bridge/gui/replay_page.py:40-70, 269-325, 450-515, 1009-1042, 1160-1310`
- Test: `tests/test_replay_page.py`

**Interfaces:**
- Add mode data value `trusted_advisor` to the existing `replay_mode_combo`.
- Add `TrustedAdviceReplayThread(QThread)` that emits progress records, completion, failure, and finished signals.
- Route the existing `truth_replay_button` to the trusted runner when the selected mode is `trusted_advisor`; keep scan and pipeline routing unchanged.

- [ ] **Step 1: Write the failing UI routing test**

  Assert the mode selector contains the trusted option and that selecting it routes the existing replay action to the trusted thread without requiring a second button.

- [ ] **Step 2: Run the UI test and verify the expected failure**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM='offscreen'; $env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_replay_page.py -k trusted
  ```

  Expected failure: the selector has no trusted mode and the routing does not exist.

- [ ] **Step 3: Implement the UI integration**

  Add one combo-box option, a worker using `DanzeroAdvisor`, progress lines in the existing diagnostics widget, a summary renderer, stop/finish cleanup, and a dynamic button label (`开始实时助手测试` for the trusted mode, existing text for the other modes). Use the selected session's `truth_log.json` and persist results under its `replay_runs` directory.

- [ ] **Step 4: Run focused UI tests**

  Run the command from Step 2 and the existing replay-page tests. Expected: all pass offscreen.

---

### Task 4: Validate the three real trusted sessions and the regression suite

**Files:**
- No source changes expected.
- Read: `data/profiles/tencent_daguandan/sessions/game_20260807_002323_616108/truth_log.json`
- Read: `data/profiles/tencent_daguandan/sessions/game_20260806_150516_1fb519/truth_log.json`
- Read: `data/profiles/tencent_daguandan/sessions/game_20260806_231123_e5ec3c/truth_log.json`

- [ ] **Step 1: Run trusted replay with a fake advisor against all three logs**

  Verify all trusted action chains complete, expected self-turn advice requests are generated, and each run has an independent summary.

- [ ] **Step 2: Run trusted replay with the real `DanzeroAdvisor` against all three logs**

  Verify the real recommendation path returns or records a clear failure for every expected self turn, with no stale result silently counted as ready.

- [ ] **Step 3: Run focused regression tests**

  ```powershell
  $env:QT_QPA_PLATFORM='offscreen'; $env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m pytest -q tests/test_live_replay.py tests/test_live_advice.py tests/test_live_orchestrator.py tests/test_live_end_to_end.py tests/test_live_controller.py tests/test_live_assistant_page.py tests/test_replay_page.py tests/test_truth_log.py tests/test_truth_log_editor.py
  ```

- [ ] **Step 4: Run compile validation**

  ```powershell
  $env:PYTHONPATH='src'; .\\.venv\\Scripts\\python.exe -m compileall -q src tests
  ```

- [ ] **Step 5: Inspect the final diff and report any unrelated baseline failures separately**

  Do not claim the full suite passes if the known Qt/Torch access violation recurs; report the exact command and failure boundary.
