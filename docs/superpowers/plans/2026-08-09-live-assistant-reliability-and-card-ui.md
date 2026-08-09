# Live Assistant Reliability and Card UI Implementation Plan

**Goal:** Correct Joker colour classification, keep pass confirmation template-only, remove DanZero cold-start latency, and show all user-visible cards as visual cards or Chinese names.

**Architecture:** Preserve raw card codes in recognition, replay, and DanZero state. Add a colour-profile filter only to Joker template candidates, remove next-player inferred-pass handling from the orchestrator, and make the live controller own one asynchronously initialized advisor. The live page presents cards with the existing `CardBadge` component while its persistent text output uses Chinese card names.

## Global constraints

- Do not modify or delete historical sessions.
- A pass is committed only after the expected player's dedicated pass template is visible.
- Do not execute pytest, compileall, UI smoke tests, or replay during this delivery.
- Preserve the current state-machine and replay data formats.

### Task 1: Joker recognition correctness

**Files:** `recognition_service.py`, `tests/test_recognition_service.py`

1. Add tests that prove a black Joker crop selects `small_joker`, a red Joker crop selects `big_joker`, and colour-incompatible template candidates are excluded.
2. Add a content-colour profile to Joker matches and use it only as a class filter after template localization.
3. Retain the existing geometry score, ordinary-card matching, template configuration, and two-deck duplicate limit.

### Task 2: Template-only pass and deck safety

**Files:** `live/orchestrator.py`, `live/consensus.py`, `tests/test_live_orchestrator.py`, `tests/test_live_consensus.py`

1. Replace inferred-next-turn-pass coverage with a test proving an empty expected zone and another active player do not create a pass without the dedicated template.
2. Remove the inferred-pass branch and keep the existing `pass_visible` lifecycle path as the only automatic pass path.
3. Add an aggregate known-card guard so a candidate producing a third physical copy is rejected before it enters confirmed history or reaches DanZero.

### Task 3: Persistent DanZero warmup

**Files:** `gui/live_controller.py`, `tests/test_live_advice.py`

1. Add controller tests proving one advisor instance is reused across sessions and warmup is dispatched without blocking the live capture start.
2. Move the advisor ownership to the controller and initialize it in the existing worker mechanism when the live page requests an initial recognition.
3. Emit non-blocking warmup status updates to the page; a failed warmup remains recoverable when the first recommendation runs.

### Task 4: Human-readable card UI and concise timeline prefix

**Files:** `gui/live_assistant_page.py`, `gui/single_image_danzero_page.py`, `live/display_text.py`, `live/session_store.py`, `tests/test_live_assistant_page.py`, `tests/test_live_session_store.py`

1. Add tests covering `[第 N 手]` prefixes, Chinese copy text, visual card badges for plays/advice, and no raw card codes in live-page output.
2. Replace raw-code live-page card strings with compact `CardBadge` strips, including initial hand, committed plays, advice, and correction candidates.
3. Preserve selectable/copyable timeline behavior by copying Chinese card names from selected rows and preserve teal advice styling plus auto-scroll.
4. Change persisted Markdown display text to Chinese card names while retaining raw JSON codes for replay.

### Task 5: Delivery checks

1. Inspect only the diff and changed Python syntax manually; do not run tests or any test-equivalent command.
2. Provide focused pytest commands and one complete regression command for the user to run later.
