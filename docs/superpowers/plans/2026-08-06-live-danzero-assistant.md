# Live DanZero Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manually started, advisory-only live GuanDan assistant that observes each player action, maintains an auditable event-sourced state, records and replays sessions, and asks DanZero for one suggestion per confirmed self turn.

**Architecture:** A persistent capture producer fans standardized frames out to a video recorder and a local action-zone lifecycle. Only the expected player's ROI is watched; after dynamic settling, a 3–5 frame burst is recognized and reduced into immutable game events. A deterministic reducer owns the confirmed state, while Qt pages subscribe to snapshots and never mutate the reducer directly.

**Tech Stack:** Python 3.12, PySide6 6.11.1, PySide6-Fluent-Widgets 1.11.2, OpenCV 4.13, NumPy 2.3, MSS 10.2, PyTorch, pytest.

## Global Constraints

- Windows 11 only; keep all existing single-image and screenshot APIs backward compatible.
- Seat order is exactly `self → right → opposite → left → self`.
- Version one starts before the first play and requires exactly 27 confirmed self cards.
- Never click, type into, or otherwise control the game client.
- Do not infer an uncertain action silently; pause state advancement and surface one-click correction.
- Use monotonic milliseconds for ordering and wall-clock ISO timestamps only for display/audit.
- Per-session data lives under `data/profiles/<profile>/sessions/game_<timestamp>_<id>/`.
- Capture target is 10 FPS; semantic recognition runs only as a 3–5 frame burst after the active ROI settles.
- Default recording is MJPEG/AVI; lossless event and incident frames are PNG.
- Existing user files and unrelated worktree changes must not be staged.

## File Map

- `src/daguandan_bridge/live/models.py`: immutable observation, candidate, event, snapshot, incident, and replay records.
- `src/daguandan_bridge/live/turns.py`: canonical seat ordering and next-active-seat calculation.
- `src/daguandan_bridge/live/reducer.py`: deterministic event application and `GuanDanState` projection.
- `src/daguandan_bridge/live/session_store.py`: JSONL, Markdown timeline, manifest, correction, and incident persistence.
- `src/daguandan_bridge/live/zone_lifecycle.py`: per-seat `WAIT_CLEAR` through `BURST_READ` transitions.
- `src/daguandan_bridge/live/consensus.py`: burst voting and validation outcomes.
- `src/daguandan_bridge/live/recorder.py`: AVI writer, timestamp index, PNG evidence, and ring buffer.
- `src/daguandan_bridge/live/replay.py`: virtual-clock event replay and video-frame replay source.
- `src/daguandan_bridge/live/orchestrator.py`: capture/recognition/reducer/advisor coordination.
- `src/daguandan_bridge/gui/live_assistant_page.py`: live controls, preview, advice, timeline, and correction bar.
- `src/daguandan_bridge/gui/replay_page.py`: session selection, playback, replay mode, and comparison output.
- `src/daguandan_bridge/recognition_service.py`: template cache and targeted ROI recognition APIs.
- `src/daguandan_bridge/capture_service.py`: persistent live capture context without changing one-shot capture.
- `src/daguandan_bridge/gui/main_window.py`: add the two new pages.

---

### Task 1: Canonical turns, event models, and deterministic reducer

**Files:**
- Create: `src/daguandan_bridge/live/__init__.py`
- Create: `src/daguandan_bridge/live/turns.py`
- Create: `src/daguandan_bridge/live/models.py`
- Create: `src/daguandan_bridge/live/reducer.py`
- Modify: `src/daguandan_bridge/recognition_service.py`
- Test: `tests/test_live_reducer.py`
- Test: `tests/test_recognition_service.py`

**Interfaces:**
- Produces: `TURN_ORDER`, `next_active_seat(current, finished)`, `LiveEvent`, `LiveSnapshot`, `LiveReducer.apply(event)`, and `LiveReducer.to_guandan_state()`.
- `LiveEvent.event_type` is one of `session_started`, `initial_state_confirmed`, `trick_started`, `turn_started`, `player_played`, `player_passed`, `player_finished`, `manual_confirmed_event`, or `event_correction`.

- [ ] **Step 1: Write failing turn-order and reducer tests**

```python
def test_turn_order_is_counter_clockwise():
    assert TURN_ORDER == ("self", "right", "opposite", "left")
    assert next_active_seat("right", frozenset()) == "opposite"
    assert next_active_seat("right", frozenset({"opposite"})) == "left"

def test_initial_state_requires_exactly_27_cards():
    reducer = LiveReducer("game-test")
    with pytest.raises(GameStateError, match="27"):
        reducer.confirm_initial_state(
            round_level="2", hand=("3S",), lead_player="right"
        )

def test_correction_rebuild_matches_clean_history():
    wrong = build_started_reducer()
    original = wrong.record_play("right", ("7S", "7H"))
    wrong.correct_event(original.event_id, cards=("8S", "8H"), is_pass=False)
    clean = build_started_reducer()
    clean.record_play("right", ("8S", "8H"))
    assert wrong.snapshot().semantic_dict() == clean.snapshot().semantic_dict()
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_reducer.py tests/test_recognition_service.py -q`

Expected: collection fails because `daguandan_bridge.live` and `TURN_ORDER` do not exist.

- [ ] **Step 3: Implement canonical turns and immutable records**

```python
# live/turns.py
TURN_ORDER: tuple[Seat, ...] = ("self", "right", "opposite", "left")

def next_active_seat(current: Seat, finished: frozenset[Seat]) -> Seat:
    start = TURN_ORDER.index(current)
    for offset in range(1, len(TURN_ORDER) + 1):
        seat = TURN_ORDER[(start + offset) % len(TURN_ORDER)]
        if seat not in finished:
            return seat
    raise GameStateError("没有仍在对局中的下一位玩家")

# live/models.py
@dataclass(frozen=True)
class LiveEvent:
    event_id: str
    event_type: str
    session_id: str
    seq: int
    monotonic_ms: int
    wall_time: str
    trick_id: int
    turn_id: int
    actor: Seat | None
    payload: dict[str, object]
    confidence: float
    source: str
    state_revision_before: int
    state_revision_after: int
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        raw = asdict(self)
        raw["evidence_refs"] = list(self.evidence_refs)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "LiveEvent":
        values = dict(raw)
        values["evidence_refs"] = tuple(values.get("evidence_refs", ()))
        return cls(**values)
```

Implement `LiveSnapshot.semantic_dict()`, `LiveReducer.confirm_initial_state()`, `record_play()`, `record_pass()`, `correct_event()`, `apply()`, `snapshot()`, `clone_empty()`, and `to_guandan_state()`. `LiveReducer` is the only mutable owner. Store the append-only event list plus a correction map; rebuild semantic state from the beginning whenever an `event_correction` is appended. Project confirmed plays into a fresh `GuanDanState`, preserving play order and starting a new trick when all other active players have passed.

- [ ] **Step 4: Replace recognition ordering with the canonical constant**

Import `TURN_ORDER` in `recognition_service.py`, replace `SEATS_IN_ORDER`, and keep `_order_events()` output backward compatible except for the corrected counter-clockwise order.

- [ ] **Step 5: Run tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_reducer.py tests/test_recognition_service.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/daguandan_bridge/live tests/test_live_reducer.py src/daguandan_bridge/recognition_service.py tests/test_recognition_service.py
git commit -m "feat: add live game event reducer"
```

### Task 2: Session store, LLM timeline, corrections, and incidents

**Files:**
- Create: `src/daguandan_bridge/live/session_store.py`
- Test: `tests/test_live_session_store.py`

**Interfaces:**
- Consumes: `LiveEvent.to_dict()` from Task 1.
- Produces: `LiveSessionStore.start()`, `append_event()`, `append_observation()`, `append_advice()`, `create_incident()`, and `seal()`.

- [ ] **Step 1: Write failing persistence tests**

```python
def test_sessions_are_physically_isolated(tmp_path):
    first = LiveSessionStore(tmp_path, "tencent_daguandan", session_id="game-a")
    second = LiveSessionStore(tmp_path, "tencent_daguandan", session_id="game-b")
    first.start(base_manifest())
    second.start(base_manifest())
    first.append_event(sample_event("game-a"))
    assert (first.directory / "timeline.jsonl").is_file()
    assert not (second.directory / "timeline.jsonl").read_text("utf-8")

def test_timeline_markdown_is_llm_readable(tmp_path):
    store = started_store(tmp_path)
    store.append_event(play_event(actor="right", cards=("7S", "7H")))
    text = (store.directory / "timeline.md").read_text("utf-8")
    assert "右家出牌：[7S, 7H]" in text
    assert "证据=" in text

def test_incident_contains_state_and_frame_references(tmp_path):
    store = started_store(tmp_path)
    path = store.create_incident(
        reason="candidate_conflict",
        state_before={"revision": 4},
        state_after={"revision": 4},
        observations=[{"id": "OBS-4"}],
    )
    assert (path / "incident.json").is_file()
    assert (path / "llm_report.md").is_file()
```

- [ ] **Step 2: Verify tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_session_store.py -q`

Expected: import failure for `LiveSessionStore`.

- [ ] **Step 3: Implement crash-tolerant append-only storage**

Use one JSON object per line, `flush()` after observations, and `flush()` plus `os.fsync()` after confirmed events/corrections. Write `manifest.json` through existing `atomic_write_json`. Generate `timeline.md` by appending a Chinese projection of each event; never parse Markdown to rebuild state.

`start()` writes application/Python versions, profile name, configuration hash, template-manifest hash, target FPS, codec, and `status="running"`. `append_advice()` writes the full JSON-safe `engine_input` and timings to `advice.jsonl`, while `timeline.jsonl` receives only the advice lifecycle event and reference. `seal()` closes handles, converts `observations.jsonl.part` to `observations.jsonl.gz`, records final counts/incident IDs, and atomically changes manifest status to `sealed`. Recovery preserves a readable `.part` file and marks the session `aborted`.

```python
class LiveSessionStore:
    def append_event(self, event: LiveEvent) -> None:
        self._append_jsonl(self.timeline_path, event.to_dict(), durable=True)
        self._append_text(self.timeline_markdown_path, format_timeline_line(event))

    def create_incident(
        self,
        *,
        reason: str,
        state_before: dict[str, object],
        state_after: dict[str, object],
        observations: list[dict[str, object]],
        frame_paths: tuple[Path, ...] = (),
        engine_input: dict[str, object] | None = None,
    ) -> Path:
        self._incident_sequence += 1
        incident_id = f"INC-{self._incident_sequence:04d}"
        directory = self.incidents_directory / incident_id
        directory.mkdir(parents=True, exist_ok=False)
        atomic_write_json(directory / "incident.json", {
            "incident_id": incident_id,
            "reason": reason,
            "state_before": state_before,
            "state_after": state_after,
            "observations": observations,
            "frame_paths": [str(path) for path in frame_paths],
        })
        if engine_input is not None:
            atomic_write_json(directory / "engine_input.json", engine_input)
        (directory / "llm_report.md").write_text(
            format_incident_report(incident_id, reason, state_before, state_after, observations),
            encoding="utf-8",
        )
        return directory
```

Treat a truncated last JSONL line as an aborted write during reading; retain all preceding lines.

- [ ] **Step 4: Run persistence tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_session_store.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/daguandan_bridge/live/session_store.py tests/test_live_session_store.py
git commit -m "feat: persist auditable live sessions"
```

### Task 3: Video recorder, frame index, and deterministic replay

**Files:**
- Create: `src/daguandan_bridge/live/recorder.py`
- Create: `src/daguandan_bridge/live/replay.py`
- Test: `tests/test_live_recorder.py`
- Test: `tests/test_live_replay.py`

**Interfaces:**
- Produces: `SessionRecorder.write_frame(frame, captured_monotonic_ms, wall_time)`, `SessionRecorder.save_evidence_frame()`, `EventReplayer.replay(events)`, and `VideoReplaySource.frames()`.

- [ ] **Step 1: Write failing recorder and virtual-clock replay tests**

```python
def test_recorder_writes_playable_video_and_index(tmp_path):
    recorder = SessionRecorder(tmp_path, size=(320, 180), fps=10)
    for index in range(5):
        recorder.write_frame(np.full((180, 320, 3), index, np.uint8), index * 100, f"t{index}")
    result = recorder.close()
    assert result.frame_count == 5
    assert cv2.VideoCapture(str(result.video_path)).isOpened()
    assert len(read_jsonl(result.index_path)) == 5

def test_event_replay_has_no_real_sleep(started_reducer, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("real sleep used"))
    result = EventReplayer(lambda: started_reducer.clone_empty()).replay(events_fixture())
    assert result.final_snapshot.current_player == "self"
```

- [ ] **Step 2: Verify tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_recorder.py tests/test_live_replay.py -q`

Expected: import failures.

- [ ] **Step 3: Implement recorder with explicit frame timestamps**

Open `cv2.VideoWriter` using `MJPG` and `.avi`; fail fast if `isOpened()` is false. Append `frame_index.jsonl` with `frame_index`, `monotonic_ms`, `wall_time`, and `dropped_before`. Maintain a deque of `(monotonic_ms, frame.copy())` capped by a configurable ten-second frame count. `save_evidence_frame()` always uses `save_image_unicode()` and PNG.

Add `save_incident_media(trigger_ms, before_ms=5000, after_ms=5000)` to select buffered frames, write a short MJPEG clip, save the trigger/confirmed frames as PNG, and generate a tiled `contact_sheet.png`. If the encoder drops a frame, increment `dropped_before` and return a typed recorder warning for the orchestrator to persist as an incident.

- [ ] **Step 4: Implement deterministic replay**

`EventReplayer` sorts by `(monotonic_ms, seq)`, feeds events directly to a fresh reducer, and records snapshot hashes after every event. `VideoReplaySource` yields `(FrameIndexRecord, np.ndarray)` and reports missing/extra decoded frames rather than silently shifting timestamps.

- [ ] **Step 5: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_recorder.py tests/test_live_replay.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/daguandan_bridge/live/recorder.py src/daguandan_bridge/live/replay.py tests/test_live_recorder.py tests/test_live_replay.py
git commit -m "feat: record and replay live sessions"
```

### Task 4: Template cache and targeted recognition APIs

**Files:**
- Modify: `src/daguandan_bridge/recognition_service.py`
- Test: `tests/test_recognition_service.py`

**Interfaces:**
- Produces: `ScreenshotRecognitionService.reload_templates()`, `recognize_play_region(image, seat, wild_rank)`, `recognize_fast_signals(image, expected_player)`, and cached `recognize()`.

- [ ] **Step 1: Write failing cache and targeted-recognition tests**

```python
def test_templates_are_loaded_once_for_repeated_recognition(service, monkeypatch, image):
    calls = 0
    original = service.template_service.list_templates
    def counted():
        nonlocal calls
        calls += 1
        return original()
    monkeypatch.setattr(service.template_service, "list_templates", counted)
    service.recognize(image)
    service.recognize(image)
    assert calls == 1

def test_targeted_play_recognition_only_returns_expected_seat(service, image):
    result = service.recognize_play_region(image, "right", wild_rank="2")
    assert result.player == "right"
```

- [ ] **Step 2: Verify tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_recognition_service.py -q`

Expected: cache call count is 2 and targeted methods are missing.

- [ ] **Step 3: Add cache with explicit invalidation**

Store loaded templates on the service instance behind a `threading.RLock`. `reload_templates()` replaces the immutable cached tuple. `recognize()` calls `_templates()` instead of reading disk. Annotation/template save flows must call `reload_templates()` through the owning service rather than relying on mtime races.

- [ ] **Step 4: Add narrow result types and methods**

```python
@dataclass(frozen=True)
class PlayRegionResult:
    player: Seat
    cards: tuple[str, ...]
    is_pass: bool
    confidence: float
    diagnostics: tuple[str, ...]
    annotations: tuple[RecognitionAnnotation, ...]

@dataclass(frozen=True)
class FastSignalResult:
    expected_player: Seat
    active_player: Seat | None
    pass_visible: bool
    self_action_buttons_visible: bool
    effect_visible: bool
```

Reuse `_recognize_cards()` and `_recognize_status()` without duplicating template matching.

- [ ] **Step 5: Run recognition tests and the existing single-image tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_recognition_service.py tests/test_annotation_page.py tests/test_danzero_api.py -q`

Expected: all tests pass and existing `RecognitionResult` stays unchanged.

- [ ] **Step 6: Commit**

```powershell
git add src/daguandan_bridge/recognition_service.py tests/test_recognition_service.py
git commit -m "perf: cache templates for targeted recognition"
```

### Task 5: Local action-zone lifecycle and burst consensus

**Files:**
- Create: `src/daguandan_bridge/live/zone_lifecycle.py`
- Create: `src/daguandan_bridge/live/consensus.py`
- Test: `tests/test_zone_lifecycle.py`
- Test: `tests/test_live_consensus.py`

**Interfaces:**
- Produces: `ZoneLifecycle.observe(ZoneFrameMetrics) -> ZoneDecision` and `BurstConsensus.decide(samples, context) -> ConsensusResult`.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_zone_waits_for_previous_content_to_clear():
    zone = ZoneLifecycle(expected_player="right", started_with_clear_zone=False)
    assert zone.observe(metrics(occupied=True, motion=0.0)).phase == ZonePhase.WAIT_CLEAR
    assert zone.observe(metrics(occupied=False, motion=0.1)).phase == ZonePhase.WAIT_ACTION

def test_effect_restarts_dynamic_settling():
    zone = ready_action_zone()
    zone.observe(metrics(occupied=True, motion=0.5))
    assert zone.observe(metrics(occupied=True, motion=0.01, effect=True)).phase == ZonePhase.CHANGING

def test_three_matching_burst_samples_confirm_cards():
    result = BurstConsensus(min_votes=3).decide(
        [play("7S", "7H"), play("7S", "7H"), play("7S", "7H"), play("7S", "7D")],
        context=legal_context(),
    )
    assert result.status == "confirmed"
    assert result.cards == ("7H", "7S")
```

- [ ] **Step 2: Verify tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_zone_lifecycle.py tests/test_live_consensus.py -q`

Expected: import failures.

- [ ] **Step 3: Implement lifecycle with monotonic timestamps, not sleeps**

```python
class ZonePhase(StrEnum):
    WAIT_CLEAR = "wait_clear"
    WAIT_ACTION = "wait_action"
    CHANGING = "changing"
    SETTLING = "settling"
    BURST_READ = "burst_read"
    VALIDATE = "validate"
    REVIEW_REQUIRED = "review_required"

@dataclass(frozen=True)
class ZoneFrameMetrics:
    monotonic_ms: int
    occupied: bool
    motion_score: float
    pass_visible: bool
    effect_visible: bool
```

Use low/high motion thresholds for hysteresis. Enter `BURST_READ` only after `settle_ms` elapsed continuously below the low threshold. On high motion/effect, return to `CHANGING` and discard collected burst samples. On `action_timeout_ms`, return `REVIEW_REQUIRED` without inventing a pass.

- [ ] **Step 4: Implement burst voting and inferred-pass policy**

Vote by normalized `(is_pass, sorted_card_multiset)`. Validate each play through existing `action_for_cards()` and deck-count constraints. A missing pass template plus an empty region and independent next-turn evidence produces `status="needs_confirmation"`, `source="inferred_pass"`; it never returns `confirmed` automatically in version one.

- [ ] **Step 5: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_zone_lifecycle.py tests/test_live_consensus.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/daguandan_bridge/live/zone_lifecycle.py src/daguandan_bridge/live/consensus.py tests/test_zone_lifecycle.py tests/test_live_consensus.py
git commit -m "feat: detect settled player actions"
```

### Task 6: Persistent capture source and live orchestrator

**Files:**
- Modify: `src/daguandan_bridge/capture_service.py`
- Create: `src/daguandan_bridge/live/orchestrator.py`
- Modify: `src/daguandan_bridge/gui/workers.py`
- Test: `tests/test_capture_service.py`
- Test: `tests/test_live_orchestrator.py`

**Interfaces:**
- Produces: `CaptureService.open_live_source(profile_name)`, `LiveCaptureSource.capture()`, `LiveOrchestrator.start()`, `ingest_frame()`, `confirm_candidate()`, `correct_latest()`, `pause()`, and `finish()`.

- [ ] **Step 1: Write failing persistent-source and orchestration tests**

```python
def test_live_source_reuses_window_and_capture_backend(service, monkeypatch):
    source = service.open_live_source("tencent_daguandan")
    source.capture()
    source.capture()
    assert source.window_lookup_count == 1
    source.close()

def test_orchestrator_commits_each_turn_once(orchestrator):
    feed_right_play_burst(orchestrator, cards=("7S", "7H"))
    feed_same_stable_frames(orchestrator, count=5)
    events = [e for e in orchestrator.events if e.event_type == "player_played"]
    assert len(events) == 1
    assert orchestrator.snapshot.current_player == "opposite"

def test_uncertain_action_pauses_but_recording_continues(orchestrator):
    feed_conflicting_burst(orchestrator)
    assert orchestrator.status == "review_required"
    assert orchestrator.recorder.frame_count > 0
```

- [ ] **Step 2: Verify tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_capture_service.py tests/test_live_orchestrator.py -q`

Expected: live source and orchestrator APIs are missing.

- [ ] **Step 3: Implement `LiveCaptureSource`**

Resolve the window and profile once, create one `LazyMssCapture`, and reuse them. Before each frame, cheaply verify the HWND still exists and client geometry still matches; on mismatch raise a typed `LiveCaptureInterrupted` so the orchestrator pauses and reopens instead of returning misaligned frames. Keep `CaptureService.capture_frame()` unchanged.

- [ ] **Step 4: Implement orchestrator as a Qt-free domain service**

The orchestrator owns one reducer, one lifecycle, one recorder, one store, and a recognition service. It accepts frames and returns immutable `LiveUpdate` objects. It does not create Qt widgets or call `QApplication.processEvents()`.

```python
@dataclass(frozen=True)
class LiveUpdate:
    status: str
    snapshot: LiveSnapshot
    event: LiveEvent | None = None
    advice: LocalAdvice | None = None
    review: ReviewRequest | None = None
```

Fast-signal checks run on each capture frame. Targeted card recognition runs only during `BURST_READ`; if a previous recognition is still running, replace the queued frame batch with the newest stable batch.

On `start()`, verify enough free disk space for the configured safety threshold before opening the recorder. On capture interruption, transition to `paused`, clear the current burst, periodically attempt to reopen `LiveCaptureSource`, and require a fresh local settling cycle after recovery. Every conflict, timeout, dropped frame, capture interruption, illegal reducer action, or advisor failure calls `create_incident()` and attaches recorder media plus state snapshots.

- [ ] **Step 5: Add a latest-only worker primitive**

Extend `gui/workers.py` with a worker whose pending capacity is one and whose stop method joins cleanly. Test it indirectly through orchestrator fakes; never invoke GUI methods from the worker thread.

- [ ] **Step 6: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_capture_service.py tests/test_live_orchestrator.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/daguandan_bridge/capture_service.py src/daguandan_bridge/live/orchestrator.py src/daguandan_bridge/gui/workers.py tests/test_capture_service.py tests/test_live_orchestrator.py
git commit -m "feat: orchestrate persistent live capture"
```

### Task 7: Early DanZero advice with idempotency and stale-result rejection

**Files:**
- Modify: `src/daguandan_bridge/live/orchestrator.py`
- Test: `tests/test_live_advice.py`

**Interfaces:**
- Consumes: `LiveReducer.to_guandan_state()` and existing `DanzeroAdvisor.recommend()`.
- Produces: `AdviceRequestKey(session_id, turn_id, state_revision)` and orchestrator `advice_ready`/`advice_stale` updates.

- [ ] **Step 1: Write failing advice timing tests**

```python
def test_advice_starts_when_reducer_predicts_self_before_timer(advisor_orchestrator):
    advisor_orchestrator.commit_left_action(("9C",))
    assert advisor_orchestrator.advisor.calls == 1
    assert advisor_orchestrator.latest_advice.visible is False
    advisor_orchestrator.ingest_fast_signal(active_player="self")
    assert advisor_orchestrator.latest_advice.visible is True

def test_corrected_state_marks_inflight_advice_stale(advisor_orchestrator):
    request = advisor_orchestrator.start_self_advice()
    advisor_orchestrator.correct_latest(cards=("10C",), is_pass=False)
    advisor_orchestrator.complete_advice(request, fake_advice())
    assert advisor_orchestrator.events[-1].event_type == "advice_stale"
```

- [ ] **Step 2: Verify tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_advice.py -q`

Expected: advice coordinator behavior is missing.

- [ ] **Step 3: Implement request keys and visibility gating**

Start advice immediately after a committed event advances `current_player` to `self`. Store in-flight requests by immutable key. A result whose state revision no longer matches is logged as stale and never displayed. A current result becomes visible after at least one self-turn corroborator (`timer_self`, self action buttons, or confirmed turn-start event) is present.

- [ ] **Step 4: Run focused tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_live_advice.py tests/test_danzero_api.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/daguandan_bridge/live/orchestrator.py tests/test_live_advice.py
git commit -m "feat: provide early idempotent DanZero advice"
```

### Task 8: Live assistant GUI and one-click correction

**Files:**
- Create: `src/daguandan_bridge/gui/live_assistant_page.py`
- Modify: `src/daguandan_bridge/gui/main_window.py`
- Test: `tests/test_live_assistant_page.py`

**Interfaces:**
- Consumes: orchestrator `LiveUpdate` snapshots and methods from Task 6.
- Produces: a main-window tab labeled `实时助手`.

- [ ] **Step 1: Write failing offscreen UI tests**

```python
def test_live_page_requires_27_cards_before_start(qtbot, fake_orchestrator):
    page = LiveAssistantPage(fake_orchestrator)
    qtbot.addWidget(page)
    page.apply_initial_recognition(initial_result(hand=("3S",)))
    assert not page.start_session_button.isEnabled()
    assert "27" in page.initialization_status.text()

def test_review_bar_confirms_candidate_with_one_click(qtbot, fake_orchestrator):
    page = LiveAssistantPage(fake_orchestrator)
    page.show_review(review_with_candidates())
    qtbot.mouseClick(page.review_candidate_buttons[0], Qt.LeftButton)
    assert fake_orchestrator.confirmed_candidate_index == 0
```

- [ ] **Step 2: Verify tests fail**

Run: `$env:QT_QPA_PLATFORM='offscreen'; ./.venv/Scripts/python.exe -m pytest tests/test_live_assistant_page.py -q`

Expected: page import failure.

- [ ] **Step 3: Implement the page without domain logic**

Build three panes: preview, timeline, and advice/status. Add manual start, pause/resume, and finish buttons. Initialization displays level, lead seat, and card count; enable start only when the orchestrator reports `initialization_ready`.

The non-modal correction bar renders one large button per candidate plus `不出` and `都不对…`. A candidate click calls `confirm_candidate(candidate_id)` once. “都不对” expands an inline minimal editor; do not open a modal dialog or interact with the game client.

- [ ] **Step 4: Add tab and shutdown wiring**

Create the page in `DaguandanBridgeWindow`, add tab `实时助手`, and call its `shutdown()` before the shared capture controller shuts down.

- [ ] **Step 5: Run UI tests**

Run: `$env:QT_QPA_PLATFORM='offscreen'; ./.venv/Scripts/python.exe -m pytest tests/test_live_assistant_page.py tests/test_capture_page.py tests/test_annotation_page.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/daguandan_bridge/gui/live_assistant_page.py src/daguandan_bridge/gui/main_window.py tests/test_live_assistant_page.py
git commit -m "feat: add live DanZero assistant page"
```

### Task 9: Replay GUI and old-versus-new comparison

**Files:**
- Modify: `src/daguandan_bridge/live/replay.py`
- Create: `src/daguandan_bridge/gui/replay_page.py`
- Modify: `src/daguandan_bridge/gui/main_window.py`
- Test: `tests/test_replay_page.py`

**Interfaces:**
- Produces: `compare_timelines(expected, actual) -> ReplayComparison` and a main-window tab labeled `对局回放`.

- [ ] **Step 1: Write failing comparison and UI tests**

```python
def test_compare_timelines_reports_changed_turn():
    result = compare_timelines(
        [event(turn_id=3, cards=("7S",))],
        [event(turn_id=3, cards=("8S",))],
    )
    assert result.changed[0].turn_id == 3

def test_replay_page_can_load_session(qtbot, recorded_session):
    page = ReplayPage(recorded_session.parent)
    qtbot.addWidget(page)
    page.select_session(recorded_session)
    assert page.play_button.isEnabled()
    assert page.session_summary.text()
```

- [ ] **Step 2: Verify tests fail**

Run: `$env:QT_QPA_PLATFORM='offscreen'; ./.venv/Scripts/python.exe -m pytest tests/test_live_replay.py tests/test_replay_page.py -q`

Expected: comparison and page APIs are missing.

- [ ] **Step 3: Implement timeline comparison**

Match formal player actions by `turn_id`; classify identical, missing, added, and changed. Report confidence and latency deltas separately so a score change does not become a semantic card change.

- [ ] **Step 4: Implement replay page**

Provide session dropdown, play/pause, single-frame, speed selection, incident jump, `状态重放`, `重新视觉识别`, and `导出大模型诊断包`. Decode frames on a worker thread; deliver copied `QImage` objects to the GUI thread.

- [ ] **Step 5: Add tab and run UI tests**

Run: `$env:QT_QPA_PLATFORM='offscreen'; ./.venv/Scripts/python.exe -m pytest tests/test_live_replay.py tests/test_replay_page.py tests/test_capture_page.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/daguandan_bridge/live/replay.py src/daguandan_bridge/gui/replay_page.py src/daguandan_bridge/gui/main_window.py tests/test_live_replay.py tests/test_replay_page.py
git commit -m "feat: add session replay diagnostics"
```

### Task 10: End-to-end regression, performance evidence, and documentation

**Files:**
- Create: `tests/fixtures/live_sessions/README.md`
- Create: `tests/test_live_end_to_end.py`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes all prior tasks.
- Produces a reproducible synthetic golden session and documented live/replay workflow.

- [ ] **Step 1: Add a synthetic golden event/observation fixture**

Create a compact JSON fixture for:

```text
右家首发 7S 7H → 对家不出 → 左家 9C 9D → 自己回合 →
DanZero 建议 JC JD → 自己实际出 JC JD → 右家回合
```

Include a long-effect sequence (`CHANGING`, brief false settle, `CHANGING`, final settle) before the left-player action.

- [ ] **Step 2: Write the end-to-end test**

```python
def test_golden_session_produces_exact_timeline(golden_observations, fake_advisor, tmp_path):
    runner = build_test_orchestrator(tmp_path, fake_advisor)
    for observation in golden_observations:
        runner.ingest_observation(observation)
    assert semantic_events(runner.events) == expected_semantic_events()
    assert fake_advisor.calls == 1
    assert runner.metrics.advice_visible_latency_ms <= 3000
    assert "右家出牌" in (runner.store.directory / "timeline.md").read_text("utf-8")
```

- [ ] **Step 3: Run full verification**

Run: `$env:QT_QPA_PLATFORM='offscreen'; $env:PYTHONPATH='src'; ./.venv/Scripts/python.exe -m pytest -q`

Expected: all old and new tests pass.

Run: `./.venv/Scripts/python.exe -m compileall -q src tests`

Expected: exit code 0.

- [ ] **Step 4: Document operation and data locations**

Add README sections describing pre-first-play start, exact 27-card gate, start/pause/finish, one-click correction, session directory contents, replay modes, and how to submit `llm_report.md` plus `contact_sheet.png` for diagnosis. Add `.superpowers/` and generated `data/profiles/*/sessions/` runtime sessions to `.gitignore`, while keeping test fixtures tracked.

- [ ] **Step 5: Commit**

```powershell
git add tests/fixtures/live_sessions/README.md tests/test_live_end_to_end.py README.md .gitignore
git commit -m "test: verify live assistant workflow"
```

- [ ] **Step 6: Record real-machine follow-up evidence**

Run one manually started real game with logging enabled. Preserve the session outside Git, replay it in both modes, and record P50/P95 stage timings in `manifest.json`. If P95 exceeds 3 seconds or any formal action differs from manual truth, keep the feature marked experimental and convert the failing clip into an incident fixture before changing thresholds.
