from __future__ import annotations

import json
from pathlib import Path
from time import monotonic_ns
from types import SimpleNamespace

import pytest

from daguandan_bridge.application.live_v2_session_runtime import LiveV2SessionRuntime
from daguandan_bridge.infrastructure import live_v2_composition as composition
from daguandan_bridge.infrastructure.live_session import DefaultLiveSessionFactory
from daguandan_bridge.infrastructure.live_v2_rule_session import ProductionRuleSession
from daguandan_bridge.live.recorder import InMemorySessionRecorder
from daguandan_bridge.live.session_store import LiveSessionStore, read_json_lines


HAND = tuple(
    f"{rank}{suit}"
    for rank in ("2", "3", "4", "5", "6", "7")
    for suit in "SHCD"
) + ("8S", "8H", "8C")


class _Source:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Capture:
    def __init__(self, root: Path) -> None:
        self.profiles_root = root
        self.profile = root / "tencent_daguandan"
        self.source = _Source()

    def load_profile(self, _name):
        return SimpleNamespace(
            paths=SimpleNamespace(
                profile_config_path=self.profile / "profile.json",
                templates_config_path=self.profile / "templates_config.json",
            ),
            config=SimpleNamespace(base_size=(64, 32)),
        )

    def open_live_source(self, _name):
        return self.source


class _VisionRuntime:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self, *, timeout=10.0):
        self.started = True

    def submit(self, *_args, **_kwargs):
        return ()

    def drain_results(self):
        return ()

    def close(self, *, timeout=5.0):
        self.closed = True


class _AdviceRuntime(_VisionRuntime):
    pass


def _factory(tmp_path: Path, monkeypatch):
    profile = tmp_path / "tencent_daguandan"
    profile.mkdir()
    (profile / "profile.json").write_text(
        json.dumps({"recording_mode": "none", "save_session_data": False}),
        encoding="utf-8",
    )
    (profile / "templates_config.json").write_text("{}", encoding="utf-8")
    visions: list[_VisionRuntime] = []
    advisers: list[_AdviceRuntime] = []

    def vision(*_args, **_kwargs):
        value = _VisionRuntime()
        visions.append(value)
        return value

    def advice(*_args, **_kwargs):
        value = _AdviceRuntime()
        advisers.append(value)
        return value

    monkeypatch.setattr(composition, "build_live_v2_vision_runtime", vision)
    monkeypatch.setattr(composition, "create_live_v2_advice_runtime", advice)
    capture = _Capture(tmp_path)
    factory = DefaultLiveSessionFactory(
        capture,
        recognizer=object(),
        advisor=None,
        profile_name=profile.name,
    )
    return factory, capture, visions, advisers


def test_default_factory_owns_one_durable_start_and_live_v2_lifecycle(
    tmp_path, monkeypatch
):
    starts: list[str] = []
    real_start = LiveSessionStore.start

    def counted_start(store, manifest):
        starts.append(store.session_id)
        return real_start(store, manifest)

    monkeypatch.setattr(LiveSessionStore, "start", counted_start)
    factory, capture, visions, advisers = _factory(tmp_path, monkeypatch)

    construction = factory.start_session(
        round_level="2",
        hand=HAND,
        lead_player="right",
        recognition_strategy="two_valid_streak",
    )
    runtime = construction.orchestrator

    assert isinstance(runtime, LiveV2SessionRuntime)
    assert isinstance(runtime.rule_session, ProductionRuleSession)
    assert isinstance(runtime.recorder, InMemorySessionRecorder)
    assert runtime.store.persistence_enabled is True
    assert len(starts) == 1
    assert read_json_lines(runtime.store.timeline_path)[0]["event_type"] == "initial_state_confirmed"

    first = runtime.bind_capture_generation(1)
    assert first.capture_generation == 1
    assert visions[-1].started and advisers[-1].started
    assert runtime.pause().status == "paused"
    assert runtime.resume(monotonic_ms=10).status == "running"
    old_vision, old_advice = visions[-1], advisers[-1]
    rebound = runtime.bind_capture_generation(2)
    assert rebound.capture_generation == 2
    assert old_vision.closed and old_advice.closed

    sealed = runtime.finish()
    assert sealed.status == "sealed"
    assert visions[-1].closed and advisers[-1].closed
    assert len(starts) == 1
    assert json.loads(runtime.store.manifest_path.read_text("utf-8"))["status"] == "sealed"
    capture.source.close()


def test_waiting_lead_opening_seed_uses_same_runtime_and_durable_history(
    tmp_path, monkeypatch
):
    factory, capture, _visions, _advisers = _factory(tmp_path, monkeypatch)
    construction = factory.start_session(
        round_level="2",
        hand=HAND,
        lead_player=None,
        recognition_strategy="two_valid_streak",
    )
    runtime = construction.orchestrator

    assert construction.initial_update.status == "waiting_lead"
    runtime.bind_capture_generation(1)
    action_ms = monotonic_ns() // 1_000_000
    update = runtime.bootstrap_opening_action(
        actor="right",
        cards=("3D",),
        expected_next_player="opposite",
        monotonic_ms=action_ms,
        confidence=1.0,
        source="factory-test",
    )

    assert update.status == "running"
    assert update.snapshot.current_player == "opposite"
    assert update.snapshot.play_history[-1].cards == ("3D",)
    assert [item["event_type"] for item in read_json_lines(runtime.store.timeline_path)] == [
        "initial_state_confirmed",
        "lead_player_confirmed",
        "player_played",
    ]
    runtime.finish()
    capture.source.close()


def test_default_factory_source_has_no_legacy_orchestrator_construction() -> None:
    source = Path("src/daguandan_bridge/infrastructure/live_session.py").read_text(
        encoding="utf-8"
    )
    assert "LiveOrchestrator" not in source
    assert "LiveReducer(" not in source
    assert "build_production_live_v2_runtime" in source


def test_composition_keeps_real_store_and_unstarted_store_fails_closed(
    tmp_path, monkeypatch
) -> None:
    profile = tmp_path / "tencent_daguandan"
    profile.mkdir()
    store = LiveSessionStore(tmp_path, profile.name)
    recorder = InMemorySessionRecorder(store.directory)
    monkeypatch.setattr(composition, "build_live_v2_vision_runtime", lambda *a, **k: _VisionRuntime())
    monkeypatch.setattr(composition, "create_live_v2_advice_runtime", lambda *a, **k: _AdviceRuntime())
    with pytest.raises(RuntimeError, match="already-started store"):
        composition.build_production_live_v2_runtime(
            store=store, recorder=recorder, recognizer=object(),
            profiles_root=tmp_path, profile_name=profile.name,
            advisor_backend="fabledan", on_update=None,
        )


def test_real_store_rejects_duplicate_start(tmp_path) -> None:
    store = LiveSessionStore(tmp_path, "tencent_daguandan")
    store.start({"runtime": "test"})
    with pytest.raises(RuntimeError, match="对局存储已经启动"):
        store.start({"runtime": "test"})


@pytest.mark.parametrize("backend", ("fabledan", "danzero"))
def test_selected_advisor_matches_manifest_and_worker_config(
    tmp_path, monkeypatch, backend
) -> None:
    profile = tmp_path / "tencent_daguandan"
    profile.mkdir()
    (profile / "profile.json").write_text(
        json.dumps({"recording_mode": "none", "advisor_strategy": backend}),
        encoding="utf-8",
    )
    (profile / "templates_config.json").write_text("{}", encoding="utf-8")
    configs = []
    monkeypatch.setattr(
        composition, "build_live_v2_vision_runtime",
        lambda *args, **kwargs: _VisionRuntime(),
    )
    monkeypatch.setattr(
        composition, "create_live_v2_advice_runtime",
        lambda config, version: configs.append(config) or _AdviceRuntime(),
    )

    class SelectedAdvisor:
        strategy_id = backend
        def audit_info(self):
            return {"backend": "test-runtime", "status": "loaded"}

    capture = _Capture(tmp_path)
    factory = DefaultLiveSessionFactory(
        capture, recognizer=object(), advisor=SelectedAdvisor(),
        profile_name=profile.name,
    )
    made = factory.start_session(
        round_level="2", hand=HAND, lead_player="right",
        recognition_strategy="two_valid_streak",
    )
    made.orchestrator.bind_capture_generation(1)
    manifest = json.loads(made.orchestrator.store.manifest_path.read_text("utf-8"))
    assert factory.advisor_strategy == backend
    assert manifest["advisor"]["strategy_id"] == backend
    assert manifest["advisor"]["advisor_backend"] == backend
    assert configs[0].advisor_backend == backend
    assert configs[0].fabledan_runtime_policy == "model_required"
    made.orchestrator.finish()
    capture.source.close()


def test_unknown_advisor_strategy_fails_closed(tmp_path) -> None:
    class UnknownAdvisor:
        strategy_id = "mystery"

    with pytest.raises(ValueError, match="不支持的建议模型"):
        DefaultLiveSessionFactory(
            _Capture(tmp_path), recognizer=object(), advisor=UnknownAdvisor(),
            profile_name="tencent_daguandan",
        )
