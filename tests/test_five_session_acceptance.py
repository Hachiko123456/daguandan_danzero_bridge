from pathlib import Path
import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_five_session_acceptance.py'


def _module():
    spec = importlib.util.spec_from_file_location('run_five_session_acceptance', SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_acceptance_set_contains_five_distinct_sessions():
    module = _module()
    assert len(module.FIVE_SESSIONS) == 5
    assert len(set(module.FIVE_SESSIONS)) == 5
    assert 'game_20260829_114157_a940ae' in module.FIVE_SESSIONS


def test_parser_defaults_to_two_bounded_workers():
    module = _module()
    args = module.build_parser().parse_args([])
    assert args.workers == 2
    assert args.run_id == 'architecture-acceptance-20260921'
