from daguandan_bridge.config import PROFILES_ROOT
from daguandan_bridge.profiles import get_profile_paths, load_profile_config


def test_tencent_daguandan_profile_and_templates_are_packaged():
    paths = get_profile_paths(PROFILES_ROOT, "tencent_daguandan")
    config = load_profile_config(paths)

    assert config.name == "tencent_daguandan"
    assert config.recording_interval_sec == 1.5
    assert len(list(paths.templates_dir.rglob("*"))) >= 93
