"""guandan-rlcard: an open Guandan (掼蛋) environment built on RLCard.

Importing this package registers the ``'guandan'`` environment with the
RLCard registry, so both forms work:

    import daguandan_bridge.danzero._vendor.guandan_rlcard
    env = daguandan_bridge.danzero._vendor.guandan_rlcard.make({'seed': 42})

    import rlcard, daguandan_bridge.danzero._vendor.guandan_rlcard
    env = rlcard.make('guandan')
"""

from rlcard.envs.registration import registry

__version__ = '0.1.0'


def _register():
    if 'guandan' not in registry.env_specs:
        registry.register(
            env_id='guandan',
            entry_point='daguandan_bridge.danzero._vendor.guandan_rlcard.envs.guandan_env:GuandanEnv',
        )


_register()

from daguandan_bridge.danzero._vendor.guandan_rlcard.envs.guandan_env import GuandanEnv  # noqa: E402


def make(config=None):
    """Create a GuandanEnv with the given config dict (all keys optional)."""
    return GuandanEnv(config)
