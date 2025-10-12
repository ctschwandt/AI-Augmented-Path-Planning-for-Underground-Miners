from __future__ import annotations

import numpy as np
from gym.spaces import Box

from .base import ObservationBuilder


class FlatObservationBuilder(ObservationBuilder):
    """Builds the original flat observation vector used by MLP policies."""

    def get_observation_space(self, env) -> Box:
        obs_dim = 8 + 2 + 1 + 1 + env.n_sensors
        return Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def get_observation(self, env):
        r, c = env.agent_pos
        neighbors = [
            (r - 1, c), (r - 1, c + 1), (r, c + 1), (r + 1, c + 1),
            (r + 1, c), (r + 1, c - 1), (r, c - 1), (r - 1, c - 1)
        ]

        def is_blocked(pos):
            return 1.0 if not env.can_move_to(pos) else 0.0

        blocked_flags = np.array([is_blocked(p) for p in neighbors], dtype=np.float32)
        norm_pos = np.array([r / (env.n_rows - 1), c / (env.n_cols - 1)], dtype=np.float32)
        last_action = env.last_action / 7.0 if env.last_action >= 0 else 0.0

        dist_to_goal = env._compute_min_distance_to_goal()

        battery_levels = np.array(
            [env.sensor_batteries.get(pos, 0.0) / 100.0 for pos in env.sensor_batteries],
            dtype=np.float32,
        )

        return np.concatenate([blocked_flags, norm_pos, [last_action], [dist_to_goal], battery_levels])
