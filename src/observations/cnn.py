from __future__ import annotations

import numpy as np
from gym.spaces import Box

from .base import ObservationBuilder
from src.constants import OBSTACLE, BASE_STATION


class CNNObservationBuilder(ObservationBuilder):
    """Builds 3D tensor observations for convolutional policies."""

    is_image_based: bool = True

    def __init__(self, num_channels: int = 6):
        self.num_channels = num_channels

    def get_observation_space(self, env) -> Box:
        return Box(
            low=0.0,
            high=1.0,
            shape=(self.num_channels, env.n_rows, env.n_cols),
            dtype=np.float32,
        )

    def get_observation(self, env):
        obs = np.zeros((self.num_channels, env.n_rows, env.n_cols), dtype=np.float32)

        # Channel 0: agent position
        agent_r, agent_c = env.agent_pos
        obs[0, agent_r, agent_c] = 1.0

        # Channel 1: blocked cells (obstacles, base stations, ...)
        for r in range(env.n_rows):
            for c in range(env.n_cols):
                if env.static_grid[r, c] in (OBSTACLE, BASE_STATION):
                    obs[1, r, c] = 1.0

        # Channel 2: sensor presence, Channel 3: sensor battery level
        obs[3, :, :] = -1.0  # default to -1 everywhere to denote absence of a sensor
        for (sensor_r, sensor_c), battery in env.sensor_batteries.items():
            obs[2, sensor_r, sensor_c] = 1.0
            obs[3, sensor_r, sensor_c] = battery / 100.0

        # Channel 4: goal positions
        for goal_r, goal_c in env.goal_positions:
            obs[4, goal_r, goal_c] = 1.0

        # Channel 5: miner positions
        for miner_r, miner_c in env.miners:
            obs[5, miner_r, miner_c] = 1.0

        return obs
