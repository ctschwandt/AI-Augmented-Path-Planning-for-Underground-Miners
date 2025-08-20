# wrappers.py

import numpy as np
import gym
from gym import ObservationWrapper

class TimeStackObservation(ObservationWrapper):
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        old_shape = env.observation_space.shape  # (C, H, W)
        self.num_frames = num_frames
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(num_frames, *old_shape),
            dtype=env.observation_space.dtype,
        )
        self.frames = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.num_frames
        return self.observation(np.array(obs)), info

    def observation(self, observation):
        self.frames.append(observation)
        self.frames = self.frames[-self.num_frames:]
        return np.stack(self.frames, axis=0)
