# DStarFallbackWrapper

import gym
import numpy as np
import torch as th
from src.constants import MOVE_TO_ACTION_MAP

class DStarFallbackWrapper(gym.Wrapper):
    """
    A wrapper that implements a hybrid control system with a D* Lite fallback.
    This wrapper MUST be applied LAST, after all observation wrappers.
    """
    def __init__(self, env, model, confidence_threshold=0.75):
        super().__init__(env)
        
        # Store a reference to the unwrapped, base environment. This allows us to
        # access .pathfinder and .agent_pos regardless of other wrappers.
        self.base_env = env.unwrapped 
        if not hasattr(self.base_env, 'pathfinder'):
             raise AttributeError("The unwrapped environment must have a 'pathfinder' attribute.")
        
        self.model = model
        self.confidence_threshold = confidence_threshold
        # This will store the observation AFTER it's processed by all inner wrappers.
        self.last_obs = None

    def reset(self, **kwargs):
        # Call the reset method of the wrapped environment stack (e.g., TimeStackObservation).
        # The 'obs' it returns is the final, fully-processed observation.
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs # Store the correctly-shaped observation.
        return self.last_obs, info

    def step(self, action_from_agent):
        expert_action = None

        with th.no_grad():
            # self.last_obs now has the correct shape (e.g., 4D for attention)
            # because it was set by the return value of the wrapped env's step/reset.
            obs_tensor, _ = self.model.policy.obs_to_tensor(self.last_obs)
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]

        max_prob = np.max(probs)
        
        if max_prob >= self.confidence_threshold:
            final_action = action_from_agent
        else:
            expert_action = self._get_dstar_action()
            final_action = expert_action if expert_action is not None else action_from_agent

        next_obs, reward, terminated, truncated, info = self.env.step(final_action)

        if isinstance(info, dict):
            # Single env case
            info['final_action'] = final_action
        else:
            # VecEnv case: info is a list of dicts, one per env
            for env_info in info:
                env_info['final_action'] = final_action
            
        self.last_obs = next_obs
        info['confidence'] = float(max_prob)
        info['used_fallback'] = (expert_action is not None and final_action == expert_action)
        '''
        print(f"[FallbackWrapper] Agent action: {action_from_agent}, "
              f"Final action: {final_action}, "
              f"Confidence: {max_prob:.3f}, "
              f"Used fallback: {final_action != action_from_agent}")
        '''
        
        return next_obs, reward, terminated, truncated, info

    def _get_dstar_action(self):
        # Use the stored unwrapped env reference to access base attributes.
        path = self.base_env.pathfinder.get_path_to_goal()
        if not path or len(path) < 2:
            return None

        current_pos_rc = tuple(self.base_env.agent_pos)
        next_pos_xy = path[1]
        move_rc = (next_pos_xy[1] - current_pos_rc[0], next_pos_xy[0] - current_pos_rc[1])
        
        return MOVE_TO_ACTION_MAP.get(move_rc)

    def set_model(self, model):
        self.model = model
