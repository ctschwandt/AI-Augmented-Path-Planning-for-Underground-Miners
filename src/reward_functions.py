# reward_functions.py

import numpy as np
from src.utils import *

##==============================================================
## Reward functions agent in GridWorldEnv may use
##==============================================================
# === Constants ===
LOWER_BOUND = -1
UPPER_BOUND = 9.0
BATTERY_THRESHOLD = 10
BASE_GOAL_REWARD = 400.0  # Reference reward for a 20x20 grid

# === Base subreward values ===
base_invalid_penalty = -0.75
base_battery_penalty = -100
base_revisit_penalty = -0.25
base_time_penalty = -0.04
min_progress_penalty = -1
base_progress_weight = 1.0

def get_reward_a(env, new_pos):
    subrewards = {}

    # === Grid-based scaling ===
    goal_reward = env.n_rows * env.n_cols
    scale = goal_reward / BASE_GOAL_REWARD

    # === Initialize subrewards ===
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = goal_reward
        subrewards["invalid_penalty"] = 0
        subrewards["battery_penalty"] = 0
        #subrewards["progress_shaping"] = 0
        subrewards["revisit_penalty"] = 0
        subrewards["time_penalty"] = 0
    else:
        subrewards["goal_reward"] = 0
        subrewards["invalid_penalty"] = scale * (base_invalid_penalty if not env.can_move_to(new_pos) else 0)
        subrewards["battery_penalty"] = scale * (base_battery_penalty if env.current_battery_level <= BATTERY_THRESHOLD else 0)

        '''
        prev_pos = env.agent_pos
        prev_dist = min(chebyshev_distances(prev_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=False))
        new_dist = min(chebyshev_distances(new_pos, env.goal_positions, env.n_cols, env.n_rows, normalize=False))
        progress = prev_dist - new_dist
        subrewards["progress_shaping"] = scale * base_progress_weight * progress
        '''

        subrewards["revisit_penalty"] = scale * (base_revisit_penalty if new_pos in env.visited else 0)
        subrewards["time_penalty"] = scale * base_time_penalty

    # === Compute raw and normalized reward ===
    raw_reward = sum(subrewards.values())
    min_reward = scale * (
        base_invalid_penalty +
        base_battery_penalty +
        base_revisit_penalty +
        base_time_penalty +
        min_progress_penalty
    )
    max_reward = goal_reward

    # normalize total reward and subrewards
    def normalize(val):
        norm = ((val - min_reward) / (max_reward - min_reward)) * (UPPER_BOUND - LOWER_BOUND) + LOWER_BOUND
        return np.clip(norm, LOWER_BOUND, UPPER_BOUND)

    norm_reward = normalize(raw_reward)
    #norm_subrewards = {k: normalize(v) for k, v in subrewards.items()}
    
    return norm_reward, subrewards

def get_reward_b(env, new_pos):
    b_inval_pen = -0.75
    b_rev_pen = -0.21
    b_t_pen = -0.04
    b_bat_pen = -100

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": env.n_rows * env.n_cols,
            "invalid_penalty": 0,
            "revisit_penalty": 0,
            "time_penalty": 0,
            "battery_penalty": 0
        }
    else:
        inval_pen = b_inval_pen if not env.can_move_to(new_pos) else 0
        rev_pen = b_rev_pen if new_pos in env.visited else 0
        t_pen = b_t_pen
        bat_pen = b_bat_pen if env.current_battery_level <= 10 else 0
        subrewards = {
            "goal_reward": 0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "time_penalty": t_pen,
            "battery_penalty": bat_pen
            }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_c(env, new_pos):
    '''
    smaller reward scale
    '''
    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": 1.0,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "time_penalty": 0.0,
            "battery_penalty": 0.0
        }
    else:
        inval_pen = -0.75 if not env.can_move_to(new_pos) else 0.0
        rev_pen = -0.25 if new_pos in env.visited else 0.0
        t_pen = -0.04
        bat_pen = -1.0 if env.current_battery_level <= 10 else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "time_penalty": t_pen,
            "battery_penalty": bat_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_d(env, new_pos):
    '''
    reward c but with distance penalty, and no time penalty
    '''
    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": 1.0,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        inval_pen = -0.75 if not env.can_move_to(new_pos) else 0.0
        rev_pen = -0.25 if new_pos in env.visited else 0.0
        bat_pen = -1.0 if env.current_battery_level <= 10 else 0.0
        dist_pen = -2.0 * env._compute_min_distance_to_goal() # product of normalized euclidean distance to closest goal

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_e(env, new_pos):
    """
    sigmoid battery penalty
    """
    # Tunable weights
    w_goal = 1.0
    w_invalid = 0.75
    w_revisit = 0.25
    w_dist = 2.0
    w_battery = 10
    k_soft = 6.0  # sigmoid sharpness
    battery_threshold = 10

    def sigmoid(x):
        return 1 / (1 + np.exp(-k_soft * x))

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": w_goal,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        # distance penalty (0–1 normalized)
        dist = env._compute_min_distance_to_goal()
        dist_pen = -w_dist * dist

        # sigmoid battery_penalty
        b = env.current_battery_level
        bat_pen = -w_battery * sigmoid((battery_threshold - b) / battery_threshold)

        inval_pen = -w_invalid if not env.can_move_to(new_pos) else 0.0
        rev_pen = -w_revisit if new_pos in env.visited else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_e2(env, new_pos):
    """
    sigmoid battery penalty
    """
    # Tunable weights
    w_goal = 1.0
    w_invalid = 0.75
    w_revisit = 0.25
    w_dist = 2.0
    w_battery = 5 # was 10
    k_soft = 6.0  # sigmoid sharpness
    battery_threshold = 10

    def sigmoid(x):
        return 1 / (1 + np.exp(-k_soft * x))

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": w_goal,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        # distance penalty (0–1 normalized)
        dist = env._compute_min_distance_to_goal()
        dist_pen = -w_dist * dist

        # sigmoid battery_penalty
        b = env.current_battery_level
        bat_pen = -w_battery * sigmoid((battery_threshold - b) / battery_threshold)

        inval_pen = -w_invalid if not env.can_move_to(new_pos) else 0.0
        rev_pen = -w_revisit if new_pos in env.visited else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_e3(env, new_pos):
    """
    sigmoid battery penalty
    """
    # Tunable weights
    w_goal = 1.0
    w_invalid = 0.75
    w_revisit = 0.25
    w_dist = 2.0
    w_battery = 2.5
    k_soft = 6.0  # sigmoid sharpness
    battery_threshold = 10

    def sigmoid(x):
        return 1 / (1 + np.exp(-k_soft * x))

    if new_pos in env.goal_positions:
        subrewards = {
            "goal_reward": w_goal,
            "invalid_penalty": 0.0,
            "revisit_penalty": 0.0,
            "battery_penalty": 0.0,
            "distance_penalty": 0.0
        }
    else:
        # distance penalty (0–1 normalized)
        dist = env._compute_min_distance_to_goal()
        dist_pen = -w_dist * dist

        # sigmoid battery_penalty
        b = env.current_battery_level
        bat_pen = -w_battery * sigmoid((battery_threshold - b) / battery_threshold)

        inval_pen = -w_invalid if not env.can_move_to(new_pos) else 0.0
        rev_pen = -w_revisit if new_pos in env.visited else 0.0

        subrewards = {
            "goal_reward": 0.0,
            "invalid_penalty": inval_pen,
            "revisit_penalty": rev_pen,
            "battery_penalty": bat_pen,
            "distance_penalty": dist_pen
        }

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_6(env, old_pos):
    """
    A proactive, path-informed reward function.

    Args:
        env: The environment object, which must contain the pathfinder.
        old_pos (tuple): The agent's position *before* the action.
    """
    # --- Tunable Weights ---
    w_goal = 1.0
    w_invalid = -0.75
    w_revisit = -0.25
    w_path_progress = 1.0
    w_dangerous = -1.0 # Penalty for moving to a state from which the pathfinder, with its knowledge of costs, can no longer find a viable or reasonable path to the goal
    path_prog_norm = 101
    w_time = -0.05

    subrewards = {
        "goal_reward": 0.0,
        "invalid_penalty": 0.0,
        "revisit_penalty": 0.0,
        "path_progress_reward": 0.0,
        "time_penalty": 0.0
    }
    
    # Case 1: Goal is reached
    new_pos = tuple(env.agent_pos)
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = w_goal
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if old_pos == new_pos:
        subrewards["invalid_penalty"] = w_invalid
        total_reward = sum(subrewards.values())
        return total_reward, subrewards

    if new_pos in env.visited:
        subrewards["revisit_penalty"] = w_revisit
    
    cost_from_old_pos = env.get_path_cost(old_pos)
    cost_from_new_pos = env.get_path_cost(new_pos)

    if cost_from_new_pos == float('inf'):
        if cost_from_old_pos == float('inf'):
            path_reward = 0.0
        else:
            path_reward = w_dangerous
    else:
        if cost_from_old_pos == float('inf'):
             path_reward = -w_dangerous
        else:
            path_reward = w_path_progress * (cost_from_old_pos - cost_from_new_pos) # scaled to [-1, 1] # try larger scale, or lower scale of other things?

    subrewards["path_progress_reward"] = path_reward if subrewards["revisit_penalty"] == 0 else 0

    subrewards["time_penalty"] = w_time

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_pathlen(env, old_pos):
    """
    Path-length-based exponential penalty reward function.
    
    Penalizes based on the number of steps remaining to the goal
    using the formula: - (1 - exp(-α * path_len))
    
    Args:
        env: The environment with a pathfinder.
        old_pos (tuple): Agent's position before the action.
    """
    # --- Tunable parameters ---
    w_goal = 1.0
    w_invalid = -0.75
    w_revisit = -0.25
    w_battery = -1 # incorporate later
    k = 3
    #diagonal_length = 1.4 * (env.n_rows + env.n_cols)
    #alpha = np.log(k) / diagonal_length # base e ln
    #alpha = 0.03
    target_penalty_at_halfway = 0.43 # try 0.6
    d_half = 0.5 * max(env.n_rows, env.n_cols)
    alpha = -np.log(1 - target_penalty_at_halfway) / d_half
    # try no invalid or revisit penalty,
    # just battery and pathlen penalty

    subrewards = {
        "goal_reward": 0.0,
        "invalid_penalty": 0.0,
        "revisit_penalty": 0.0,
        "path_length_penalty": 0.0,
    }

    new_pos = tuple(env.agent_pos)

    # Case 1: Reached goal
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = w_goal
        return sum(subrewards.values()), subrewards

    # Case 2: Invalid move (into wall, obstacle, etc.)
    if not env.can_move_to(new_pos):
        subrewards["invalid_penalty"] = w_invalid
        return sum(subrewards.values()), subrewards

    # Case 3: Revisit penalty
    if new_pos in env.visited:
        subrewards["revisit_penalty"] = w_revisit

    # Case 4: Exponential penalty based on path length
    path = env.pathfinder.get_path_to_goal()
    path_len = len(path) - 1  # Exclude current position
    penalty = -(1 - np.exp(-alpha * path_len))  # In range [-1, 0]
    
    subrewards["path_length_penalty"] = penalty

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def get_reward_follow_dstar(env, old_pos):
    """
    Rewards the agent for following the path provided by D* Lite.
    Penalizes deviation from the optimal path.
    
    Args:
        env: Environment with pathfinder and agent info
        old_pos (tuple): Agent position before the action
        
    Returns:
        total_reward: float
        subrewards: dict with detailed components
    """
    # --- Tunable parameters ---
    w_goal = 1.0
    w_wall = -0.75
    w_revisit = -0.25
    w_path_follow = 0.3     # reward for correct next step
    w_path_deviation = -0.2 # penalty for deviating from path
    # try without path deviation penalty
    subrewards = {
        "goal_reward": 0.0,
        "wall_penalty": 0.0,
        "revisit_penalty": 0.0,
        "path_follow_reward": 0.0,
    }

    new_pos = tuple(env.agent_pos)

    # Reached goal
    if new_pos in env.goal_positions:
        subrewards["goal_reward"] = w_goal
        return sum(subrewards.values()), subrewards

    # Wall collision
    if not env.can_move_to(new_pos):
        subrewards["wall_penalty"] = w_wall
        return sum(subrewards.values()), subrewards

    # Revisit penalty
    if new_pos in env.visited:
        subrewards["revisit_penalty"] = w_revisit

    # Path-follow reward
    path = env.pathfinder.get_path_to_goal(start=old_pos)
    if path and len(path) > 1:
        expected_next = path[1]  # the next step D* suggests
        if new_pos == expected_next:
            subrewards["path_follow_reward"] = w_path_follow
        else:
            subrewards["path_follow_reward"] = w_path_deviation

    total_reward = sum(subrewards.values())
    return total_reward, subrewards

def compute_reward(env, old_pos, reward_fn):
    old_pos = tuple(old_pos)
    return reward_fn(env, old_pos)
