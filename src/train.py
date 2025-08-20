# train.py
from src.grid_env import *
import os
import numpy as np
import gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from src.constants import *
from src.plot_metrics import *
from src.cnn_feature_extractor import CustomGridCNNWrapper, GridCNNExtractor, AgentFeatureMatrixWrapper, FeatureMatrixCNNExtractor
import src.reward_functions as reward_functions
#import gymnasium as gym
from src.attention import AttentionCNNExtractor
from src.wrappers import TimeStackObservation
from src.DStarFallbackWrapper import *
import re
from src.PPOFallback import *

##==============================================================
## Unified PPO Training Function
##==============================================================
def train_PPO_model(reward_fn,
                    grid_file: str,
                    timesteps: int,
                    folder_name: str,
                    use_hybrid_control: bool = False,
                    confidence_threshold: float = 0.75,
                    reset_kwargs: dict = {},
                    arch: str | None = None,
                    features_dim: int = 128,
                    battery_truncation=False,
                    is_att: bool = False,
                    num_frames: int = 8
                    ):

    # --- Step 1: Create the foundational environment ---
    base_env = GridWorldEnv(
        reward_fn=reward_fn,
        grid_file=grid_file,
        is_cnn=(arch is not None or is_att),
        reset_kwargs=reset_kwargs,
        battery_truncation=battery_truncation
    )

    # --- Step 2: Apply observation wrappers ---
    obs_wrapped_env = base_env
    if (arch is not None) or is_att:
        obs_wrapped_env = CustomGridCNNWrapper(obs_wrapped_env)
    if is_att:
        obs_wrapped_env = TimeStackObservation(obs_wrapped_env, num_frames=num_frames)

    # --- Step 3: Define the policy and model kwargs ---
    policy_kwargs = None
    policy_name = "MlpPolicy"

    if is_att:
        print("INFO: Using AttentionCNNExtractor for 4D time-stacked observations.")
        policy_kwargs = {
            "features_extractor_class": AttentionCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "grid_file": grid_file,
                "temporal_len": num_frames
            },
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        }
    elif arch is not None:
        print(f"INFO: Using GridCNNExtractor (backbone: {arch}) for 3D image observations.")
        policy_kwargs = {
            "features_extractor_class": GridCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "grid_file": grid_file,
                "backbone": arch.lower()
            },
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        }
    else:
        print("INFO: Using default MLP for flat vector observations.")

    # --- Step 4: Choose environment and model based on hybrid flag ---
    if use_hybrid_control:
        print(f"--- Applying D* Lite Fallback Wrapper (OUTERMOST) ---")
        final_env = DStarFallbackWrapper(obs_wrapped_env, None, confidence_threshold)
        vec_env = DummyVecEnv([lambda: final_env])
        model = PPOFallback(
            policy=policy_name,
            env=vec_env,
            ent_coef=0.1,
            gae_lambda=0.90,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            clip_range_vf=0.5,
            tensorboard_log=os.path.join("saved_experiments", folder_name),
            verbose=1,
            policy_kwargs=policy_kwargs
        )
        final_env.set_model(model)  # So the wrapper has access to the model for confidence
    else:
        vec_env = DummyVecEnv([lambda: obs_wrapped_env])
        model = PPO(
            policy=policy_name,
            env=vec_env,
            ent_coef=0.1,
            gae_lambda=0.90,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            clip_range_vf=0.5,
            tensorboard_log=os.path.join("saved_experiments", folder_name),
            verbose=1,
            policy_kwargs=policy_kwargs
        )

    # --- Step 5: Train ---
    print("Starting model training...")
    callback = CustomTensorboardCallback(verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)

    # --- Step 6: Save ---
    log_dir = callback.logger.dir
    model_save_path = os.path.join(log_dir, "model.zip")
    model.save(model_save_path)
    print(f"\nPPO training complete. Logs and model stored to {log_dir}")

    # --- Step 7: Metrics ---
    num_points = 50
    plots = plot_all_metrics(log_dir=log_dir, num_points=num_points)
    print("\n=== Metrics Plots Generated ===")
    for csv_file, plot_list in plots.items():
        print(f"\n{csv_file}:")
        for p in plot_list:
            print(f"  {p}")

    return model

'''
def train_PPO_model(reward_fn,
                    grid_file: str,
                    timesteps: int,
                    folder_name: str,
                    reset_kwargs: dict = {},
                    arch: str | None = None,
                    features_dim: int = 128,
                    battery_truncation=False,
                    is_att = False,
                    num_frames = 8
                    ):

    # Initialize environment
    is_cnn = arch is not None
    env = GridWorldEnv(reward_fn=reward_fn, grid_file=grid_file, is_cnn=is_cnn, reset_kwargs=reset_kwargs, battery_truncation=battery_truncation)
    if is_cnn:
        env = CustomGridCNNWrapper(env)
    if is_att:
        env = TimeStackObservation(env, num_frames = num_frames)

    vec_env = DummyVecEnv([lambda: env])
    base_log_path = os.path.join(SAVE_DIR, folder_name)

    policy_kwargs = None
    if is_cnn:
        policy_kwargs = {
            "features_extractor_class": GridCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "grid_file": grid_file,
                "backbone": arch.lower()
            },
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        }
    
    elif is_att:
        policy_kwargs = {
            "features_extractor_class": AttentionCNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
                "grid_file": grid_file
            },
            "net_arch": dict(pi=[64, 64], vf=[64, 64])
        }

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        ent_coef=0.1,
        gae_lambda=0.90,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        clip_range_vf=0.5,
        tensorboard_log=base_log_path,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    callback = CustomTensorboardCallback()
    model.learn(total_timesteps=timesteps, callback=callback)

    log_dir = get_latest_run_dir(base_log_path)
    model_save_path = os.path.join(log_dir, "model.zip")
    model.save(model_save_path)
    print(f"\nPPO training complete. Logs and model stored to {log_dir}")

    # generate graphs from csvs using chunked smoothing
    grid_area = env.n_rows * env.n_cols
    num_points = 50
    #num_points = int(max(20, grid_area // 10))
    plots = plot_all_metrics(log_dir=log_dir, num_points=num_points)

    print("\n=== Metrics Plots Generated ===")
    for csv_file, plot_list in plots.items():
        print(f"\n{csv_file}:")
        for p in plot_list:
            print(f"  {p}")

    return model
'''

def infer_reward_fn(experiment_name: str):
    """
    Extracts and returns the reward function used in a given experiment name.
    Looks for parts like 'reward_d' in the experiment name (split by '__'),
    then returns the corresponding function 'get_reward_d' from reward_functions.
    """
    parts = experiment_name.split("__")
    
    for part in parts:
        if part.startswith("reward_"):
            reward_key = part.split("_")[0] + "_" + part.split("_")[1]  # "reward_d"
            fn_name = f"get_{reward_key}"
            if hasattr(reward_functions, fn_name):
                return getattr(reward_functions, fn_name)
            else:
                raise ValueError(f"No reward function named {fn_name} in reward_functions.py")
    
    raise ValueError(f"No reward key found in experiment name: {experiment_name}")

def best_matching_grid(experiment_name: str, grid_dir: str) -> str:
    grid_name = experiment_name.split("__")[0] + ".txt"
    print(grid_name)
    grid_file_path = os.path.join(grid_dir, grid_name)
    if not os.path.exists(grid_file_path):
        raise FileNotFoundError(f"Grid file not found: {grid_file_path}")
    return grid_name

# training utils
#-------------------------------------------------
def evaluate_model(env, model, n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True,
                   halfsplit=False):
    """
    Evaluate the model in the given environment for a number of episodes,
    printing agent's position, reward, and action at every timestep, and summarizing performance.
    """
    total_reward_sum = 0.0
    total_steps = 0
    success_count = 0
    total_collisions = 0
    total_revisits = 0
    total_battery_sum = 0.0
    leq_10s = 0
    leq_0s = 0

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_battery_sum = 0.0
        step_num = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            agent_pos = info.get('agent_pos', None)
            subrewards = info.get('subrewards', {})
            reached_exit = env.agent_reached_exit()
            success_count += int(reached_exit)

            # Track battery level at each step
            curr_bat = info.get("current_battery", 0)
            ep_battery_sum += curr_bat
            if curr_bat <= 10:
                leq_10s += 1
                if curr_bat == 0:
                    leq_0s += 1

            # ===== Printout Info ===== #
            if verbose:
                action_dir = ACTION_NAMES.get(int(action), f"Unknown({action})")

                # 1. Print the main step information
                print(f"Step {step_num}: Pos={agent_pos}, Action={action_dir} ({action_dir}), Reward={reward:.2f}")

                # 2. Print the confidence and decision information (if available)
                if 'used_fallback' in info:
                    confidence = info.get('confidence', 0)
                    if info.get('used_fallback'):
                        print(f"  └─ Decision: Agent uncertain (conf: {confidence:.3f}). Falling back to D* Lite.")
                    else:
                        print(f"  └─ Decision: Agent confident (conf: {confidence:.3f}). Using learned policy.")
                
                # 3. Print the subreward breakdown
                for name, val in subrewards.items():
                    print(f"  └─ {name}: {val:.2f}")
            #===========================#

            if render:
                env.render_pygame()
                time.sleep(sleep_time)

            step_num += 1

        # Episode-level stats
        total_collisions += info.get("obstacle_hits", 0)
        total_revisits += info.get("revisit_count", 0)
        total_steps += step_num
        total_reward_sum += ep_reward
        total_battery_sum += (ep_battery_sum / step_num) if step_num > 0 else 0

        if verbose:
            print(f"Episode {ep + 1} complete: Total Reward = {ep_reward:.2f}")

    # Aggregate stats
    mean_reward = total_reward_sum / n_eval_episodes
    success_rate = success_count / n_eval_episodes
    mean_col = total_collisions / n_eval_episodes
    avg_steps = total_steps / n_eval_episodes
    avg_rev = total_revisits / n_eval_episodes
    mean_battery = total_battery_sum / n_eval_episodes
    
    print("\n=== Evaluation Summary ===")
    print(f"Total Episodes: {n_eval_episodes}")
    print(f"Reached Exit: {success_count}/{n_eval_episodes} ({success_rate:.1%})")
    print(f"Mean Reward per Episode: {mean_reward:.2f}")
    print(f"Mean Obstacle Hits per Episode: {mean_col:.2f}")
    print(f"Mean Steps per Episode: {avg_steps:.1f}")
    print(f"Mean Revisits per Episode: {avg_rev:.1f}")
    print(f"Mean Battery Level per Episode: {mean_battery:.1f}")
    print(f"Timesteps Where Battery Level <= 10: {leq_10s}/{total_steps} ({leq_10s/total_steps * 100:.4f}%)")
    print(f"Timesteps Where Battery Level <= 0: {leq_0s}/{total_steps} ({leq_0s/total_steps * 100:.4f}%)")

def load_model(experiment_folder: str, 
               experiment_name: str,
               reset_kwargs: dict = {}):
    """
    Loads a PPO model and reconstructs its full environment stack.
    It uses the provided 'experiment_name' to infer the configuration.
    """
    model_path = os.path.join(experiment_folder, "model.zip")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # --- 1. Parse the full configuration from the PROVIDED experiment_name ---
    name_lower = experiment_name.lower()
    
    inferred_grid = best_matching_grid(experiment_name, FIXED_GRID_DIR)
    
    is_att = "att" in name_lower
    is_cnn = "cnn" in name_lower and not is_att
    
    use_fallback = "fb_" in name_lower
    confidence_threshold = 0.75 # Default
    if use_fallback:
        match = re.search(r'fb_(\d\.\d+)', name_lower)
        if match:
            confidence_threshold = float(match.group(1))

    reward_fn = infer_reward_fn(experiment_name)
    
    print(f"\n--- Loading Model: {experiment_name} from folder {os.path.basename(experiment_folder)} ---")
    print(f"  Grid: {inferred_grid}, Reward Fn: {reward_fn.__name__}")
    print(f"  CNN: {is_cnn}, Attention: {is_att}, Fallback: {use_fallback} (conf={confidence_threshold})")
    
    # --- 2. Load the model weights FIRST to break the circular dependency ---
    if use_fallback:
        print("Loading PPOFallback model (with fallback support).")
        model_class = PPOFallback
    else:
        print("Loading standard PPO model.")
        model_class = PPO

    model = model_class.load(model_path)
        
    # --- 3. Build the full environment stack based on the parsed flags ---
    num_frames = 8 # Assume this is a constant for your attention models

    base_env = GridWorldEnv(reward_fn=reward_fn, grid_file=inferred_grid, is_cnn=(is_cnn or is_att), reset_kwargs=reset_kwargs)

    obs_wrapped_env = base_env
    if is_cnn or is_att:
        obs_wrapped_env = CustomGridCNNWrapper(obs_wrapped_env)
    if is_att:
        obs_wrapped_env = TimeStackObservation(obs_wrapped_env, num_frames=num_frames)
    
    if use_fallback:
        final_env = DStarFallbackWrapper(obs_wrapped_env, model, confidence_threshold)
    else:
        final_env = obs_wrapped_env

    # --- 4. Connect the model to the final, fully-wrapped environment ---
    vec_env = DummyVecEnv([lambda: final_env])
    model.set_env(vec_env)
    
    return model

def load_model_and_evaluate(model_folder: str, grid_file: str, is_cnn: bool = False, reset_kwargs: dict = {},
                            n_eval_episodes=20, sleep_time=0.1, render: bool = True, verbose: bool = True):
    """
    Load a model by experiment folder and evaluate it in the matching environment.
    """
    model = load_model(model_folder, grid_file, is_cnn, reset_kwargs)
    evaluate_model(env=model.get_env().envs[0], model=model, n_eval_episodes=n_eval_episodes,
                   sleep_time=sleep_time, render=render, verbose=verbose)

def get_halfsplit_battery_overrides(grid_path: str) -> dict:
    """
    Returns a battery override dictionary where:
    - Top half of sensors have 100.0 battery
    - Bottom half of sensors have 0.0 battery
    """
    sensor_positions = []
    with open(grid_path, "r") as f:
        lines = [line.strip() for line in f]
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == SENSOR:
                    sensor_positions.append((r, c))

    if not sensor_positions:
        raise ValueError(f"No sensors found in: {grid_path}")

    n_rows = len(lines)
    mid_row = n_rows // 2

    battery_overrides = {
        (r, c): 100.0 if r < mid_row else 0.0
        for r, c in sensor_positions
    }

    return battery_overrides

def train_quick_junk_model(grid_file: str, is_cnn: bool = False):
    '''
    Very small training for quick testing
    '''
    junk_folder_name = grid_file[0:len(grid_file) - 3] + "__" + "junk"
    if is_cnn:
        junk_folder_name += "_cnn"
    timesteps = 500 
    
    model = train_PPO_model(
        grid_file=grid_file,
        timesteps=timesteps,
        folder_name=junk_folder_name,
        is_cnn=is_cnn
    )
    
    print(f"Trained junk model '{junk_folder_name}' for {timesteps} timesteps.")
    return junk_folder_name
