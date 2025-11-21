import os
import copy
from typing import List, Dict, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

from src.constants import FIXED_GRID_DIR, SAVE_DIR
from src.grid_env import GridWorldEnv, CustomTensorboardCallback
from src.cnn_feature_extractor import GridCNNExtractor
import src.reward_functions as reward_functions
from src.federated_train_split import split_grid_into_quadrants


# ============================================================
# Aggregation Algorithm (FedAvg)
# ============================================================
def fedavg(policy_state_dicts: List[Dict[str, torch.Tensor]],
           weights: List[float] | None = None) -> Dict[str, torch.Tensor]:
    """
    Standard Federated Averaging algorithm (FedAvg).
    """
    n = len(policy_state_dicts)
    if n == 0:
        raise ValueError("No models to aggregate.")
    if weights is None:
        weights = [1.0 / n] * n
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    new_state = {}
    for k in policy_state_dicts[0].keys():
        new_state[k] = sum(sd[k] * weights[i] for i, sd in enumerate(policy_state_dicts))
    return new_state


# ============================================================
# Environment + PPO Builders
# ============================================================
def make_env(grid_file: str, reward_fn: Callable, obs_profile: str = "cnn7") -> GridWorldEnv:
    return GridWorldEnv(
        grid_file=grid_file,
        reward_fn=reward_fn,
        obs_profile=obs_profile,
        is_cnn=True,
    )


def make_ppo_for_env(
    env: GridWorldEnv,
    grid_file: str,
    features_dim: int = 128,
) -> PPO:
    """
    Create a PPO model for a given env.

    NOTE: tensorboard_log=None so this model will NOT create any TB event files.
    Only the custom SummaryWriters (one per region) log metrics.
    """
    vec_env = DummyVecEnv([lambda: env])

    policy_kwargs = {
        "features_extractor_class": GridCNNExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "grid_file": grid_file,
            "backbone": "seq",
        },
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
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
        verbose=1,
        tensorboard_log=None,  # no SB3 TB logs here
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model


# ============================================================
# Single-process local client training (no multiprocessing)
# ============================================================
def train_single_client(
    global_policy_state: Dict[str, torch.Tensor],
    quadrant_file: str,
    reward_fn: Callable,
    local_steps: int,
    obs_profile: str,
    client_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Train ONE federated client sequentially in the current process.
    Returns the client's updated policy state_dict.
    """
    # Build env
    env = make_env(quadrant_file, reward_fn, obs_profile=obs_profile)
    model = make_ppo_for_env(env, quadrant_file)

    # Load global weights into local model
    model.policy.load_state_dict(global_policy_state)

    callback = CustomTensorboardCallback(verbose=1)

    print(f"  ▶️ Client {client_id} starting local training for {local_steps:,} timesteps...")
    model.learn(total_timesteps=local_steps, callback=callback)
    print(f"  ✅ Client {client_id} finished training.")

    # Return only the policy weights
    # (keep them on CPU to make averaging easy)
    client_state = {k: v.detach().cpu() for k, v in model.policy.state_dict().items()}
    return client_state


# ============================================================
# Federated Learning Orchestration
# ============================================================
def run_federated_split_training(
    global_grid_file: str = "mine_100x100.txt",
    rounds: int = 20,
    local_steps: int = 50_000,
    reward_fn_name: str = "get_reward_d",
    obs_profile: str = "cnn7",
    aggregation_fn: Callable = fedavg,
    n_eval_episodes: int = 10000,
    folder_name: str | None = None,
    identical_start: bool = False,
):
    """
    Perform federated learning by splitting the 100x100 grid into 4 50x50 local clients.
    Each client trains locally; the server aggregates with FedAvg after every round.

    - Training is done SEQUENTIALLY (no multiprocessing) → easier on CPU.
    - After each round, evaluate the GLOBAL model on each quadrant and
      log (mean_cumulative_reward, mean_obstacle_hits, mean_battery) into:

        <SAVE_DIR>/<folder_name>/region_1
        <SAVE_DIR>/<folder_name>/region_2
        <SAVE_DIR>/<folder_name>/region_3
        <SAVE_DIR>/<folder_name>/region_4

      There is exactly ONE SummaryWriter per region, so for one full
      federated run you get exactly 4 event files (one per region).
    """
    if not hasattr(reward_functions, reward_fn_name):
        raise ValueError(f"Reward function {reward_fn_name} not found.")
    reward_fn = getattr(reward_functions, reward_fn_name)

    # resolve base directory for this experiment
    if folder_name is None:
        folder_name = "Federated_PPO_Split"
    base_dir = os.path.join(SAVE_DIR, folder_name)
    os.makedirs(base_dir, exist_ok=True)

    print("\n[SETUP] Splitting the 100x100 map into four 50x50 quadrants...")
    quadrant_files = split_grid_into_quadrants(global_grid_file)

    # Optional debug mode: make all 4 quadrants identical at the start
    if identical_start:
        import shutil
        q1_path = os.path.join(FIXED_GRID_DIR, quadrant_files[0])
        for i in range(1, 4):
            qi_path = os.path.join(FIXED_GRID_DIR, quadrant_files[i])
            shutil.copyfile(q1_path, qi_path)
        print("[DEBUG] identical_start=True → All 4 quadrant grids now have identical contents.")

    # Initialize global model using quadrant 1, with correct obs_profile
    ref_env = make_env(quadrant_files[0], reward_fn, obs_profile=obs_profile)
    global_model = make_ppo_for_env(ref_env, quadrant_files[0])

    # Per-region eval envs (quadrants)
    region_envs = [make_env(qf, reward_fn, obs_profile=obs_profile) for qf in quadrant_files]

    # Per-region TensorBoard writers  (ONE per region per whole run)
    region_writers: List[SummaryWriter] = []
    for i in range(4):
        region_logdir = os.path.join(base_dir, f"region_{i+1}")
        os.makedirs(region_logdir, exist_ok=True)
        region_writers.append(SummaryWriter(log_dir=region_logdir))

    cumulative_steps = 0  # will be our x-axis (round * local_steps)

    for r in range(1, rounds + 1):
        print(f"\n================ Federated Round {r}/{rounds} ================")

        # ---- Train each local client SEQUENTIALLY ----
        print("  [Server] Training clients sequentially...")
        # Always export global weights on CPU for averaging
        global_policy_state = {
            k: v.detach().cpu() for k, v in global_model.policy.state_dict().items()
        }

        client_states: List[Dict[str, torch.Tensor]] = []
        for i, qf in enumerate(quadrant_files):
            client_id = i + 1
            client_state = train_single_client(
                global_policy_state=global_policy_state,
                quadrant_file=qf,
                reward_fn=reward_fn,
                local_steps=local_steps,
                obs_profile=obs_profile,
                client_id=client_id,
            )
            client_states.append(client_state)

        # ---- Aggregate client models ----
        print(f"  [Server] Aggregating {len(client_states)} clients using {aggregation_fn.__name__} ...")
        new_state = aggregation_fn(client_states)

        # Load aggregated weights into global model (move to current device)
        device = global_model.device
        new_state_device = {k: v.to(device) for k, v in new_state.items()}
        global_model.policy.load_state_dict(new_state_device)

        # Update global timestep counter
        cumulative_steps += local_steps

        # ---- Evaluate GLOBAL model on each region ----
        print(f"  [Eval] Testing global model on each of the 4 regions...")

        # ---- Save global checkpoint ----
        ckpt_path = os.path.join(base_dir, "checkpoints", f"round_{r}.zip")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        global_model.save(ckpt_path)
        print(f"  [Server] Saved global model to {ckpt_path}")

    # ---- Close writers ----
    for w in region_writers:
        w.close()

    # ---- Final save ----
    final_path = os.path.join(base_dir, "global_final.zip")
    global_model.save(final_path)
    print(f"\n[✓] Federated Split Training Complete. Final model saved at {final_path}")
    return global_model


def train_federated_100x100_split():
    """
    Default launcher:
    - Splits 100x100 map into four 50x50 clients
    - Runs 20 federated rounds
    """
    return run_federated_split_training(
        global_grid_file="mine_100x100.txt",
        rounds=20,
        local_steps=50_000,
        reward_fn_name="get_reward_d",
        obs_profile="cnn7",
        aggregation_fn=fedavg,
        n_eval_episodes=10,
        folder_name="Federated_PPO_Split",  # default name if called directly
        identical_start=False,
    )
