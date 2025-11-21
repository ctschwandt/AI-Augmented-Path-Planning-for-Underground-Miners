import os
from typing import List, Dict, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

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

    Federated mode does not use SB3 TensorBoard logging.
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
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return model


# ============================================================
# Single process local client training
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
    Returns the client's updated policy state_dict (on CPU).
    """
    env = make_env(quadrant_file, reward_fn, obs_profile=obs_profile)
    model = make_ppo_for_env(env, quadrant_file)

    # Load global weights into local model
    model.policy.load_state_dict(global_policy_state)

    callback = CustomTensorboardCallback(verbose=1)

    print(f"  Client {client_id} starting local training for {local_steps:,} timesteps...")
    model.learn(total_timesteps=local_steps, callback=callback)
    print(f"  Client {client_id} finished training.")

    # Return only the policy weights on CPU for FedAvg
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
    folder_name: str | None = None,
    identical_start: bool = False,
):
    """
    Perform federated learning by splitting the 100x100 grid into 4 50x50 local clients.
    Each client trains locally; the server aggregates with FedAvg after every round.

    No evaluation is performed inside the loop or at the end.
    Only checkpoints are saved.
    """
    # get reward function
    if not hasattr(reward_functions, reward_fn_name):
        raise ValueError(f"Reward function {reward_fn_name} not found.")
    reward_fn = getattr(reward_functions, reward_fn_name)

    # base directory for this federated experiment
    if folder_name is None:
        folder_name = "Federated_PPO_Split"
    base_dir = os.path.join(SAVE_DIR, folder_name)
    os.makedirs(base_dir, exist_ok=True)

    print("\n[SETUP] Splitting the 100x100 map into four 50x50 quadrants...")
    quadrant_files = split_grid_into_quadrants(global_grid_file)

    # optional debug: make all quadrants identical initially
    if identical_start:
        import shutil
        q1_path = os.path.join(FIXED_GRID_DIR, quadrant_files[0])
        for i in range(1, 4):
            qi_path = os.path.join(FIXED_GRID_DIR, quadrant_files[i])
            shutil.copyfile(q1_path, qi_path)
        print("[DEBUG] identical_start=True -> All 4 quadrant grids now have identical contents.")

    # initialize global model using first quadrant
    ref_env = make_env(quadrant_files[0], reward_fn, obs_profile=obs_profile)
    global_model = make_ppo_for_env(ref_env, quadrant_files[0])

    cumulative_steps = 0

    for r in range(1, rounds + 1):
        print(f"\n================ Federated Round {r}/{rounds} ================")
        print("  Server: training clients sequentially")

        # export global weights on CPU
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

        print(f"  Server: aggregating {len(client_states)} clients using {aggregation_fn.__name__}")
        new_state = aggregation_fn(client_states)

        # load aggregated weights back into global model
        device = global_model.device
        new_state_device = {k: v.to(device) for k, v in new_state.items()}
        global_model.policy.load_state_dict(new_state_device)

        cumulative_steps += local_steps

        # save checkpoint after this round
        ckpt_path = os.path.join(base_dir, "checkpoints", f"round_{r}.zip")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        global_model.save(ckpt_path)
        print(f"  Server: saved global checkpoint to {ckpt_path}")

    # final save
    final_path = os.path.join(base_dir, "global_final.zip")
    global_model.save(final_path)
    print(f"\n[Done] Federated split training complete. Final model saved at {final_path}")
    return global_model


def train_federated_100x100_split():
    """
    Helper launcher.
    """
    return run_federated_split_training(
        global_grid_file="mine_100x100.txt",
        rounds=20,
        local_steps=50_000,
        reward_fn_name="get_reward_d",
        obs_profile="cnn7",
        aggregation_fn=fedavg,
        folder_name="Federated_PPO_Split",
        identical_start=False,
    )
