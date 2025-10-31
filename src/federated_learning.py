import os
import copy
import numpy as np
import torch
from typing import List, Dict, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.constants import FIXED_GRID_DIR, SAVE_DIR
from src.grid_env import GridWorldEnv
from src.cnn_feature_extractor import GridCNNExtractor
import src.reward_functions as reward_functions
from src.federated_train_split import split_grid_into_quadrants
from src.grid_env import CustomTensorboardCallback


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
        is_cnn=True
    )


def make_ppo_for_env(env: GridWorldEnv,
                     grid_file: str,
                     features_dim: int = 128,
                     round_id: int = None,
                     client_id: int = None) -> PPO:
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    # choose per-round/client log dir if provided
    if round_id is not None and client_id is not None:
        log_dir = os.path.join(SAVE_DIR, "Federated_PPO_Split",
                               f"round_{round_id}_client_{client_id}")
    else:
        log_dir = os.path.join(SAVE_DIR, "Federated_PPO_Split")
    os.makedirs(log_dir, exist_ok=True)

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
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
    )
    return model


# ============================================================
# Federated Learning Orchestration
# ============================================================
def train_local_model(global_model: PPO, grid_file: str,
                      reward_fn: Callable, local_steps: int = 1_000_000,
                      round_id: int = 1, client_id: int = 1) -> PPO:
    """
    Train a local PPO model on one quadrant (grid_file) for a few steps.
    Each client logs to its own TensorBoard directory with custom metrics.
    """
    local_env = make_env(grid_file, reward_fn)
    log_dir = os.path.join(SAVE_DIR, "Federated_PPO_Split",
                           f"round_{round_id}_client_{client_id}")
    os.makedirs(log_dir, exist_ok=True)

    local_model = make_ppo_for_env(local_env, grid_file,
                                   round_id=round_id, client_id=client_id)

    # attach your custom metrics callback
    callback = CustomTensorboardCallback(verbose=1)

    # copy global policy weights
    local_model.policy.load_state_dict(copy.deepcopy(global_model.policy.state_dict()))

    print(f"  ▶️ Client {client_id} starting local training for {local_steps:,} timesteps...")
    local_model.learn(total_timesteps=local_steps, callback=callback)
    print(f"  ✅ Client {client_id} finished training.")
    return local_model


def run_federated_split_training(
    global_grid_file: str = "mine_100x100.txt",
    rounds: int = 20,
    local_steps: int = 50_000,
    reward_fn_name: str = "get_reward_d",
    obs_profile: str = "cnn7",
    aggregation_fn: Callable = fedavg,
    n_eval_episodes: int = 10000
    ):
    """
    Perform federated learning by splitting the 100x100 grid into 4 50x50 local clients.
    Each client trains locally; the server aggregates with FedAvg after every round.
    """
    if not hasattr(reward_functions, reward_fn_name):
        raise ValueError(f"Reward function {reward_fn_name} not found.")
    reward_fn = getattr(reward_functions, reward_fn_name)

    print("\n[SETUP] Splitting the 100x100 map into four 50x50 quadrants...")
    quadrant_files = split_grid_into_quadrants(global_grid_file)

    # Initialize global model
    ref_env = make_env(quadrant_files[0], reward_fn)
    global_model = make_ppo_for_env(ref_env, quadrant_files[0])

    # Prepare eval environment
    eval_env = make_env(quadrant_files[0], reward_fn)

    for r in range(1, rounds + 1):
        print(f"\n================ Federated Round {r}/{rounds} ================")
        local_models = []

        # ---- Train each local client ----
        for i, qf in enumerate(quadrant_files):
            print(f"  [Client {i+1}] Training on {qf} ...")
            local_model = train_local_model(global_model,
                                            qf,
                                            reward_fn,
                                            local_steps,
                                            round_id=r,
                                            client_id=i+1)
            local_models.append(local_model)

        # ---- Aggregate client models ----
        print(f"  [Server] Aggregating {len(local_models)} clients using {aggregation_fn.__name__} ...")
        client_states = [m.policy.state_dict() for m in local_models]
        new_state = aggregation_fn(client_states)
        global_model.policy.load_state_dict(new_state)

        # ---- Evaluate global model ----
        print(f"  [Eval] Testing global model on full 100x100 map...")

        # local import avoids circular import with train.py
        from src.train import evaluate_model
        from stable_baselines3.common.vec_env import DummyVecEnv

        # temporarily swap env, evaluate, then restore
        _prev_env = global_model.get_env()
        try:
            eval_vec = DummyVecEnv([lambda: eval_env])
            global_model.set_env(eval_vec)
            evaluate_model(eval_env, global_model, n_eval_episodes=n_eval_episodes, render=False, verbose=False)
        finally:
            # restore the original training env
            global_model.set_env(_prev_env)

        # ---- Save checkpoint ----
        ckpt_path = os.path.join(SAVE_DIR, "Federated_PPO_Split", f"round_{r}.zip")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        global_model.save(ckpt_path)
        print(f"  [Server] Saved global model to {ckpt_path}")

    # ---- Final save ----
    final_path = os.path.join(SAVE_DIR, "Federated_PPO_Split", "global_final.zip")
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
        n_eval_episodes=3
    )
