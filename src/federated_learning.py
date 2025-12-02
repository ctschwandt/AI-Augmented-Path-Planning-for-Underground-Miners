import os
from typing import List, Dict, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback

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
# Custom Callback for Continuous Federated Logging
# ============================================================
class FederatedClientCallback(BaseCallback):
    """
    Logs metrics to an external SummaryWriter to ensure continuous logging
    across federated rounds (which involve re-instantiating the model).
    """
    def __init__(self, writer: SummaryWriter, initial_steps: int, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.initial_steps = initial_steps

    def _flatten_info(self, info):
        flat = {}
        subrewards = {}
        for k, v in info.items():
            if k == "subrewards" and isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    subrewards[f"{sub_k}"] = sub_v
            elif not isinstance(v, (dict, list)):
                flat[k] = v
        return flat, subrewards

    def _on_step(self) -> bool:
        # Current global timestep
        current_step = self.initial_steps + self.num_timesteps

        infos = self.locals.get("infos", [])
        for info in infos:
            if not info:
                continue

            flat_info, subrewards_info = self._flatten_info(info)

            # Split metrics
            timestep_keys = [
                "current_reward", "current_battery",
                "distance_to_goal",
                "terminated", "truncated"
            ]
            episode_keys = [
                "cumulative_reward", "obstacle_hits",
                "visited_count", "average_battery_level",
                "episode_length", "revisit_count"
            ]

            timestep_data = {k: flat_info.get(k) for k in timestep_keys if k in flat_info}
            episode_data = {k: flat_info.get(k) for k in episode_keys if k in flat_info}

            # TensorBoard logging
            for k, v in timestep_data.items():
                self.writer.add_scalar(f"timestep/{k}", v, current_step)
            for k, v in episode_data.items():
                self.writer.add_scalar(f"episode/{k}", v, current_step)
            for k, v in subrewards_info.items():
                self.writer.add_scalar(f"subrewards/{k}", v, current_step)
        
        return True


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
    tensorboard_log: str | None = None,
) -> PPO:
    """
    Create a PPO model for a given env.

    Federated mode can optionally log SB3 metrics via tensorboard_log.
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
        tensorboard_log=tensorboard_log,
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
    base_dir: str,
    client_writer: SummaryWriter,
    initial_steps: int
) -> Dict[str, torch.Tensor]:
    """
    Train ONE federated client sequentially in the current process.
    Returns the client's updated policy state_dict (on CPU).
    """
    # We do NOT use SB3's tensorboard_log here to avoid fragmentation.
    # Instead we use our custom FederatedClientCallback with the external writer.
    
    env = make_env(quadrant_file, reward_fn, obs_profile=obs_profile)
    model = make_ppo_for_env(env, quadrant_file, tensorboard_log=None)

    model.policy.load_state_dict(global_policy_state)

    # Use our custom callback for continuous logging
    callback = FederatedClientCallback(writer=client_writer, initial_steps=initial_steps, verbose=1)

    print(f"  Client {client_id} starting local training for {local_steps:,} timesteps (Global Step: {initial_steps})...")
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
    n_eval_episodes: int = 10,
    folder_name: str | None = None,
    identical_start: bool = False,
):
    """
    Federated learning over 4 quadrants.

    - Clients are trained sequentially.
    - After each round, the GLOBAL model is evaluated on each region.
    - For each region we log:
        global/mean_cumulative_reward
        global/mean_obstacle_hits
        global/mean_battery

      into:
        <SAVE_DIR>/<folder_name>/region_1
        ...
        <SAVE_DIR>/<folder_name>/region_4
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

    # initialize global model using first quadrant (no TB logging for global training)
    ref_env = make_env(quadrant_files[0], reward_fn, obs_profile=obs_profile)
    global_model = make_ppo_for_env(ref_env, quadrant_files[0], tensorboard_log=None)

    # envs per region for global evaluation
    region_envs = [make_env(qf, reward_fn, obs_profile=obs_profile) for qf in quadrant_files]

    # writers per region for global metrics
    region_writers: List[SummaryWriter] = []
    for i in range(4):
        region_logdir = os.path.join(base_dir, f"region_{i+1}")
        os.makedirs(region_logdir, exist_ok=True)
        region_writers.append(SummaryWriter(log_dir=region_logdir))

    # writers per client for local training metrics (continuous)
    client_writers: List[SummaryWriter] = []
    for i in range(4):
        client_logdir = os.path.join(base_dir, f"client_{i+1}")
        os.makedirs(client_logdir, exist_ok=True)
        client_writers.append(SummaryWriter(log_dir=client_logdir))

    cumulative_steps = 0 # Global steps (rounds * local_steps)
    total_client_steps = 0 # Accumulated steps for clients

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
                base_dir=base_dir,
                client_writer=client_writers[i],
                initial_steps=total_client_steps
            )
            client_states.append(client_state)

        print(f"  Server: aggregating {len(client_states)} clients using {aggregation_fn.__name__}")
        new_state = aggregation_fn(client_states)

        # load aggregated weights back into global model
        device = global_model.device
        new_state_device = {k: v.to(device) for k, v in new_state.items()}
        global_model.policy.load_state_dict(new_state_device)

        cumulative_steps += local_steps
        total_client_steps += local_steps

        # ==== Evaluate GLOBAL model on each region and log metrics ====
        print("  Server: evaluating global model on 4 regions")

        from src.train import evaluate_model

        _prev_env = global_model.get_env()
        try:
            for idx, (region_env, writer) in enumerate(zip(region_envs, region_writers)):
                eval_vec = DummyVecEnv([lambda env=region_env: env])
                global_model.set_env(eval_vec)

                stats = evaluate_model(
                    env=region_env,
                    model=global_model,
                    n_eval_episodes=n_eval_episodes,
                    render=False,
                    verbose=False,
                    return_metrics=True,
                )

                writer.add_scalar(
                    "global/mean_cumulative_reward",
                    stats["mean_cumulative_reward"],
                    cumulative_steps,
                )
                writer.add_scalar(
                    "global/mean_obstacle_hits",
                    stats["mean_obstacle_hits"],
                    cumulative_steps,
                )
                writer.add_scalar(
                    "global/mean_battery",
                    stats["mean_battery"],
                    cumulative_steps,
                )
                writer.flush()

                print(
                    f"    Region {idx+1}: "
                    f"Reward={stats['mean_cumulative_reward']:.2f}, "
                    f"Obstacles={stats['mean_obstacle_hits']:.2f}, "
                    f"Battery={stats['mean_battery']:.2f}"
                )
        finally:
            if _prev_env is not None:
                global_model.set_env(_prev_env)

        # save checkpoint after this round
        ckpt_path = os.path.join(base_dir, "checkpoints", f"round_{r}.zip")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        global_model.save(ckpt_path)
        print(f"  Server: saved global checkpoint to {ckpt_path}")

    # close writers
    for w in region_writers:
        w.close()
    for w in client_writers:
        w.close()

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
        n_eval_episodes=10,           # keep this small to avoid long eval times
        folder_name="Federated_PPO_Split",
        identical_start=False,
    )
