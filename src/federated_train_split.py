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

def split_grid_into_quadrants(grid_file: str) -> List[str]:
    """
    Reads a 100x100 ASCII grid file and splits it into 4 50x50 quadrant text files.
    Returns the filenames of the 4 new grid files.
    """
    src_path = os.path.join(FIXED_GRID_DIR, grid_file)
    with open(src_path, "r") as f:
        lines = [list(line.strip()) for line in f if line.strip()]

    grid = np.array(lines)
    n_rows, n_cols = grid.shape
    assert n_rows == 100 and n_cols == 100, "Input map must be 100x100."

    quadrants = []
    coords = [
        (0, 50, 0, 50),     # top-left
        (0, 50, 50, 100),   # top-right
        (50, 100, 0, 50),   # bottom-left
        (50, 100, 50, 100)  # bottom-right
    ]

    for i, (r0, r1, c0, c1) in enumerate(coords):
        subgrid = grid[r0:r1, c0:c1]
        fname = f"{os.path.splitext(grid_file)[0]}_q{i+1}.txt"
        fpath = os.path.join(FIXED_GRID_DIR, fname)
        with open(fpath, "w") as f:
            for row in subgrid:
                f.write("".join(row) + "\n")
        quadrants.append(fname)
        print(f"[INFO] Saved quadrant {i+1} to {fname}")
    return quadrants