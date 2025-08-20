# constants.py
import os
import numpy as np

# grid symbols
EMPTY = '.'
OBSTACLE = '#'
GOAL = 'G'
SENSOR = 'S'
AGENT = 'A'
FINISHED = 'F'
TRAIL_OUTSIDE = '*'
TRAIL_INSIDE = 'T'
# RADAR_BG = 'b'
BASE_STATION = 'B'
MINER = 'M'

# rgb colors for pygame rendering
RENDER_COLORS = {
    EMPTY: (255, 255, 255),
    OBSTACLE: (100, 100, 100),
    TRAIL_OUTSIDE: (255, 255, 0),
    AGENT: (0, 0, 255),
    GOAL: (0, 255, 0),
    FINISHED: (0, 255, 255),
    SENSOR: (255, 0, 0),
    TRAIL_INSIDE: (173, 216, 230),
    # RADAR_BG: (255, 165, 0),
    BASE_STATION: (128, 0, 128),
    MINER: (0, 100, 0)
}

# actions agent can take (cardinal directions and diagonals)
DIRECTION_MAP = {
    0: (-1,  0), 1: (-1, +1), 2: (0, +1), 3: (+1, +1),
    4: (+1,  0), 5: (+1, -1), 6: (0, -1), 7: (-1, -1),
}

ACTION_NAMES = {
    0: "UP",
    1: "UP-RIGHT",
    2: "RIGHT",
    3: "DOWN-RIGHT",
    4: "DOWN",
    5: "DOWN-LEFT",
    6: "LEFT",
    7: "UP-LEFT"
}

MOVE_TO_ACTION_MAP = {move: action for action, move in DIRECTION_MAP.items()}

# model saving and logging
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


MODEL_DIR = os.path.join(BASE_DIR, "SavedModels")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MODELS = {
    "ppo": os.path.join(MODEL_DIR, "PPO_custom_grid"),
    "dqn": os.path.join(MODEL_DIR, "DQN_custom_grid")
}

LOGS = {
    "ppo": os.path.join(LOG_DIR, "PPO_custom_grid"),
    "dqn": os.path.join(LOG_DIR, "DQN_custom_grid"),
}

SAVE_DIR = os.path.join(BASE_DIR, "saved_experiments")

# directory for storing generated ASCII grids
GRID_DIR = os.path.join(BASE_DIR, "saved_grids")
RANDOM_GRID_DIR = os.path.join(GRID_DIR, "random")
FIXED_GRID_DIR = os.path.join(GRID_DIR, "fixed")

##=================================
# Constants (for the Energy Model)
##=================================
BATTERY_CAPACITY_JOULES = 10.0 # Total Capacity of each battery
ALPHA_ELEC = 50e-9          # 50 nJ/bit
ALPHA_SHORT = 10e-12        # 10 pJ/bit/m^2
ALPHA_LONG = 0.0013e-12     # 0.0013 pJ/bit/m^4
THRESHOLD_DIST = 15.0  # Threshold distance in the grid
MAX_COMM_RANGE = 141 # 25.0           # Max Communication range (in grid cells)
# have to adjust other variables according to max_comm_range
#========================
# Message Sizes (in bits)
#========================
K_BROADCAST = 10_000 # Sensor-to-Sensor
K_TO_MINER = 5_000 # Miner-to-Sensor
K_TO_BASE = 2_000  # Sensor-to-base




