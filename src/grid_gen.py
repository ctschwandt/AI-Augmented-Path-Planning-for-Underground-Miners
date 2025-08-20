# grid_gen.py
import numpy as np
import random
from src.constants import EMPTY, OBSTACLE, GOAL, SENSOR, AGENT, BASE_STATION, MINER

##==============================================================
## Kush
##==============================================================
def get_safe_zone_around(coords, rows, cols, radius=2):
    """
    Returns a set of coordinates within `radius` of each point in coords,
    ensuring all are within grid bounds.
    """
    safe_zone = set()
    for x, y in coords:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    safe_zone.add((nx, ny))
    return safe_zone

def generate_and_save_obstacles(rows, cols, n_obstacles=None, exclude_coords=[], filename="obstacle_coords.txt"):
    """
    Generate a set of obstacles while excluding certain zones (e.g., around goals).
    Writes coordinates to a text file.
    """
    if n_obstacles is None:
        n_obstacles = int(0.3 * rows * cols)

    coords = set()
    exclude_zone = get_safe_zone_around(exclude_coords, rows, cols, radius=2)

    attempts = 0
    max_attempts = n_obstacles * 10

    while len(coords) < n_obstacles and attempts < max_attempts:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) not in exclude_zone:
            coords.add((r, c))
        attempts += 1

    with open(filename, "w") as f:
        for r, c in coords:
            f.write(f"{r},{c}\n")

    print(f"[INFO] {len(coords)} obstacles saved to {filename}.")

def generate_and_save_sensors(rows, cols, n_sensors, obstacle_file="obstacle_coords.txt", sensor_file="sensor_coords.txt", goal_coords=[]):
    """
    Randomly generate sensor positions and battery levels.
    Ensures sensors do not overlap with obstacles or goals.
    """
    if not os.path.exists(obstacle_file):
        raise FileNotFoundError(f"Obstacle file '{obstacle_file}' not found.")

    obstacles = set()
    with open(obstacle_file, "r") as f:
        for line in f:
            r, c = map(int, line.strip().split(","))
            obstacles.add((r, c))

    forbidden = obstacles.union(goal_coords)
    sensors = set()
    sensor_data = []

    attempts = 0
    max_attempts = n_sensors * 20

    while len(sensors) < n_sensors and attempts < max_attempts:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) not in sensors and (r, c) not in forbidden:
            battery = random.randint(1, 100)
            sensor_data.append((r, c, battery))
            sensors.add((r, c))
        attempts += 1

    with open(sensor_file, "w") as f:
        for r, c, battery in sensor_data:
            f.write(f"{r},{c},{battery}\n")

    print(f"[INFO] {len(sensor_data)} sensors saved to {sensor_file}.")

'''
def compute_sensor_radar_zone(sensor_coords, rows, cols, radius=2):
    """
    Return set of all grid cells within radar radius of sensors.
    """
    radar_zone = set()
    for x, y in sensor_coords:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    radar_zone.add((nx, ny))
    return radar_zone
'''

##==============================================================
## Cole
##==============================================================
def gen_goals(grid):
    """
    Place fixed goal positions on the grid.

    Goals are placed in three corners:
    - bottom-right
    - top-right
    - bottom-left

    Args:
        grid (np.ndarray): 2D grid array to modify in place.

    Returns:
        list of tuple: Positions of placed goals.
    """
    rows, cols = grid.shape
    goal_positions = [
        (rows - 1, cols - 1),
        (0, cols - 1),
        (rows - 1, 0)
    ]
    for r, c in goal_positions:
        grid[r, c] = GOAL
    return goal_positions

def gen_obstacles(grid, n_obstacles, exclude_coords):
    """
    Randomly place obstacles on the grid while avoiding a safe zone.

    Args:
        grid (np.ndarray): 2D grid array to modify in place.
        n_obstacles (int): Number of obstacles to place.
        exclude_coords (list of tuple): Coordinates to exclude around (safe zone).

    Returns:
        set of tuple: Positions of placed obstacles.
    """
    rows, cols = grid.shape
    obstacles = set()
    safe_zone = get_safe_zone_around(exclude_coords, rows, cols, radius=2)

    attempts = 0
    max_attempts = n_obstacles * 10
    while len(obstacles) < n_obstacles and attempts < max_attempts:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) not in obstacles and (r, c) not in safe_zone and grid[r, c] == EMPTY:
            obstacles.add((r, c))
        attempts += 1

    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    return obstacles

def gen_sensors(grid, n_sensors):
    """
    Randomly place sensors on empty grid cells and assign default battery levels.

    Args:
        grid (np.ndarray): 2D grid array to modify in place.
        n_sensors (int): Number of sensors to place.

    Returns:
        dict: Mapping of sensor positions (tuple) to battery levels (float).
    """
    rows, cols = grid.shape
    sensors = set()
    sensor_batteries = {}
    attempts = 0
    max_attempts = n_sensors * 20
    while len(sensors) < n_sensors and attempts < max_attempts:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if grid[r, c] == EMPTY:
            sensors.add((r, c))
            sensor_batteries[(r, c)] = 100.0  # default battery
        attempts += 1

    for r, c in sensors:
        grid[r, c] = SENSOR
    return sensor_batteries

def gen_agent(grid):
    """
    Place the agent on a random empty cell in the grid.

    Args:
        grid (np.ndarray): 2D grid array to modify in place.

    Returns:
        tuple or None: Position of the agent or None if no empty cell found.
    """
    rows, cols = grid.shape
    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == EMPTY]
    agent_pos = None
    if empty_cells:
        agent_pos = random.choice(empty_cells)
        grid[agent_pos] = AGENT
    return agent_pos

def gen_grid(rows, cols, obstacle_percentage=0.3, n_sensors=5, place_agent=False):
    """
    Generate a complete grid with goals, obstacles, sensors, and an optional agent.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        obstacle_percentage (float, optional): Fraction of grid cells to fill with obstacles.
        n_sensors (int, optional): Number of sensors to place.
        place_agent (bool, optional): Whether to place the agent randomly.

    Returns:
        tuple:
            np.ndarray: Generated grid.
            tuple or None: Agent position.
            list of tuple: Goal positions.
            dict: Sensor positions mapped to battery levels.
    """
    grid = np.full((rows, cols), EMPTY, dtype='<U1')

    # Goals
    goal_positions = gen_goals(grid)

    # Obstacles
    n_obstacles = int(obstacle_percentage * rows * cols)
    gen_obstacles(grid, n_obstacles, exclude_coords=goal_positions)

    # Sensors
    sensor_batteries = gen_sensors(grid, n_sensors)

    # Agent
    agent_pos = None
    if place_agent:
        agent_pos = gen_agent(grid)

    return grid, agent_pos, goal_positions, sensor_batteries

def save_grid(grid, filename):
    """
    Save grid (2D numpy array) as ASCII text file.
    """
    with open(filename, "w") as f:
        for row in grid:
            f.write("".join(row) + "\n")
    print(f"[INFO] Grid saved to {filename}")

def gen_and_save_grid(rows, cols, obstacle_percentage=0.3, n_sensors=5, place_agent=False, save_path="generated_grid.txt"):
    """
    Generate a complete grid and save it to a file.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        obstacle_percentage (float, optional): Fraction of grid cells to fill with obstacles.
        n_sensors (int, optional): Number of sensors to place.
        place_agent (bool, optional): Whether to place the agent randomly.
        save_path (str, optional): Path to save the generated ASCII grid.

    Returns:
        tuple:
            np.ndarray: Generated grid.
            tuple or None: Agent position.
            list of tuple: Goal positions.
            dict: Sensor positions mapped to battery levels.
    """
    grid, agent_pos, goal_positions, sensor_batteries = gen_grid(
        rows, cols, obstacle_percentage, n_sensors, place_agent
    )
    save_grid(grid, save_path)
    return grid, agent_pos, goal_positions, sensor_batteries
    
def load_grid(filename):
    """
    Load grid from ASCII text file.

    Returns:
        grid (np.ndarray): 2D array of grid symbols (agent replaced with EMPTY)
        agent_pos (tuple or None)
        goal_positions (list)
        sensor_batteries (dict)
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n_rows = len(lines)
    n_cols = len(lines[0])
    grid = np.array([list(row) for row in lines], dtype='<U1')

    agent_pos = None
    goal_positions = []
    sensor_batteries = {}
    base_station_positions = []

    for r in range(n_rows):
        for c in range(n_cols):
            symbol = grid[r, c]
            if symbol == AGENT:
                agent_pos = (r, c)
                grid[r, c] = EMPTY  # agent not part of static grid
            elif symbol == GOAL:
                goal_positions.append((r, c))
            elif symbol == SENSOR:
                sensor_batteries[(r, c)] = 100.0
            elif symbol == BASE_STATION:
                base_station_positions.append((r, c))

    return grid, agent_pos, goal_positions, sensor_batteries, base_station_positions

def generate_random_grid(n_rows=20, n_cols=20, obstacle_pct=0.15, seed=None):
    """
    Generates a random grid with the specified number of rows and columns,
    and randomly places obstacles (#) to match the given obstacle percentage.

    Returns a list of strings, each representing a row.
    """
    if seed is not None:
        random.seed(seed)

    total_cells = n_rows * n_cols
    num_obstacles = int(total_cells * obstacle_pct)

    # Start with all empty cells
    grid = [['.' for _ in range(n_cols)] for _ in range(n_rows)]

    # Randomly place obstacles
    available_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    obstacle_positions = random.sample(available_positions, num_obstacles)
    for r, c in obstacle_positions:
        grid[r][c] = '#'

    # Convert each row to a string
    return [''.join(row) for row in grid]

'''
grid = generate_random_grid(100, 100, 0.30)
for row in grid:
    print(row)
'''
