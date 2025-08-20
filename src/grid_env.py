# grid_env.py
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict
import numpy as np
import random 
import os
import csv
import pygame
import src.grid_gen as grid_gen
from src.constants import *
from src.utils import *
from src.reward_functions import compute_reward
from src.sensor import transmission_energy, reception_energy, compute_sensor_energy_loss, update_single_sensor_battery
from stable_baselines3.common.callbacks import BaseCallback
import datetime
from src.path_planning import *

##==============================================================
## Logs custom metrics stored in `info` dict to TensorBoard
## and csv
##==============================================================
class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        # tracker
        self.episode_count = 0

        # CSV files and writers
        self.timestep_csv_file = None
        self.timestep_csv_writer = None
        self.timestep_fieldnames = []

        self.episode_csv_file = None
        self.episode_csv_writer = None
        self.episode_fieldnames = []

        self.subrewards_csv_file = None
        self.subrewards_csv_writer = None
        self.subrewards_fieldnames = []

        self.sensor_battery_csv_file = None
        self.sensor_battery_csv_writer = None
        self.sensor_battery_fieldnames = []

        # file to save non-line graph data
        self.txt = None
        
        self.verbose = verbose

    def _on_training_start(self) -> None:
        log_dir = self.logger.dir
        print(f"Logger directory resolved to: {log_dir}")

        self.timestep_csv_path = os.path.join(log_dir, f"timestep_metrics.csv")
        self.episode_csv_path = os.path.join(log_dir, f"episode_metrics.csv")
        self.subrewards_csv_path = os.path.join(log_dir, f"subrewards_metrics.csv")
        self.sensor_battery_csv_path = os.path.join(log_dir, "sensor_battery_levels.csv")
        
        self.timestep_csv_file = open(self.timestep_csv_path, "w", newline="")
        self.episode_csv_file = open(self.episode_csv_path, "w", newline="")
        self.subrewards_csv_file = open(self.subrewards_csv_path, "w", newline="")
        self.sensor_battery_csv_file = open(self.sensor_battery_csv_path, "w", newline="")

        self.txt_path = os.path.join(log_dir, "log.txt")
        self.txt_file = open(self.txt_path, "w")
        self.success_count = 0

        if self.verbose > 0:
            print(f"Timestep CSV logging to: {self.timestep_csv_path}")
            print(f"Episode CSV logging to: {self.episode_csv_path}")
            print(f"Subrewards CSV logging to: {self.subrewards_csv_path}")
            print(f"Sensor Battery CSV logging to: {self.sensor_battery_csv_path}")
            
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
                self.logger.record(f"timestep/{k}", v)
            for k, v in episode_data.items():
                self.logger.record(f"episode/{k}", v)
            for k, v in subrewards_info.items():
                self.logger.record(f"subrewards/{k}", v)

            # CSV: timestep
            if not self.timestep_fieldnames:
                self.timestep_fieldnames = list(timestep_data.keys())
                self.timestep_csv_writer = csv.DictWriter(self.timestep_csv_file, fieldnames=self.timestep_fieldnames)
                self.timestep_csv_writer.writeheader()
            timestep_row = {k: timestep_data.get(k, "") for k in self.timestep_fieldnames}
            self.timestep_csv_writer.writerow(timestep_row)

            # CSV: subrewards
            if subrewards_info:
                # Optionally, attach step number if available
                if not self.subrewards_fieldnames:
                    self.subrewards_fieldnames = list(subrewards_info.keys())
                    self.subrewards_csv_writer = csv.DictWriter(self.subrewards_csv_file, fieldnames=self.subrewards_fieldnames)
                    self.subrewards_csv_writer.writeheader()
                subrewards_row = {k: subrewards_info.get(k, "") for k in self.subrewards_fieldnames}
                self.subrewards_csv_writer.writerow(subrewards_row)

            # CSV: episode
            if flat_info.get("terminated", False) or flat_info.get("truncated", False):
                # add episode count to csv data
                self.episode_count += 1
                episode_row = {"episode": self.episode_count}
                episode_row.update(episode_data)
                
                if not self.episode_fieldnames:
                    self.episode_fieldnames = ["episode"] + list(episode_data.keys())
                    self.episode_csv_writer = csv.DictWriter(self.episode_csv_file, fieldnames=self.episode_fieldnames)
                    self.episode_csv_writer.writeheader()
                    
                self.episode_csv_writer.writerow(episode_row)

                reached_exit = flat_info.get("terminated")
                self.success_count += int(reached_exit)

                # log to txt
                success_ratio = self.success_count / self.episode_count
                if self.txt_file:
                    self.txt_file.write(
                        f"Episode {self.episode_count}: Success Ratio = {self.success_count}/{self.episode_count} ({success_ratio:.1%})\n"
                    )
                    self.txt_file.flush()

            sensor_battery_snapshot = info.get("sensor_batteries")
            if sensor_battery_snapshot:
                if not self.sensor_battery_fieldnames:
                    # Use stringified sensor positions as column names
                    self.sensor_battery_fieldnames = list(map(str, sorted(sensor_battery_snapshot.keys())))
                    self.sensor_battery_csv_writer = csv.DictWriter(
                        self.sensor_battery_csv_file, fieldnames=self.sensor_battery_fieldnames
                    )
                    self.sensor_battery_csv_writer.writeheader()
                
                # Write battery values in sorted sensor position order
                row = {str(k): sensor_battery_snapshot[k] for k in sorted(sensor_battery_snapshot.keys())}
                self.sensor_battery_csv_writer.writerow(row)
            
        return True

    def _on_training_end(self) -> None:
        for f in [
                (self.timestep_csv_file, self.timestep_csv_path),
                (self.episode_csv_file, self.episode_csv_path),
                (self.subrewards_csv_file, self.subrewards_csv_path),
                (self.sensor_battery_csv_file, self.sensor_battery_csv_path)
        ]:
            if f[0]:
                f[0].close()
                if self.verbose > 0:
                    print(f"Closed CSV: {f[1]}")

        if self.txt_file:
            self.txt_file.close()
            if self.verbose > 0:
                print(f"Closed TXT: {self.txt_path}")

##==============================================================
## GridWorldEnv Class
##==============================================================
class GridWorldEnv(Env):
    def __init__(self,
                 reward_fn,
                 grid_file: str = None,
                 grid_height: int = None,
                 grid_width: int = None,
                 obstacle_percentage=None,
                 n_sensors=None, reset_kwargs={},
                 is_cnn=False, battery_truncation=False,
                 n_miners=12,
                 ):
        super(GridWorldEnv, self).__init__()

        ##=============== member variables ===============##
        # grid config
        self._init_grid_config(grid_file,
                               grid_height, grid_width,
                               obstacle_percentage,
                               n_sensors)
        self.max_steps = 2000
        self.reset_kwargs = reset_kwargs
        self.is_cnn = is_cnn
        self.battery_truncation = battery_truncation
        
        # pygame rendering
        self.pygame_initialized = False
        self.fixed_window_size = None
        self.cell_size = None
        self.screen = None
        self.font = None
        self.clock = None
        self.render_fps = 30

        # environment state variables
        self.grid = self.static_grid.copy()
        self.agent_pos = None
        self.visited = None
        self.current_battery_level = 0.0
        # self.battery_values_in_radar = None
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0
        self.last_action = -1
        self.miners = []
        self.n_miners = 12
        self.OBSTACLE_VALS = (OBSTACLE, SENSOR, BASE_STATION)
        self.n_miners = n_miners
        self.reward_fn = reward_fn

        # d*lite pathplanning assitance
        self.pathfinder = None
        self.cost_func_calls = 0
        
        # exclusively for graphing
        self.battery_levels_during_episode = []
            
        # observation/action spaces
        self._init_spaces()

        '''
        # radar around sensors
        self._init_radar_zone()
        '''
        ##================================================##

        self.reset()

    def _init_grid_config(self, grid_file, grid_height, grid_width, obstacle_percentage, n_sensors):
        if grid_file:
            grid_path = os.path.join(FIXED_GRID_DIR, grid_file)
            self.static_grid, self.agent_pos, self.goal_positions, self.sensor_batteries, self.base_station_positions = grid_gen.load_grid(grid_path)
            self.n_rows, self.n_cols = self.static_grid.shape
            self.n_sensors = len(self.sensor_batteries)

        else:
            assert grid_height is not None and grid_width is not None, \
                "Must provide grid_height and grid_width when not using a grid_file."
            self.n_rows = grid_height
            self.n_cols = grid_width
            save_path = os.path.join(RANDOM_GRID_DIR, f"grid_{self.n_rows}x{self.n_cols}{obstacle_percentage*100}p.txt")
            
            self.static_grid, self.agent_pos, self.goal_positions, self.sensor_batteries = grid_gen.gen_and_save_grid(
                self.n_rows, self.n_cols,
                obstacle_percentage=obstacle_percentage,
                n_sensors=n_sensors,
                place_agent=False,
                save_path=save_path
            )
            self.base_station_positions = []
            self.n_sensors = n_sensors
            
        self.original_sensor_batteries = self.sensor_batteries

    def _init_spaces(self):
        # 4 cardinals dirs + 4 diagonals
        self.action_space = Discrete(8)
        if self.is_cnn:
            '''
            self.observation_space = Box(
            low=0.0,
            high=5.0,
            shape=(2, self.n_rows, self.n_cols),
            dtype=np.float32
            )
            '''
            self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(5, self.n_rows, self.n_cols),
            dtype=np.float32
            )
            
        else:
            
            # [0, 7] - space around agent
            # [8, 9] - agent r, c
            # [10] - last action
            # [11] - distance to closest goal
            # [12, n_sensors-1] - battery levels of all sensors
            obs_dim = 8 + 2 + 1 + 1 + self.n_sensors

            # cnn observation space is set in wrapper
            self.observation_space = Box(
                low=0.0,
                high=1.0,
                shape=(obs_dim, ),
                dtype=np.float32
            )
            
    def _get_goal_exclusion_zone(self):
        # goal is not placed in sensor radar range???
        return grid_gen.get_safe_zone_around(self.goal_positions,
                                             self.n_rows,
                                             self.n_cols,
                                             radius=2)
        
    def is_empty(self, pos):
        return self.static_grid[pos[0], pos[1]] == EMPTY

    def _generate_random_obstacles(self):
        obstacles = set()
        while len(obstacles) < self.n_obstacles:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if self.is_empty((r, c)):
                obstacles.add((r, c))
        return obstacles

    def _generate_random_sensors(self):
        sensors = {}
        while len(sensors) < self.n_sensors:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if (r, c) not in sensors and self.is_empty((r, c)):
                sensors[(r, c)] = 100.0
        return sensors

    def init_pygame(self):
        if not self.pygame_initialized:
            pygame.init()

            # Desired max window size
            MAX_WIDTH = 1000
            MAX_HEIGHT = 750

            # Compute scaling factors
            scale_x = MAX_WIDTH / self.n_cols
            scale_y = MAX_HEIGHT / self.n_rows

            # Choose the smaller scaling factor to fit both dimensions
            cell_size = int(min(scale_x, scale_y))

            # Compute the final window size
            window_width = self.n_cols * cell_size
            window_height = self.n_rows * cell_size

            # Store
            self.cell_size = cell_size
            self.screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("GridWorld Visualization")
            self.font = pygame.font.SysFont("Arial", max(10, self.cell_size // 3))
            self.clock = pygame.time.Clock()
            self.pygame_initialized = True
    
    def render_pygame(self, uncap_fps=False, show_miners=False):
        if not self.pygame_initialized:
            self.init_pygame()

        screen = self.screen
        font = self.font
        cell_size = self.cell_size
        colors = RENDER_COLORS
        
        grid_copy = self.grid.copy()

        # Update trail cells
        for pos in self.visited:
            row, col = pos
            if (row, col) != tuple(self.agent_pos) and (row, col) not in self.goal_positions:
                if self.can_move_to((row, col)):
                    grid_copy[row, col] = TRAIL_OUTSIDE  # trail outside radar (yellow)

        # Draw each cell
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                val = grid_copy[row, col]

                # Always start by painting the cell white
                pygame.draw.rect(screen, colors[EMPTY], (col * cell_size, row * cell_size, cell_size, cell_size))

    
                
                # Then foreground item (trail, agent, goal, etc.)
                if val != EMPTY:
                    color = colors.get(val, (0, 0, 0))
                    pygame.draw.rect(screen, color, (col * cell_size, row * cell_size, cell_size, cell_size))

                # Border
                pygame.draw.rect(screen, (0, 0, 0), (col * cell_size, row * cell_size, cell_size, cell_size), 1)

        if self.pathfinder:
            path = self.pathfinder.get_path_to_goal()
            # We need at least two points to draw a line
            if path and len(path) > 1:
                # Convert grid coordinates (row, col) to pixel coordinates (center of each cell)
                # Assuming the path from D* Lite is a list of (col, row) tuples
                path_points = [
                (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
                for x, y in path
            ]
            
                # Define color and thickness for the path line
                path_color = (30, 144, 255)  # A nice, visible "dodger blue"
                path_thickness = 3
            
                # Draw the series of connected lines on the screen
                pygame.draw.lines(screen, path_color, False, path_points, path_thickness)

        # draw miners
        if show_miners:
            for row, col in self.miners:
                pygame.draw.rect(screen, colors[MINER],
                         (col * cell_size, row * cell_size,
                          cell_size, cell_size)) 

        # draw agent on top of miners
        row, col = self.agent_pos
        color = colors[AGENT]
        pygame.draw.rect(screen, color, (col * cell_size,
                                         row * cell_size, cell_size,
                                         cell_size))

        # draw sensors and battery labels (on top of all)
        for (row, col), battery in self.sensor_batteries.items():
            pygame.draw.rect(screen, colors[SENSOR], (col * cell_size, row * cell_size, cell_size, cell_size))
            label = font.render(f"{int(battery)}", True, (0, 0, 0))
            text_rect = label.get_rect(center=(col * cell_size + cell_size // 2, row * cell_size + cell_size // 2))
            screen.blit(label, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    exit()

        if uncap_fps:
            self.clock.tick(60)
        else:
            self.clock.tick(self.render_fps) # use default render_fps
       
    def get_sensor_distances(self, pos, normalize=True):
        if normalize:
            return euclidean_distances(pos, list(self.sensor_batteries.keys()), self.n_cols, self.n_rows)
        else:
            return euclidean_distances(pos, list(self.sensor_batteries.keys()))

    def get_exit_distances(self, normalize=True):
        if normalize:
            return euclidean_distances(self.agent_pos, self.goal_positions, self.n_cols, self.n_rows)
        else:
            return euclidean_distances(self.agent_pos, self.goal_positions)

    def get_observation(self):
        if self.is_cnn:
            # 5 channels      
            obs = np.zeros((5, self.n_rows, self.n_cols), dtype=np.float32)

            # Channel 0: agent
            r, c = self.agent_pos
            obs[0, r, c] = 1.0

            # Channel 1: blocked (obstacle, base station, etc.)
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    if self.static_grid[r, c] in (OBSTACLE, BASE_STATION):  # add others if needed
                        obs[1, r, c] = 1.0

            # Channel 2: sensor presence, Channel 3: sensor battery
            for (r, c), battery in self.sensor_batteries.items():
                obs[2, r, c] = 1.0
                obs[3, r, c] = battery / 100.0  # normalized battery

            # Channel 4: goal
            for r, c in self.goal_positions:
                obs[4, r, c] = 1.0

            return obs
        else:
            # Flat vector
            r, c = self.agent_pos
            neighbors = [
            (r - 1, c), (r - 1, c + 1), (r, c + 1), (r + 1, c + 1),
            (r + 1, c), (r + 1, c - 1), (r, c - 1), (r - 1, c - 1)
            ]

            def is_blocked(pos):
                return 1.0 if not self.can_move_to(pos) else 0.0

            blocked_flags = np.array([is_blocked(p) for p in neighbors], dtype=np.float32)
            norm_pos = np.array([r / (self.n_rows - 1), c / (self.n_cols - 1)], dtype=np.float32)
            last_action = self.last_action / 7.0 if self.last_action >= 0 else 0.0

            dist_to_goal = self._compute_min_distance_to_goal()

            battery_levels = np.array(
                [self.sensor_batteries.get(pos, 0.0) / 100.0 for pos in self.sensor_batteries],
            dtype=np.float32
            )

            return np.concatenate([blocked_flags, norm_pos, [last_action], [dist_to_goal], battery_levels])


    def reset(self, seed=None, options = None):
        """
        Resets the environment to the starting state for a new episode.
        Returns the initial observation and an empty info dict.
        """    
        # register seed
        super().reset(seed=seed)

        battery_overrides = self.reset_kwargs.get("battery_overrides", {})
        agent_override = self.reset_kwargs.get("agent_override", {})

        # restore static grid layout
        self.grid = np.copy(self.static_grid)

        # reset sensor battery levels
        self.sensor_batteries = {
            pos: random.uniform(0.0, 100.0)
            for pos in self.original_sensor_batteries
        }

        # optionally override specific sensor batteries
        if battery_overrides:
            for pos, value in battery_overrides.items():
                if pos in self.sensor_batteries:
                    self.sensor_batteries[pos] = value
                    
        # reset counters
        self.episode_steps = 0
        self.total_reward = 0
        self.obstacle_hits = 0
        self.episode_revisits = 0
        self.battery_levels_during_episode = []

        # place agent
        if agent_override:
            self.agent_pos = np.array(agent_override)
        else:
            while True:
                x, y = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
                if self.grid[x, y] == EMPTY:
                    self.agent_pos = np.array([x, y])
                    break

        self.visited = {tuple(self.agent_pos)}

        # === Spawn miners === #
        self.miners = []
        while len(self.miners) < self.n_miners:
            r, c = random.randint(0, self.n_rows - 1), random.randint(0, self.n_cols - 1)
            if self.is_empty((r, c)) and (r, c) not in self.miners:
                self.miners.append((r, c))

        # Create a grid for planning where ALL impassable cells are marked as OBSTACLE (1).
        # The pathfinder only understands 0 (passable) and 1 (impassable).
        impassable_grid = np.zeros_like(self.static_grid) # Start with all-passable grid
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                # If the cell is an obstacle, sensor, or base station, mark it as impassable.
                if self.static_grid[r, c] in self.OBSTACLE_VALS:
                    impassable_grid[r, c] = 1
                else:
                    impassable_grid[r, c] = 0
            
        start_xy = (int(self.agent_pos[1]), int(self.agent_pos[0]))
        goals_xy = [(int(pos[1]), int(pos[0])) for pos in self.goal_positions]
    
        # === Compute initial path === #
        self.pathfinder = DStarLite(
            grid=impassable_grid,
            start=start_xy,
            goals=goals_xy,
            cost_function=self._dstar_cost_function
        )
        self.pathfinder._compute_shortest_path()

        return self.get_observation(), {}

    def can_move_to(self, pos):
        '''
        Return True if the position is in bounds
        and not blocked
        '''
        row, col = pos
        in_bounds = self.in_bounds(pos)
        return in_bounds and not self.static_grid[row, col] in self.OBSTACLE_VALS
    
    def in_bounds(self, pos):
        return 0 <= pos[0] < self.n_rows \
    and 0 <= pos[1] < self.n_cols
                                
    def agent_reached_exit(self):
        return tuple(self.agent_pos) in self.goal_positions

    def _pathfinder_step(self):
        """
        Updates the D* Lite pathfinder. Assumes all sensor costs may have changed.
        """
        if not self.pathfinder:
            return

        # Convert sensor positions from (row, col) to (col, row) for the update.
        # Assumes self.sensor_batteries.keys() are (row, col) tuples.
        all_sensor_positions_xy = [(pos[1], pos[0]) for pos in self.sensor_batteries.keys()]

        if all_sensor_positions_xy:
            self.pathfinder.update_costs(all_sensor_positions_xy)

        # Convert the agent's new position to (col, row) for the pathfinder
        agent_pos_xy = (self.agent_pos[1], self.agent_pos[0])
        self.pathfinder.move_and_replan(agent_pos_xy)
    
    def step(self, action):
        move = DIRECTION_MAP[int(action)]
        old_pos = self.agent_pos
        new_pos = self.agent_pos + move
        self.episode_steps += 1

        self._update_sensor_batteries()
        self._update_agent_position(new_pos)
        self._move_miners_and_update_sensors()
        self._pathfinder_step()
        reward, subrewards = self._compute_reward_and_update(old_pos) 
        if tuple(new_pos) in self.visited:
            self.episode_revisits += 1
        self.visited.add(tuple(new_pos))
        
        terminated = self.agent_reached_exit()
        truncated = self.episode_steps >= self.max_steps or \
            (self.battery_truncation and self.current_battery_level <= 10)

        info = self._build_info_dict(terminated, truncated, reward, subrewards)
        
        return self.get_observation(), reward, terminated, truncated, info

    def _compute_reward_and_update(self, old_pos):
        reward, subrewards = compute_reward(self, old_pos, self.reward_fn)
        self.total_reward += reward
        return reward, subrewards

    def _update_agent_position(self, new_pos):
        hit_wall = not self.can_move_to(new_pos)
        if not hit_wall:
            self.agent_pos = new_pos
        else:
            self.obstacle_hits += 1

    def _update_sensor_batteries(self):
        closest_sensor = self._get_closest_sensor(self.agent_pos)
        if closest_sensor is not None:
            self.current_battery_level = self.sensor_batteries[closest_sensor]
            self.battery_levels_during_episode.append(self.current_battery_level)
            self.sensor_batteries = update_single_sensor_battery(
                self.sensor_batteries,
                sensor_pos=closest_sensor,
                miner=self.agent_pos,
                base_stations=self.base_station_positions
            )
        else:
            self.current_battery_level = 0.0

    def _move_miners_and_update_sensors(self):
        self.move_miners_randomly()
        for miner_pos in self.miners:
            closest_sensor = self._get_closest_sensor(miner_pos)
            if closest_sensor:
                self.sensor_batteries = update_single_sensor_battery(
                    self.sensor_batteries,
                    sensor_pos=closest_sensor,
                    miner=miner_pos,
                    base_stations=self.base_station_positions
                )

    def _build_info_dict(self, terminated, truncated, reward, subrewards):
        info = {
            "agent_pos": self.agent_pos,
            "current_reward": reward,
            "cumulative_reward": self.total_reward,
            "obstacle_hits": self.obstacle_hits,
            "current_battery": self.current_battery_level,
            "distance_to_goal": self._compute_min_distance_to_goal(),
            "visited_count": len(self.visited),
            "terminated": terminated,
            "truncated": truncated,
            "subrewards": subrewards,
            "sensor_batteries": dict(self.sensor_batteries),
            "revisit_count": self.episode_revisits
        }
        if terminated or truncated:
            avg_battery = (
                sum(self.battery_levels_during_episode) / len(self.battery_levels_during_episode)
                if self.battery_levels_during_episode else 0.0
        )
            info["average_battery_level"] = avg_battery
            info["episode_length"] = self.episode_steps

        return info
        
    def episode_summary(self):
        print(f"   Episode Summary:")
        print(f"   Total Steps     : {self.episode_steps}")
        print(f"   Obstacles Hit   : {self.obstacle_hits}")
        print(f"   Total Reward    : {self.total_reward}")
        print(f"   Sensor Battery Levels (Sample):")
        for i, ((x, y), battery) in enumerate(self.sensor_batteries.items()):
            print(f"     Sensor {i+1} at ({x},{y}) → {battery:.2f}")

        '''
        if self.battery_values_in_radar:
            avg_battery = sum(self.battery_values_in_radar) / len(self.battery_values_in_radar)
            print(f"   Avg Battery Level While in Sensor Radar: {avg_battery:.2f}")
        else:
            print("   Agent never entered a sensor radar zone.")
        '''

    def close(self):
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

    def _in_radar(self, sensor_pos, agent_pos, radius=2):
        dx = abs(sensor_pos[0] - agent_pos[0])
        dy = abs(sensor_pos[1] - agent_pos[1])
        return dx <= radius and dy <= radius

    def manual_control_loop(self):
        self.init_pygame()
        obs, _ = self.reset()
        done = False

        key_to_action = {
            # 8 directional WASDQEZC
            pygame.K_w: 0,  # North
            pygame.K_e: 1,  # North-East
            pygame.K_d: 2,  # East
            pygame.K_c: 3,  # South-East
            pygame.K_s: 4,  # South
            pygame.K_z: 5,  # South-West
            pygame.K_a: 6,  # West
            pygame.K_q: 7,  # North-West

            # arrow keys (cardinal directions only)
            pygame.K_UP:    0,  # N
            pygame.K_RIGHT: 2,  # E
            pygame.K_DOWN:  4,  # S
            pygame.K_LEFT:  6,  # W
        }

        last_move_time = 0
        move_delay = 80 # milliseconds between moves

        step_count = 1
        while not done:
            self.render_pygame(uncap_fps=True)

            current_time = pygame.time.get_ticks()
        
            # Event handling (quit + escape)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.close()
                    return

            # Handle held-down keys
            if current_time - last_move_time >= move_delay:
                keys = pygame.key.get_pressed()
                for key, action in key_to_action.items():
                    if keys[key]:
                        step_count += 1
                        obs, reward, terminated, truncated, info = self.step(action)
                        subrew_str = ', '.join([f"{k}:{v:.2f}" for k, v in info['subrewards'].items()])
                        print(
                            f"Step {step_count} | Pos: {info['agent_pos'].tolist()} | "
                            f"Reward: {reward:.2f} | Battery: {info['current_battery']:.2f} | "
                            f"Dist: {info['distance_to_goal']:.3f} | Subrewards: [{subrew_str}]"
                        )

                        done = terminated or truncated
                        last_move_time = current_time
                        break # only process one direction at a time
        self.close()

    def _get_closest_sensor(self, pos):
        if not self.sensor_batteries:
            return None

        distances = {sensor: euclidean_distance(pos, sensor)
                    for sensor in self.sensor_batteries}
        return min(distances, key=distances.get)
    
    def move_miners_randomly(self):
        updated_miners = []
        for miner_pos in self.miners:
            action = random.randint(0, 7)  # 8 directions
            move = DIRECTION_MAP[action]
            new_pos = (miner_pos[0] + move[0], miner_pos[1] + move[1])

            if self.in_bounds(new_pos) and self.can_move_to(new_pos) and new_pos not in updated_miners:
                updated_miners.append(new_pos)
            else:
                updated_miners.append(miner_pos)  # stay in place if invalid
        self.miners = updated_miners

    def _compute_min_distance_to_goal(self):
        if not self.goal_positions:
            return 0
        distances = euclidean_distances(
            self.agent_pos,
            self.goal_positions,
            self.n_cols,
            self.n_rows
        )
        return min(distances)
    
    def update_observation_agent(self):
        if hasattr(self, "prev_agent_pos"):
            prev_r, prev_c = self.prev_agent_pos
            prev_idx = prev_r * self.n_cols + prev_c
            self.obs[prev_idx] = 4.0  # EMPTY

        curr_r, curr_c = self.agent_pos
        curr_idx = curr_r * self.n_cols + curr_c
        self.obs[curr_idx] = 2.0

        self.prev_agent_pos = (curr_r, curr_c)

    # === PATHFINDER ===
    def _dstar_cost_function(self, position_xy):
        """
        Defines the traversal cost for D* Lite based on the environment rule
        that each cell is covered by its single closest sensor. Includes debugging prints.
        """
        # Convert (col, row) to (row, col) for env logic
        pos_rc = (position_xy[1], position_xy[0])
    
        # Find the sensor that covers this cell.
        closest_sensor_pos_rc = self._get_closest_sensor(pos_rc)
    
        if closest_sensor_pos_rc is None:
            return 0.0

        battery_level = self.sensor_batteries.get(closest_sensor_pos_rc, 0.0)
    
        # Determine cost based on the covering sensor's battery
        cost = 0.0
        '''
        if battery_level == 0:
            cost = 200.0
        '''
        if battery_level <= 10:
            cost = 100.0
        elif battery_level <= 30:
            cost = 25.0

        return cost

    # === PATHFINDER ===
    def get_path_cost(self, position_rc):
        """
        Gets the total cost of the optimal path from a given position to the goal.
        This is a wrapper around the pathfinder's g-score.
        """
        if self.pathfinder is None:
            return float('inf') # Should not happen after reset
            
        # Return infinite cost if the move itself is invalid (e.g., into a wall)
        if not self.can_move_to(position_rc):
            return float('inf')
            
        # We query the pathfinder using (col, row) coordinates
        position_xy = (position_rc[1], position_rc[0])
        cost = self.pathfinder.g.get(position_xy, float('inf'))
        return cost
    
