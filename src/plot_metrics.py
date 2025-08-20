# plot_metrics.py

import os
from src.constants import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

def get_latest_run_dir(base_dir):
    """
    Returns the subdirectory with the highest PPO_N number.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Log base directory does not exist: {base_dir}")

    ppo_dirs = []
    for name in os.listdir(base_dir):
        if name.startswith("PPO_") and os.path.isdir(os.path.join(base_dir, name)):
            try:
                num = int(name.split("_")[1])
                ppo_dirs.append((num, name))
            except (IndexError, ValueError):
                continue

    if not ppo_dirs:
        raise FileNotFoundError(f"No PPO_N directories found in {base_dir}")

    latest_num, latest_dir = sorted(ppo_dirs, reverse=True)[0]
    return os.path.join(base_dir, latest_dir)

def plot_csv(csv_path, output_dir, num_points=40):
    """
    Reads a CSV file, applies chunk-based smoothing (instead of rolling mean),
    and generates line plots for numeric columns.

    Args:
        csv_path (str): Path to the CSV file.
        output_dir (str): Directory to save the PNG plots.
        num_points (int): Number of smoothing chunks (approximate).
    """
    
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Skipping empty CSV file: {csv_path}")
            return []
    except pd.errors.EmptyDataError:
        print(f"Skipping empty or malformed CSV file: {csv_path}")
        return []

    print(f"\nLoaded {csv_path}: {df.shape[0]} rows, columns: {list(df.columns)}")

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    numeric_cols = df.select_dtypes(include='number').columns
    plots = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if "sensor_battery_levels" in base_name:
        plt.figure(figsize=(12, 6))
        for col in numeric_cols:
            y_values = df[col]
            plt.plot(df.index, y_values, label=f"Sensor {col}")
        
        plt.title("Sensor Battery Levels (Raw Data)")
        plt.xlabel("Timestep")
        plt.ylabel("Battery Level")
        plt.grid(True)
        plt.legend(loc="upper right", ncol=3, fontsize="small")

        out_path = os.path.join(output_dir, f"{base_name}_all_sensors_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        plots.append(out_path)
        print(f"Saved sensor battery plot: {out_path}")
    else:
        is_episode_metrics = "episode_metrics" in base_name

        for col in numeric_cols:
            raw_values = df[col].values
            x_base = df["episode"].values if is_episode_metrics else np.arange(len(df))

            # Smoothing logic: chunk-average
            n = len(raw_values)
            interval = max(1, n // num_points)
            xs, ys = [], []

            for i in range(0, n, interval):
                window = raw_values[i: min(i + interval, n)]
                x_chunk = x_base[i: min(i + interval, n)]
                xs.append(np.mean(x_chunk))
                ys.append(np.mean(window))

            plt.figure(figsize=(10, 4))
            plt.plot(xs, ys, marker='o', label=f"Smoothed ({num_points} pts)")
            plt.title(f"{base_name}: {col} (Smoothed)")
            plt.xlabel("Episode" if is_episode_metrics else "Timestep")
            plt.ylabel(col)
            plt.grid(True)
            plt.legend()

            safe_col_name = col.replace("/", "_").replace(" ", "_")
            filename = f"{base_name}_{safe_col_name}_{timestamp}.png"
            out_path = os.path.join(output_dir, filename)

            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()

            plots.append(out_path)
            print(f"Saved plot: {out_path}")

    return plots

def plot_all_metrics(
    log_dir,
    output_dir=None,
    num_points=40
):
    """
    Generates plots for all metrics CSVs in a log directory.

    Args:
        log_dir (str): Directory containing CSV files.
        output_dir (str): Directory to save PNG plots.
        num_points (int): Number of smoothing chunks (approximate).

    Returns:
        dict: Mapping of CSV filenames to lists of saved plot file paths.
    """
    base_name = os.path.basename(os.path.normpath(log_dir))
    if base_name.startswith("PPO_") and os.path.isdir(log_dir):
        run_dir = log_dir
    else:
        run_dir = get_latest_run_dir(log_dir)

    if output_dir is None:
        output_dir = os.path.join(run_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    all_plots = {}

    # Look for all metrics CSVs
    csv_files = [
        f for f in os.listdir(run_dir)
        if f.endswith(".csv") and (
            f.startswith("timestep_metrics") or
            f.startswith("episode_metrics") or
            f.startswith("subrewards_metrics") or
            f.startswith("sensor_battery_levels")
        )
    ]

    for f in csv_files:
        csv_path = os.path.join(run_dir, f)
        plots = plot_csv(csv_path, output_dir, num_points=num_points)
        all_plots[f] = plots

    return all_plots

def generate_all_plots(base_dir=None, num_points=40):
    """
    Recursively search for folders containing metrics CSVs
    and generate plots for each.

    Args:
        base_dir (str): Base directory to start search.
        num_points (int): Number of smoothing chunks (approximate).
    """
    if base_dir is None:
        base_dir = LOG_DIR
    for root, dirs, files in os.walk(base_dir):
        csvs = [f for f in files if f.endswith(".csv") and (
            f.startswith("timestep_metrics") or
            f.startswith("episode_metrics") or
            f.startswith("subrewards_metrics") or
            f.startswith("sensor_battery_levels")
        )]

        if csvs:
            print(f"\n=== Processing CSVs in {root} ===")
            output_dir = os.path.join(root, "plots")
            plots = plot_all_metrics(log_dir=root, output_dir=output_dir, num_points=num_points)
            print(f"Generated {sum(len(v) for v in plots.values())} plots in {output_dir}")
        else:
            print(f"No CSVs found in {root}")
