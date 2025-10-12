# main.py

from src.test import *
from src.plot_metrics import generate_all_plots
from src.train import *

def ensure_directories_exist():
    directories = [
        SAVE_DIR,
        FIXED_GRID_DIR
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)

def main():
    #test_manual_control("mine_100x100.txt")
    #generate_all_plots(rolling_window=50_000)
    train_all_models(100)
    evaluate_all_models(n_eval_episodes=100, render=True, verbose=True)
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()

