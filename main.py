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
    train_all_models(1_000_000)
    #evaluate_all_models(n_eval_episodes=100, render=False, verbose=False)
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()

# tests that need to be performed:
# - why is it that reward 6 with super low path progress penalty works the
#   best? compare reward6 with reward6 without path penalty
# - reward_pathlen (pathlen based) performs worse than reward_d (dist based).
#   I need to compare their penalty values using test_manual_control.
#   one possibility for why is that the agent is given dist in observation
#   space for both. I should test reward_pathlen with patheln in observation
#   space.
#   once a pathlen reward function becomes the best, test fallback
    

