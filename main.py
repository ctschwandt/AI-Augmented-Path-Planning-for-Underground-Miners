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
    #evaluate_all_models(n_eval_episodes=10000, render=False, verbose=False)
    '''
    ppo_path = "saved_experiments/mine_50x50__reward_d__None__flat_cnn6/PPO_1"   # path to your PPO_N folder
    experiment_name = "mine_50x50__reward_d__cnn6"                     # name of experiment
    evaluate_ppo_run(
        ppo_path=ppo_path,
        experiment_name=experiment_name,
        n_eval_episodes=10000, 
        render=False,
        verbose=False
    )
    '''
    

    
    
    
if __name__ == "__main__":
    ensure_directories_exist()
    main()

