import os
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Import training logic
# We need to hack sys.path to ensure imports inside train.py work
import sys
# Current: src/extras/experiments.py -> Parent: src/ -> Root: IAPV/
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from train import train, make_env_helper, LIMIT_STEPS

import json

# File paths
# Root is src_dir's parent
root_dir = os.path.dirname(src_dir)
RESULTS_JSON = os.path.join(root_dir, "output", "benchmark_results.json")

def evaluate_model(model, env_name, n_episodes=20):
    """Evaluates a model and returns detailed metrics."""
    # Create valid env
    if env_name == "Custom": gym_id = "GridWorld-v0"
    else: gym_id = "CartPole-v1"
    
    # Eval env must match training env wrapping (Basic)
    env = make_env_helper(gym_id)
    # We use bare env loop to easier count success/steps
    
    successes = 0
    total_steps = 0
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rew = 0
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_rew += reward
            steps += 1
            
        total_rewards.append(ep_rew)
        total_steps += steps
        
        # Success Logic
        if env_name == "Custom":
            if ep_rew > 0: successes += 1
        else:
            if ep_rew >= 195: successes += 1
            
    mean_rew = float(np.mean(total_rewards))
    std_rew = float(np.std(total_rewards))
    success_rate = float(successes / n_episodes)
    avg_len = float(total_steps / n_episodes)
    
    return mean_rew, std_rew, success_rate, avg_len

def generate_intervals(env_name, max_demos, step=10, spawn_mode="random"):
    """
    Step 1: Generates incremental dataset files from the master demos.pkl.
    Saves to output/{env}/intervals/demos_{N}.pkl.
    Returns: List of (n_demos, file_path)
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # IAPV Root
    if env_name == "Custom":
        env_folder = os.path.join("grid", spawn_mode)
    else:
        env_folder = "cartpole"
    base_output = os.path.join(root, "output", env_folder)
    intervals_dir = os.path.join(base_output, "intervals")
    os.makedirs(intervals_dir, exist_ok=True)
    
    # Master demos are now in the 'simple' subfolder for GridWorld
    # For CartPole it's still clean, but we can assume 'simple' logic doesn't apply there or is handled by top level.
    # Actually, for consistency let's check both or standard location.
    # In Streamlit we definde demos_file as .../simple/demos.pkl.
    source_folder = os.path.join(base_output, "simple") if env_name == "Custom" else base_output
    demos_path = os.path.join(source_folder, "demos.pkl")
    
    if not os.path.exists(demos_path):
        raise FileNotFoundError(f"Master demos file not found: {demos_path}")
        
    import pickle
    with open(demos_path, "rb") as f:
        all_demos = pickle.load(f)
        
    total_available = len(all_demos)
    valid_max = min(max_demos, total_available)
    
    generated_files = []
    
    demo_counts = range(step, valid_max + 1, step)
    for n in demo_counts:
        subset = all_demos[:n]
        fname = f"demos_{n}.pkl"
        fpath = os.path.join(intervals_dir, fname)
        with open(fpath, "wb") as f:
            pickle.dump(subset, f)
        generated_files.append((n, fpath))
        
    return generated_files

def run_benchmark_loop(env_name, generated_files, seed=42, progress_callback=None, gail_epochs=100, bc_epochs=50, spawn_mode="random", eval_episodes=100, force_retrain=False):
    """
    Step 2: Runs training on pre-generated files with a FIXED seed.
    Saves results to JSON.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # IAPV Root
    if env_name == "Custom":
        env_folder = os.path.join("grid", spawn_mode)
    else:
        env_folder = "cartpole"
    base_output = os.path.join(root, "output", env_folder)
    intervals_dir = os.path.join(base_output, "intervals")
    os.makedirs(intervals_dir, exist_ok=True)
    
    # Define results path inside INTERVALS folder
    results_path = os.path.join(intervals_dir, "benchmark_results.json")
    
    # We iterate ALGO -> FILES (Define BEFORE using)
    algorithms = ["BC", "GAIL"]
    total_steps = len(generated_files) * len(algorithms)
    current_step = 0
    
    # Initialize nested structure
    results = {algo: {} for algo in algorithms}
    
    for algo in algorithms:
        for n_demos, fpath in generated_files:
            current_step += 1
            if progress_callback:
                # Show progress at START of step
                pct = (current_step - 1) / total_steps
                msg = f"[{current_step}/{total_steps}] A processar {algo} com {n_demos} demos..."
                progress_callback(pct, msg)
                
            model_name = f"{algo}_{n_demos}_seed{seed}.zip"
            # Models inside intervals/models
            model_out = os.path.join(intervals_dir, "models", model_name)
            
            # User defined settings
            epochs = bc_epochs if algo == "BC" else gail_epochs
            
            # Placeholder for metrics
            metrics = {}
            
            try:
                # CACHE CHECK
                if os.path.exists(model_out) and not force_retrain:
                    print(f"Cache hit: {model_name}")
                else:
                    # Run Training only if not cached
                    from train import train as run_train_func
                    print(f"DEBUG: Starting benchmark train for {algo} with {n_demos} demos...")
                    run_train_func(
                        demos_path=fpath,
                        output_path=model_out,
                        env_name=env_name,
                        algorithm=algo,
                        epochs=epochs,
                        spawn_mode=spawn_mode,
                        eval_episodes=eval_episodes
                    )
                
                # Evaluate (Always evaluate to ensure metrics are fresh/present)
                from stable_baselines3.common.policies import ActorCriticPolicy
                from stable_baselines3 import PPO
                
                # Torch Load Hack
                _safe_load = torch.load
                def _patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return _safe_load(*args, **kwargs)
                torch.load = _patched_load
                
                try:
                    # Suppress harmless "learning_rate" code object warnings (Py3.12 vs Py3.11 issue)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        print(f"DEBUG: Loading model for eval: {model_out}")
                        try:
                            model = PPO.load(model_out)
                        except:
                            model = ActorCriticPolicy.load(model_out)
                finally:
                    torch.load = _safe_load
                
                print("DEBUG: Evaluating model...")
                mean, std, success, avg_len = evaluate_model(model, env_name, n_episodes=20)
                step_ratio = avg_len / LIMIT_STEPS
                
                metrics = {
                    "Mean Reward": mean,
                    "Std Dev": std,
                    "Success Rate": success,
                    "Step Ratio": step_ratio
                }
                
            except Exception as e:
                import traceback
                print(f"Error {algo}-{n_demos}: {e}")
                traceback.print_exc()
                metrics = {
                    "Mean Reward": -100.0,
                    "Std Dev": 0.0,
                    "Success Rate": 0.0,
                    "Step Ratio": 1.0,
                    "Error": str(e)
                }
            
            # Assign to nested dict
            results[algo][int(n_demos)] = metrics
            
            # INCREMENTAL SAVE
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
    
    return results, results_path
