import argparse
import gymnasium as gym
import numpy as np
import os
import sys
import torch
import random
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.data import serialize
from imitation.util.networks import RunningNorm

# Registration
from gymnasium.envs.registration import register
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Formula for Max Steps
GRID_SIZE = 8
LIMIT_STEPS = int(((GRID_SIZE * GRID_SIZE) / 2) / 10) * 10

try:
    import custom_env
    # GridWorld is registered dynamically in train() with spawn mode
except Exception:
    pass

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def make_env_helper(env_name, **kwargs):
    if env_name == "GridWorld-v0":
        return gym.make(env_name, **kwargs)
    return gym.make(env_name)

from stable_baselines3.common.evaluation import evaluate_policy

def train(demos_path, output_path, env_name, algorithm, epochs, spawn_mode="random", eval_episodes=100, seed=None):
    # Set global seeds if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # SB3 doesn't have a global set_seed anymore in simple sense, 
        # but we pass it to algorithms.
    
    # ... (existing setup code) ...
    # --- DYNAMIC STEPS LOGIC ---
    # Define GRID_SIZE locally first to avoid UnboundLocalError
    try:
        from demos import GRID_SIZE
    except ImportError:
        GRID_SIZE = 8 # Fallback
        
    grid_size = GRID_SIZE 
    limit_steps = int(((grid_size * grid_size) / 2) / 10) * 10

    if env_name == "Custom" or env_name == "GridWorld-v0":
        # Register GridWorld with spawn mode
        random_start = (spawn_mode == "random")
        print(f"DEBUG: Registering GridWorld-v0 with random_start={random_start}", flush=True)
        
        # Clean re-registration to ensure params are updated
        try:
            if "GridWorld-v0" in gym.envs.registration.registry:
                del gym.envs.registration.registry["GridWorld-v0"]
        except:
            pass # Safety for older gym versions/structures
            
        register(
            id="GridWorld-v0",
            entry_point="custom_env:GridWorldEnv",
            max_episode_steps=LIMIT_STEPS,
            kwargs={'size': GRID_SIZE, 'random_start': random_start}
        )
        gym_id = "GridWorld-v0"
    elif env_name == "CartPole":
        gym_id = "CartPole-v1"
        limit_steps = 500 # Default for CartPole-v1 is 500
    else:
        gym_id = env_name
        limit_steps = 200
        
    display_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {algorithm} on {gym_id} using {demos_path}...")
    print(f"Hardware Acceleration: {display_device.upper()}")

    # 16 episodes per update (The instructor logic)
    dynamic_n_steps = int(limit_steps * 16)
    if dynamic_n_steps < 256: dynamic_n_steps = 256
    
    print(f"Dynamic Config: Steps/Ep={limit_steps}, Update Buffer={dynamic_n_steps}")
    # ---------------------------
    
    # Load Demos (Robust Pickle Load)
    
    # Handle directory input
    if os.path.isdir(demos_path):
        demos_path = os.path.join(demos_path, "demos.pkl")
        
    try:
        if not os.path.exists(demos_path):
            raise FileNotFoundError(f"Demos file not found: {demos_path}")
            
        with open(demos_path, "rb") as f:
            raw_transitions = pickle.load(f)
            
        # --- DATA CONVERSION & VALIDATION ---
        from imitation.data.types import TrajectoryWithRew
        transitions = []
        
        print(f"Loading raw data, found {len(raw_transitions)} items.")
        
        for i, t in enumerate(raw_transitions):
            # 1. Handle Streamlit Dict Format
            if isinstance(t, dict):
                obs = t["obs"]
                acts = t["actions"]
                rews = t["rewards"]
            # 2. Handle TrajectoryWithRew object (from demos.py)
            elif hasattr(t, "obs"):
                obs = t.obs
                acts = t.acts
                rews = t.rews
            else:
                print(f"Skipping unknown data format at index {i}")
                continue
                
            # Validation: Obs length must be Acts + 1
            if len(obs) != len(acts) + 1:
                # Try to fix by trimming or skipping
                if len(obs) == len(acts):
                    # Likely missing last obs. This is fatal for imitation.
                    print(f"Warning: Episode {i} has equal Obs/Acts length. Skipping.")
                    continue
                else:
                    print(f"Warning: Episode {i} length mismatch (Obs={len(obs)}, Acts={len(acts)}). Skipping.")
                    continue
            
            # Create proper object
            # infos needs to be None or list of dicts. None is safer if empty.
            transitions.append(TrajectoryWithRew(
                obs=np.array(obs),
                acts=np.array(acts),
                rews=np.array(rews),
                terminal=True, # GridWorld episodes usually end in terminal state
                infos=None 
            ))
            
        if len(transitions) == 0:
            raise ValueError("No valid trajectories found after validation!")
            
        # Print Stats
        lens = [len(t) for t in transitions]
        avg_rew = np.mean([np.sum(t.rews) for t in transitions])
        print(f"--- Dataset Stats ---")
        print(f"Total Trajectories: {len(transitions)}")
        print(f"Avg Length: {np.mean(lens):.1f} steps")
        print(f"Avg Reward: {avg_rew:.2f}")
        print(f"---------------------")
            
    except Exception as e:
        print(f"CRITICAL ERROR LOADING DATA: {e}")
        return

    # Create Envs (Train and Eval)
    # CRITICAL DECISION: NO NORMALIZATION for GAIL
    # Why? Demos are RAW [0-7], but VecNormalize makes agent rollouts ~0
    # This creates trivial discriminator task that doesn't help learning
    
    # Wrap in Monitor and DummyVecEnv (NO VecNormalize!)
    venv = DummyVecEnv([lambda: Monitor(make_env_helper(gym_id))])
    eval_env = DummyVecEnv([lambda: Monitor(make_env_helper(gym_id))])
    
    # Output dir for best model
    log_dir = os.path.dirname(output_path)
    
    if algorithm == "BC":
        rng = np.random.default_rng()
        # BC uses unwrapped env observation space, not the wrapped venv
        fresh_env = make_env_helper(gym_id)
        trainer = bc.BC(
            observation_space=fresh_env.observation_space,
            action_space=fresh_env.action_space,
            demonstrations=transitions,
            rng=rng,
            l2_weight=0.001,  # Added regularization
            batch_size=32,
        )
        print(f"Starting BC Training ({epochs} Epochs)...")
        print(f"Starting BC Training ({epochs} Epochs)...")
        trainer.train(n_epochs=epochs, progress_bar=True) 
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trainer.policy.save(output_path)
        print(f"BC Policy saved to {output_path}")
        
    elif algorithm == "GAIL":
        from imitation.util.util import make_vec_env
        from imitation.data.wrappers import RolloutInfoWrapper
        
        # Correctly create environment using imitation's utility
        # This handles wrapping and vectorization properly
        venv = make_vec_env(
            gym_id,
            n_envs=1,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
            rng=np.random.default_rng(42)
        )
        
        # Reward Net - Using RunningNorm as validated in diagnostics/notebook
        from imitation.util.networks import RunningNorm
        reward_net = BasicRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            normalize_input_layer=RunningNorm,
        )
        
        # Generator (Agent) parameters - Validated Config
        # Increased n_steps to 1024 for better stability
        step_size = 1024
        
        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            n_steps=step_size,
            batch_size=64,
            ent_coef=0.0,               # Zero entropy (let discriminator force diversity)
            learning_rate=0.0004,       # Slightly higher learning rate
            n_epochs=5,
            gamma=0.95,                 # Lower gamma for short horizon tasks
            verbose=1
        )
        
        # Calculate total transitions to avoid batch size error
        total_transitions = sum(len(t) for t in transitions)
        actual_demo_batch_size = min(128, total_transitions)
        
        trainer = GAIL(
            demonstrations=transitions,
            demo_batch_size=actual_demo_batch_size, # Dynamic batch size
            gen_replay_buffer_capacity=step_size,  # Buffer matches n_steps
            n_disc_updates_per_round=8,      # Train discriminator 8x more than generator
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
        )
        
        # Callback logic
        best_reward = -np.inf
        def gail_callback(round_num):
            nonlocal best_reward
            if round_num % 5 == 0: # Eval every 5 rounds
                try:
                    # No normalization sync needed anymore!
                    mean, _ = evaluate_policy(trainer.gen_algo, eval_env, n_eval_episodes=5)
                    print(f"Round {round_num} Eval (Raw): {mean:.2f}")
                except Exception as e:
                    print(f"Eval Error: {e}")
        
        # Total Steps: Direct control via epochs
        total_steps = epochs * step_size 
        
        print(f"Starting GAIL Training (Total: {total_steps} Steps)...")
        
        # tqdm for GAIL
        from tqdm import tqdm
        pbar = tqdm(total=total_steps, desc="GAIL Training", file=sys.stdout)
        
        def gail_progress_callback(round_num):
            # Update pbar by step_size (approx)
            # round_num is roughly number of updates
            # Actually, let's just update by step_size every call
            pbar.update(step_size)
            gail_callback(round_num)
            
        try:
            # Removed incorrect safety check that used non-existent trainer.batch_size
            trainer.train(total_timesteps=total_steps, callback=gail_progress_callback)
        except Exception as e:
            print(f"GAIL Training Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            pbar.close()
        
        # Save Model (NO VecNormalize stats since we removed it!)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trainer.gen_algo.save(output_path)
        print(f"Saved: {output_path}")

    # --- FINAL EVALUATION ---
    print(f"\nRunning Final Evaluation ({eval_episodes} Episodes)...")
    try:
        # Determine model object for evaluation
        if algorithm == "BC":
            model_to_eval = trainer.policy # BC stores policy in .policy
        else:
            model_to_eval = trainer.gen_algo # GAIL PPO is gen_algo
            
        # Manual Eval with TQDM
        eval_rewards = []
        from tqdm import tqdm
        
        print(f"Starting Validation of {eval_episodes} episodes...", flush=True)
        # Use simple range with tqdm
        for i in tqdm(range(eval_episodes), desc="Validating", file=sys.stdout):
            # We create a fresh non-vectorized env to facilitate precise per-episode seeding
            # This ensures the robustness plot has a real distribution
            eval_env = make_env_helper(gym_id)
            ep_seed = (seed + 1000 + i) if seed is not None else None
            obs, _ = eval_env.reset(seed=ep_seed)
            
            done = False
            total_reward = 0
            while not done:
                action, _ = model_to_eval.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    eval_rewards.append(total_reward)
                    break
            eval_env.close()
                    
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        print(f"Final Result: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        
    except Exception as e:
        print(f"Final Eval Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("DEBUG: train.py STARTED", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to demonstrations file")
    parser.add_argument("--output", type=str, required=True, help="Path to save trained policy")
    parser.add_argument("--gym", type=str, required=True, help="Environment name: CartPole or Custom")
    parser.add_argument("--algorithm", type=str, required=True, choices=["BC", "GAIL"], help="Algorithm: BC or GAIL")
    parser.add_argument("--epochs", type=int, default=100, help="Number of Epochs (BC) or Timesteps/1000 (GAIL)")
    parser.add_argument("--spawn", type=str, default="random", choices=["random", "fixed"], help="Spawn mode for GridWorld (random or fixed)")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of evaluation episodes at the end")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args() # argparse automatically handles --help
    print(f"DEBUG: ARGS RECEIVED: {args}", flush=True)
    
    train(args.file, args.output, args.gym, args.algorithm, args.epochs, args.spawn, args.eval_episodes, args.seed)
