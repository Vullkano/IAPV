import argparse
import gymnasium as gym
import numpy as np
import os
import sys
import torch
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
    register(
        id="GridWorld-v0",
        entry_point="custom_env:GridWorldEnv",
        max_episode_steps=LIMIT_STEPS,
        kwargs={'size': GRID_SIZE}
    )
except Exception:
    pass

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def make_env_helper(env_name, **kwargs):
    if env_name == "GridWorld-v0":
        return gym.make(env_name, **kwargs)
    return gym.make(env_name)

from stable_baselines3.common.evaluation import evaluate_policy

def train(demos_path, output_path, env_name, algorithm):
    # ... (existing setup code) ...
    if env_name == "Custom":
        gym_id = "GridWorld-v0"
    elif env_name == "CartPole":
        gym_id = "CartPole-v1"
    else:
        gym_id = env_name
        
    display_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {algorithm} on {gym_id} using {demos_path}...")
    print(f"Hardware Acceleration: {display_device.upper()}")
    
    # Load Demos (Robust Pickle Load)
    import pickle
    
    # Handle directory input
    if os.path.isdir(demos_path):
        demos_path = os.path.join(demos_path, "demos.pkl")
        print(f"Input is a directory. Loading from: {demos_path}")
        
    try:
        if not os.path.exists(demos_path):
            raise FileNotFoundError(f"Demos file not found: {demos_path}")
            
        with open(demos_path, "rb") as f:
            transitions = pickle.load(f)
        print(f"Loaded {len(transitions)} trajectories from {demos_path}")
    except Exception as e:
        print(f"Error loading demos: {e}")
        print("Trying fallback to imitation.serialize...")
        transitions = serialize.load(demos_path)

    # Create Envs (Train and Eval)
    env = make_env_helper(gym_id)
    eval_env = make_env_helper(gym_id)
    
    # Wrap in Monitor for Stats
    venv = DummyVecEnv([lambda: Monitor(env)])
    
    # Output dir for best model
    log_dir = os.path.dirname(output_path)
    
    if algorithm == "BC":
        # ... (Same BC Code) ...
        rng = np.random.default_rng()
        trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            rng=rng,
            l2_weight=0.001,
            ent_weight=1e-3,
            batch_size=32,
        )
        print("Starting BC Training (100 Epochs)...")
        trainer.train(n_epochs=100) 
        trainer.policy.save(output_path)
        print(f"BC Policy saved to {output_path}")
        
    elif algorithm == "GAIL":
        reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        
        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            batch_size=64,
            ent_coef=0.05, # Higher entropy to encourage exploration in GridWorld
            learning_rate=0.0003,
            n_epochs=20, # Increased from 10 to squeeze more juice out of each rollout
            verbose=1
        )
        
        trainer = GAIL(
            demonstrations=transitions,
            demo_batch_size=32,
            gen_replay_buffer_capacity=2048, # Standard PPO buffer size (was 512 - too small)
            n_disc_updates_per_round=2,      # Lower discriminator updates to prevent overpowering (was 4)
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
        )
        
        # Custom Callback for GAIL (The official way for 'imitation' lib is a simple function)
        best_reward = -np.inf
        
        def gail_callback(round_num):
            nonlocal best_reward
            # Check performance every 5 rounds
            if round_num % 5 == 0:
                print(f"--- Eval Round {round_num} ---")
                mean_reward, _ = evaluate_policy(trainer.gen_algo, eval_env, n_eval_episodes=5)
                print(f"Mean Reward: {mean_reward}")
                
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    print("New Best Model! Saving...")
                    trainer.gen_algo.save(os.path.join(log_dir, "best_model.zip"))
        
        print(f"Starting GAIL Training (100,000 Timesteps)...")
        trainer.train(total_timesteps=100000, callback=gail_callback)
        
        # Check if best_model was created and overwrite output_path
        best_model_path = os.path.join(log_dir, "best_model.zip")
        if os.path.exists(best_model_path):
            print(f"Found best model at {best_model_path}. Overwriting output file...")
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(best_model_path, output_path)
            print(f"Best GAIL Policy saved to {output_path}")
        else:
            trainer.gen_algo.save(output_path)
            print(f"GAIL Final Policy saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to demonstrations file")
    parser.add_argument("--output", type=str, required=True, help="Path to save trained policy")
    parser.add_argument("--gym", type=str, required=True, help="Environment name: CartPole or Custom")
    parser.add_argument("--algorithm", type=str, required=True, choices=["BC", "GAIL"], help="Algorithm: BC or GAIL")
    
    args = parser.parse_args() # argparse automatically handles --help
    
    train(args.file, args.output, args.gym, args.algorithm)
