import argparse
import gymnasium as gym
import numpy as np
import os
import sys
import torch
from stable_baselines3 import PPO
from imitation.algorithms import bc

# Add src to path to find custom_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gymnasium.envs.registration import register

# Register GridWorld (Copying registration logic to be standalone)
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

def run_policy(policy_path, env_name, algorithm, mode):
    print(f"\n--- RUNNING POLICY ---")
    print(f"Policy: {policy_path}")
    print(f"Env: {env_name}")
    print(f"Mode: {mode}")
    print("-" * 30)

    # 1. Load Env
    render_mode = "human"
    env = gym.make(env_name, render_mode=render_mode)
    
    # 2. Load Model
    try:
        if algorithm == "BC":
            model = bc.reconstruct_policy(policy_path)
        else: # GAIL
            model = PPO.load(policy_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Simulation Loop
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    
    try:
        while True:
            # Step control
            if mode == "step":
                input(f"Step {steps} | Reward {total_reward:.2f} > Press Enter...")
            else:
                import time
                time.sleep(0.05) # Small delay for smooth viz
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                print(f"Episode Finished! Total Reward: {total_reward} | Steps: {steps}")
                obs, _ = env.reset()
                total_reward = 0
                steps = 0
                if mode == "step":
                    if input("Press Enter to restart or 'q' to quit: ").lower() == 'q':
                        break
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to policy file")
    parser.add_argument("--gym", type=str, required=True, help="Environment: CartPole or Custom")
    parser.add_argument("--algorithm", type=str, default="BC", choices=["BC", "GAIL"], help="Algorithm used (needed for loading)")
    parser.add_argument("--mode", type=str, default="continuous", choices=["continuous", "step"], help="Execution mode")
    
    args = parser.parse_args()
    run_policy(args.file, args.gym, args.algorithm, args.mode)
