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
try:
    from demos import GRID_SIZE
except ImportError:
    GRID_SIZE = 8 # Fallback

LIMIT_STEPS = int(((GRID_SIZE * GRID_SIZE) / 2) / 10) * 10
try:
    import custom_env
    # GridWorld is registered dynamically in run_policy with spawn mode
except Exception:
    pass

def run_policy(policy_path, env_name, algorithm, mode, spawn_mode="random"):
    print(f"\n--- RUNNING POLICY ---")
    print(f"Policy: {policy_path}")
    print(f"Env: {env_name}")
    print(f"Mode: {mode}")
    if env_name == "Custom":
        print(f"Spawn Mode: {spawn_mode}")
    print("-" * 30)

    # 1. Load Env
    if env_name == "Custom":
        # Register GridWorld with spawn mode
        random_start = (spawn_mode == "random")
        try:
            register(
                id="GridWorld-v0",
                entry_point="custom_env:GridWorldEnv",
                max_episode_steps=LIMIT_STEPS,
                kwargs={'size': GRID_SIZE, 'random_start': random_start}
            )
        except:
            pass
        gym_id = "GridWorld-v0"
    elif env_name == "CartPole":
        gym_id = "CartPole-v1"
    else:
        gym_id = env_name

    render_mode = "human"
    env = gym.make(gym_id, render_mode=render_mode)
    
    # 2. Load Model
    try:
        # Patch torch.load to handle weights_only=True default in newer PyTorch
        _original_load = torch.load
        def _safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = _safe_load

        if algorithm == "BC":
            # BC saves a standard FeedForward/ActorCritic policy. 
            # reconstruct_policy is for full algorithm checkpoints, but we saved policy directly.
            # Try loading as a generic ActorCriticPolicy
            from stable_baselines3.common.policies import ActorCriticPolicy
            model = ActorCriticPolicy.load(policy_path)
        else: # GAIL
            model = PPO.load(policy_path)
            
        # Restore original
        torch.load = _original_load
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
    parser.add_argument("--spawn", type=str, default="random", choices=["random", "fixed"], help="Spawn mode for GridWorld (random or fixed)")
    
    args = parser.parse_args()
    run_policy(args.file, args.gym, args.algorithm, args.mode, args.spawn)
