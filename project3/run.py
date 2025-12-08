import argparse
import gymnasium as gym
import sys
import os
import time
import msvcrt
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch

# Registration
from gymnasium.envs.registration import register
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Formula for Max Steps: int(((n*m)/2)/10) * 10
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

def run(policy_path, env_name):
    if env_name == "Custom":
        gym_id = "GridWorld-v0"
    elif env_name == "CartPole":
        gym_id = "CartPole-v1"
    else:
        gym_id = env_name

    env = gym.make(gym_id, render_mode="human")
    
    print(f"Loading policy from {policy_path}...")
    
    # Load Logic (Handling BC vs GAIL/PPO formats)
    # Spec doesn't specify how to distinguish, but we can try to guess or just load PPO.
    # BC saves as ActorCriticPolicy (via .save), GAIL saves as PPO (via .save).
    # PPO.load can often load Policy objects if structure matches? No.
    # We should probably use a try-except strategy or infer.
    # Let's assume PPO load first (GAIL), then Policy load (BC).
    
    # Monkeypatch for torch.load security
    _original_load = torch.load
    def _safe_load(*args, **kwargs):
        if 'weights_only' not in kwargs: kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _safe_load
    
    model = None
    policy = None
    
    try:
        model = PPO.load(policy_path)
        predict_fn = lambda obs: model.predict(obs, deterministic=True)
        print("Loaded as PPO/GAIL model.")
    except Exception:
        try:
            policy = ActorCriticPolicy.load(policy_path)
            predict_fn = lambda obs: policy.predict(obs, deterministic=True)
            print("Loaded as BC Policy.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    finally:
        torch.load = _original_load
    
    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Clear Screen
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Episode: {ep+1}/{episodes} | Step: {steps} | Total Reward: {total_reward:.2f}")
            env.render()
            
            time.sleep(0.3)
            
            action, _ = predict_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
        print(f"Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to policy file")
    parser.add_argument("--gym", type=str, required=True, help="Environment name: CartPole or Custom")
    
    args = parser.parse_args()
    
    run(args.file, args.gym)
