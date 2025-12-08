import argparse
import gymnasium as gym
import numpy as np
import os
import sys
import msvcrt
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import TrajectoryWithRew
from huggingface_sb3 import load_from_hub
import pickle

# --- Registration ---
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
except Exception as e:
    print(f"Failed to register GridWorld-v0: {e}")

# --- Helper for Human Input ---
def get_key():
    """Reads a keypress from stdin (Windows)."""
    # Key Mapping for GridWorld-v0 (0=Right, 1=Up, 2=Left, 3=Down)
    
    key = msvcrt.getch()
    if key == b'\xe0':
        key = msvcrt.getch()
        mapping = {
            b'H': 1, # Up
            b'P': 3, # Down
            b'K': 2, # Left
            b'M': 0, # Right
        }
        return mapping.get(key, None)
    return None

def collect_demos(env_name, n_episodes, file_path, use_pretrained):
    print(f"Collecting {n_episodes} episodes on {env_name}...")
    
    if env_name == "Custom":
        gym_id = "GridWorld-v0"
    elif env_name == "CartPole":
        gym_id = "CartPole-v1"
    else:
        gym_id = env_name

    try:
        # Create Env
        env = gym.make(gym_id, render_mode="human")
    except Exception as e:
        print(f"Error creating environment '{gym_id}': {e}")
        return

    if use_pretrained and env_name == "CartPole":
        print("Using Pretrained Agent (HuggingFace)...")
        # Load from HuggingFace
        checkpoint = load_from_hub(
            repo_id="sb3/ppo-CartPole-v1",
            filename="ppo-CartPole-v1.zip",
        )
        model = PPO.load(checkpoint)
        
        # Use imitation rollout helper
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(gym_id))])
        rollouts = rollout.rollout(
            model,
            venv,
            rollout.make_sample_until(min_episodes=n_episodes),
            rng=np.random.default_rng(),
        )
        # Use pickle for robust saving
        with open(file_path, "wb") as f:
            pickle.dump(rollouts, f)
        print(f"Saved {len(rollouts)} trajectories to {file_path} (PICKLE)")
        
    else:
        print("Using Human Input (Keyboard)...")
        if env_name == "Custom":
            print("Control: ARROW KEYS")
        else:
            print("Control: Warning - CartPole manual control not fully implemented.")

        trajectories = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            obs_list = [obs]
            acts_list = []
            rews_list = []
            
            print(f" Episode {ep+1} Start")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Episode {ep+1} - Start")
            env.render()
            
            while not done:
                action = None
                while action is None:
                    k = get_key()
                    if env_name == "CartPole":
                        # Map Left(2)->0, Right(0)->1
                        if k == 2: action = 0
                        elif k == 0: action = 1
                    else:
                        action = k
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Clear and Render
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Episode {ep+1} | Reward: {sum(rews_list) + reward}")
                env.render()
                
                obs_list.append(next_obs)
                acts_list.append(action)
                rews_list.append(reward)
                obs = next_obs
            
            traj = TrajectoryWithRew(
                obs=np.array(obs_list),
                acts=np.array(acts_list),
                infos=None,
                terminal=True,
                rews=np.array(rews_list)
            )
            trajectories.append(traj)
            
        # Handle directory input (User has output/demos_grid.pkl as a folder)
        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, "demos.pkl")
            print(f"Target is a directory. Saving to: {file_path}")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Robust Saving Logic: Write to temp -> Rename
        temp_path = file_path + ".tmp"
        
        try:
            # 1. Write to temporary file (Atomic write)
            with open(temp_path, "wb") as f:
                pickle.dump(trajectories, f)
            print(f"Saved to temporary file: {temp_path}")
            
            # 2. Try to rename (replace) the target file
            max_retries = 3
            for i in range(max_retries):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path) # Force remove in case of Windows issues
                    os.rename(temp_path, file_path)
                    print(f"✅ Successfully saved {len(trajectories)} trajectories to {file_path}")
                    break
                except PermissionError:
                    if i < max_retries - 1:
                        print(f"⚠️ Target file locked. Retrying in 2s ({i+1}/{max_retries})...")
                        time.sleep(2)
                    else:
                        print(f"❌ PERMISSION DENIED: Could not rename to {file_path}.")
                        print(f"⚠️ DATA IS SAFE! Your episodes are in: {temp_path}")
                        print("Please rename this file manually when the other program closes.")
                except Exception as e:
                     print(f"❌ Error renaming file: {e}")
                     print(f"⚠️ DATA IS SAFE! Your episodes are in: {temp_path}")
                     break
                     
        except Exception as e:
            print(f"❌ CRITICAL ERROR saving data: {e}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym", type=str, required=True, help="Environment name: CartPole or Custom")
    parser.add_argument("--episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--file", type=str, required=True, help="Output file path")
    parser.add_argument("--use-pretrained", action="store_true", help="Use pretrained agent (CartPole only)")
    
    args = parser.parse_args()
    
    collect_demos(args.gym, args.episodes, args.file, args.use_pretrained)
