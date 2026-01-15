
"""
demos.py - Script para recolher demonstrações (trajetórias) para Imitation Learning.
Permite modo manual (teclado) e automático (PPO HuggingFace para CartPole).
Salva as demonstrações em formato compatível com a biblioteca imitation.
"""
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
from gymnasium.envs.registration import register

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import custom_env
except ImportError:
    print("[ERRO] Não foi possível importar custom_env. Verifica se o ficheiro está em src.")

# Parâmetros do ambiente custom
GRID_SIZE = 8
LIMIT_STEPS = int(((GRID_SIZE * GRID_SIZE) / 2) / 10) * 10
try:
    # GridWorld is registered dynamically in collect_demos with spawn mode
    pass
except Exception as e:
    print(f"[AVISO] Falha ao registar GridWorld-v0: {e}")

def get_key():
    """
    Lê uma tecla do teclado (Windows). Mapeia setas para ações do GridWorld.
    Returns: int | None
    """
    key = msvcrt.getch()
    if key == b'\x1b': # ESC
        return "QUIT"
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

def get_key_pygame():
    """Reads arrow keys from Pygame directly."""
    import pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "QUIT"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "QUIT"
            if event.key == pygame.K_LEFT:
                return 0 # Left
            if event.key == pygame.K_RIGHT:
                return 1 # Right
            # GridWorld mappings
            if event.key == pygame.K_UP:
                return 1 # Up for GridWorld? (Check mapping)
                # Wait, GridWorld uses: 
                # 0: Right, 1: Up, 2: Left, 3: Down
            pass
    
    # For continuous polling or state check
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]: return "QUIT"
    if keys[pygame.K_LEFT]: return "LEFT"
    if keys[pygame.K_RIGHT]: return "RIGHT"
    if keys[pygame.K_UP]: return "UP"
    if keys[pygame.K_DOWN]: return "DOWN"
    
    return None

def get_action_pygame(env_name):
    """
    Unified input handler using Pygame.
    Non-blocking: Returns action if key is held, else None.
    Pumps events to keep window alive.
    """
    import pygame
    
    # Pump events handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
             return "QUIT"
        if event.type == pygame.KEYDOWN:
             if event.key == pygame.K_ESCAPE: return "QUIT"

    # Continuous check
    keys = pygame.key.get_pressed()
    
    if env_name == "Custom":
        if keys[pygame.K_RIGHT]: return 0
        if keys[pygame.K_UP]: return 1
        if keys[pygame.K_LEFT]: return 2
        if keys[pygame.K_DOWN]: return 3
    else:
        # CartPole Mapping
        if keys[pygame.K_LEFT]: return 0
        if keys[pygame.K_RIGHT]: return 1
        
    return None

def save_trajectories(trajectories, file_path):
    """
    Salva as trajetórias de forma robusta (temp file + rename).
    """
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, "demos.pkl")
        print(f"Target is a directory. Saving to: {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    temp_path = file_path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"Saved to temporary file: {temp_path}")
        max_retries = 3
        for i in range(max_retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                os.rename(temp_path, file_path)
                print(f"\u2705 Successfully saved {len(trajectories)} trajectories to {file_path}")
                break
            except PermissionError:
                if i < max_retries - 1:
                    print(f"\u26a0\ufe0f Target file locked. Retrying in 2s ({i+1}/{max_retries})...")
                    time.sleep(2)
                else:
                    print(f"\u274c PERMISSION DENIED: Could not rename to {file_path}.")
                    print(f"\u26a0\ufe0f DATA IS SAFE! Your episodes are in: {temp_path}")
                    print("Please rename this file manually when the other program closes.")
            except Exception as e:
                print(f"\u274c Error renaming file: {e}")
                print(f"\u26a0\ufe0f DATA IS SAFE! Your episodes are in: {temp_path}")
                break
    except Exception as e:
        print(f"\u274c CRITICAL ERROR saving data: {e}")

def collect_demos(env_name, num_episodes, output_file, use_pretrained=False, spawn_mode="random"):
    """
    Recolhe demonstrações (trajetórias) no ambiente escolhido.
    Se use_pretrained=True e env_name=CartPole, usa PPO HuggingFace.
    Caso contrário, usa input manual do utilizador.
    """
    print(f"Collecting {num_episodes} episodes on {env_name}...")
    if env_name == "Custom":
        # Register GridWorld with spawn mode
        print(f"DEBUG: GridWorld Spawn Mode Arg: {spawn_mode}")
        random_start = (spawn_mode == "random")
        print(f"DEBUG: random_start calculated as: {random_start}")
        try:
            register(
                id="GridWorld-v0",
                entry_point="custom_env:GridWorldEnv",
                max_episode_steps=LIMIT_STEPS,
                kwargs={'size': GRID_SIZE, 'random_start': random_start}
            )
        except Exception as e:
            print(f"[DEBUG] Registration warning: {e}")
        
        gym_id = "GridWorld-v0"
        try:
            # Force random_start in make to ensure it overrides defaults/cache
            print(f"[DEBUG] Calling gym.make with random_start={random_start}")
            env = gym.make(gym_id, render_mode="human", random_start=random_start)
        except Exception as e:
            print(f"Error creating environment '{gym_id}': {e}")
            return
    elif env_name == "CartPole" or env_name == "CartPole-v1":
        gym_id = "CartPole-v1"
        # Only render if NOT using pretrained (Manual mode)
        render_mode = "human" if not use_pretrained else None
        env = gym.make(gym_id, render_mode=render_mode)
    else:
        gym_id = env_name
        render_mode = "human"
        env = gym.make(gym_id, render_mode=render_mode)
    
    if env is None: # Should not happen if env_name is valid
        print(f"Error: Could not create environment for '{env_name}'.")
        return

    if use_pretrained and (env_name == "CartPole" or env_name == "CartPole-v1"):
        print("Using Pretrained Agent (HuggingFace)...")
        checkpoint = load_from_hub(
            repo_id="sb3/ppo-CartPole-v1",
            filename="ppo-CartPole-v1.zip",
        )
        model = PPO.load(checkpoint)
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(gym_id))])
        rollouts = rollout.rollout(
            model,
            venv,
            rollout.make_sample_until(min_episodes=num_episodes),
            rng=np.random.default_rng(),
        )
        save_trajectories(rollouts, output_file)
    else:
        print("Using Human Input (Keyboard)...")
        if env_name == "Custom":
            print("Control: ARROW KEYS")
        else:
            print("Control: Use LEFT/RIGHT ARROWS to balance the pole. Press ESC to quit.")
        trajectories = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            obs_list = [obs]
            acts_list = []
            rews_list = []
            print(f" Episode {ep+1} Start")
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Episode {ep+1} - Start")
            # Only render if window exists
            if hasattr(env, 'render_mode') and env.render_mode == "human":
                env.render()
            while not done:
                action = None
                while action is None:
                    # Use Pygame input if available (CartPole uses pygame via gymnasium)
                    # GridWorld might use simple console or pygame depending on render mode
                    
                    # Check if we can use pygame logic
                    USE_PYGAME_INPUT = False
                    if (env_name == "CartPole" or env_name == "CartPole-v1") and hasattr(env, 'render_mode') and env.render_mode == "human":
                        # If human render, getting input from window is best
                        USE_PYGAME_INPUT = True
                        
                    if USE_PYGAME_INPUT:
                        res = get_action_pygame(env_name)
                        if res == "QUIT":
                            print("Exiting...")
                            env.close()
                            sys.exit(0)
                        
                        if res is not None:
                            action = res
                        else:
                            # Render and short sleep to allow "Pause" effect
                            if hasattr(env, 'render_mode') and env.render_mode == "human":
                                env.render()
                            time.sleep(0.02) # 50Hz polling when idle
                            
                    else:
                        # Fallback to MSVCRT (Console) for GridWorld
                        if msvcrt.kbhit():
                            k = get_key() # Old function
                            if k == "QUIT":
                                print("Exiting...")
                                env.close()
                                sys.exit(0)
                            if env_name == "Custom":
                                action = k
                        else:
                            time.sleep(0.05)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"Episode {ep+1} | Reward: {sum(rews_list) + reward}")
                if hasattr(env, 'render_mode') and env.render_mode == "human":
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
        
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_trajectories(trajectories, output_file)
    env.close()

def validate_args(args):
    """
    Valida argumentos da linha de comandos e avisa o utilizador de erros comuns.
    """
    if not args.gym in ["CartPole", "Custom"]:
        print("[AVISO] --gym deve ser 'CartPole' ou 'Custom'.")
    if args.episodes <= 0:
        print("[ERRO] --episodes deve ser > 0.")
        exit(1)
    if not args.file:
        print("[ERRO] --file é obrigatório.")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recolhe demonstrações para Imitation Learning.")
    parser.add_argument("--gym", type=str, required=True, help="Environment name: CartPole ou Custom")
    parser.add_argument("--episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--file", type=str, required=True, help="Output file path")
    parser.add_argument("--use-pretrained", action="store_true", help="Use pretrained agent (CartPole only)")
    parser.add_argument("--spawn", type=str, default="random", choices=["random", "fixed"], help="Spawn mode for GridWorld (random or fixed)")
    args = parser.parse_args()
    validate_args(args)
    args = parser.parse_args()
    validate_args(args)
    try:
        collect_demos(args.gym, args.episodes, args.file, args.use_pretrained, args.spawn)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] The program crashed: {e}")
    finally:
        print("\nPress Enter to exit...")
        input()
        print("\n" + "="*50)
        print("CRITICAL ERROR DURING DEMONSTRATION RECORDING")
        print("="*50)
        traceback.print_exc()
        print("="*50)
        input("\nPress ENTER to close this window...")
