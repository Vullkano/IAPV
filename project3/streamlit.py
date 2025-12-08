import streamlit as st
import gymnasium as gym
import numpy as np
import time
import torch
import os
import sys
import matplotlib.pyplot as plt
import io
import subprocess
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import threading

# --- Boilerplate for Import ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gymnasium.envs.registration import register

# --- Page Config ---
st.set_page_config(
    page_title="IAPV Imitation Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for "Ultra Bonito" Look ---
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Elegant Headers */
    h1, h2, h3 {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
    }
    
    /* Card Style Containers */
    .css-1r6slb0, .stMarkdown {
        border-radius: 15px;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(215, 109, 119, 0.4);
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: white;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-bottom: 2px solid #84fab0;
    }
</style>
""", unsafe_allow_html=True)

# --- Dynamic Env Registration ---
GRID_SIZE = 8
# Formula: int(((n*m)/2)/10) * 10
LIMIT_STEPS = int(((GRID_SIZE * GRID_SIZE) / 2) / 10) * 10

try:
    register(
        id="GridWorld-v0",
        entry_point="custom_env:GridWorldEnv",
        max_episode_steps=LIMIT_STEPS,
        kwargs={'size': GRID_SIZE}
    )
except Exception:
    pass

# --- Helper Functions ---
@st.cache_resource
def load_model(algo, gym_name):
    """Loads the model intelligently."""
    # Monkeypatch for torch load
    _original_load = torch.load
    def _safe_load(*args, **kwargs):
        if 'weights_only' not in kwargs: kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _safe_load
    
    # Define paths based on standard structure
    folder = "output"
    
    if algo == "BC" and gym_name == "Custom":
        path = f"{folder}/bc_grid.zip"
    elif algo == "GAIL" and gym_name == "Custom":
        path = f"{folder}/gail_grid.zip"
    elif algo == "BC" and gym_name == "CartPole":
        path = f"{folder}/bc_cartpole.zip"
    elif algo == "GAIL" and gym_name == "CartPole":
        path = f"{folder}/gail_cartpole.zip"
    else:
        path = None

    if not path or not os.path.exists(path):
        return None, f"File not found: {path}"
        
    try:
        model = PPO.load(path)
        return model, None
    except:
        try:
            model = ActorCriticPolicy.load(path)
            return model, None
        except Exception as e:
            return None, str(e)
    finally:
        torch.load = _original_load

def run_training_subprocess(command):
    """Runs a training command in a subprocess and yields output."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
        bufsize=1,
        universal_newlines=True
    )
    return process

# --- Main App ---
st.title("🌌 IAPV Neural Nexus")
st.caption(f"Advanced Imitation Learning Studio • Grid Limit: {LIMIT_STEPS} Steps")

# Tabs for workflow
tab_data, tab_train, tab_viz = st.tabs(["📁 1. Data Collection", "🧠 2. Training Center", "👁️ 3. Neural Visualization"])

# ==========================================
# 1. DATA COLLECTION TAB
# ==========================================
with tab_data:
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.subheader("🛠️ Capture Demonstrations")
        st.info("Launch the external data collector. For **Custom**, a window will open for arrow keys.")
        
        d_env = st.selectbox("Select Environment", ["Custom", "CartPole"], key="d_env")
        d_episodes = st.number_input("Episodes to Collect", 10, 200, 50 if d_env=="CartPole" else 20)
        
        file_name = "demos_grid.pkl" if d_env == "Custom" else "demos_cartpole.pkl"
        output_path = f"output/{file_name}"
        
        # Ensure output dir exists
        os.makedirs("output", exist_ok=True)
        
        cmd = f"python demos.py --gym {d_env} --episodes {d_episodes} --file {output_path}"
        if d_env == "CartPole":
            cmd += " --use-pretrained"
            
        st.code(cmd, language="bash")
        
        if st.button("🚀 Launch Collector (External Window)"):
            try:
                # Use Shell=True for windows piping
                subprocess.Popen(f'start cmd /k "{cmd}"', shell=True)
                st.success("Collector launched in new terminal! Please provide input there.")
            except Exception as e:
                st.error(f"Failed to launch: {e}")

    with col_d2:
        st.subheader("📊 Dataset Status")
        
        check_path = output_path
        if os.path.isdir(output_path):
            check_path = os.path.join(output_path, "demos.pkl")
            
        if os.path.exists(check_path):
            st.success(f"✅ Found {file_name}")
            try:
                stats = os.stat(check_path)
                st.metric("File Size", f"{stats.st_size / 1024:.2f} KB")
                st.caption("Ready for Training!")
            except:
                pass
        else:
            st.warning(f"⚠️ {file_name} not found.")
            st.error("You MUST collect demos before training.")

# ==========================================
# 2. TRAINING TAB
# ==========================================
with tab_train:
    col_t1, col_t2 = st.columns([1, 2])
    
    with col_t1:
        st.subheader("⚙️ Hyperparameters")
        t_env = st.selectbox("Environment", ["Custom", "CartPole"], key="t_env")
        t_algo = st.selectbox("Algorithm", ["BC", "GAIL"], key="t_algo")
        
        demos_file = "output/demos_grid.pkl" if t_env == "Custom" else "output/demos_cartpole.pkl"
        output_file = f"output/{t_algo.lower()}_{'grid' if t_env == 'Custom' else 'cartpole'}.zip"
        
        st.markdown("---")
        st.markdown("**Configurations:**")
        st.text(f"Input: {demos_file}")
        st.text(f"Output: {output_file}")
        
    with col_t2:
        st.subheader("🖥️ Training Console")
        
        if st.button("🔥 IGNITE TRAINING"):
            if not os.path.exists(demos_file):
                st.error(f"❌ DATA MISSING: {demos_file}")
                st.markdown("Please go to **Data Collection** tab and collect data first!")
            else:
                st.info(f"Training {t_algo} on {t_env}...")
                
                cmd = f"python train.py --gym {t_env} --file {demos_file} --output {output_file} --algorithm {t_algo}"
                
                output_area = st.empty()
                logs = []
                
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        shell=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            logs.append(line.strip())
                            # Auto-scroll log
                            output_area.code("\n".join(logs[-15:]), language="text")
                    
                    if process.returncode == 0:
                        st.success("✨ Training Completed Successfully!")
                        st.balloons()
                    else:
                        st.error("Training Failed. Check logs above.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

# ==========================================
# 3. VISUALIZATION TAB
# ==========================================
with tab_viz:
    col1, col2 = st.columns([3, 1])

    # Sidebar for Viz settings (Moved to column logic for tab purity)
    with col2:
        st.markdown("### 🎮 Control Center")
        
        v_env_name = st.selectbox("Env", ["Custom (GridWorld)", "CartPole"], key="v_env")
        v_algo = st.selectbox("Model", ["BC", "GAIL"], key="v_algo")
        
        clean_algo = "BC" if "BC" in v_algo else "GAIL"
        clean_env = "Custom" if "Custom" in v_env_name else "CartPole"
        
        n_test_episodes = st.slider("Episodes", 1, 10, 3)
        speed = st.slider("Speed Factor", 0.1, 2.0, 1.0)
        
        model, err = load_model(clean_algo, clean_env)
        
        st.markdown("---")
        if err:
            st.error("❌ Model Not Found")
            st.caption(err)
        elif model:
            st.success("✅ Model Ready")
        
        ep_placeholder = st.empty()
        reward_placeholder = st.empty()
        
        start_btn = st.button("▶️ PLAY SIMULATION", disabled=not model)
        stop_btn = st.button("⏹️ STOP")
        
        if start_btn:
            st.session_state['viz_running'] = True
        if stop_btn:
            st.session_state['viz_running'] = False

    if st.session_state.get('viz_running') and model:
        gym_id = "GridWorld-v0" if clean_env == "Custom" else "CartPole-v1"
        try:
            # Re-register if needed to ensure limits apply
            env = gym.make(gym_id, render_mode="rgb_array")
        except:
             register(
                id="GridWorld-v0",
                entry_point="custom_env:GridWorldEnv",
                max_episode_steps=LIMIT_STEPS,
                kwargs={'size': GRID_SIZE}
            )
             env = gym.make(gym_id, render_mode="rgb_array")
        
        with col1:
            img_placeholder = st.empty()
            
            for ep in range(1, n_test_episodes + 1):
                if not st.session_state.get('viz_running'): break
                
                obs, _ = env.reset()
                done = False
                total_reward = 0
                steps = 0
                
                ep_placeholder.metric("Episode", f"{ep}/{n_test_episodes}")
                
                while not done:
                    # Predict
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    steps += 1
                    
                    # Render
                    if clean_env == "Custom":
                        unwrapped = env.unwrapped
                        grid_size = unwrapped.size
                        
                        fig, ax = plt.subplots(figsize=(6,6))
                        ax.set_xlim(0, grid_size)
                        ax.set_ylim(0, grid_size)
                        ax.invert_yaxis()
                        
                        # Style match dark theme
                        fig.set_facecolor('#0e1117')
                        ax.set_facecolor('#0e1117')
                        
                        # Custom Grid
                        ax.set_xticks(np.arange(0, grid_size, 1))
                        ax.set_yticks(np.arange(0, grid_size, 1))
                        ax.grid(which='both', color='#2c3e50', linestyle='-', linewidth=1, alpha=0.5)
                        
                        # Walls (Neon Style)
                        for obs_pos in unwrapped._obstacles:
                            rect = plt.Rectangle((obs_pos[1], obs_pos[0]), 1, 1, color='#34495e', ec='#7f8c8d', lw=2) 
                            ax.add_patch(rect)
                        
                        # Glow Effects (Fake it with stacking)
                        target = unwrapped._target_location
                        # Outer Glow
                        ax.add_patch(plt.Circle((target[1] + 0.5, target[0] + 0.5), 0.4, color='#e74c3c', alpha=0.3))
                        # Core
                        ax.add_patch(plt.Circle((target[1] + 0.5, target[0] + 0.5), 0.25, color='#c0392b', zorder=10))
                        ax.text(target[1]+0.5, target[0]+0.5, "G", color='white', ha='center', va='center', weight='bold', fontsize=12)

                        agent = unwrapped._agent_location
                        # Outer Glow
                        ax.add_patch(plt.Circle((agent[1] + 0.5, agent[0] + 0.5), 0.4, color='#3498db', alpha=0.3))
                        # Core
                        ax.add_patch(plt.Circle((agent[1] + 0.5, agent[0] + 0.5), 0.25, color='#2980b9', zorder=20))
                        # Directional Arrow for flair could be added here if we knew orientation, but we don't.
                        
                        ax.axis('off')
                        
                        # Buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1, facecolor='#0e1117')
                        buf.seek(0)
                        img_placeholder.image(buf)
                        plt.close(fig)
                        
                    else:
                        frame = env.render()
                        img_placeholder.image(frame, use_container_width=True)
                    
                    # Update Metrics
                    reward_placeholder.metric("Total Reward", f"{total_reward:.2f}")
                    
                    delay = 1.0 / (speed * 4) if clean_env == "Custom" else 1.0 / (speed * 40)
                    time.sleep(delay)
                    
                    if not st.session_state.get('viz_running'):
                        break
                
                time.sleep(0.5)
            
            st.session_state['viz_running'] = False
            st.success("Test sequence complete.")
