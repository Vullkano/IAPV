
import streamlit as st
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import os
import sys
import pickle
import time
import subprocess
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import base64

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="IAPV Suite de Imitação",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Adicionar path para imports locais e DEFINIR PASTA RAIZ E OUTPUT
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) # Sobbe um nível para a raiz do projeto (c:\Users\diogo\Desktop\IAPV)
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

sys.path.append(CURRENT_DIR)

# --- 2. CSS "ELEGANCE & COMPLEXITY" (Dark, Premium, Professional) ---
def inject_premium_css():
    css_path = os.path.join(CURRENT_DIR, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"Erro: Arquivo de estilo {css_path} não encontrado.")

inject_premium_css()

# --- 3. LÓGICA DO AMBIENTE ---
GRID_SIZE = 8
LIMIT_STEPS = int(((GRID_SIZE * GRID_SIZE) / 2) / 10) * 10

try:
    register(
        id="GridWorld-v0",
        entry_point="custom_env:GridWorldEnv",
        max_episode_steps=LIMIT_STEPS,
        kwargs={'size': GRID_SIZE}
    )
except:
    pass

@st.cache_resource
def load_model(algo, gym_name):
    # USE OUTPUT_DIR DEFINED AT TOP
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{algo.lower()}_{'grid' if gym_name=='Custom' else 'cartpole'}.zip"
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path): return None, f"Modelo não encontrado: {path}"
    try:
        _original_load = torch.load
        def _safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = _safe_load
        try: model = PPO.load(path)
        except: model = ActorCriticPolicy.load(path)
        torch.load = _original_load
        return model, None
    except Exception as e: return None, str(e)

# --- 4. RENDERIZAÇÃO MODERNA (HTML RETORNADO) ---
def get_grid_html(size, agent_pos, target_pos, walls):
    html = f'<div class="grid-container" style="grid-template-columns: repeat({size}, 1fr);">'
    for r in range(size):
        for c in range(size):
            pos = (r, c)
            is_wall = pos in walls
            is_agent = (pos == tuple(agent_pos))
            is_target = (pos == tuple(target_pos))
            cell_class = "grid-cell"; content = ""
            if is_wall: cell_class += " wall"
            if is_target: content += '<div class="token">🏁</div>'
            if is_agent: content += '<div class="token">🤖</div>'
            html += f'<div class="{cell_class}">{content}</div>'
    html += '</div>'
    return html

def render_monitor_frame(env_name, content_html):
    return f"""<div class="monitor-frame"><div class="monitor-header"><div style="display:flex; gap:8px; align-items:center;"><span style="color:#ef4444;">●</span> LIVE REC</div><div style="opacity:0.6;">{env_name}</div></div><div class="monitor-screen">{content_html}</div></div>"""

# --- 5. INTERFACE PRINCIPAL ---

st.markdown("""
<div style='display:flex; align-items:center; gap:20px; margin-bottom:2rem; padding-bottom:1rem; border-bottom:1px solid #27272a;'>
    <div style='width:64px; height:64px; background: linear-gradient(135deg, #6366f1 0%, #a5b4fc 100%); border-radius:16px; display:flex; align-items:center; justify-content:center; box-shadow:0 8px 32px rgba(99,102,241,0.3);'>
        <span style='font-size:32px;'>⚡</span>
    </div>
    <div>
        <h1 style='margin:0 !important; font-size:2.5rem !important; line-height:1.1; background: linear-gradient(90deg, #fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>IAPV Suite de Imitação</h1>
        <div style='font-size:1.1rem; color:#a1a1aa; font-family:"Inter", sans-serif; font-weight:400; margin-top:4px;'>Painel Avançado de Aprendizagem por Reforço <span style="opacity:0.6">&mdash; Grupo 7</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

selected_env = st.session_state.get("selected_env", "GridWorld (Custom)")
# Ensure env name is simple for UI
env_display = "GridWorld" if "Grid" in selected_env else "CartPole-v1"

# --- SIDEBAR GLOBAL ---
with st.sidebar:
    st.markdown("""<div style="background:linear-gradient(180deg, #18181b 0%, #09090b 100%); border-radius:16px; padding:24px; margin-bottom:24px; border:1px solid #27272a; box-shadow:0 4px 24px rgba(0,0,0,0.4);"><div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;"><div style="width:40px; height:40px; background:rgba(16, 185, 129, 0.1); border-radius:10px; display:flex; align-items:center; justify-content:center; color:#10b981; font-size:1.4rem;">🧭</div><div><div style="font-size:1.1rem; font-weight:700; color:#ededed; font-family:'Outfit', sans-serif;">Ambiente RL</div><div style="font-size:0.85rem; color:#71717a; font-weight:500;">Espaço de Trabalho</div></div></div><div style="font-size:0.9rem; color:#a1a1aa; line-height:1.5;">Selecione o desafio.</div></div>""", unsafe_allow_html=True)
    if "selected_env" not in st.session_state: st.session_state.selected_env = "GridWorld (Custom)"
    envs = ["GridWorld (Custom)", "CartPole-v1"]; env_icons = ["🌐", "🎢"]
    cols = st.columns(2, gap="small")
    grid_selected = st.session_state.selected_env == "GridWorld (Custom)"
    cart_selected = st.session_state.selected_env == "CartPole-v1"
    
    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-of-type(1) button {{ background: {'linear-gradient(135deg, #4338ca 0%, #6366f1 100%)' if grid_selected else '#18181b'} !important; border: 1px solid {'#6366f1' if grid_selected else '#27272a'} !important; color: {'#fff' if grid_selected else '#71717a'} !important; }}
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-of-type(2) button {{ background: {'linear-gradient(135deg, #4338ca 0%, #6366f1 100%)' if cart_selected else '#18181b'} !important; border: 1px solid {'#6366f1' if cart_selected else '#27272a'} !important; color: {'#fff' if cart_selected else '#71717a'} !important; }}
    </style>
    """, unsafe_allow_html=True)
    
    for i, (env, icon) in enumerate(zip(envs, env_icons)):
        with cols[i]:
            if st.button(f"{icon} {env.split()[0]}", key=f"envbtn_{i}", use_container_width=True):
                st.session_state.selected_env = env; st.rerun()

    st.markdown("""<div style="margin-top:auto; padding-top:2rem;"><div style="display:flex; justify-content:space-between; background:#121215; padding:12px; border-radius:12px; border:1px solid #27272a;"><span style="color:#71717a; font-size:0.85em; font-family:'JetBrains Mono';">v 2.0.0</span><span style="background:#27272a; color:#a1a1aa; font-size:0.8em; font-weight:600; border-radius:6px; padding:2px 8px;">BETA</span></div></div>""", unsafe_allow_html=True)

selected_env = st.session_state.selected_env
is_grid = selected_env == "GridWorld (Custom)"
clean_env_name = "Custom" if is_grid else "CartPole"

# CHECK DATA EXISTENCE - FIXED PATH RESOLUTION
demos_file = os.path.join(OUTPUT_DIR, "demos_grid.pkl") if is_grid else os.path.join(OUTPUT_DIR, "demos_cartpole.pkl")
has_demos = os.path.exists(demos_file)

# TABS LOGIC
if has_demos:
    tab_dash, tab_rec, tab_train, tab_viz = st.tabs(["Painel", "Gravar", "Treinar", "Visualizar"])
else:
    tab_dash, tab_rec = st.tabs(["Painel", "Gravar"])

# --- DASHBOARD ---
with tab_dash:
    st.markdown(f"""
<div class="premium-card"><div style="display:flex; justify-content:space-between; align-items:flex-start;"><div>
<h2 style="margin-top:0 !important; font-size:1.8rem;">Bem-vindo ao Suite</h2><div style="color:#a1a1aa; margin-top:0.5rem; font-size:1.1rem;">Ambiente Ativo: <strong style="color:#fff;">{selected_env}</strong></div></div>
<div style="background:rgba(99,102,241,0.1); color:#818cf8; padding:8px 16px; border-radius:20px; font-weight:600; font-size:0.9rem;">{env_display}</div></div>
</div>
""", unsafe_allow_html=True)
    
    # Overview Cards
    st.markdown("""
    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:20px; margin-top:24px;">
        <div style="background:#121215; border:1px solid #27272a; padding:24px; border-radius:12px;">
            <div style="font-size:1.5rem; margin-bottom:12px; color:#ef4444;">🔴</div>
            <div style="font-weight:700; font-size:1.1rem; color:#fff;">1. Gravar</div>
            <div style="color:#71717a; font-size:0.9rem; margin-top:4px;">Geração de demonstrações.</div>
        </div>
        <div style="background:#121215; border:1px solid #27272a; padding:24px; border-radius:12px;">
            <div style="font-size:1.5rem; margin-bottom:12px; color:#f59e0b;">🧠</div>
            <div style="font-weight:700; font-size:1.1rem; color:#fff;">2. Treinar</div>
            <div style="color:#71717a; font-size:0.9rem; margin-top:4px;">Behavioral Cloning ou GAIL.</div>
        </div>
        <div style="background:#121215; border:1px solid #27272a; padding:24px; border-radius:12px;">
            <div style="font-size:1.5rem; margin-bottom:12px; color:#6366f1;">👁️</div>
            <div style="font-weight:700; font-size:1.1rem; color:#fff;">3. Visualizar</div>
            <div style="color:#71717a; font-size:0.9rem; margin-top:4px;">Validação de métricas.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- GRAVAR (RECORD) ---
with tab_rec:
    # Header
    # Header
    header_color = "var(--brand-primary)" if is_grid else "#f59e0b"
    extra_info = "" if is_grid else f'<div style="color:#71717a; margin-top:4px;">Fonte: <span style="color:{header_color};">Agente Pré-treinado (HuggingFace)</span></div>'
    
    st.markdown(f"""
    <div style="background:var(--bg-card); border-left:4px solid {header_color}; padding:16px 24px; border-radius:0 8px 8px 0; margin-bottom:32px;">
        <h2 style="margin:0 !important; font-size:1.5rem !important;">Gravar Demonstrações</h2>
        <div style="color:var(--text-secondary); margin-top:4px;">Challenge: <strong style="color:#fff;">{env_display}</strong></div>
        {extra_info}
    </div>
    """, unsafe_allow_html=True)

    if is_grid:
        # GRIDWORLD LAYOUT
        col_config, col_game = st.columns([0.35, 0.65], gap="large")
        
        with col_config:
            st.markdown('<div style="font-size:1.1rem; font-weight:600; margin-bottom:20px; color:#fff;">⚙️ Configuração</div>', unsafe_allow_html=True)
            
            st.markdown('<label style="font-size:0.9rem; font-weight:500; color:#d4d4d8;">Episódios</label>', unsafe_allow_html=True)
            d_episodes = st.number_input("", min_value=5, value=20, step=5, key="d_episodes", label_visibility="collapsed")
            st.markdown('<div style="margin-bottom:24px;"></div>', unsafe_allow_html=True)

            st.markdown('<label style="font-size:0.9rem; font-weight:500; color:#d4d4d8;">Estratégia</label>', unsafe_allow_html=True)
            vocab_strategy = st.radio(
                "Estratégia", ["Único (n)", "Intervalo (1..n)"], 
                horizontal=True, key="training_strat_sel", label_visibility="collapsed"
            )
            st.markdown('<div style="margin-bottom:16px;"></div>', unsafe_allow_html=True)

            st.markdown('<label style="font-size:0.9rem; font-weight:500; color:#d4d4d8;">Interface</label>', unsafe_allow_html=True)
            rec_mode = st.radio(
                "Interface", ["CMD (Rápido)", "In-App (Web)"], 
                horizontal=True, key="rec_mode_sel", label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action Launcher for CMD
            if "CMD" in rec_mode:
                is_interval = "Intervalo" in vocab_strategy
                if is_interval:
                    file_name = f"demos_grid_{d_episodes}.pkl"
                    output_path = os.path.join(OUTPUT_DIR, "intervals", file_name)
                else:
                    file_name = "demos_grid.pkl"
                    output_path = os.path.join(OUTPUT_DIR, file_name)
                
                demos_script = os.path.join(CURRENT_DIR, "demos.py")
                cmd = f'python "{demos_script}" --gym {clean_env_name} --episodes {d_episodes} --file "{output_path}"'
                if st.button("🚀 Lançar Gravador Externo", use_container_width=True, type="primary"):
                    try:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        subprocess.Popen(f'start cmd /k "{cmd}"', shell=True)
                        st.toast("Terminal iniciado!")
                    except Exception as e: st.error(f"Erro: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # In-App Controls
            if "In-App" in rec_mode:
                if "rec_state" not in st.session_state: st.session_state.rec_state = "IDLE"
                
                # Show Start Button if IDLE or FINISHED
                if st.session_state.rec_state in ["IDLE", "FINISHED"]:
                    if st.button("🔴 Gravar Agora", use_container_width=True, type="primary"):
                        st.session_state.rec_state = "RECORDING"
                        st.session_state.ep_count = 0
                        st.session_state.demos_buffer = []
                        st.session_state.env = gym.make("GridWorld-v0", render_mode="rgb_array")
                        obs, _ = st.session_state.env.reset()
                        st.session_state.obs = obs
                        st.session_state.curr_ep_obs = [obs]
                        st.session_state.curr_ep_acts = []
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                elif st.session_state.rec_state == "RECORDING":
                    st.progress(st.session_state.ep_count / d_episodes, text=f"Episódio {st.session_state.ep_count + 1}/{d_episodes}")
                    if st.button("Cancelar", use_container_width=True): st.session_state.rec_state = "IDLE"; st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        with col_game:
            # Monitor
            if "In-App" in rec_mode and st.session_state.get("rec_state") == "RECORDING":
                if "env" in st.session_state: 
                    uw = st.session_state.env.unwrapped
                    content_html = get_grid_html(uw.size, uw._agent_location, uw._target_location, uw._obstacles)
                else: content_html = "..."
            else:
                icon = "⌨️" if "CMD" in rec_mode else "🛑"
                msg = "A aguardar input..." if "CMD" in rec_mode else "Inicie a gravação."
                content_html = f'<div style="text-align:center; color:#52525b;"><div style="font-size:3rem; margin-bottom:16px; opacity:0.5;">{icon}</div><div style="font-family:\'JetBrains Mono\';">{msg}</div></div>'
            
            st.markdown(render_monitor_frame(selected_env, content_html), unsafe_allow_html=True)

            # BEAUTIFUL DPAD CONTROL
            if "In-App" in rec_mode and st.session_state.get("rec_state") == "RECORDING":
                st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
                
                step = None
                # DPAD UI
                _, c_pad, _ = st.columns([1, 0.6, 1])
                
                with c_pad:
                    # CS1: Up
                    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
                    with c2:
                        if st.button("▴", key="btn_u", use_container_width=True): step = 1
                    
                    # CS2: Left, Down, Right
                    c4, c5, c6 = st.columns([1, 1, 1], gap="small")
                    with c4:
                        if st.button("◂", key="btn_l", use_container_width=True): step = 2
                    with c5:
                        if st.button("▾", key="btn_d", use_container_width=True): step = 3
                    with c6:
                        if st.button("▸", key="btn_r", use_container_width=True): step = 0

                if step is not None:
                    next_obs, reward, term, trunc, info = st.session_state.env.step(step)
                    done = term or trunc
                    st.session_state.curr_ep_acts.append(step)
                    if done:
                        # ONLY SAVE IF SUCCESS (Reward > 0 implies reached target in GridWorld)
                        if reward > 0:
                            st.session_state.demos_buffer.append({
                                "obs": np.array(st.session_state.curr_ep_obs),
                                "actions": np.array(st.session_state.curr_ep_acts), 
                                "rewards": np.array([1.0]*len(st.session_state.curr_ep_acts))
                            })
                            st.session_state.ep_count += 1
                            st.toast(f"Episódio {st.session_state.ep_count} Completo! 🎉")
                        else:
                            st.toast("Falhou (Colisão/Limite). A reiniciar episódio...", icon="❌")
                        
                        if st.session_state.ep_count >= d_episodes:
                            st.session_state.rec_state = "FINISHED"
                            
                            is_interval = "Intervalo" in vocab_strategy
                            if is_interval:
                                folder_path = os.path.join(OUTPUT_DIR, "intervals")
                                file_name = f"demos_grid_{d_episodes}.pkl"
                            else:
                                folder_path = OUTPUT_DIR
                                file_name = "demos_grid.pkl"
                            
                            try:
                                os.makedirs(folder_path, exist_ok=True)
                                save_path = os.path.join(folder_path, file_name)
                                # SAFETY SAVE
                                with open(save_path, "wb") as f: pickle.dump(st.session_state.demos_buffer, f)
                                st.balloons()
                                # RESET STATE TO IDLE SO BUTTON REAPPEARS
                                st.session_state.rec_state = "IDLE"
                                time.sleep(2)
                                st.rerun()
                            except PermissionError:
                                # FALLBACK SAVE
                                backup_name = f"demos_grid_{int(time.time())}.pkl"
                                backup_path = os.path.join(folder_path, backup_name)
                                with open(backup_path, "wb") as f: pickle.dump(st.session_state.demos_buffer, f)
                                st.error(f"⚠️ Ficheiro principal bloqueado! Dados salvos em: {backup_name}")
                                st.toast("Ficheiro principal bloqueado. Cópia criada!", icon="⚠️")
                                st.session_state.rec_state = "IDLE"
                                time.sleep(3) # Give time to read
                                st.rerun()
                        else:
                            obs, _ = st.session_state.env.reset(); st.session_state.obs = obs; st.session_state.curr_ep_obs = [obs]; st.session_state.curr_ep_acts = []; st.rerun()
                    else:
                        st.session_state.obs = next_obs; st.session_state.curr_ep_obs.append(next_obs)
                    st.rerun()
    
    else:
        # CARTPOLE LAYOUT (PREMIUM)
        c_cp_conf, c_cp_status = st.columns([0.4, 0.6], gap="large")
        
        with c_cp_conf:
            st.markdown('<div style="font-size:1.1rem; font-weight:600; margin-bottom:16px; color:#fff;">⚙️ Parâmetros</div>', unsafe_allow_html=True)
            cp_episodes = st.number_input("Episódios a Gerar", min_value=10, value=20, step=10, key="cp_episodes")
            
            output_path = os.path.join(OUTPUT_DIR, "demos_cartpole.pkl")
            demos_script = os.path.join(CURRENT_DIR, "demos.py")
            cmd = f'python "{demos_script}" --gym CartPole --episodes {cp_episodes} --file "{output_path}" --use-pretrained'
            
            st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
            if st.button("🚀 Iniciar Geração (Terminal)", key="btn_cp_gen", use_container_width=True, type="primary"):
                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    subprocess.Popen(f'start cmd /k "{cmd}"', shell=True)
                    st.toast("Terminal lançado! Aguarde a conclusão...", icon="⏳")
                except Exception as e: st.error(f"Erro: {e}")

        with c_cp_status:
             st.markdown('<div style="font-size:1.1rem; font-weight:600; margin-bottom:16px; color:#fff;">📂 Estado do Dataset</div>', unsafe_allow_html=True)
             exists = os.path.exists(output_path)
             status_color = "#10b981" if exists else "#ef4444"
             status_text = "✅ Dataset Disponível" if exists else "❌ Não Encontrado"
             
             st.markdown(f"""
             <div class="premium-card" style="padding:20px; display:flex; flex-direction:column; gap:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#a1a1aa; font-size:0.9rem;">Caminho:</span>
                    <span style="font-family:'JetBrains Mono'; font-size:0.8rem; color:#71717a;">output/demos_cartpole.pkl</span>
                </div>
                <div style="width:100%; height:1px; background:#27272a;"></div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="width:12px; height:12px; border-radius:50%; background:{status_color}; box-shadow:0 0 10px {status_color};"></div>
                    <div style="font-size:1rem; font-weight:500; color:#ededed;">{status_text}</div>
                </div>
                <p style="margin:0; font-size:0.85rem; color:#71717a; margin-top:8px;">
                    Este ficheiro será utilizado na etapa de <strong>Treino</strong>. Se gerar novos dados, este ficheiro será sobrescrito.
                </p>
             </div>
             """, unsafe_allow_html=True)


# --- TREINAR (TRAIN) ---
if "tab_train" in locals():
    with tab_train:
        st.markdown(f"""
        <div style="background:var(--bg-card); border-left:4px solid #10b981; padding:16px 24px; border-radius:0 8px 8px 0; margin-bottom:32px;">
            <h2 style="margin:0 !important; font-size:1.5rem !important;">Treinar Agente</h2>
            <div style="color:var(--text-secondary); margin-top:4px;">Algoritmo: <strong style="color:#fff;">BC / GAIL</strong></div>
        </div>
        """, unsafe_allow_html=True)
        c_config, c_logs = st.columns([0.35, 0.65], gap="large")
        with c_config:
            st.markdown('<div style="font-size:1.1rem; font-weight:600; margin-bottom:16px; color:#fff;">⚙️ Parâmetros</div>', unsafe_allow_html=True)
            t_algo = st.selectbox("Algoritmo", ["BC", "GAIL"], key="t_algo")
            st.markdown(f"<div style='font-size:0.8rem; color:#71717a; margin-top:-10px; margin-bottom:12px;'>{'Behavioral Cloning (Supervisionado)' if t_algo=='BC' else 'Generative Adversarial Imitation Learning'}</div>", unsafe_allow_html=True)
            
            t_epochs = st.number_input("Épocas / Iterações", min_value=10, value=20, step=10, key="t_epochs")
            
            demos_file = os.path.join(OUTPUT_DIR, "demos_grid.pkl") if is_grid else os.path.join(OUTPUT_DIR, "demos_cartpole.pkl")
            output_file = os.path.join(OUTPUT_DIR, f"{t_algo.lower()}_{'grid' if is_grid else 'cartpole'}.zip")
            available = os.path.exists(demos_file)
            
            st.markdown(f"""<div style="background:rgba(255,255,255,0.03); padding:12px; border-radius:6px; margin:12px 0; border:1px solid #27272a;"><div style="font-size:0.85rem; color:#71717a; margin-bottom:4px;">Dataset Entrada</div><div class="mono" style="font-size:0.8rem; color:#a1a1aa; word-break:break-all;">{demos_file}</div><div style="margin-top:8px; font-weight:600; font-size:0.9rem; color: {'#10B981' if available else '#EF4444'}">{'✅ Disponível' if available else '❌ Não Encontrado'}</div></div>""", unsafe_allow_html=True)
            if st.button("Iniciar Treino", disabled=not available, key="btn_train", use_container_width=True, type="primary"):
                cmd = f"python -u train.py --gym {clean_env_name} --file {demos_file} --output {output_file} --algorithm {t_algo} --epochs {t_epochs}"
                # Start process unbuffered
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True, bufsize=1, universal_newlines=True)
                st.session_state.training_process = process
                st.session_state.training_logs = [] # Init buffer
                st.toast("Treino Iniciado!")
                st.rerun() # Rerun to enter the log loop below
            
            st.markdown('</div>', unsafe_allow_html=True)
        with c_logs:
            # Single placeholder for the entire endpoint
            terminal_placeholder = st.empty()
            
            def render_terminal(log_content, status_line=""):
                return f"""
                <div class="premium-card" style="height:100%; min-height:500px; display:flex; flex-direction:column; padding:0; overflow:hidden; border:1px solid #3f3f46;">
                    <div style="background:#09090b; padding:12px 24px; border-bottom:1px solid #27272a; display:flex; justify-content:space-between; align-items:center;">
                        <div style="font-family:'JetBrains Mono', monospace; font-size:0.9rem; color:#e4e4e7;">
                            <span style="color:#10b981;">●</span> TERMINAL DE TREINO
                        </div>
                    </div>
                    <div style="flex:1; background:#000; padding:20px; font-family:'JetBrains Mono'; color:#a1a1aa; font-size:0.85rem; overflow-y:auto; display:flex; flex-direction:column-reverse;">
                        <div style='white-space: pre-wrap;'>{log_content}</div>
                        {status_line}
                    </div>
                </div>
                """

            if "training_process" in st.session_state:
                process = st.session_state.training_process
                if "training_logs" not in st.session_state: st.session_state.training_logs = []
                
                # ANSI Escape Pattern (simple)
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                
                title_shown = False
                while True:
                    # Read character by character to handle \r properly or read line
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    
                    if line:
                        # 1. REMOVE ASCII ART BANNER & NOISE
                        # Triggers: 8b, d8, adP, Yba, 8a, 88 88 (common in 'imitation' banner)
                        ascii_triggers = ["8b", "d8", "adP", "Yba", "8a", "88  88", "IAPV Suite", "Histórico"]
                        if any(trigger in line for trigger in ascii_triggers):
                            # Safety: Keep if looks like valid info
                            if not ("Epoch" in line or "batch" in line or "Training" in line or "Reward" in line or "Step" in line or "Loading" in line):
                                continue 
                        
                        # 2. HANDLE CURSOR MOVEMENTS (The [A stuff)
                        # Check for Cursor Up \x1b[A
                        if "\x1b[A" in line or "[A" in line:
                            if st.session_state.training_logs:
                                st.session_state.training_logs.pop()
                            # Often combined with text, so parse the rest? 
                            # Usually independent code in tqdm
                            
                            # Clean the code from the line so we can see if there is content left
                            line = line.replace("\x1b[A", "").replace("[A", "")
                            if not line.strip(): continue

                        # 3. HANDLE CARRIAGE RETURN \r (Replace last line)
                        # If line starts with \r or seems to be a dedicated update line
                        is_update = "\r" in line
                        
                        # 4. HEURISTIC FOR EPOCH/BATCH UPDATES (Force replace)
                        # "255batch" or "Epoch 82/100" often come repeatedly
                        is_progress = "batch" in line or "Epoch" in line or "%" in line
                        
                        clean_line = ansi_escape.sub('', line) # Remove colors/codes for clean display
                        
                        # 5. PREVENT MARKDOWN INTERPRETATION (Fix "Gigantic" tables)
                        # SCENARIO: Streamlit markdown parser sees |...| as tables and ---- as headers/HRs
                        # FIX: Replace with Box Drawing characters to break markdown syntax but keep look
                        if "|" in clean_line:
                            clean_line = clean_line.replace("|", "│") # U+2502
                        if "---" in clean_line:
                            clean_line = clean_line.replace("-", "─") # U+2500
                        
                        # 6. TABLE CLEANING (User Request: "Clean before showing")
                        # Logic: If we see a Top Border (Separator preceded by non-table content), clear the specific log buffer.
                        is_separator = "──────────" in clean_line
                        is_top_border = False
                        
                        if is_separator:
                            # It's a border. Check context.
                            if st.session_state.training_logs:
                                last = st.session_state.training_logs[-1]
                                # If last line had a pipe, this is likely the bottom border -> Don't clear
                                if "│" not in last:
                                    is_top_border = True
                            else:
                                is_top_border = True
                        
                        if is_top_border:
                            # CLEAR LOGS to focus on new table
                            st.session_state.training_logs = [clean_line]
                        elif (is_update or is_progress) and st.session_state.training_logs:
                            # Check if previous line was similar type
                            last_line = st.session_state.training_logs[-1]
                            if "batch" in last_line or "Epoch" in last_line or "%" in last_line:
                                st.session_state.training_logs[-1] = clean_line
                            else:
                                st.session_state.training_logs.append(clean_line)
                        else:
                            st.session_state.training_logs.append(clean_line)

                        # Render FULL terminal with updates
                        log_str = "".join(st.session_state.training_logs[-40:])
                        terminal_placeholder.markdown(render_terminal(log_str, "<div class='blink'>_</div><style>.blink { animation: blinker 1s linear infinite; } @keyframes blinker { 50% { opacity: 0; } }</style>"), unsafe_allow_html=True)
                        time.sleep(0.005) # Very fast updates
                
                # Finished
                log_text = "".join(st.session_state.training_logs[-40:])
                if process.poll() is not None:
                    status_html = "<div style='color:#10b981; margin-top:12px; border-top:1px solid #333; padding-top:10px;'>> Processo Terminado com Sucesso. 🎉</div>"
                    del st.session_state.training_process
                else:
                    status_html = "<div class='blink'>_</div>"
                
                terminal_placeholder.markdown(render_terminal(log_text, status_html), unsafe_allow_html=True)
                
            else:
                # Show stored logs
                if "training_logs" in st.session_state and st.session_state.training_logs:
                    log_text = "".join(st.session_state.training_logs[-40:])
                    terminal_placeholder.markdown(render_terminal(log_text, "<div style='color:#10b981; margin-top:12px; border-top:1px solid #333; padding-top:10px;'>> Histórico de Treino.</div>"), unsafe_allow_html=True)
                else:
                    terminal_placeholder.markdown(render_terminal("", "<div>> Aguardando comando de início...</div><div class='blink'>_</div><style>.blink { animation: blinker 1s linear infinite; } @keyframes blinker { 50% { opacity: 0; } }</style>"), unsafe_allow_html=True)

# --- VISUALIZAR (VIZ) ---
if "tab_viz" in locals():
    with tab_viz:
        st.markdown(f"""
        <div style="background:var(--bg-card); border-left:4px solid #6366f1; padding:16px 24px; border-radius:0 8px 8px 0; margin-bottom:32px;">
            <h2 style="margin:0 !important; font-size:1.5rem !important;">Visualizar Modelo</h2>
            <div style="color:var(--text-secondary); margin-top:4px;">Inferência em Tempo Real</div>
        </div>
        """, unsafe_allow_html=True)
        col_v_conf, col_v_g = st.columns([0.35, 0.65], gap="large")
        with col_v_conf:
            st.markdown('<div style="font-size:1.1rem; font-weight:600; margin-bottom:16px; color:#fff;">⚙️ Configuração</div>', unsafe_allow_html=True)
            v_algo = st.selectbox("Algoritmo", ["BC", "GAIL"], key="v_algo_viz")
            v_episodes = st.number_input("Simulações", min_value=1, value=10, step=1, key="v_episodes")
            
            model_loaded, err = load_model(v_algo, clean_env_name)
            status_color = "#10b981" if model_loaded else "#ef4444"; status_msg = "Pronto a Simular" if model_loaded else "Modelo em falha"
            st.markdown(f"""<div style="background:rgba(255,255,255,0.03); padding:12px; border-radius:6px; margin:12px 0; border:1px solid #27272a; display:flex; align-items:center; gap:10px;"><div style="width:10px; height:10px; border-radius:50%; background:{status_color}; box-shadow:0 0 8px {status_color};"></div><div style="font-size:0.9rem; color:#ededed;">{status_msg}</div></div>""", unsafe_allow_html=True)
            if model_loaded:
                if st.button("▶️ Simular", use_container_width=True, type="primary"): st.session_state.viz_active = True
            else: st.error("Ficheiro do modelo não encontrado.")
            st.markdown('</div>', unsafe_allow_html=True)
            if st.session_state.get("viz_active"):
                if st.button("⏹️ Parar Simulação", use_container_width=True): st.session_state.viz_active = False; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col_v_g:
            if st.session_state.get("viz_active"):
                env_id = "GridWorld-v0" if is_grid else "CartPole-v1"; env = gym.make(env_id, render_mode="rgb_array"); obs, _ = env.reset(); place = st.empty()
                stats_place = st.empty() # Placeholder for final stats
                
                header_html = f"""<div class="monitor-frame"><div class="monitor-header"><div style="display:flex; gap:8px; align-items:center;"><span style="color:#6366f1;">●</span> LIVE INFERENCE</div><div style="opacity:0.6;">{selected_env}</div></div><div class="monitor-screen">"""
                footer_html = "</div></div>"
                
                total_success = 0
                total_steps = 0
                episodes_completed = 0
                current_steps = 0
                
                while episodes_completed < v_episodes:
                    if not st.session_state.get("viz_active"): break
                    
                    action, _ = model_loaded.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    current_steps += 1
                    
                    if is_grid: 
                        uw = env.unwrapped; 
                        grid_html = get_grid_html(uw.size, uw._agent_location, uw._target_location, uw._obstacles)
                        place.markdown(header_html + grid_html + footer_html, unsafe_allow_html=True)
                    else:
                        img = env.render()
                        place.image(img, width=600)
                        
                    time.sleep(0.05) # Slightly faster for batch viz
                    
                    if terminated or truncated:
                        episodes_completed += 1
                        total_steps += current_steps
                        current_steps = 0
                        # Check success logic (GridWorld specific: Reward 1.0 = Goal)
                        if reward > 0: total_success += 1
                        obs, _ = env.reset()
                        
                st.session_state.viz_active = False
                
                # Show Stats
                if episodes_completed > 0:
                    success_rate = (total_success / episodes_completed) * 100
                    avg_steps = total_steps / episodes_completed
                    
                    st.toast("Simulação Terminada!", icon="🏁")
                    
                    # Custom Result Card
                    place.empty() # Clear game
                    stats_place.markdown(f"""
                    <div class="premium-card" style="text-align:center; padding:32px;">
                        <h2 style="color:#fff; margin-bottom:24px;">Relatório de Simulação</h2>
                        <div style="display:flex; justify-content:space-around; align-items:center;">
                            <div>
                                <div style="font-size:3rem; font-weight:700; color:{'#10b981' if success_rate > 80 else '#f59e0b'};">{success_rate:.0f}%</div>
                                <div style="color:#a1a1aa; font-size:0.9rem;">Taxa de Sucesso</div>
                            </div>
                            <div style="height:50px; width:1px; background:#3f3f46;"></div>
                            <div>
                                <div style="font-size:3rem; font-weight:700; color:#fff;">{avg_steps:.1f}</div>
                                <div style="color:#a1a1aa; font-size:0.9rem;">Média de Passos</div>
                            </div>
                        </div>
                        <div style="margin-top:24px; color:#52525b; font-size:0.8rem;">Baseado em {episodes_completed} episódios</div>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                content = f"""<div style="text-align:center; color:#52525b;"><div style="font-size:3rem; margin-bottom:16px; opacity:0.5;">👁️</div><div style="font-family:'JetBrains Mono';">A aguardar início da simulação...</div></div>"""
                st.markdown(render_monitor_frame(selected_env, content), unsafe_allow_html=True)
