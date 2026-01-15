
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
import pandas as pd

# --- 1. CONFIGURAÇÃO INICIAL ---
st.set_page_config(
    page_title="IAPV Ambiente de Imitação",
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
        st.error(f"Erro: Ficheiro de estilo {css_path} não encontrado.")

inject_premium_css()

# --- 3. LÓGICA DO AMBIENTE ---
GRID_SIZE = 8
LIMIT_STEPS = int(((GRID_SIZE * GRID_SIZE) / 2) / 10) * 10

try:
    # GridWorld is registered dynamically with spawn mode later
    pass
except:
    pass

@st.cache_resource
def load_model(algo, gym_name, spawn_mode="random"):
    # USE OUTPUT_DIR DEFINED AT TOP
    if gym_name == "Custom":
        subfolder = os.path.join("grid", spawn_mode, "simple")
    else:
        subfolder = "cartpole"
    folder_path = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(folder_path, exist_ok=True)
    
    filename = f"{algo.lower()}_{'grid' if gym_name=='Custom' else 'cartpole'}.zip"
    path = os.path.join(folder_path, filename)
    
    # FALLBACK: If specific manual model missing, look in intervals/models (Benchmark)
    if not os.path.exists(path):
        bench_dir = os.path.join(folder_path, "intervals", "models")
        if os.path.exists(bench_dir):
            # Find matching algo files
            candidates = [f for f in os.listdir(bench_dir) if f.startswith(algo) and f.endswith(".zip")]
            if candidates:
                # Pick the one with highest demos number (usually nicely named Algo_N_seed.zip)
                # Simple sort by name works for 10, 20... wait. 10 vs 100. Length sorting or specific?
                # Just sorting by modification time or name is a decent heuristic.
                candidates.sort() # Algo_10, Algo_20, Algo_50
                path = os.path.join(bench_dir, candidates[-1])
                # Warn/Toast would be nice but we are in a cached function.
                
    if not os.path.exists(path): return None, f"Modelo não encontrado: {path}", None
    
    try:
        _original_load = torch.load
        def _safe_load(*args, **kwargs):
            if 'weights_only' not in kwargs: kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        torch.load = _safe_load
        
        # Suppress warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try: model = PPO.load(path)
            except: model = ActorCriticPolicy.load(path)
            
        torch.load = _original_load
        return model, None, path
    except Exception as e: return None, str(e), path

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
<div class="main-header">
    <div class="main-header-icon">
        <span>⚡</span>
    </div>
    <div class="main-header-text">
        <h1>IAPV Ambiente de Imitação</h1>
        <div class="main-header-subtitle">Painel Avançado de Aprendizagem por Reforço <span style="opacity:0.6">&mdash; Grupo 7</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

selected_env = st.session_state.get("selected_env", "GridWorld (Custom)")
# Ensure env name is simple for UI
env_display = "GridWorld" if "Grid" in selected_env else "CartPole-v1"

# --- SIDEBAR GLOBAL ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-header">
            <div class="sidebar-info-icon">🧭</div>
            <div>
                <div class="sidebar-info-title">Ambiente RL</div>
                <div class="sidebar-info-label">Espaço de Trabalho</div>
            </div>
        </div>
        <div class="sidebar-info-body">Selecionar o desafio:</div>
    </div>
    """, unsafe_allow_html=True)
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

    selected_env = st.session_state.selected_env
    is_grid = selected_env == "GridWorld (Custom)"
    clean_env_name = "Custom" if is_grid else "CartPole"

    # SPAWN MODE SELECTOR (only for GridWorld)
    if is_grid:
        if "spawn_mode" not in st.session_state:
            st.session_state.spawn_mode = "random"
        
        st.divider()
        st.caption("Modo de Aplicação")
        
        # New "pills" style selection
        spawn_mode_map = {"🎲 Aleatório": "random", "📍 Fixo": "fixed"}
        rev_spawn_mode_map = {v: k for k, v in spawn_mode_map.items()}
        
        current_selection = rev_spawn_mode_map.get(st.session_state.spawn_mode, "🎲 Aleatório")
        
        new_selection = st.radio(
            "Spawn Mode",
            options=list(spawn_mode_map.keys()),
            index=list(spawn_mode_map.values()).index(st.session_state.spawn_mode),
            horizontal=True,
            label_visibility="collapsed",
            key="spawn_radio"
        )
        
        if spawn_mode_map[new_selection] != st.session_state.spawn_mode:
            st.session_state.spawn_mode = spawn_mode_map[new_selection]
            st.rerun()

        # Custom CSS for Radio Buttons to look like premium pills/segments
        st.markdown("""
        <style>
        div[role="radiogroup"] {
            background: #18181b;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid #27272a;
            display: flex;
            gap: 4px;
            margin: 0 auto;
            width: fit-content;
            justify-content: center;
        }
        div[role="radiogroup"] label {
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 4px 12px;
            margin: 0 !important;
            transition: all 0.2s;
            flex: 1;
            text-align: center;
            justify-content: center;
            display: flex;
            align-items: center;
            color: #71717a !important;
        }
        div[role="radiogroup"] label[data-checked="true"] {
            background: #27272a !important;
            color: #fbbf24 !important;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        div[role="radiogroup"] label:hover {
            color: #e4e4e7 !important;
        }
        /* Hide default radio circle */
        div[role="radiogroup"] input {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
        spawn_mode = st.session_state.spawn_mode
        
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
    else:
        spawn_mode = "random"  # Not applicable for CartPole

    st.markdown("""
    <div class="footer-container">
        <div class="footer-content">
            <span class="version-tag">v 2.0.0</span>
            <span class="beta-badge">BETA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



# CHECK DATA EXISTENCE - UPDATED PATH WITH SPAWN MODE
if is_grid:
    env_folder = os.path.join("grid", spawn_mode)
    simple_folder = "simple"
else:
    env_folder = "cartpole"
    simple_folder = ""

demos_file = os.path.join(OUTPUT_DIR, env_folder, simple_folder, "demos.pkl")
has_demos = os.path.exists(demos_file)

# TABS LOGIC
if has_demos:
    tab_dash, tab_rec, tab_train, tab_viz, tab_bench = st.tabs(["Painel", "Gravar", "Treinar", "Visualizar", "Relatório"])
else:
    tab_dash, tab_rec = st.tabs(["Painel", "Gravar"])

# --- DASHBOARD ---
# ... (Dashboard content unchanged) ...

# ... (Previous tabs unchanged) ...

# --- RELATÓRIO / BENCHMARK (NEW) ---
if "tab_bench" in locals():
    with tab_bench:
        # --- HEADER ---
        st.markdown(f"""
        <div class="benchmark-header">
            <h2>🏆 Área Extra: Benchmark & Relatório</h2>
            <div>Gera dados, treina modelos e cria gráficos profissionais para o teu relatório.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- TOP CONFIG ---
        st.markdown("### ⚙️ Configuração da Campanha")
        c_cfg1, c_cfg2 = st.columns([0.3, 0.7])
        
        # Load max available demos count for slider limit
        try:
            import pickle
            d_path = os.path.join(OUTPUT_DIR, env_folder, "demos.pkl")
            with open(d_path, "rb") as f: d_len = len(pickle.load(f))
        except: d_len = 0
        
        help_text = "Cada conjunto adiciona 1 demo (ex: 3 conjuntos = 1, 2, 3 demos)." if not is_grid else "Cada conjunto adiciona 10 demos (ex: 3 conjuntos = 10, 20, 30 demos)."
        b_num_sets = c_cfg1.number_input("Nº de Conjuntos", value=3, min_value=2, step=1, help=help_text, label_visibility="collapsed")
        
        if is_grid:
            targets = list(range(10, (b_num_sets * 10) + 1, 10))
        else:
            targets = list(range(1, b_num_sets + 1))
        
        with c_cfg2:
            # Create higher-end chip components for targets (Single line to avoid markdown break)
            pills = "".join([f'<div class="benchmark-pill"><div class="benchmark-pill-dot"></div><span class="benchmark-pill-text">{t} Demos</span></div>' for t in targets])
            
            st.markdown(f"""
            <div class="benchmark-pills-container">
                <div class="benchmark-pills-inner">
                    <div class="benchmark-plan-label">
                        <span class="benchmark-plan-icon">🎯</span>
                        <span class="benchmark-plan-text">PLANO:</span>
                    </div>
                    <div style="display:flex; gap:2px;">{pills}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")

        # --- MAIN WORKSPACE ---
        c_rec, c_lab = st.columns([0.45, 0.55], gap="large")
        
        # === LEFT: RECORDING STUDIO ===
        with c_rec:
            st.markdown('<div class="section-title">🎥 Estúdio de Gravação</div>', unsafe_allow_html=True)
            st.caption("Grava as demonstrações necessárias para cada cenário.")
            
            import extras.experiments as experiments
            import importlib; importlib.reload(experiments)
            
            # Dynamic Path based on Env (use top-level env_folder)
            intervals_dir = os.path.join(OUTPUT_DIR, env_folder, "intervals")
            os.makedirs(intervals_dir, exist_ok=True)
            
            valid_files = [] 
            
            for t_n in targets:
                fname = f"demos_{t_n}.pkl"
                # ... (rest of loop logic logic remains relying on intervals_dir)
                fpath = os.path.join(intervals_dir, fname)
                exists = os.path.exists(fpath)
                
                # Card-like row
                with st.container():
                    r1, r2, r3 = st.columns([0.15, 0.5, 0.35])
                    with r1:
                        st.markdown("✅" if exists else "🔴")
                    with r2:
                        st.markdown(f"**{t_n} Demos**")
                        if exists:
                            try:
                                with open(fpath, "rb") as f: cnt = len(pickle.load(f))
                                st.caption(f"{cnt} eps ok")
                            except: st.caption("Erro")
                        else: st.caption("Em falta")
                    with r3:
                        # CMD Launcher
                        script_path = os.path.join(ROOT_DIR, "src", "demos.py")
                        safe_path = fpath # Absolute path is correct
                        
                        py_exe = sys.executable
                        use_pretrained_arg = " --use-pretrained" if not is_grid else ""
                        cmd = f'"{py_exe}" "{script_path}" --gym {"Custom" if is_grid else "CartPole"} --episodes {t_n} --file "{safe_path}" --spawn {spawn_mode if is_grid else "random"}{use_pretrained_arg}'
                        
                        btn_label = "Gravar" if is_grid else "Gerar"
                        if st.button(btn_label, key=f"rec_{t_n}", disabled=False):
                            subprocess.Popen(f'start "IAPV-BatchRec" cmd /k "{cmd}"', shell=True)
                            msg = f"A abrir gravador para {t_n}..." if is_grid else f"A gerar {t_n} amostras (HuggingFace)..."
                            st.toast(msg, icon="🎮" if is_grid else "🤖")
                
                if exists: valid_files.append((t_n, fpath))
                st.divider()

        # === RIGHT: AI LAB ===
        with c_lab:
            # ... (Expander UI remains same)
            st.markdown('<div class="section-title lab">🧪 Laboratório de Treino</div>', unsafe_allow_html=True)
            
            with st.expander("⚙️ Definições Avançadas (Epochs & Seeds)", expanded=True):
                p1, p2 = st.columns(2)
                b_seed = p1.number_input("Seed (Semente)", value=42)
                if is_grid:
                    b_epoch_bc = p2.number_input("Epochs BC", value=100, step=10)
                    b_epoch_gail = st.number_input("Epochs GAIL", value=100, step=10, help="Mais epochs = Melhores resultados, mas demora mais.")
                else:
                    b_epoch_bc = 10
                    b_epoch_gail = 10
                    st.markdown('<div class="info-notice">Configuração CartPole: <strong>10 Epochs</strong></div>', unsafe_allow_html=True)
                b_eval_eps = 20
                st.markdown('<div style="font-size:0.85rem; color:#a1a1aa; margin-top:5px; margin-bottom:12px;">Simulação de Validação: <strong>Fixa a 20 episódios</strong></div>', unsafe_allow_html=True)
                b_force = st.checkbox("Forçar Re-treino (Ignorar Cache)", value=False, key="b_force", help="Se ativo, treina mesmo que o modelo já exista.")
            
            # Action Button
            st.write("")
            start_btn = st.button("🚂 Iniciar Treino em Batch", type="primary", use_container_width=True, disabled=len(valid_files)<1)
            if len(valid_files) < 1:
                st.warning("Grava pelo menos 1 cenário para começar.")
            
            # Dynamic Logs
            log_box = st.status("Estado da Execução", expanded=False)
            
            if start_btn:
                log_box.update(label="A processar...", state="running", expanded=True)
                try:
                    def prog_callback(pct, msg):
                        log_box.write(f"{pct*100:.0f}% - {msg}")
                    
                    # Force reload to get updated logic
                    import extras.experiments as experiments
                    import importlib
                    importlib.reload(experiments)
                    
                    st.write(f"DEBUG: Force Retrain = {b_force}") # Visible proof
                    
                    # Run and get results + path (results is now a nested dict)
                    results_dict, json_path = experiments.run_benchmark_loop(
                        env_name=clean_env_name,
                        generated_files=valid_files,
                        seed=b_seed,
                        progress_callback=prog_callback,
                        gail_epochs=b_epoch_gail,
                        bc_epochs=b_epoch_bc,
                        spawn_mode=spawn_mode if is_grid else "random",
                        eval_episodes=b_eval_eps,
                        force_retrain=b_force
                    )
                    
                    # Flatten Dict to List for CSV/DataFrame
                    flat_results = []
                    for algo, demo_dict in results_dict.items():
                        for n_demos, metrics in demo_dict.items():
                            row = {"Algorithm": algo, "Demos": int(n_demos)}
                            row.update(metrics)
                            flat_results.append(row)
                    
                    import pandas as pd
                    df = pd.DataFrame(flat_results)
                    csv_path = json_path.replace(".json", ".csv")
                    df.to_csv(csv_path, index=False)
                    
                    log_box.update(label="Treino Concluído!", state="complete", expanded=False)
                    st.toast(f"Resultados salvos em {csv_path}", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
                    log_box.update(label="Erro!", state="error")
            
            st.markdown("---")
            
            # === RESULTS SECTION ===
            st.markdown('<div style="font-size:1.2rem; font-weight:700; margin-bottom:12px; color:#fbbf24;">📊 Resultados & Gráficos</div>', unsafe_allow_html=True)
            
            # Use top-level env_folder (already computed with spawn mode)
            results_csv = os.path.join(OUTPUT_DIR, env_folder, "intervals", "benchmark_results.csv")
            import extras.plots as plots; importlib.reload(plots)
            
            c_r1, c_r2 = st.columns([0.4, 0.6])
            with c_r1:
                if st.button("🔄 Atualizar Gráficos"):
                    if os.path.exists(results_csv):
                        # Use wrapper to get correct subfolder structure logic
                        plots.plot_benchmark_results(results_csv)
                        st.toast("Updated!", icon="🎨")
                        st.rerun()
            
            if os.path.exists(results_csv):
                # Calculate expected path based on updated logic: docs/plots/{env}/{mode}/graficos
                # env_folder is 'grid/random' etc.
                plot_dir = os.path.join(ROOT_DIR, "docs", "plots", env_folder, "graficos")
                # Define tabs based on environment
                tab_titles = ["Sucesso", "Duração", "Reward", "Robustez", "Dados"] if is_grid else ["Sucesso", "Reward", "Robustez", "Dados"]
                tabs = st.tabs(tab_titles)
                
                # Assign tabs based on titles
                if is_grid:
                    t_suc, t_dur, t_rew, t_rob, t_dat = tabs
                else:
                    t_suc, t_rew, t_rob, t_dat = tabs
                    t_dur = None
                
                with t_suc:
                    p = os.path.join(plot_dir, "success_rate.png")
                    if os.path.exists(p): 
                        st.image(p)
                        with open(p, "rb") as f: st.download_button("📥 PNG", f, "suc.png", key="dl_suc")
                
                if t_dur:
                    with t_dur:
                        p = os.path.join(plot_dir, "step_ratio.png")
                        if os.path.exists(p): 
                            st.image(p)
                            with open(p, "rb") as f: st.download_button("📥 PNG", f, "step.png", key="dl_step")
                
                with t_rew:
                    p = os.path.join(plot_dir, "reward_efficiency.png")
                    if os.path.exists(p): 
                        st.image(p)
                        with open(p, "rb") as f: st.download_button("📥 PNG", f, "rew.png", key="dl_rew")
                with t_rob:
                    p = os.path.join(plot_dir, "robustness.png")
                    if os.path.exists(p): 
                        st.image(p)
                        with open(p, "rb") as f: st.download_button("📥 PNG", f, "rob.png", key="dl_rob")
                with t_dat:
                    df = pd.read_csv(results_csv)
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("A aguardar resultados do treino...")
with tab_dash:
    st.markdown(f"""
<div class="premium-card"><div style="display:flex; justify-content:space-between; align-items:flex-start;"><div>
<h2 style="margin-top:0 !important; font-size:1.8rem;">Bem-vindo ao Ambiente de Testes!</h2><div style="color:#a1a1aa; margin-top:0.5rem; font-size:1.1rem;">Ambiente Ativo: <strong style="color:#fff;">{selected_env}</strong></div></div>
<div style="background:rgba(99,102,241,0.1); color:#818cf8; padding:8px 16px; border-radius:20px; font-weight:600; font-size:0.9rem;">{env_display}</div></div>
</div>
""", unsafe_allow_html=True)
    
    # Overview Cards
    st.markdown("""
    <div class="dashboard-grid">
        <div class="dashboard-card">
            <div class="dashboard-card-icon">🎮</div>
            <div class="dashboard-card-title">1. Gravar</div>
            <div class="dashboard-card-body">Recolha de expert demos.</div>
        </div>
        <div class="dashboard-card">
            <div class="dashboard-card-icon">🧠</div>
            <div class="dashboard-card-title">2. Treinar</div>
            <div class="dashboard-card-body">Behavioral Cloning ou GAIL.</div>
        </div>
        <div class="dashboard-card">
            <div class="dashboard-card-icon">👁️</div>
            <div class="dashboard-card-title">3. Visualizar</div>
            <div class="dashboard-card-body">Validação de métricas.</div>
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
    <div class="tab-header" style="border-left: 4px solid {header_color};">
        <h2>Gravar Demonstrações</h2>
        <div class="tab-header-subtitle">Challenge: <strong style="color:#fff;">{env_display}</strong></div>
        {extra_info}
    </div>
    """, unsafe_allow_html=True)

    if is_grid:
        # GRIDWORLD LAYOUT
        col_config, col_game = st.columns([0.4, 0.6], gap="large")
        
        with col_config:
            st.markdown("""
            <div class="mission-control-card">
                <div class="mission-control-header">
                    <div class="mission-control-icon" style="color:#f59e0b;">🧭</div>
                    <div class="mission-control-title">Mission Control</div>
                </div>
                <div class="mission-control-body">Configuração de recolha para o ambiente matricial.</div>
            </div>
            """, unsafe_allow_html=True)
            
            p1, p2 = st.columns([0.4, 0.6])
            with p1:
                st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#a1a1aa; margin-bottom:8px; display:block;">Episódios</label>', unsafe_allow_html=True)
                d_episodes = st.number_input("Episódios", min_value=1, value=20, step=5, key="d_episodes", label_visibility="collapsed")
            with p2:
                st.markdown('<label style="font-size:0.85rem; font-weight:600; color:#a1a1aa; margin-bottom:8px; display:block;">Interface</label>', unsafe_allow_html=True)
                rec_mode = st.radio(
                    "Interface", ["CMD (Rápido)", "In-App (Web)"], 
                    horizontal=True, key="rec_mode_sel", label_visibility="collapsed"
                )
            
            st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
            
            # Action Launcher for CMD
            if "CMD" in rec_mode:
                output_path = demos_file
                demos_script = os.path.join(CURRENT_DIR, "demos.py")
                spawn_arg = spawn_mode if is_grid else "random"
                py_exe = sys.executable
                cmd = f'"{py_exe}" "{demos_script}" --gym {clean_env_name} --episodes {d_episodes} --file "{output_path}" --spawn {spawn_arg}'
                
                if st.button("🚀 Lançar Gravador Externo", use_container_width=True, type="primary"):
                    try:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        # Fixed quoting for Windows shell
                        subprocess.Popen(f'start "IAPV-Recorder" cmd /k "{cmd}"', shell=True)
                        st.toast("Terminal iniciado!", icon="🚀")
                    except Exception as e: st.error(f"Erro: {e}")
            
            # In-App Controls
            if "In-App" in rec_mode:
                if "rec_state" not in st.session_state: st.session_state.rec_state = "IDLE"
                
                if st.session_state.rec_state in ["IDLE", "FINISHED"]:
                    if st.button("🔴 Iniciar Gravação Web", use_container_width=True, type="primary"):
                        st.session_state.rec_state = "RECORDING"
                        st.session_state.ep_count = 0
                        st.session_state.demos_buffer = []
                        
                        if is_grid:
                            try:
                                from custom_env import GridWorldEnv
                                register(
                                    id="GridWorld-v0",
                                    entry_point="custom_env:GridWorldEnv",
                                    max_episode_steps=LIMIT_STEPS,
                                    kwargs={'size': GRID_SIZE, 'random_start': (spawn_mode == "random")}
                                )
                            except: pass

                        st.session_state.env = gym.make("GridWorld-v0", render_mode="rgb_array")
                        obs, _ = st.session_state.env.reset()
                        st.session_state.obs = obs
                        st.session_state.curr_ep_obs = [obs]
                        st.session_state.curr_ep_acts = []
                        st.rerun()
                
                elif st.session_state.rec_state == "RECORDING":
                    st.progress(st.session_state.ep_count / d_episodes, text=f"Progresso: {st.session_state.ep_count}/{d_episodes} episódios")
                    if st.button("✖️ Cancelar Sessão", use_container_width=True): 
                        st.session_state.rec_state = "IDLE"
                        st.rerun()
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
                            
                            st.session_state.rec_state = "FINISHED"
                            
                            # Use correct path defined globally (demos_file)
                            # This ensures it matches what Train tab looks for (grid/{mode}/simple/demos.pkl)
                            target_save_path = demos_file
                            folder_path = os.path.dirname(target_save_path)
                            file_name = os.path.basename(target_save_path)
                            
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
        c_cp_conf, c_cp_status = st.columns([0.45, 0.55], gap="large")
        
        with c_cp_conf:
            st.markdown("""
            <div class="mission-control-card">
                <div class="mission-control-header">
                    <div class="mission-control-icon" style="color:#6366f1;">🎮</div>
                    <div class="mission-control-title">Mission Control</div>
                </div>
                <div class="mission-control-body">Configure e execute a recolha de dados para o equilibrador.</div>
            </div>
            """, unsafe_allow_html=True)
            
            cp_episodes = st.number_input("Episódios a Gerar", min_value=1, value=20, step=1, key="cp_episodes")
            
            # Paths
            cp_folder = os.path.join(OUTPUT_DIR, env_folder)
            output_path = os.path.join(cp_folder, "demos.pkl")
            manual_out = os.path.join(cp_folder, "manual_test.pkl")
            demos_script = os.path.join(CURRENT_DIR, "demos.py")
            
            # Using sys.executable to ensure we use the same environment
            py_exe = sys.executable
            cmd_auto = f'"{py_exe}" "{demos_script}" --gym CartPole --episodes {cp_episodes} --file "{output_path}" --use-pretrained'
            cmd_manual = f'"{py_exe}" "{demos_script}" --gym CartPole --episodes 3 --file "{manual_out}"'
            
            st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
            
            # Action Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Auto-Gerar", key="btn_cp_gen", use_container_width=True, type="primary", help="Usa um agente especialista do HuggingFace."):
                    try:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        subprocess.Popen(f'start "IAPV-Auto" cmd /k "{cmd_auto}"', shell=True)
                        st.toast("A gerar demonstrações experientes...", icon="⏳")
                    except Exception as e: st.error(f"Erro: {e}")
            
            with col2:
                if st.button("🕹️ Jogar Manual", key="btn_cp_manual", use_container_width=True, help="Abre uma janela para jogares (Setas Esquerda/Direita)."):
                    try:
                        os.makedirs(os.path.dirname(manual_out), exist_ok=True)
                        subprocess.Popen(f'start "IAPV-Manual" cmd /k "{cmd_manual}"', shell=True)
                        st.toast("Abre o CMD e Janela! Usa as SETAS.", icon="🕹️")
                    except Exception as e: st.error(f"Erro: {e}")

            st.markdown("""
            <div class="control-guide">
                <div class="control-guide-title">
                    <span>⌨️ CONTROLO MANUAL</span>
                </div>
                <div class="control-guide-body">Utilize as setas Esquerda/Direita para equilibrar a barra. ESC para sair.</div>
            </div>
            """, unsafe_allow_html=True)

        with c_cp_status:
             st.markdown('<div style="font-size:1.1rem; font-weight:600; margin-bottom:16px; color:#fff;">📂 Estado do Dataset</div>', unsafe_allow_html=True)
             exists = os.path.exists(output_path)
             status_color = "#10b981" if exists else "#ef4444"
             status_text = "✅ Dataset Disponível" if exists else "❌ Não Encontrado"
             
             st.markdown(f"""
             <div class="premium-card dataset-status-card">
                <div class="status-row">
                    <span class="status-label">Caminho:</span>
                    <span class="status-value">output/cartpole/demos.pkl</span>
                </div>
                <div style="width:100%; height:1px; background:var(--border-subtle);"></div>
                <div class="status-indicator-row">
                    <div class="status-dot" style="background:{status_color}; box-shadow:0 0 10px {status_color};"></div>
                    <div class="status-text">{status_text}</div>
                </div>
                <p class="mission-control-body" style="margin-top:8px;">
                    Este ficheiro será utilizado na etapa de <strong>Treino</strong>. Se gerar novos dados, este ficheiro será sobrescrito.
                </p>
             </div>
             """, unsafe_allow_html=True)


# --- TREINAR (TRAIN) ---
if "tab_train" in locals():
    with tab_train:
        st.markdown(f"""
        <div class="tab-header" style="border-left:4px solid #10b981;">
            <h2>Treinar Agente</h2>
            <div class="tab-header-subtitle">Algoritmo: <strong style="color:#fff;">BC / GAIL</strong></div>
        </div>
        """, unsafe_allow_html=True)
        c_config, c_logs = st.columns([0.35, 0.65], gap="large")
        with c_config:
            st.markdown('<div style="font-size:1.1rem; font-weight:600; margin-bottom:16px; color:#fff;">⚙️ Parâmetros</div>', unsafe_allow_html=True)
            t_algo = st.selectbox("Algoritmo", ["BC", "GAIL"], key="t_algo")
            st.markdown(f"<div class='help-text'>{'Behavioral Cloning (Supervisionado)' if t_algo=='BC' else 'Generative Adversarial Imitation Learning'}</div>", unsafe_allow_html=True)
            
            if is_grid:
                t_epochs = st.number_input("Épocas / Iterações", min_value=10, value=100, step=10, key="t_epochs")
            else:
                t_epochs = 10
                st.markdown('<div style="background:rgba(16,185,129,0.05); padding:8px 12px; border-radius:6px; border:1px solid #3f3f46; font-size:0.85rem; color:#a1a1aa; margin-bottom:12px;">Configuração CartPole: <strong>10 Epochs</strong></div>', unsafe_allow_html=True)
            t_eval_episodes = st.number_input("Simulações de Validação", min_value=0, value=20, step=5, key="t_eval_eps")
            
            # Use env_folder which already has spawn mode
            # For GridWorld, single runs go to 'simple' subfolder
            t_folder = os.path.join(env_folder, "simple") if is_grid else env_folder
            
            output_file = os.path.join(OUTPUT_DIR, t_folder, f"{t_algo.lower()}_{'grid' if is_grid else 'cartpole'}.zip")
            available = os.path.exists(demos_file)
            
            st.markdown(f"""
            <div class="info-notice" style="background:rgba(255,255,255,0.03); padding:12px; border-radius:6px; margin:12px 0; border:1px solid var(--border-subtle);">
                <div style="font-size:0.85rem; color:var(--text-tertiary); margin-bottom:4px;">Dataset Entrada</div>
                <div style="font-family:'JetBrains Mono'; font-size:0.8rem; color:var(--text-secondary); word-break:break-all;">{demos_file}</div>
                <div style="margin-top:8px; font-weight:600; font-size:0.9rem; color: {'#10B981' if available else '#EF4444'}">{'✅ Disponível' if available else '❌ Não Encontrado'}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Iniciar Treino", disabled=not available, key="btn_train", use_container_width=True, type="primary"):
                train_script = os.path.join(CURRENT_DIR, "train.py")
                spawn_arg = spawn_mode if is_grid else "random"
                py_exe = sys.executable
                cmd = f'"{py_exe}" -u "{train_script}" --gym {clean_env_name} --file "{demos_file}" --output "{output_file}" --algorithm {t_algo} --epochs {t_epochs} --spawn {spawn_arg} --eval_episodes {t_eval_episodes}'
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
                <div class="terminal-container">
                    <div class="terminal-header">
                        <div class="terminal-title">
                            <span class="terminal-led">●</span> TERMINAL DE TREINO
                        </div>
                    </div>
                    <div class="terminal-body">
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
            if is_grid:
                v_episodes = st.number_input("Simulações", min_value=1, value=1, step=1, key="v_episodes")
            else:
                v_episodes = 1
                st.markdown('<div style="background:rgba(255,255,255,0.05); padding:8px 12px; border-radius:6px; border:1px solid #3f3f46; font-size:0.85rem; color:#a1a1aa; margin-bottom:12px;">Simulação Fixa: <strong>1 Episódio</strong></div>', unsafe_allow_html=True)
            
            model_loaded, err, loaded_path = load_model(v_algo, clean_env_name, spawn_mode=spawn_mode)
            status_color = "#10b981" if model_loaded else "#ef4444"; status_msg = "Pronto a Simular" if model_loaded else "Modelo em falha"
            st.markdown(f"""<div style="background:rgba(255,255,255,0.03); padding:12px; border-radius:6px; margin:12px 0; border:1px solid #27272a; display:flex; align-items:center; gap:10px;"><div style="width:10px; height:10px; border-radius:50%; background:{status_color}; box-shadow:0 0 8px {status_color};"></div><div style="font-size:0.9rem; color:#ededed;">{status_msg}</div></div>""", unsafe_allow_html=True)
            if model_loaded:
                if st.button("▶️ Simular", use_container_width=True, type="primary"): 
                    st.session_state.viz_active = True
                    st.session_state.loaded_model_path = loaded_path
            else: st.error("Ficheiro do modelo não encontrado.")
            st.markdown('</div>', unsafe_allow_html=True)
            if st.session_state.get("viz_active"):
                if st.button("⏹️ Parar Simulação", use_container_width=True): st.session_state.viz_active = False; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col_v_g:
            if st.session_state.get("viz_active"):
                # --- SETUP: VECNORMALIZE & ENV ---
                status_box = st.empty()
                env_id = "GridWorld-v0" if is_grid else "CartPole-v1"
                
                model_path = st.session_state.get("loaded_model_path", "")
                vec_norm_path = model_path.replace(".zip", "_vecnorm.pkl") if model_path else ""
                
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                base_env = gym.make(env_id, render_mode="rgb_array")
                
                if os.path.exists(vec_norm_path):
                    status_box.info(f"Loading Normalization from {vec_norm_path}...")
                    venv = DummyVecEnv([lambda: base_env])
                    venv = VecNormalize.load(vec_norm_path, venv)
                    venv.training = False 
                    venv.norm_reward = False
                    env_to_use = venv
                    is_norm = True
                else:
                    env_to_use = base_env
                    is_norm = False

                # Reset
                if is_norm:
                     obs = env_to_use.reset() # returns (n_envs, obs_shape)
                else:
                     obs, _ = env_to_use.reset()

                place = st.empty()
                stats_place = st.empty()
                
                header_html = f"""<div class="monitor-frame"><div class="monitor-header"><div style="display:flex; gap:8px; align-items:center;"><span style="color:#6366f1;">●</span> LIVE INFERENCE</div><div style="opacity:0.6;">{selected_env}</div></div><div class="monitor-screen">"""
                footer_html = "</div></div>"
                
                total_success = 0
                total_steps = 0
                episodes_completed = 0
                current_steps = 0
                
                while episodes_completed < v_episodes:
                    if not st.session_state.get("viz_active"): break
                    
                    action, _ = model_loaded.predict(obs, deterministic=True)
                    
                    if is_norm:
                        obs, rewards, dones, infos = env_to_use.step(action)
                        done = dones[0]
                        # For rendering, we need the real underlying env instance
                        render_env = env_to_use.venv.envs[0] 
                    else:
                        obs, reward, terminated, truncated, info = env_to_use.step(action)
                        done = terminated or truncated
                        render_env = env_to_use
                    
                    current_steps += 1
                    
                    if is_grid: 
                        uw = render_env.unwrapped
                        grid_html = get_grid_html(uw.size, uw._agent_location, uw._target_location, uw._obstacles)
                        place.markdown(header_html + grid_html + footer_html, unsafe_allow_html=True)
                    else:
                        img = render_env.render()
                        place.image(img, width=600)
                        
                    time.sleep(0.05)
                    
                    if done:
                        episodes_completed += 1
                        total_steps += current_steps
                        current_steps = 0
                        
                        # Logic for stats (GridWorld reward > 0 means success)
                        # We need the ACTUAL reward, not normalized. 
                        # If is_norm, rewards is array.
                        r = rewards[0] if is_norm else reward
                        if r > 0: total_success += 1
                        
                        if is_norm:
                            obs = env_to_use.reset()
                        else:
                            obs, _ = env_to_use.reset()
                        
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
