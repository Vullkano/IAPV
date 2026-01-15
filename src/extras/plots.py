import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob

# Set Style - Professional & Academic
# Manually setting params for consistency across environments
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Colors
PALETTE = {"BC": "#2563eb", "GAIL": "#ea580c"} # Blue and Orange

def save_plot(fig, name, subfolder=None):
    """Saves figure to docs/plots/{subfolder}/ directory."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    plot_dir = os.path.join(root, "docs", "plots")
    
    if subfolder:
        plot_dir = os.path.join(plot_dir, subfolder)
        
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_success_rates(csv_path, subfolder=None):
    """Generates Success Rate Line Plot (User Req #2)."""
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Convert to percentage
    df["Success %"] = df["Success Rate"] * 100
    
    sns.lineplot(
        data=df, 
        x="Demos", 
        y="Success %", 
        hue="Algorithm", 
        style="Algorithm",
        markers=True, 
        dashes=False,
        palette=PALETTE,
        ax=ax,
        linewidth=2.5,
        markersize=8
    )
    
    ax.set_title("Success Rate by Dataset Size", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Demonstrations", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    
    return save_plot(fig, "success_rate.png", subfolder)

def plot_step_ratios(csv_path, subfolder=None):
    """Generates Avg Steps / Max Steps Ratio Bar Plot (User Req #2)."""
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    
    # We want to see how efficient they are. 
    # Use the MAX dataset size for the "Final Comparison", or plot all?
    # Barplots with X-axis as continuous variable (Demos) can be crowded.
    # Let's use LinePlot if X is Demos, OR BarPlot if we just compare Final.
    # User said: "range... saltando de 10 em 10...". 
    # A barplot with groups for 10, 20, 30... is okay.
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.barplot(
        data=df,
        x="Demos",
        y="Step Ratio",
        hue="Algorithm",
        palette=PALETTE,
        ax=ax,
        capsize=.05,
        errorbar=('ci', 95),
        alpha=0.9
    )
    
    ax.set_title("Episode Duration Ratio (Avg Steps / Limit)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Demonstrations", fontsize=12)
    ax.set_ylabel("Duration Ratio (Lower is Faster)", fontsize=12)
    ax.set_ylim(0, 1.1)
    
    # Add a reference line for 1.0 (Timeout)
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.5, label="Timeout Limit")
    
    return save_plot(fig, "step_ratio.png", subfolder)

def plot_efficiency(csv_path, subfolder=None):
    """Generates Mean Reward Line Plot (Baseline)."""
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.lineplot(
        data=df, 
        x="Demos", 
        y="Mean Reward", 
        hue="Algorithm", 
        style="Algorithm",
        markers=True, 
        dashes=False,
        palette=PALETTE,
        ax=ax,
        linewidth=2.5
    )
    
    ax.set_title("Average Reward vs Demonstrations", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Demonstrations", fontsize=12)
    ax.set_ylabel("Mean Reward", fontsize=12)
    
    return save_plot(fig, "reward_efficiency.png", subfolder)

def plot_robustness(csv_path, subfolder=None):
    """Generates Box Plot for max dataset size."""
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    
    # Filter for max demos only
    max_d = df["Demos"].max()
    sub_df = df[df["Demos"] == max_d]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.boxplot(
        data=sub_df, 
        x="Algorithm", 
        y="Mean Reward", 
        palette=PALETTE,
        width=0.4,
        ax=ax,
        boxprops=dict(alpha=0.8) # prettier
    )
    sns.swarmplot(data=sub_df, x="Algorithm", y="Mean Reward", color=".2", size=5)
    
    ax.set_title(f"Robustness (N={max_d})", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Final Reward Distribution", fontsize=12)
    
    return save_plot(fig, "robustness.png", subfolder)

def plot_benchmark_results(csv_path):
    """Wrapper to generate all plots for the given CSV."""
    if not os.path.exists(csv_path): return
    
    # Extract Context from CSV Path for Subfolder
    # Expected: .../output/{challenge}/{mode}/intervals/benchmark.csv 
    # OR .../output/{challenge}/intervals/benchmark.csv
    
    try:
        norm_path = os.path.normpath(csv_path)
        parts = norm_path.split(os.sep)
        
        # Find 'output' index
        if 'output' in parts:
            idx = parts.index('output')
            # Check if grid or cartpole
            challenge = parts[idx+1] # grid or cartpole
            
            if challenge == "grid":
                mode = parts[idx+2]      # random or fixed
                subfolder = os.path.join(challenge, mode, "graficos")
            else:
                subfolder = os.path.join(challenge, "graficos") # No mode for cartpole
        else:
            subfolder = "misc"
            
    except Exception:
        subfolder = "misc"

    # Pass subfolder to plotting functions
    plot_success_rates(csv_path, subfolder)
    if challenge != "cartpole":
        plot_step_ratios(csv_path, subfolder)
    plot_efficiency(csv_path, subfolder)
    plot_robustness(csv_path, subfolder)
    
    return subfolder

    return subfolder

