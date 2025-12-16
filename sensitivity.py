import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
DATASETS = {
    'Amazon-Google': {
        'path_b': './data/Amazon.csv',
        'path_a': './data/GoogleProducts.csv',
        'cols': ['title', 'description', 'manufacturer'],
        'encoding': 'unicode_escape'
    },
    'DBLP-ACM': {
        'path_a': './data/ACM.csv',
        'path_b': './data/DBLP.csv',
        'cols': ['title', 'authors', 'venue'],
        'encoding': 'unicode_escape'
    }
}

CONFIGS = [
    {"label": "Sluggish ($W=800$)", "W": 800, "lr": 0.05, "color": "#1f77b4", "style": "--", "width": 2},
    {"label": "Balanced ($W=200$)", "W": 200, "lr": 0.05, "color": "#2ca02c", "style": "-", "width": 3}
]

# --- 1. DATA LOADING (Keep your existing functions) ---
def preprocess_text(df, cols):
    df['text'] = ""
    for col in cols:
        if col in df.columns:
            df['text'] += df[col].astype(str).fillna('') + " "
    return df['text']

def get_natural_stream(ds_name, top_k=10):
    cfg = DATASETS[ds_name]
    print(f"[{ds_name}] Loading & Embedding...")
    df_a = pd.read_csv(cfg['path_a'], encoding=cfg['encoding'])
    df_b = pd.read_csv(cfg['path_b'], encoding=cfg['encoding'])
    
    df_a['text'] = preprocess_text(df_a, cfg['cols'])
    df_b['text'] = preprocess_text(df_b, cfg['cols'])
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vec_a = model.encode(df_a['text'].tolist(), show_progress_bar=True)
    vec_b = model.encode(df_b['text'].tolist(), show_progress_bar=True)
    
    faiss.normalize_L2(vec_a); faiss.normalize_L2(vec_b)
    index = faiss.IndexHNSWFlat(vec_a.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.add(vec_a)
    sims, _ = index.search(vec_b, top_k)
    return sims.flatten()

# --- 2. CONTROLLER LOGIC ---
def get_optimal_alpha(scores, target_budget):
    def budget_diff(a):
        return np.sum(np.minimum(1.0, a * scores)) - target_budget
    try:
        opt_alpha = brentq(budget_diff, 0.0, 10.0)
    except:
        opt_alpha = 0.3
    return opt_alpha

def run_controller(scores, win, lr, target_budget):
    stream_len = len(scores)
    # Heuristic init
    alpha = target_budget / (stream_len * 0.5)
    B_w = target_budget / (stream_len / win)
    
    traj_x, traj_y = [], []
    total_utility = 0.0
    current_alpha = alpha
    
    for i in range(0, stream_len, win):
        end = min(i + win, stream_len)
        chunk = scores[i:end]
        
        # Calculate probabilities and utility
        probs = np.minimum(1.0, current_alpha * chunk)
        m_w = np.sum(probs)
        # Utility = Sum of weights * probability of selection (Expected Utility)
        total_utility += np.sum(probs * chunk) 
        
        # Update Controller
        if B_w > 0:
            dev = (B_w - m_w) / B_w
            current_alpha = current_alpha * (1 + lr * dev)
            current_alpha = max(0.01, min(current_alpha, 10.0))
        
        traj_x.append((end / stream_len) * 100)
        traj_y.append(current_alpha)
        
    return traj_x, traj_y, total_utility

# --- 3. SENSITIVITY ANALYSIS FOR W ---
def run_w_sensitivity(scores, target_budget, w_values):
    utilities = []
    
    # Calculate Offline Optimal Utility (Top-K Oracle)
    # We sort all candidates and take the Top-Budget sum
    k_budget = int(target_budget)
    sorted_scores = np.sort(scores)[::-1]
    optimal_utility = np.sum(sorted_scores[:k_budget])
    
    for w in w_values:
        # Run with fixed learning rate (e.g. 0.1) for varying W
        _, _, util = run_controller(scores, w, lr=0.1, target_budget=target_budget)
        utilities.append(util / optimal_utility)
        
    return w_values, utilities

# --- 4. PLOTTING FUNCTION ---
def plot_combined_row():
    # 1. INCREASE BASE FONT SCALE
    sns.set_theme(style="whitegrid", context="talk", font_scale=2.0)
    
    # Width=28 to accommodate larger text
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    
    ds_names = ['Amazon-Google', 'DBLP-ACM']
    w_range = [50, 100, 200,300,400, 500] 
    
    for idx, name in enumerate(ds_names):
        print(f"Processing {name}...")
        scores = get_natural_stream(name)
        target_budget = len(scores) * 0.15
        opt_alpha = get_optimal_alpha(scores, target_budget)
        
        # --- PLOT A: ALPHA DYNAMICS (Columns 0 and 2) ---
        ax_dyn = axes[idx * 2] 
        
        # Track min/max values for dynamic zooming
        all_y_values = [opt_alpha] 
        
        # Plot Ideal Line
        ax_dyn.axhline(opt_alpha, color='#d62728', linestyle=':', linewidth=5.0, label='Ideal $\\alpha$')
        
        for cfg in CONFIGS:
            x, y, _ = run_controller(scores, cfg['W'], cfg['lr'], target_budget)
            # Smooth lightly 
            y_smooth = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values
            
            # Store values to calculate zoom later
            all_y_values.extend(y_smooth)
            
            ax_dyn.plot(x, y_smooth, label=cfg['label'], color=cfg['color'], 
                        linestyle=cfg['style'], linewidth=cfg['width'] + 2) # Thicker lines
            
        ax_dyn.set_title(f'{name}\nController Stability', fontweight='bold', fontsize=30, pad=20)
        ax_dyn.set_xlabel('Progress of S (%)', fontsize=24, fontweight='bold')
        ax_dyn.set_ylabel(r'Scaling Factor $\alpha$', fontsize=24, fontweight='bold')
        ax_dyn.tick_params(axis='both', which='major', labelsize=20)

        # --- DYNAMIC Y-LIMIT ZOOM ---
        # Calculate strict bounds based on data range to show fluctuations
        y_min, y_max = min(all_y_values), max(all_y_values)
        y_range = y_max - y_min
        # Add small 10% padding so lines don't touch the edge
        ax_dyn.set_ylim(y_min - (y_range * 0.1), y_max + (y_range * 0.1))

        # --- PLOT B: W SENSITIVITY (Columns 1 and 3) ---
        ax_sens = axes[idx * 2 + 1]
        
        ws, utils = run_w_sensitivity(scores, target_budget, w_range)
        
        ax_sens.plot(ws, utils, marker='o', color='#4b0082', linewidth=5, markersize=14, label='Perf. vs $W$')
        #ax_sens.set_xscale('log')
        ax_sens.set_title(f'{name}\nWindow Sensitivity', fontweight='bold', fontsize=30, pad=20)
        ax_sens.set_xlabel('Window Size $W$', fontsize=24, fontweight='bold')
        ax_sens.set_ylabel('Norm. Cumul. Utility', fontsize=24, fontweight='bold')
        ax_sens.tick_params(axis='both', which='major', labelsize=20)
        
        # Highlight W=200
        ax_sens.axvline(200, color='#2ca02c', linestyle='--', linewidth=4, alpha=0.6, label='Selected $W=200$')
        
        min_util = min(utils)
        ax_sens.set_ylim(max(0.0, min_util - 0.02), 1.01)

    # --- SHARED LEGEND ---
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    unique = dict(zip(l1 + l2, h1 + h2))
    
    # Larger Legend Font
    fig.legend(unique.values(), unique.keys(), loc='lower center', 
               bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, fontsize=24)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, wspace=0.35) 
    plt.savefig('sensitivity.pdf', bbox_inches='tight', dpi=500)
    plt.show()

if __name__ == "__main__":
    plot_combined_row()
