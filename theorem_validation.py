import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATASETS = {
    'Abt-Buy': {
        'path_a': './data/Buy.csv', 'path_b': './data/Abt.csv',
        'cols': ['name', 'description'], 'encoding': 'unicode_escape', 'k': 5
    },
    'DBLP-ACM': {
        'path_a': './data/DBLP.csv', 'path_b': './data/ACM.csv',
        'cols': ['title', 'authors', 'venue', 'year'], 'encoding': 'unicode_escape', 'k': 5
    }
}
CONFIG = {"W": 200, "lr": 0.1}

def preprocess_text(df, cols):
    df['text'] = ""
    for col in cols:
        if col in df.columns:
            df['text'] += df[col].astype(str).fillna('') + " "
    return df['text']

def get_natural_stream_utility(ds_name):
    cfg = DATASETS[ds_name]
    print(f"[{ds_name}] Loading & Embedding...")
    try:
        df_a = pd.read_csv(cfg['path_a'], encoding=cfg['encoding'])
        df_b = pd.read_csv(cfg['path_b'], encoding=cfg['encoding'])
    except:
        print(f"[{ds_name}] File not found, using dummy data.")
        return np.abs(np.random.normal(0.5, 0.15, 5000))

    df_a['text'] = preprocess_text(df_a, cfg['cols'])
    df_b['text'] = preprocess_text(df_b, cfg['cols'])

    model = SentenceTransformer('all-MiniLM-L6-v2')
    vec_a = model.encode(df_a['text'].tolist(), show_progress_bar=False)
    vec_b = model.encode(df_b['text'].tolist(), show_progress_bar=False)

    faiss.normalize_L2(vec_a)
    faiss.normalize_L2(vec_b)

    index = faiss.IndexHNSWFlat(vec_a.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.add(vec_a)
    dists, _ = index.search(vec_b, cfg['k'])
    return dists.flatten()

def run_budget_comparison(scores):
    total_pairs = len(scores)
    # Filter negative scores for fair comparison (utility >= 0)
    scores = np.maximum(scores, 0)
    
    sorted_scores = np.sort(scores)[::-1]
    optimal_utility = np.cumsum(sorted_scores)

    target_budget_total = total_pairs #* 0.15
    # Initial alpha estimation
    alpha = target_budget_total / (total_pairs * 0.5) 
    win, lr = CONFIG['W'], CONFIG['lr']
    B_w, m_w = 0.15 * win, 0

    sper_x, sper_y = [0], [0.0]
    pairs_cnt, curr_util = 0, 0.0
    np.random.seed(42)
    print(scores)
    for i, score in enumerate(scores):
        prob = min(1.0, alpha * score)
        if np.random.rand() < prob:
            pairs_cnt += 1
            curr_util += score
            sper_x.append(pairs_cnt)
            sper_y.append(curr_util)

        m_w += prob
        if (i + 1) % win == 0:
            if B_w > 0:
                dev = (B_w - m_w) / B_w
                alpha = alpha * (1 + lr * dev)
                alpha = max(0.01, min(alpha, 10.0))
            m_w = 0

    return optimal_utility, (np.array(sper_x), np.array(sper_y))

def plot_theorem_vertical_clean():
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 2))

    datasets = ['Abt-Buy', 'DBLP-ACM']
    axes = [ax1, ax2]
    titles = ['Abt-Buy', 'DBLP-ACM']

    for ax, ds, title in zip(axes, datasets, titles):
        scores = get_natural_stream_utility(ds)
        # Ensure scores are positive for theoretical calculation
        valid_scores = scores[scores > 0]
        
        opt_util_curve, (sper_x, sper_y) = run_budget_comparison(scores)

        max_k = sper_x[-1]
        opt_x = np.arange(max_k + 1)
        
        # Calculate Normalization Factor (Total Utility of Top-K)
        # Note: We normalize by the Optimal utility at the *end point* # to show percentage of optimality.
        opt_y = opt_util_curve[:max_k + 1]
        norm_factor = opt_y[-1]

        # --- 1. Optimal Line ---
        ax.plot(opt_x, opt_y / norm_factor, color='gray', linestyle='--', linewidth=3.0,
                label=r'Optimal $\mathcal{S}^*$ (Sorted)', alpha=0.6)

        # --- 2. Theoretical Expectation (Theorem 4.1) ---
        # E[U] = B * (Sum(w^2) / Sum(w))
        # This is a linear projection based on the second moment.
        w_sum = np.sum(valid_scores)
        w_sq_sum = np.sum(valid_scores**2)
        theo_slope = w_sq_sum / w_sum if w_sum > 0 else 0
        
        theo_curve = (opt_x * theo_slope) / norm_factor
        
        ax.plot(opt_x, theo_curve, color='black', linestyle=':', linewidth=3.0,
                label=r'Theor. Exp. (Th. 4.1)', zorder=20)

        # --- 3. SPER Line ---
        ax.plot(sper_x, sper_y / norm_factor, color='#2ca02c', linewidth=4.0,
                label='SPER', zorder=10)

        # --- Fill Between ---
        # Interpolate SPER to match opt_x for filling
        sper_y_interp = np.interp(opt_x, sper_x, sper_y / norm_factor)
        
        # Fill between SPER and Theory to highlight the match/gain
        ax.fill_between(opt_x, sper_y_interp, theo_curve, color='#2ca02c', alpha=0.1)

        ax.set_title(title, fontweight='bold', pad=15, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_k)
        ax.set_ylim(0, 1.05)

    ax1.tick_params(axis='both', which='major', labelsize=15, width=2, length=6)
    ax2.tick_params(axis='both', which='major', labelsize=15, width=2, length=6)

    # Labels and Legend
    ax1.set_xlabel('Budget B', fontweight='bold', fontsize=16)
    ax2.set_xlabel('Budget B', fontweight='bold', fontsize=16)
    ylabel = fig.text(0.025, 0.55, 'Norm. Cum. Utility',
                      va='center', rotation='vertical', fontweight='bold', fontsize=16)

    handles, labels = ax1.get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.37),
                     ncol=3, frameon=False, fontsize=16)

    plt.subplots_adjust(left=0.13, right=0.95, top=0.92, bottom=0.17, hspace=0.39)

    print("Saving figure to theorem_validation.pdf...")
    plt.savefig('theorem_validation.pdf', bbox_extra_artists=(lgd,ylabel), bbox_inches='tight', dpi=500)
    plt.show()

if __name__ == "__main__":
    plot_theorem_vertical_clean()
