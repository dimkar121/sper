import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import csv
import os
import time

# --- Configuration ---
BUDGET_STEPS = [2000, 5000, 8_000, 10_000, 12_700 ]
CSV_FILENAME = './results/walmart.csv'

# Dataset Paths & Columns
PATH_RAW_A = './data/amazon_products.csv'
PATH_RAW_B = './data/walmart_products.csv'
PATH_GT = './data/truth_amazon_walmart.tsv' # Assumed TSV based on typical data format
ID_COL_A = 'id1'
ID_COL_B = 'id2'
COLS_TO_USE = [ "longdescr", "shortdescr", "title"]


def log_results_to_csv(budget, cost, adherence, recall, precision, time_taken, filename=CSV_FILENAME):
    """Appends a single experiment result to the CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Target_Budget', 'Actual_Cost', 'Adherence_Pct', 'Recall_Pct', 'Precision_Pct', 'Time_Sec'])
            
        writer.writerow([budget, cost, f"{adherence:.2f}", f"{recall:.2f}", f"{precision:.2f}", f"{time_taken:.4f}"])

def preprocess_text(df, cols):
    """Concatenates specified columns into a single text string."""
    df['text'] = ""
    for col in cols:
        if col in df.columns:
            # Cast to string to handle prices/numbers safely
            df['text'] += df[col].astype(str).fillna('') + " "
    return df['text']

def load_and_embed_data(
    path_a=PATH_RAW_A, 
    path_b=PATH_RAW_B, 
    path_gt=PATH_GT
):
    print("1. Loading & Preprocessing Data...")
    try:
        # Load CSVs (Handle encoding gracefully)
        try:
            df_a = pd.read_csv(path_a, encoding="unicode_escape")
            df_b = pd.read_csv(path_b, encoding="unicode_escape")
        except UnicodeDecodeError:
            df_a = pd.read_csv(path_a, encoding='latin1')
            df_b = pd.read_csv(path_b, encoding='latin1')
            
        # Load Ground Truth (TSV)
        df_gt = pd.read_csv(path_gt, sep='\t', encoding="unicode_escape", keep_default_na=False)
        
    except FileNotFoundError:
        print("Error: Files not found. Please check paths.")
        return None, None, None, None, None

    # Text Preprocessing

    df_a['id'] = pd.to_numeric(df_a['id'], errors='coerce')
    df_a.dropna(subset=['id'], inplace=True)
    df_a['id'] = df_a['id'].astype(int)
    df_b['id'] = pd.to_numeric(df_b['id'], errors='coerce')
    df_b.dropna(subset=['id'], inplace=True)
    df_b['id'] = df_b['id'].astype(int)
    df_a.reset_index(drop=True, inplace=True)
    df_b.reset_index(drop=True, inplace=True)

    df_a['text'] = preprocess_text(df_a, COLS_TO_USE)
    df_b['text'] = preprocess_text(df_b, COLS_TO_USE)
    
    # --- Gold Standard Filtering (Crucial Step) ---
    print("   Filtering Gold Standard...")
    # Create Sets of Valid IDs for O(1) lookup
    valid_ids_a = set(df_a['id'].values) 
    valid_ids_b = set(df_b['id'].values)
    
    gt_pairs = set()
    
    for index, row in df_gt.iterrows():
        id_walmart = row[ID_COL_B] 
        id_amazon = row[ID_COL_A]
        
        # Exclude pairs where IDs are missing from source files
        if id_walmart not in valid_ids_b or id_amazon not in valid_ids_a:
            continue 
            
        gt_pairs.add((id_amazon, id_walmart)) # Store as (Amazon, Walmart) tuple
        
    print(f"   Valid GT Pairs Loaded: {len(gt_pairs)}")

    # --- Embedding ---
    print("2. Generating MiniLM Embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed Amazon (Index)
    vectors_a = model.encode(df_a['text'].tolist(), show_progress_bar=True)
    # Embed Walmart (Stream)
    vectors_b = model.encode(df_b['text'].tolist(), show_progress_bar=True)
    
    # Normalize for Cosine Similarity (FAISS Inner Product)
    faiss.normalize_L2(vectors_a)
    faiss.normalize_L2(vectors_b)

    stream_size = len(vectors_b)
    total_matches = len(gt_pairs)
    print(f"Stream size: {stream_size} entities {stream_size*5} .")
    print(f"Total matches: {total_matches}")

    
    return vectors_a, df_a['id'].values, vectors_b, df_b['id'].values, gt_pairs

def build_faiss_index(vectors, d):
    """Builds an HNSW index for Inner Product search."""
    M = 32 
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40 
    index.add(vectors)
    return index

def run_single_sper_pass(
    vectors_idx, ids_idx, # Index Data (Amazon)
    vectors_stm, ids_stm, # Stream Data (Walmart)
    gt_pairs, 
    target_budget,
    window_size=200, 
    top_k=5,
    learning_rate=0.15
):
    start_time = time.time()
    
    stream_size = len(ids_stm)
    total_matches = len(gt_pairs)
    dim = vectors_idx.shape[1]
    
    # --- Build Index ---
    index = build_faiss_index(vectors_idx, dim)
    index.hnsw.efSearch = 64

    # --- Initialization ---
    total_candidates_est = stream_size * top_k
    alpha = target_budget / (total_candidates_est * 0.5)
    B_w = target_budget / (stream_size / window_size)
    
    cumulative_comparisons = 0
    cumulative_matches = 0
    
    # --- Stream Loop ---
    for i in range(0, stream_size, window_size):
        end_idx = min(i + window_size, stream_size)
        window_vectors = vectors_stm[i : end_idx]
        window_ids = ids_stm[i : end_idx]
        m_w = 0 
        
        # 1. Retrieval (FAISS)
        sims, idxs = index.search(window_vectors, top_k)
        
        # Iterate through queries in this window
        for local_idx in range(len(window_ids)):
            q_id = window_ids[local_idx] # Walmart ID
            
            # 2. Stochastic Selection
            for rank in range(top_k):
                idx_neighbor = idxs[local_idx][rank]
                if idx_neighbor < 0: continue
                
                w_score = sims[local_idx][rank]
                target_id = ids_idx[idx_neighbor] # Amazon ID
                
                # SPER Probability
                p_select = min(1.0, alpha * w_score)
                
                if np.random.random() < p_select:
                    m_w += 1
                    cumulative_comparisons += 1
                    
                    # Check Match (Amazon_ID, Walmart_ID)
                    if (target_id, q_id) in gt_pairs:
                    #if (q_id, target_id) in gt_pairs:
                        cumulative_matches += 1
        
        # 3. Budget Update
        if B_w > 0:
            deviation = (B_w - m_w) / B_w
            alpha = alpha * (1 + learning_rate * deviation)
            alpha = max(0.0001, alpha)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # --- Metrics ---
    if total_matches > 0:
        final_recall = (cumulative_matches / total_matches) * 100
    else:
        final_recall = 0.0
        
    adherence = (cumulative_comparisons / target_budget) * 100
    
    if cumulative_comparisons > 0:
        precision = (cumulative_matches / cumulative_comparisons) * 100
    else:
        precision = 0.0
    
    return final_recall, adherence, cumulative_comparisons, precision, elapsed_time

def run_experiment_logging():
    data = load_and_embed_data()
    if data[0] is None: return
    
    # Unpack Data
    vec_idx, id_idx, vec_stm, id_stm, gt = data
    
    print(f"\n3. Starting Experiment (Walmart-Amazon). Results to '{CSV_FILENAME}'...")
    print(f"{'Budget':<10} | {'Cost':<10} | {'Adh.%':<8} | {'Recall%':<8} | {'Prec.%':<8} | {'Time(s)':<8}")
    print("-" * 75)
    
    if os.path.exists(CSV_FILENAME):
        os.remove(CSV_FILENAME)
    
    for b in BUDGET_STEPS:
        recall, adherence, cost, prec, t_sec = run_single_sper_pass(
            vec_idx, id_idx, 
            vec_stm, id_stm, 
            gt, 
            target_budget=b
        )
        
        log_results_to_csv(b, cost, adherence, recall, prec, t_sec)
        print(f"{b:<10} | {cost:<10} | {adherence:.1f}%   | {recall:.2f}%   | {prec:.2f}%   | {t_sec:.4f}")

    print(f"\nDone! Data saved to {CSV_FILENAME}")

# Run Experiment
if __name__ == "__main__":
    run_experiment_logging()
