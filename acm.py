import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import csv
import os
import time

# --- Configuration ---
BUDGET_STEPS = [3_000, 5_000, 8_000, 10_000,  13_000]
CSV_FILENAME = './results/acm.csv'

def log_results_to_csv(budget, cost, adherence, recall, precision, time_taken, filename=CSV_FILENAME):
    """Appends a single experiment result to the CSV file."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Target_Budget', 'Actual_Cost', 'Adherence_Pct', 'Recall_Pct', 'Precision_Pct', 'Time_Sec'])
            
        writer.writerow([budget, cost, f"{adherence:.2f}", f"{recall:.2f}", f"{precision:.2f}", f"{time_taken:.4f}"])

def load_and_embed_data(abt_path='./data/ACM.csv', buy_path='./data/DBLP.csv', gt_path='./data/truth_ACM_DBLP.csv'):
    print("1. Loading & Preprocessing Data...")
    try:
        df_abt = pd.read_csv(abt_path, encoding='unicode_escape')
        df_buy = pd.read_csv(buy_path, encoding='unicode_escape')
        df_gt = pd.read_csv(gt_path)
    except FileNotFoundError:
        print("Error: CSV files not found.")
        return None, None, None, None, None

    gt_pairs = set(zip(df_gt['idACM'], df_gt['idDBLP']))
    
    # Text Preprocessing
    df_abt['text'] = (df_abt['authors'].fillna('') + " " + df_abt['title'].fillna('') + " " + df_abt['venue'].fillna('')  ).astype(str)
    df_buy['text'] = (df_buy['authors'].fillna('') + " " + df_buy['title'].fillna('') + " " + df_buy['venue'].fillna('') ).astype(str)
    
    # Embedding
    print("2. Generating MiniLM Embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors_abt = model.encode(df_abt['text'].tolist(), show_progress_bar=True)
    vectors_buy = model.encode(df_buy['text'].tolist(), show_progress_bar=True)
    
    # Normalize for Cosine Similarity via Inner Product
    faiss.normalize_L2(vectors_abt)
    faiss.normalize_L2(vectors_buy)
 
    stream_size = len(vectors_buy)
    total_matches = len(gt_pairs)
    print(f"Stream size: {stream_size} entities. {stream_size*5}")
    print(f"Total matches: {total_matches}")    

    return vectors_abt, df_abt['id'].values, vectors_buy, df_buy['id'].values, gt_pairs

def build_faiss_index(vectors, d):
    """Builds an HNSW index for Inner Product search."""
    M = 32 
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40 
    index.add(vectors)
    return index

def run_single_sper_pass(
    vectors_abt, ids_abt, 
    vectors_buy, ids_buy, 
    gt_pairs, 
    target_budget,
    window_size=200, 
    top_k=5,
    learning_rate=0.1
):
    start_time = time.time()
    
    stream_size = len(ids_buy)
    total_matches = len(gt_pairs)

    dim = vectors_abt.shape[1]
    
    # --- Build Index ---
    index_abt = build_faiss_index(vectors_abt, dim)
    index_abt.hnsw.efSearch = 64

    # --- Initialization ---
    total_candidates_est = stream_size * top_k
    alpha = target_budget / (total_candidates_est * 0.5)
    B_w = target_budget / (stream_size / window_size)
    
    cumulative_comparisons = 0
    cumulative_matches = 0
    
    # --- Stream Loop ---
    for i in range(0, stream_size, window_size):
        end_idx = min(i + window_size, stream_size)
        window_vectors = vectors_buy[i : end_idx]
        window_ids = ids_buy[i : end_idx]
        m_w = 0 
        
        # 1. Retrieval (FAISS)
        sims, idxs = index_abt.search(window_vectors, top_k)
        
        # Iterate through queries in this window
        for local_idx in range(len(window_ids)):
            q_id = window_ids[local_idx]
            
            # 2. Stochastic Selection
            for rank in range(top_k):
                idx_neighbor = idxs[local_idx][rank]
                if idx_neighbor < 0: continue
                
                w_score = sims[local_idx][rank]
                target_id = ids_abt[idx_neighbor]
                
                p_select = min(1.0, alpha * w_score)
                 
                if np.random.random() < p_select:
                   m_w += 1
                   cumulative_comparisons += 1                    
                   if (target_id, q_id) in gt_pairs:
                        cumulative_matches += 1
        
        # 3. Budget Update
        if B_w > 0:
            deviation = (B_w - m_w) / B_w
            alpha = alpha * (1 + learning_rate * deviation)
            alpha = max(0.0001, alpha)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # --- Metrics Calculation ---
    final_recall = (cumulative_matches / total_matches) * 100
    adherence = (cumulative_comparisons / target_budget) * 100
    
    if cumulative_comparisons > 0:
        precision = (cumulative_matches / cumulative_comparisons) * 100
    else:
        precision = 0.0
    
    return final_recall, adherence, cumulative_comparisons, precision, elapsed_time

def run_experiment_logging():
    data = load_and_embed_data()
    if data[0] is None: return
    vec_abt, id_abt, vec_buy, id_buy, gt = data
    
    print(f"\n3. Starting Experiment (FAISS). Results will be saved to '{CSV_FILENAME}'...")
    print(f"{'Budget':<10} | {'Cost':<10} | {'Adh.%':<8} | {'Recall%':<8} | {'Prec.%':<8} | {'Time(s)':<8}")
    print("-" * 75)
    
    # Optional: Clear previous file if needed
    if os.path.exists(CSV_FILENAME):
        os.remove(CSV_FILENAME)
    
    for b in BUDGET_STEPS:
        recall, adherence, cost, prec, t_sec = run_single_sper_pass(
            vec_abt, id_abt, vec_buy, id_buy, gt, 
            target_budget=b
        )
        
        # Log to CSV
        log_results_to_csv(b, cost, adherence, recall, prec, t_sec)
        
        # Print to Console
        print(f"{b:<10} | {cost:<10} | {adherence:.1f}%   | {recall:.2f}%   | {prec:.2f}%   | {t_sec:.4f}")

    print(f"\nDone! Data saved to {CSV_FILENAME}")

# Run Experiment
if __name__ == "__main__":
    run_experiment_logging()
