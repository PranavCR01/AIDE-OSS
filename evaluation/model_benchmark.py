# evaluation/model_benchmark.py

import pandas as pd
import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

QUERY_FILE = "data/embedding_model_benchmark.csv"
RESULTS_FILE = "reports/embedding_model_results.csv"
METRICS_FILE = "reports/embedding_model_metrics.csv"

MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "models/faiss_index.bin"
TEXT_STORE_PATH = "models/log_texts.pkl"

def compute_mrr(ranks):
    return np.mean([1 / (r + 1) for r in ranks]) if ranks else 0.0

def evaluate_model(model_name, query_df):
    embedder = SentenceTransformer(model_name)
    index = faiss.read_index(FAISS_INDEX_PATH)
    logs = joblib.load(TEXT_STORE_PATH)

    top1_hits, ranks, detailed_rows = 0, [], []

    for _, row in tqdm(query_df.iterrows(), total=len(query_df)):
        query = row["query"]
        expected = str(row["expected_substring"]).lower()

        vec = embedder.encode([query]).astype("float32")
        _, indices = index.search(vec, 10)
        results = [logs[i] for i in indices[0]]

        match_rank = next((i for i, log in enumerate(results) if expected in log.lower()), -1)
        if match_rank == 0:
            top1_hits += 1
        if match_rank >= 0:
            ranks.append(match_rank)

        detailed_rows.append({
            "model": model_name,
            "query": query,
            "expected": expected,
            "match_rank": match_rank if match_rank >= 0 else "miss",
            "top1_hit": match_rank == 0
        })

    return detailed_rows, top1_hits / len(query_df), compute_mrr(ranks)

def run_benchmark():
    query_df = pd.read_csv(QUERY_FILE)
    all_results = []
    metrics_summary = []

    print(f"\n Evaluating {MODEL_NAME} ...")
    results, acc, mrr = evaluate_model(MODEL_NAME, query_df)
    all_results.extend(results)
    metrics_summary.append({
        "model": MODEL_NAME,
        "accuracy@1": round(acc, 4),
        "MRR": round(mrr, 4)
    })

    pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
    pd.DataFrame(metrics_summary).to_csv(METRICS_FILE, index=False)
    print(f"\n Benchmark complete. Results saved to:\n- {RESULTS_FILE}\n- {METRICS_FILE}")

if __name__ == "__main__":
    run_benchmark()
