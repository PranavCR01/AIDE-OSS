# evaluation/embedding_evaluator.py

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os
import json

DATA_PATH = "data/logs_with_anomaly_score.csv"
EVAL_RESULTS_PATH = "reports/embedding_eval_results.json"
BATCH_SIZE = 64
TOP_K = 5

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "BAAI/bge-base-en-v1.5",
    "sentence-transformers/sentence-t5-base"
]

def encode_in_batches(model, texts, batch_size=64):
    return np.vstack([
        model.encode(texts[i:i + batch_size], convert_to_numpy=True)
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding")
    ])

def evaluate_model(model_name, eval_queries, texts, labels):
    print(f"\n Evaluating model: {model_name}")
    try:
        model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        query_vecs = encode_in_batches(model, eval_queries, BATCH_SIZE)
        db_vecs = encode_in_batches(model, texts, BATCH_SIZE)

        faiss_index = faiss.IndexFlatL2(db_vecs.shape[1])
        faiss_index.add(db_vecs)

        correct = 0
        for i in tqdm(range(len(eval_queries)), desc="Running evaluation"):
            _, I = faiss_index.search(np.array([query_vecs[i]]), TOP_K + 1)
            I_filtered = [j for j in I[0] if j != i][:TOP_K]
            retrieved_templates = [labels[j] for j in I_filtered]
            correct += int(labels[i] in retrieved_templates)

        precision_at_k = correct / len(eval_queries)
        return {
            "model": model_name,
            "precision@5": round(precision_at_k, 4),
            "matches": int(correct),
            "total_queries": len(eval_queries)
        }

    except Exception as e:
        return {"model": model_name, "error": str(e)}

def evaluate_all():
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["template"], inplace=True)
    texts = df["log"].tolist()
    labels = df["template"].tolist()
    eval_queries = texts.copy()

    results = []
    for model in MODELS:
        result = evaluate_model(model, eval_queries, texts, labels)
        results.append(result)

    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Evaluation complete. Results saved to: {EVAL_RESULTS_PATH}")
