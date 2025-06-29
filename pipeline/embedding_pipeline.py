# aideoss/pipeline/embedding_pipeline.py

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import faiss

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# -------------------------------
# Config
# -------------------------------
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INPUT = "data/logs_with_summaries.csv"
DEFAULT_OUTPUT = "models/"
DEFAULT_TEXT_SOURCE = "summary"

os.makedirs(DEFAULT_OUTPUT, exist_ok=True)

# -------------------------------
# Load model (HF or ST)
# -------------------------------
def load_model(model_name: str):
    if "sentence-transformers" in model_name or "e5" in model_name or "bge" in model_name:
        return SentenceTransformer(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device

# -------------------------------
# Embed using SentenceTransformers
# -------------------------------
def encode_st_model(model, texts: List[str]):
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# -------------------------------
# Embed using HuggingFace Transformers
# -------------------------------
def encode_hf_model(tokenizer, model, device, texts: List[str], batch_size=16):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = output.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(last_hidden.cpu().numpy())
    return np.vstack(embeddings)

# -------------------------------
# Main Pipeline
# -------------------------------
def build_faiss_index(
    model_name: str = DEFAULT_MODEL,
    input_file: str = DEFAULT_INPUT,
    output_dir: str = DEFAULT_OUTPUT,
    text_source: str = DEFAULT_TEXT_SOURCE
):
    df = pd.read_csv(input_file)
    df[text_source] = df[text_source].fillna("")
    texts = df[text_source].tolist()

    print(f"🔍 Loaded {len(texts)} texts from '{input_file}' using source column '{text_source}'")

    if "sentence-transformers" in model_name or "e5" in model_name or "bge" in model_name:
        model = load_model(model_name)
        embeddings = encode_st_model(model, texts)
    else:
        tokenizer, model, device = load_model(model_name)
        embeddings = encode_hf_model(tokenizer, model, device, texts)

    print(f"✅ Embeddings shape: {embeddings.shape}")

    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))

    # Save texts and embeddings
    with open(os.path.join(output_dir, "log_texts.pkl"), "wb") as f:
        pickle.dump(texts, f)

    np.save(os.path.join(output_dir, "log_embeddings.npy"), embeddings)

    print(f"📦 Saved FAISS index and metadata to '{output_dir}'")

# -------------------------------
# CLI Entrypoint
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index for AIDE-OSS")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Embedding model name")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--text-source", type=str, default=DEFAULT_TEXT_SOURCE, help="Column to use: summary | log | template")

    args = parser.parse_args()
    build_faiss_index(args.model, args.input, args.output, args.text_source)
