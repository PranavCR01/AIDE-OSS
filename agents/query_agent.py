# agents/query_agent.py

import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import subprocess

INDEX_PATH = "models/faiss_index.bin"
LOGS_PATH = "models/log_texts.pkl"

def load_resources():
    index = faiss.read_index(INDEX_PATH)
    logs = joblib.load(LOGS_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, logs, model

def query_logs(query, index, logs, model, top_k=5):
    query_vector = model.encode([query])
    scores, indices = index.search(np.array(query_vector), top_k)
    return [logs[i] for i in indices[0]]

def build_prompt(matching_logs, query):
    return f"""
You are a cybersecurity assistant. Here is a user query:
"{query}"

The following logs were retrieved as relevant:

{chr(10).join(matching_logs)}

Please summarize what’s going on and provide possible explanations or recommendations.
"""

def ask_mistral(prompt):
    result = subprocess.run(["ollama", "run", "mistral"], input=prompt.encode(), stdout=subprocess.PIPE)
    print("\nMistral's RAG-based Response:\n")
    print(result.stdout.decode())

if __name__ == "__main__":
    index, logs, model = load_resources()
    query = input("\n Enter your log query: ")
    matches = query_logs(query, index, logs, model)
    prompt = build_prompt(matches, query)
    ask_mistral(prompt)
