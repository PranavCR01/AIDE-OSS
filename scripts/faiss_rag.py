# scripts/faiss_rag.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Load summarized logs
df = pd.read_csv("data/logs_with_summaries.csv")
texts = df["summary"].fillna("").tolist()

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(texts, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load summarization model (FLAN-T5)
llm = pipeline("text2text-generation", model="google/flan-t5-base")

def ask(query, top_k=3):
    query_vec = embedder.encode([query])
    _, indices = index.search(query_vec, top_k)

    retrieved_logs = [texts[i] for i in indices[0]]
    context = " ".join(retrieved_logs)

    prompt = f"summarize: {context}"
    result = llm(prompt, max_length=100, do_sample=False)
    
    print("\n Top Results:")
    for i, log in enumerate(retrieved_logs, 1):
        print(f"{i}. {log}")
    print("\n💡 Summary:")
    print(result[0]['generated_text'])

# Example loop
if __name__ == "__main__":
    print(" AIDE-OSS | Ask about suspicious logs (type 'exit' to quit)")
    while True:
        q = input("\nYour question: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        ask(q)
