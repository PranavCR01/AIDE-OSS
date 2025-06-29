# utils/ollama_summarizer.py

import requests
from typing import List
from tqdm import tqdm

_summary_cache = {}

def summarize_log(text: str) -> str:
    if text in _summary_cache:
        return _summary_cache[text]

    payload = {
        "model": "mistral",
        "prompt": f"Summarize the following log message in one sentence:\n{text}",
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        summary = response.json().get("response", "").strip()
    except Exception as e:
        summary = f"[ Summary failed: {e}]"
        print(f"⚠️  Summarization failed for:\n{text[:100]}...\nError: {e}\n")

    _summary_cache[text] = summary
    return summary



def batch_summarize_logs(logs: List[str], batch_size: int = 10) -> List[str]:
    summaries = []
    for i in tqdm(range(0, len(logs), batch_size), desc=" Summarizing unique templates"):
        batch = logs[i:i+batch_size]
        batch_summaries = [summarize_log(log) for log in batch]
        summaries.extend(batch_summaries)
    return summaries
