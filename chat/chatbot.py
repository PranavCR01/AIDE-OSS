## chat/ chatbot.py

import faiss
import numpy as np
import pandas as pd
import joblib
import subprocess
from sentence_transformers import SentenceTransformer
from datetime import datetime
import dateparser
import re
import codecs


class RAGChatbot:
    def __init__(self):
        self.INDEX_PATH = "models/faiss_index.bin"
        self.EMBEDDINGS_PATH = "models/log_embeddings.npy"
        self.LOGS_PATH = "models/log_texts.pkl"
        self.CSV_PATH = "data/logs_with_anomalies.csv" 

        self.index = faiss.read_index(self.INDEX_PATH)
        self.logs = joblib.load(self.LOGS_PATH)
        self.embeddings = np.load(self.EMBEDDINGS_PATH)

        with codecs.open(self.CSV_PATH, 'r', encoding='utf-8-sig') as f:
            self.df_logs = pd.read_csv(f)

        self.df_logs.columns = (
            self.df_logs.columns
            .str.strip()
            .str.lower()
            .str.replace('\u200b', '')
        )
        self.df_logs.rename(columns=lambda x: x.replace("’", "'").replace("“", '"').replace("”", '"'), inplace=True)

        print(" Cleaned columns:", list(self.df_logs.columns))
        if "log" not in self.df_logs.columns:
            raise ValueError(" Column 'log' not found. Available columns: " + str(self.df_logs.columns.tolist()))

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chat_history = []

    def extract_datetime(self, log_line):
        cleaned = re.sub(r'\s+', ' ', log_line)  # normalize spacing
        match = re.search(r'\w{3} \d{1,2}, \d{4} \d{2}:\d{2}:\d{2}\.\d+', cleaned)
        if match:
            ts = match.group(0)
            parsed = dateparser.parse(ts)
            return parsed
        return None

    def search_logs(self, query, top_k=5, start_time_str=None, end_time_str=None, ip=None, anomaly_only=False):
        vec = self.embedder.encode([query])
        _, indices = self.index.search(vec.astype("float32"), top_k)
        matched_logs = [self.logs[i] for i in indices[0]]

        print("\n Matched logs BEFORE filtering:")
        for log in matched_logs:
            print("•", log[:150])

        results_df = self.df_logs[self.df_logs["log"].isin(matched_logs)].copy()

        # Anchor: first parseable timestamp
        anchor_dt = next((self.extract_datetime(log) for log in results_df["log"] if self.extract_datetime(log)), None)

        if start_time_str and end_time_str:
            if anchor_dt:
                try:
                    start_dt = dateparser.parse(start_time_str, settings={"RELATIVE_BASE": anchor_dt})
                    end_dt = dateparser.parse(end_time_str, settings={"RELATIVE_BASE": anchor_dt})

                    def within_time_range(log):
                        dt = self.extract_datetime(log)
                        return dt and start_dt <= dt <= end_dt

                    results_df["time_check"] = results_df["log"].apply(within_time_range)
                    results_df = results_df[results_df["time_check"]]
                    print(f" Time filtering: {len(results_df)} logs matched from {start_dt.time()} to {end_dt.time()}")
                except Exception as e:
                    print(f" Time filtering error: {e}")
            else:
                print(" Skipping time filtering — no valid datetime found.")

        if ip:
            results_df = results_df[results_df["log"].str.casefold().str.contains(ip.casefold())]

        if anomaly_only and "is_anomaly" in results_df.columns:
            results_df = results_df[results_df["is_anomaly"] == -1]

        return results_df["log"].tolist()

def build_prompt(self, user_query, matched_logs):
    # Build limited chat memory
    session_context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in self.chat_history[-3:]])
    
    # Truncate each log for context window
    log_context = "\n".join(f"- {log[:300]}" for log in matched_logs[:10])

    return f"""
You are a cybersecurity assistant helping a SOC analyst investigate system logs.

Below is the recent conversation context (if any) and a new user query. Respond with clear, concise insights. 
Do not repeat log content. Do not just rephrase the logs. Summarize patterns, causes, and recommendations.

Conversation so far:
{session_context or 'N/A'}

Relevant log entries:
{log_context}

Now answer the following query with helpful reasoning and triage advice:
Query: {user_query}
""".strip()


    def call_mistral(self, prompt):
        try:
            result = subprocess.run(
                ["ollama", "run", "mistral"],
                input=prompt.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180
            )
            return result.stdout.decode().strip()
        except Exception as e:
            return f" Error calling mistral: {e}"

    def chat_loop(self):
        print("\n AIDE-OSS RAG Chatbot (Mistral + FAISS + Anomaly Filtering)")
        print("Type 'exit' to quit.\n")
        while True:
            user_query = input("You: ")
            if user_query.lower().strip() in {"exit", "quit"}:
                break
            ip = input(" Filter by IP (press Enter to skip): ").strip() or None
            start_time_str = input(" Start time (e.g., '6:55 AM') or skip: ").strip()
            end_time_str = input(" End time (e.g., '7:05 AM') or skip: ").strip()
            anomaly_input = input(" Only show anomaly logs? (y/n): ").strip().lower()
            anomaly_only = anomaly_input == "y"

            matched_logs = self.search_logs(user_query, 10, start_time_str, end_time_str, ip, anomaly_only)

            if not matched_logs:
                print(" No matching logs found.\n")
                continue

            print("\n Logs used in this response:")
            for log in matched_logs:
                print("•", log[:200], "...\n")

            prompt = self.build_prompt(user_query, matched_logs)
            answer = self.call_mistral(prompt)
            self.chat_history.append((user_query, answer))
            print(f"\n Mistral:\n{answer}\n")


if __name__ == "__main__":
    bot = RAGChatbot()
    bot.chat_loop()
