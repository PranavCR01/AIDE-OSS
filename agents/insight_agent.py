# agents/insight_agent.py

import pandas as pd
import subprocess

CSV_PATH = "data/logs_with_anomaly_score.csv"

def get_anomalies():
    df = pd.read_csv(CSV_PATH)
    anomalies = df[df["is_anomaly"] == -1]["log"].tolist()
    return anomalies

def generate_prompt(anomalies):
    joined_logs = "\n".join(anomalies[:10])
    return f"""
You are a cybersecurity analyst reviewing anomalous log lines detected by our ML-based threat detection engine.

Logs:
{joined_logs}

Instructions:
Analyze the logs and provide a structured summary with:
1. A brief description of the types of events or patterns detected.
2. A likely explanation of their causes (e.g., misconfiguration, reconnaissance, etc.).
3. Recommended triage steps.

Please respond with **only one clearly numbered list** without repeating steps or sections.
""".strip()


def query_mistral(prompt):
    result = subprocess.run(["ollama", "run", "mistral"], input=prompt.encode(), stdout=subprocess.PIPE)
    print("\n Mistral's Summary:\n")
    print(result.stdout.decode())

def run_summary(): 
    anomalies = get_anomalies()
    prompt = generate_prompt(anomalies)
    query_mistral(prompt)

# Optional: allow running directly
if __name__ == "__main__":
    run_summary()
