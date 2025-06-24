# scripts/insight_agent.py

from transformers import pipeline
import pandas as pd

# Load logs
df = pd.read_csv("data/logs_with_anomaly_score.csv")
anomalous_logs = df[df["is_anomaly"] == 1]["log"].dropna().tolist()

# Step 1: Deduplicate & prepare a joined context
unique_logs = sorted(set(anomalous_logs))
context = "\n".join(unique_logs)[:800]  # fit within safe input limit

# Step 2: Prompt FLAN-T5 to summarize patterns in logs
prompt = f"""
You are a cybersecurity analyst. Analyze the following server logs and summarize any suspicious behaviors, such as:
- repeated access to admin pages,
- use of curl or automated agents for login,
- any signs of brute force, anomalies, or unauthorized access.

Logs:
{context}
"""

# Step 3: Load FLAN-T5
summarizer = pipeline("text2text-generation", model="google/flan-t5-base")

# Step 4: Get high-level summary
result = summarizer(prompt, max_new_tokens=200, do_sample=False)
summary = result[0]["generated_text"]

# Step 5: Save back into DataFrame (same summary for all anomalies for now)
df["summary"] = ""
df.loc[df["is_anomaly"] == 1, "summary"] = summary

# Save
df.to_csv("data/logs_with_summaries.csv", index=False)
print("Contextual summary saved for all anomalies.")
