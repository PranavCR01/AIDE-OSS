# log_pipeline.py

import pandas as pd
import os
from tqdm import tqdm

from utils.drain3_parser import parse_with_drain3
from utils.anomaly_detector import detect_anomalies
from utils.ollama_summarizer import summarize_log
from configs.config import RAW_LOG_FILE, ANOMALY_CSV_FILE, SUMMARY_CSV_FILE, MODEL_PATH, ENCODER_PATH

def run_log_pipeline():
    print("\n  Running Log Parsing + Anomaly Detection + Summarization Pipeline")

    if not os.path.exists(RAW_LOG_FILE):
        print(f"  Raw log file not found: {RAW_LOG_FILE}")
        return

    # Step 1: Parse raw logs with Drain3
    print("\n  Parsing logs with Drain3...")
    parsed_logs = parse_with_drain3(RAW_LOG_FILE)
    df = pd.DataFrame(parsed_logs)

    if df.empty:
        print("  No logs were parsed.")
        return

    # Step 2: Run anomaly detection
    print("\n  Running anomaly detection...")
    df = detect_anomalies(df, ANOMALY_CSV_FILE, MODEL_PATH, ENCODER_PATH)

    # Step 3: Summarize unique templates only
    print("\n   Generating summaries for unique templates...")
    template_to_summary = {}
    unique_templates = df["template"].dropna().unique()

    for template in tqdm(unique_templates, desc=" Summarizing templates"):
        summary = summarize_log(template)
        template_to_summary[template] = summary

    df["summary"] = df["template"].map(template_to_summary)

    # Step 4: Save final output
    print("\n  Saving summarized logs to CSV...")
    df.to_csv(SUMMARY_CSV_FILE, index=False)
    print(f"  Final output saved: {SUMMARY_CSV_FILE}")

if __name__ == "__main__":
    run_log_pipeline()
