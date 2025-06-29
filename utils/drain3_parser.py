import re
import pandas as pd
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence

# Timestamp extractor (unchanged)
def extract_timestamp(line: str) -> str:
    iso_match = re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', line)
    if iso_match:
        return iso_match.group(0)

    syslog_match = re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', line)
    if syslog_match:
        return syslog_match.group(0)

    return "N/A"

# ✅ This is what your pipeline expects
def parse_with_drain3(input_path: str) -> list:
    miner = TemplateMiner(FilePersistence("drain3_state.json"))
    logs = []

    with open(input_path, "r") as f:
        for line in tqdm(f, desc="Parsing logs"):
            line = line.strip()
            if not line:
                continue
            result = miner.add_log_message(line)
            logs.append({
                "log": line,
                "timestamp": extract_timestamp(line),
                "cluster_id": result.get("cluster_id", "N/A"),
                "template": result.get("template_mined", "N/A")
            })

    return logs

# Optional: Keep the original parse_logs() if you still want CSV functionality
def parse_logs(input_path: str, output_path: str):
    logs = parse_with_drain3(input_path)
    df = pd.DataFrame(logs)
    df.to_csv(output_path, index=False)
    print(f" Structured logs saved to: {output_path}")
