# scripts/log_parser.py

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import pandas as pd
from tqdm import tqdm
import re

LOG_FILE = "data/zeek/sample_logs_small.txt"
STRUCTURED_OUTPUT = "data/normalized_logs.csv"

# Basic timestamp extraction (assumes ISO format like '2025-06-24T13:22:45' or syslog-like 'Jun 24 13:22:45')
def extract_timestamp(line: str) -> str:
    iso_match = re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', line)
    if iso_match:
        return iso_match.group(0)
    
    syslog_match = re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', line)
    if syslog_match:
        return syslog_match.group(0)

    return "N/A"

def main():
    persistence = FilePersistence("drain3_state.json")
    miner = TemplateMiner(persistence)

    logs = []
    with open(LOG_FILE, "r") as f:
        for line in tqdm(f, desc="Parsing logs"):
            line = line.strip()
            result = miner.add_log_message(line)
            logs.append({
                "log": line,
                "timestamp": extract_timestamp(line),
                "cluster_id": result.get("cluster_id", "N/A"),
                "template": result.get("template_mined", "N/A")
            })

    df = pd.DataFrame(logs)
    df.to_csv(STRUCTURED_OUTPUT, index=False)
    print(f"\n Saved structured logs to: {STRUCTURED_OUTPUT}")

if __name__ == "__main__":
    main()
