# scripts/log_parser.py

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
import pandas as pd
from tqdm import tqdm

LOG_FILE = "data/sample_logs.txt"
STRUCTURED_OUTPUT = "data/normalized_logs.csv"

def main():
    persistence = FilePersistence("drain3_state.json")
    miner = TemplateMiner(persistence)

    logs = []
    with open(LOG_FILE, "r") as f:
        for line in tqdm(f, desc="Parsing logs"):
            result = miner.add_log_message(line.strip())
            if result["change_type"] != "none":
                logs.append({
                    "log": line.strip(),
                    "cluster_id": result["cluster_id"],
                    "template": result["template_mined"]
                })

    df = pd.DataFrame(logs)
    df.to_csv(STRUCTURED_OUTPUT, index=False)
    print(f"\n Saved structured logs to: {STRUCTURED_OUTPUT}")

if __name__ == "__main__":
    main()
