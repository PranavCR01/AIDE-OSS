import pandas as pd
import re

def parse_filters(query):
    filters = {}
    if match := re.search(r'ip:(\d{1,3}(?:\.\d{1,3}){3})', query):
        filters["ip"] = match.group(1)
    if match := re.search(r'after:(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})', query):
        filters["after"] = pd.to_datetime(match.group(1), errors="coerce")
    if "anomaly:true" in query.lower():
        filters["anomaly"] = 1
    return filters

def apply_filters(df: pd.DataFrame, filters: dict, timestamps: pd.Series) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if "ip" in filters:
        mask &= df["log"].str.contains(filters["ip"])
    if "after" in filters and timestamps.notna().any():
        mask &= timestamps >= filters["after"]
    if "anomaly" in filters:
        mask &= df["is_anomaly"] == filters["anomaly"]
    return df[mask]
