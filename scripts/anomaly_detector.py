# scripts/anomaly_detector.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import joblib

INPUT_FILE = "data/normalized_logs.csv"
OUTPUT_FILE = "data/logs_with_anomaly_score.csv"
MODEL_PATH = "models/isolation_forest.pkl"

def preprocess(df):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(df[["template"]])
    return X, enc

def main():
    df = pd.read_csv(INPUT_FILE)
    X, encoder = preprocess(df)

    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)

    df["anomaly_score"] = clf.decision_function(X)
    df["is_anomaly"] = clf.predict(X)  # -1 = anomaly, 1 = normal

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Anomaly scores saved to {OUTPUT_FILE}")

    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
