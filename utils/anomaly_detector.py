## anomaly_detector.py

import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

def preprocess(df):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X = enc.fit_transform(df[["template"]])
    return X, enc

def detect_anomalies(df: pd.DataFrame, output_csv: str, model_path: str, encoder_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    X, encoder = preprocess(df)

    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)

    df["anomaly_score"] = clf.decision_function(X)
    df["is_anomaly"] = clf.predict(X)

    df.to_csv(output_csv, index=False)
    joblib.dump(clf, model_path)
    joblib.dump(encoder, encoder_path)

    print(f"✅ Anomalies saved to: {output_csv}")
    print(f"💾 Model saved to: {model_path}")
    print(f"💾 Encoder saved to: {encoder_path}")
    print(f"🔎 Anomalies detected: {(df['is_anomaly'] == -1).sum()} of {len(df)} logs")

    return df
