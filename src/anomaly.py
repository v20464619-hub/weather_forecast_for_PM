from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .config import FIGURE_DIR, RANDOM_STATE, TABLE_DIR
from .utils import save_fig, save_table


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02, max_rows: int = 30000) -> pd.DataFrame:
    features = [
        "temperature_celsius", "feels_like_celsius", "wind_kph", "pressure_mb", "precip_mm",
        "humidity", "cloud", "visibility_km", "uv_index", "air_quality_PM2.5", "air_quality_PM10"
    ]
    features = [c for c in features if c in df.columns]
    if len(features) < 2:
        return df.assign(anomaly=0, anomaly_score=0.0)

    work = df.sample(min(len(df), max_rows), random_state=RANDOM_STATE).copy()
    X = work[features].fillna(work[features].median(numeric_only=True))
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_estimators=50)
    labels = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    work["anomaly"] = (labels == -1).astype(int)
    work["anomaly_score"] = scores

    anomaly_sample = work.loc[work["anomaly"] == 1, ["date", "country", "location_name"] + features + ["anomaly_score"]].head(500)
    save_table(anomaly_sample, TABLE_DIR / "anomaly_sample.csv")

    if "temperature_celsius" in work.columns:
        plt.figure(figsize=(10, 5))
        plt.scatter(work["date"], work["temperature_celsius"], s=4, alpha=0.35)
        anomaly_pts = work[work["anomaly"] == 1]
        if len(anomaly_pts) > 0:
            plt.scatter(anomaly_pts["date"], anomaly_pts["temperature_celsius"], s=8, marker="x")
        plt.title("Temperature Anomaly Detection with Isolation Forest")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        save_fig(FIGURE_DIR / "temperature_anomalies.png")

    return work
