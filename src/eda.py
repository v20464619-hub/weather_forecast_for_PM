from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURE_DIR, TABLE_DIR
from .utils import save_fig, save_table


def run_eda(df: pd.DataFrame, daily: pd.DataFrame) -> dict:
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "countries": int(df["country"].nunique()) if "country" in df.columns else None,
        "locations": int(df["location_name"].nunique()) if "location_name" in df.columns else None,
    }

    missing = df.isna().mean().sort_values(ascending=False).reset_index()
    missing.columns = ["column", "missing_ratio"]
    save_table(missing, TABLE_DIR / "missing_values.csv")

    # Daily global temperature trend
    if "temperature_celsius" in daily.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(daily["date"], daily["temperature_celsius"], linewidth=1.5)
        plt.title("Global Average Daily Temperature")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        save_fig(FIGURE_DIR / "daily_temperature_trend.png")

    # Precipitation trend
    if "precip_mm" in daily.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(daily["date"], daily["precip_mm"], linewidth=1.2)
        plt.title("Global Average Daily Precipitation")
        plt.xlabel("Date")
        plt.ylabel("Precipitation (mm)")
        save_fig(FIGURE_DIR / "daily_precipitation_trend.png")

    # Correlation matrix for major weather variables
    corr_cols = [
        "temperature_celsius", "feels_like_celsius", "wind_kph", "pressure_mb", "precip_mm",
        "humidity", "cloud", "visibility_km", "uv_index", "air_quality_PM2.5", "air_quality_PM10"
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        save_table(corr.reset_index().rename(columns={"index": "feature"}), TABLE_DIR / "correlation_matrix.csv")
        plt.figure(figsize=(9, 7))
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=60, ha="right")
        plt.yticks(range(len(corr_cols)), corr_cols)
        plt.colorbar(label="Pearson correlation")
        plt.title("Weather Feature Correlation Matrix")
        save_fig(FIGURE_DIR / "correlation_matrix.png")

    # Top country comparisons
    if "country" in df.columns and "temperature_celsius" in df.columns:
        top_temp = (
            df.groupby("country", as_index=False)["temperature_celsius"]
            .mean().sort_values("temperature_celsius", ascending=False).head(20)
        )
        save_table(top_temp, TABLE_DIR / "top20_countries_temperature.csv")
        plt.figure(figsize=(10, 6))
        plt.barh(top_temp["country"], top_temp["temperature_celsius"])
        plt.gca().invert_yaxis()
        plt.title("Top 20 Countries by Average Temperature")
        plt.xlabel("Temperature (°C)")
        save_fig(FIGURE_DIR / "top20_countries_temperature.png")

    return summary
