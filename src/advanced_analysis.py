from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import FIGURE_DIR, RANDOM_STATE, TABLE_DIR
from .utils import save_fig, save_table


def climate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["country", "year", "month"]
    if not all(c in df.columns for c in group_cols + ["temperature_celsius"]):
        return pd.DataFrame()
    climate = (
        df.groupby(group_cols, as_index=False)
        .agg(avg_temperature_celsius=("temperature_celsius", "mean"),
             avg_precip_mm=("precip_mm", "mean") if "precip_mm" in df.columns else ("temperature_celsius", "size"),
             observations=("temperature_celsius", "size"))
    )
    save_table(climate, TABLE_DIR / "climate_country_monthly.csv")

    global_monthly = df.groupby(["year", "month"], as_index=False)["temperature_celsius"].mean()
    global_monthly["period"] = pd.to_datetime(global_monthly[["year", "month"]].assign(day=1))
    plt.figure(figsize=(10, 5))
    plt.plot(global_monthly["period"], global_monthly["temperature_celsius"])
    plt.title("Global Monthly Average Temperature")
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    save_fig(FIGURE_DIR / "global_monthly_temperature.png")
    return climate


def air_quality_analysis(df: pd.DataFrame) -> pd.DataFrame:
    aq_cols = [
        "air_quality_PM2.5", "air_quality_PM10", "air_quality_Ozone",
        "air_quality_Carbon_Monoxide", "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
        "air_quality_us-epa-index", "air_quality_gb-defra-index"
    ]
    weather_cols = ["temperature_celsius", "humidity", "wind_kph", "pressure_mb", "precip_mm", "uv_index"]
    cols = [c for c in aq_cols + weather_cols if c in df.columns]
    if len(cols) < 3:
        return pd.DataFrame()
    corr = df[cols].corr(numeric_only=True)
    save_table(corr.reset_index().rename(columns={"index": "feature"}), TABLE_DIR / "air_quality_weather_correlations.csv")

    if "air_quality_PM2.5" in df.columns and "temperature_celsius" in df.columns:
        sample = df.sample(min(len(df), 8000), random_state=RANDOM_STATE)
        plt.figure(figsize=(8, 5))
        plt.scatter(sample["temperature_celsius"], sample["air_quality_PM2.5"], s=5, alpha=0.35)
        plt.title("PM2.5 vs Temperature")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("PM2.5")
        save_fig(FIGURE_DIR / "pm25_vs_temperature.png")
    return corr


def feature_importance(df: pd.DataFrame, target: str = "temperature_celsius") -> pd.DataFrame:
    feature_cols = [
        "latitude", "longitude", "wind_kph", "pressure_mb", "precip_mm", "humidity", "cloud",
        "visibility_km", "uv_index", "gust_kph", "air_quality_PM2.5", "air_quality_PM10",
        "air_quality_Ozone", "air_quality_Carbon_Monoxide", "month", "dayofyear"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns and c != target]
    if target not in df.columns or len(feature_cols) < 2:
        return pd.DataFrame()
    data = df[[target] + feature_cols].dropna()
    if len(data) > 10000:
        data = data.sample(10000, random_state=RANDOM_STATE)
    X, y = data[feature_cols], data[target]
    model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=3)
    model.fit(X, y)
    imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    save_table(imp, TABLE_DIR / "feature_importance_random_forest.csv")

    plt.figure(figsize=(8, 6))
    top = imp.head(15).sort_values("importance")
    plt.barh(top["feature"], top["importance"])
    plt.title("Random Forest Feature Importance for Temperature")
    plt.xlabel("Importance")
    save_fig(FIGURE_DIR / "feature_importance_temperature.png")
    return imp


def spatial_analysis(df: pd.DataFrame) -> pd.DataFrame:
    required = ["latitude", "longitude", "temperature_celsius"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    loc_cols = ["country", "location_name", "latitude", "longitude"]
    metrics = ["temperature_celsius"] + [c for c in ["precip_mm", "humidity", "air_quality_PM2.5"] if c in df.columns]
    loc = df.groupby(loc_cols, as_index=False)[metrics].mean()
    save_table(loc, TABLE_DIR / "spatial_location_summary.csv")

    plt.figure(figsize=(10, 5))
    plt.scatter(loc["longitude"], loc["latitude"], c=loc["temperature_celsius"], s=12, alpha=0.7)
    plt.colorbar(label="Avg Temperature (°C)")
    plt.title("Global Spatial Temperature Pattern")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    save_fig(FIGURE_DIR / "spatial_temperature_scatter.png")

    # Optional KMeans clustering can be added here; the location summary already supports spatial/geographical analysis.
    return loc
