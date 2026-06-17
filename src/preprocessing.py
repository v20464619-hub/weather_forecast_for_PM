from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import DATE_COL


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Required time column '{DATE_COL}' not found.")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()
    df["date"] = df[DATE_COL].dt.date
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["dayofyear"] = df[DATE_COL].dt.dayofyear
    df["dayofweek"] = df[DATE_COL].dt.dayofweek
    return df


def clean_data(df: pd.DataFrame, cap_outliers: bool = True) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number, "datetime64[ns]"]).columns.tolist()

    if num_cols:
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = imputer.fit_transform(df[num_cols])

    for col in cat_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    if cap_outliers:
        # Winsorize major physical weather variables with IQR fences instead of deleting rows.
        weather_cols = [
            "temperature_celsius", "feels_like_celsius", "wind_kph", "gust_kph",
            "pressure_mb", "precip_mm", "humidity", "cloud", "visibility_km", "uv_index",
            "air_quality_PM2.5", "air_quality_PM10", "air_quality_Ozone",
            "air_quality_Carbon_Monoxide", "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide"
        ]
        for col in [c for c in weather_cols if c in df.columns]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)

    return df


def add_scaled_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    df = df.copy()
    scale_cols = [
        "temperature_celsius", "feels_like_celsius", "wind_kph", "pressure_mb",
        "precip_mm", "humidity", "cloud", "visibility_km", "uv_index",
        "air_quality_PM2.5", "air_quality_PM10"
    ]
    scale_cols = [c for c in scale_cols if c in df.columns]
    scaler = StandardScaler()
    if scale_cols:
        scaled = scaler.fit_transform(df[scale_cols])
        for i, col in enumerate(scale_cols):
            df[f"scaled_{col}"] = scaled[:, i]
    return df, scaler, scale_cols


def make_daily_series(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in numeric_cols if not c.startswith("last_updated_epoch")]
    groupers = ["date"] + ([group_col] if group_col and group_col in df.columns else [])
    daily = df.groupby(groupers, as_index=False)[agg_cols].mean()
    daily = daily.sort_values(groupers).reset_index(drop=True)
    return daily
