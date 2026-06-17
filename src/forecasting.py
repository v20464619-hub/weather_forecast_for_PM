from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import FIGURE_DIR, RANDOM_STATE, TABLE_DIR, TARGET_COL
from .utils import save_fig, save_table


def _rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def _mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100)


def prepare_supervised_series(daily: pd.DataFrame, target_col: str = TARGET_COL, lags: int = 7) -> pd.DataFrame:
    data = daily[["date", target_col]].dropna().sort_values("date").copy()
    data["time_index"] = np.arange(len(data))
    data["dayofyear"] = pd.to_datetime(data["date"]).dt.dayofyear
    data["month"] = pd.to_datetime(data["date"]).dt.month
    for lag in range(1, lags + 1):
        data[f"lag_{lag}"] = data[target_col].shift(lag)
    data["rolling_mean_7"] = data[target_col].shift(1).rolling(7).mean()
    data["rolling_std_7"] = data[target_col].shift(1).rolling(7).std()
    data = data.dropna().reset_index(drop=True)
    return data


def evaluate_forecasting_models(daily: pd.DataFrame, target_col: str = TARGET_COL, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = prepare_supervised_series(daily, target_col=target_col)
    if len(data) < 30:
        raise ValueError("Not enough daily observations for forecasting. Need at least 30 after lag construction.")

    feature_cols = [c for c in data.columns if c not in ["date", target_col]]
    split = int(len(data) * (1 - test_ratio))
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    models = {
        "Naive_Last_Day": None,
        "Linear_Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random_Forest": RandomForestRegressor(n_estimators=80, random_state=RANDOM_STATE, min_samples_leaf=2, n_jobs=-1),
        "Gradient_Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=100),
    }

    # XGBoost can be added here if the environment supports it; sklearn models keep the repo lightweight and reproducible.

    predictions = pd.DataFrame({"date": test["date"].values, "actual": y_test.values})
    metrics = []
    model_preds = []

    for name, model in models.items():
        if name == "Naive_Last_Day":
            pred = test["lag_1"].values
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        predictions[name] = pred
        model_preds.append(pred)
        metrics.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": _rmse(y_test, pred),
            "MAPE_percent": _mape(y_test, pred),
            "R2": r2_score(y_test, pred),
        })

    # Simple ensemble over non-naive predictive models.
    ensemble_cols = [c for c in predictions.columns if c not in ["date", "actual", "Naive_Last_Day"]]
    if ensemble_cols:
        predictions["Ensemble_Average"] = predictions[ensemble_cols].mean(axis=1)
        pred = predictions["Ensemble_Average"].values
        metrics.append({
            "model": "Ensemble_Average",
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": _rmse(y_test, pred),
            "MAPE_percent": _mape(y_test, pred),
            "R2": r2_score(y_test, pred),
        })

    metrics_df = pd.DataFrame(metrics).sort_values("RMSE")
    save_table(metrics_df, TABLE_DIR / "forecast_model_metrics.csv")
    save_table(predictions, TABLE_DIR / "forecast_predictions.csv")

    plt.figure(figsize=(11, 5))
    plt.plot(predictions["date"], predictions["actual"], label="Actual", linewidth=2)
    best_model = metrics_df.iloc[0]["model"]
    plt.plot(predictions["date"], predictions[best_model], label=f"Best: {best_model}", linewidth=1.5)
    plt.title(f"Temperature Forecast: Actual vs Best Model ({best_model})")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    save_fig(FIGURE_DIR / "forecast_actual_vs_best.png")

    return metrics_df, predictions
