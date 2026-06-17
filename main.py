from __future__ import annotations

import argparse
import json

from src.advanced_analysis import air_quality_analysis, climate_analysis, feature_importance, spatial_analysis
from src.anomaly import detect_anomalies
from src.config import CLEAN_DATA_PATH, DAILY_DATA_PATH, OUTPUT_DIR, RAW_DATA_PATH, TARGET_COL
from src.eda import run_eda
from src.forecasting import evaluate_forecasting_models
from src.preprocessing import add_scaled_features, clean_data, load_data, make_daily_series
from src.utils import write_json


def main():
    parser = argparse.ArgumentParser(description="Weather trend forecasting assessment project")
    parser.add_argument("--data", default=str(RAW_DATA_PATH), help="Path to GlobalWeatherRepository.csv")
    parser.add_argument("--target", default=TARGET_COL, help="Forecast target column")
    parser.add_argument("--skip-advanced", action="store_true", help="Run only cleaning, EDA and forecasting")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.data)

    print("Cleaning and preprocessing...")
    clean = clean_data(df)
    clean, _, _ = add_scaled_features(clean)
    clean.to_csv(CLEAN_DATA_PATH, index=False)

    print("Creating daily time series using last_updated...")
    daily = make_daily_series(clean)
    daily.to_csv(DAILY_DATA_PATH, index=False)

    print("Running EDA...")
    summary = run_eda(clean, daily)

    print("Training and comparing forecasting models...")
    metrics, predictions = evaluate_forecasting_models(daily, target_col=args.target)

    advanced_outputs = {}
    if not args.skip_advanced:
        # Use a reproducible sample for computationally heavier advanced methods so the assessment runs quickly on normal laptops.
        advanced_sample = clean.sample(min(len(clean), 30000), random_state=42)

        print("Running anomaly detection...")
        anomalized = detect_anomalies(advanced_sample)
        advanced_outputs["anomaly_rate"] = float(anomalized["anomaly"].mean()) if "anomaly" in anomalized.columns else None

        print("Running climate analysis...")
        climate = climate_analysis(clean)
        advanced_outputs["climate_rows"] = int(len(climate))

        print("Running air quality analysis...")
        aq = air_quality_analysis(advanced_sample)
        advanced_outputs["air_quality_analysis_shape"] = list(aq.shape)

        print("Running feature importance...")
        imp = feature_importance(advanced_sample, target=args.target)
        advanced_outputs["top_features"] = imp.head(10).to_dict(orient="records") if len(imp) else []

        print("Running spatial analysis...")
        spatial = spatial_analysis(advanced_sample)
        advanced_outputs["spatial_locations"] = int(len(spatial))

    project_summary = {
        "dataset_summary": summary,
        "best_model_by_rmse": metrics.iloc[0].to_dict(),
        "advanced_outputs": advanced_outputs,
        "outputs_folder": str(OUTPUT_DIR),
    }
    write_json(project_summary, OUTPUT_DIR / "project_summary.json")

    print("Done. Key result:")
    print(json.dumps(project_summary["best_model_by_rmse"], indent=2))


if __name__ == "__main__":
    main()
