# Weather Trend Forecasting — Data Scientist / Analyst Tech Assessment

This repository contains a complete, reproducible data science project for the **Global Weather Repository** dataset. It covers data cleaning, exploratory data analysis, weather trend forecasting, anomaly detection, environmental impact analysis, feature importance, and spatial/geographical analysis.

## PM Accelerator Mission

PM Accelerator mission statement displayed for the assessment requirement:

> We are committed to offering free Product Management education. Our mission is to break down financial barriers and achieve educational fairness.

Source: PM Accelerator official website.

## Project Objectives

- Clean and preprocess global weather data.
- Use the `last_updated` feature for time-series analysis.
- Explore temperature, precipitation, air quality, and geographical patterns.
- Build and compare multiple forecasting models.
- Create an ensemble model to improve forecast robustness.
- Detect anomalous weather observations.
- Analyze feature importance and environmental relationships.
- Generate charts, tables, and reproducible outputs for report/presentation use.

## Repository Structure

```text
weather_forecast_project/
├── data/
│   └── GlobalWeatherRepository.csv
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── forecasting.py
│   ├── anomaly.py
│   ├── advanced_analysis.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── clean_weather.csv
│   ├── daily_weather.csv
│   └── project_summary.json
├── report/
│   └── Weather_Trend_Forecasting_Report.md
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py --data data/GlobalWeatherRepository.csv
```

For a faster basic run:

```bash
python main.py --data data/GlobalWeatherRepository.csv --skip-advanced
```

## Methods Used

### Data Cleaning

- Converted `last_updated` to datetime.
- Extracted `date`, `year`, `month`, `dayofyear`, and `dayofweek`.
- Imputed numerical missing values with the median.
- Imputed categorical missing values with the mode.
- Applied IQR-based winsorization to cap extreme weather outliers.
- Added standardized numerical features.

### EDA

Generated:

- Daily global temperature trend.
- Daily global precipitation trend.
- Correlation matrix for key weather and air-quality variables.
- Top 20 countries by average temperature.
- Missing-value summary.

### Forecasting

The project creates a global daily average temperature series based on `last_updated`, then builds lag-based supervised features:

- Time index
- Day of year
- Month
- 1–7 day lags
- 7-day rolling mean
- 7-day rolling standard deviation

Models compared:

- Naive last-day baseline
- Linear Regression
- Random Forest
- Gradient Boosting
- Average ensemble of non-naive models

Metrics:

- MAE
- RMSE
- MAPE
- R²

### Advanced Analyses

- **Anomaly Detection:** Isolation Forest to flag unusual weather and air-quality observations.
- **Climate Analysis:** Monthly country-level climate summaries.
- **Environmental Impact:** Correlation between air-quality indicators and weather variables.
- **Feature Importance:** Random Forest feature importance for temperature prediction.
- **Spatial Analysis:** Latitude-longitude visualization and KMeans weather clustering.

## Key Output Files

After running `main.py`, check:

```text
outputs/project_summary.json
outputs/tables/forecast_model_metrics.csv
outputs/tables/forecast_predictions.csv
outputs/tables/feature_importance_random_forest.csv
outputs/tables/air_quality_weather_correlations.csv
outputs/figures/forecast_actual_vs_best.png
outputs/figures/correlation_matrix.png
outputs/figures/temperature_anomalies.png
outputs/figures/spatial_temperature_scatter.png
```

## Suggested Demo Video Script

1. Briefly introduce the dataset and objective.
2. Show the GitHub repository structure.
3. Run `python main.py --data data/GlobalWeatherRepository.csv`.
4. Open `outputs/project_summary.json` and `forecast_model_metrics.csv`.
5. Explain the best model and ensemble result.
6. Show generated figures: trend, correlation, anomaly detection, feature importance, and spatial plot.
7. Summarize insights and next steps.

## Possible Improvements

- Build city-specific forecasting models.
- Add Prophet or SARIMA for stronger time-series baselines.
- Add SHAP explanations for model interpretability.
- Build a Streamlit dashboard for interactive exploration.
- Use continent mapping to strengthen geographical comparison.
