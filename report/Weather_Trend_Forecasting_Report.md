# Weather Trend Forecasting Report

## 1. Introduction

This project analyzes the Global Weather Repository dataset to forecast future weather trends and demonstrate a complete data science workflow. The dataset contains daily weather observations for cities around the world and includes meteorological, air-quality, astronomical, and geographical features.

## 2. PM Accelerator Mission

PM Accelerator mission statement included as required by the assessment:

> We are committed to offering free Product Management education. Our mission is to break down financial barriers and achieve educational fairness.

## 3. Dataset Overview

The pipeline reads `GlobalWeatherRepository.csv`, parses the `last_updated` timestamp, and creates daily time features. The global daily average weather series is used for the main forecasting task.

Main feature groups:

- Location: country, location name, latitude, longitude, timezone
- Weather: temperature, wind, pressure, precipitation, humidity, cloud, visibility, UV index
- Air quality: CO, Ozone, NO2, SO2, PM2.5, PM10, EPA/DEFRA index
- Astronomy: sunrise, sunset, moon phase, moon illumination

## 4. Data Cleaning and Preprocessing

The preprocessing module performs:

- Datetime parsing of `last_updated`
- Date feature extraction: year, month, day of year, day of week
- Median imputation for numerical features
- Mode imputation for categorical features
- IQR-based outlier capping for major weather variables
- Standard scaling for selected numerical features

Outliers are capped rather than removed to avoid discarding valid extreme weather events.

## 5. Exploratory Data Analysis

The EDA module generates:

- Daily average temperature trend
- Daily average precipitation trend
- Weather-feature correlation matrix
- Top 20 countries by average temperature
- Missing-value summary

These charts help identify global temporal trends, variable relationships, and geographical differences.

## 6. Forecasting Models

The forecasting task predicts global daily average temperature. The pipeline uses the `last_updated` feature to build a chronological daily series.

Supervised time-series features:

- Time index
- Month
- Day of year
- Lag 1 to lag 7 temperatures
- 7-day rolling mean
- 7-day rolling standard deviation

Models compared:

1. Naive last-day baseline
2. Linear Regression
3. Random Forest
4. Gradient Boosting
5. Average ensemble of non-naive models

Evaluation metrics:

- MAE
- RMSE
- MAPE
- R²

Results are saved to `outputs/tables/forecast_model_metrics.csv`.

## 7. Advanced Analysis

### 7.1 Anomaly Detection

Isolation Forest is used to detect unusual observations across temperature, wind, pressure, precipitation, humidity, cloud, visibility, UV index, and air-quality features.

Output:

- `outputs/tables/anomaly_sample.csv`
- `outputs/figures/temperature_anomalies.png`

### 7.2 Climate Analysis

The climate module aggregates monthly country-level temperature and precipitation patterns.

Output:

- `outputs/tables/climate_country_monthly.csv`
- `outputs/figures/global_monthly_temperature.png`

### 7.3 Environmental Impact

The project analyzes correlations between air-quality indicators and weather features such as temperature, humidity, wind, pressure, precipitation, and UV index.

Output:

- `outputs/tables/air_quality_weather_correlations.csv`
- `outputs/figures/pm25_vs_temperature.png`

### 7.4 Feature Importance

A Random Forest model estimates which variables are most important for predicting temperature. This provides interpretability and helps identify the strongest drivers of the forecasting target.

Output:

- `outputs/tables/feature_importance_random_forest.csv`
- `outputs/figures/feature_importance_temperature.png`

### 7.5 Spatial Analysis

The spatial module visualizes average temperature patterns using latitude and longitude. It summarizes geographical patterns by latitude and longitude, with average temperature and related weather variables by location.

Output:

- `outputs/tables/spatial_location_summary.csv`
- `outputs/figures/spatial_temperature_scatter.png`

## 8. Conclusions

This project provides a complete assessment-ready workflow from raw weather data to cleaned datasets, visualizations, model comparison, ensemble forecasting, anomaly detection, feature importance, environmental analysis, and spatial insights.

The code is modular and can be extended with city-specific models, Prophet/SARIMA, SHAP, or a Streamlit dashboard.
