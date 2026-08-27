"""
Main Training Pipeline
Orchestrates: data acquisition → preprocessing → ML training → DL training → explainability
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.data.acquisition import fetch_all_cities
from src.data.preprocessing import run_pipeline, split_train_test, scale_features
from src.models.ml_models import train_all
from src.models.dl_models import train_dl_models
from src.models.timeseries_models import run_all_statistical
from src.models.explainability import explain_best_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

FEATURE_COLS = [
    "month_sin", "month_cos", "doy_sin", "doy_cos", "is_monsoon",
    "temp_range", "precip_roll_7d", "precip_roll_30d",
    "temp_roll_7d", "temp_roll_30d",
    "precip_lag_1", "precip_lag_3", "precip_lag_7",
    "temp_lag_1", "temp_lag_3", "temp_lag_7",
    "wind_speed_10m_max", "et0_fao_evapotranspiration",
    "shortwave_radiation_sum"
]
TARGET = "precipitation_sum"


def run(steps: list):
    print("=" * 60)
    print("  Weather Intelligence Platform — Training Pipeline")
    print("=" * 60)

    if "data" in steps:
        print("\n[1/5] Fetching historical weather data...")
        fetch_all_cities(start="2015-01-01", end="2024-12-31")

    needs_df = any(s in steps for s in ["ml", "dl", "stats", "explain"])

    if "preprocess" in steps:
        print("\n[2/5] Preprocessing & feature engineering...")
        df = run_pipeline()
    elif needs_df:
        df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])

    if "ml" in steps:
        print("\n[3/5] Training ML models...")
        train, test = split_train_test(df)
        X_train, X_test, y_train, y_test, _, _ = scale_features(
            train, test, FEATURE_COLS, TARGET
        )
        results = train_all(X_train, y_train, X_test, y_test)
        print(results)

    if "dl" in steps:
        print("\n[4/5] Training Deep Learning models (Mumbai)...")
        city_df = df[df["city"] == "Mumbai"].sort_values("date")
        scaler = MinMaxScaler()
        series = scaler.fit_transform(city_df[[TARGET]]).ravel()
        results = train_dl_models(series, epochs=50)
        print(results)

    if "stats" in steps:
        print("\n[4b/5] Training Statistical models (Mumbai)...")
        city_df = df[df["city"] == "Mumbai"].copy()
        results = run_all_statistical(city_df)
        print(results)

    if "explain" in steps:
        print("\n[5/5] Generating SHAP explanations...")
        train, test = split_train_test(df)
        X_train, X_test, _, _, _, _ = scale_features(train, test, FEATURE_COLS, TARGET)
        importance = explain_best_model(X_train=X_train, X_test=X_test,
                                        feature_names=FEATURE_COLS)
        print(importance.head(10))

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Intelligence Training Pipeline")
    parser.add_argument(
        "--steps", nargs="+",
        default=["data", "preprocess", "ml", "stats", "explain"],
        choices=["data", "preprocess", "ml", "dl", "stats", "explain"],
        help="Pipeline steps to run"
    )
    args = parser.parse_args()
    run(args.steps)
