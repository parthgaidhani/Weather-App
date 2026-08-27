"""
Preprocessing & Feature Engineering Module
Cleans raw weather data and engineers monsoon-relevant features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


# Monsoon season: June–September (months 6–9)
MONSOON_MONTHS = [6, 7, 8, 9]


def load_raw(path: str = "data/raw/all_cities_daily.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and outliers."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Forward-fill short gaps (≤3 days), then interpolate
    df[numeric_cols] = (
        df.sort_values("date")
        .groupby("city")[numeric_cols]
        .transform(lambda x: x.ffill(limit=3).interpolate(method="linear"))
    )

    # Cap extreme precipitation outliers at 99.9th percentile per city
    for city, grp in df.groupby("city"):
        cap = grp["precipitation_sum"].quantile(0.999)
        df.loc[df["city"] == city, "precipitation_sum"] = df.loc[
            df["city"] == city, "precipitation_sum"
        ].clip(upper=cap)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal, monsoon, and lag features."""
    df = df.copy()
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Temporal features
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)

    # Cyclical encoding for month and day_of_year
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]     = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]     = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Monsoon flag
    df["is_monsoon"]  = df["month"].isin(MONSOON_MONTHS).astype(int)

    # Temperature range
    df["temp_range"]  = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Rolling statistics (7-day and 30-day) per city
    for city, grp in df.groupby("city"):
        idx = grp.index
        for window, suffix in [(7, "7d"), (30, "30d")]:
            df.loc[idx, f"precip_roll_{suffix}"] = (
                grp["precipitation_sum"].rolling(window, min_periods=1).mean().values
            )
            df.loc[idx, f"temp_roll_{suffix}"] = (
                grp["temperature_2m_mean"].rolling(window, min_periods=1).mean().values
            )

    # Lag features (1, 3, 7 days)
    for lag in [1, 3, 7]:
        df[f"precip_lag_{lag}"] = df.groupby("city")["precipitation_sum"].shift(lag)
        df[f"temp_lag_{lag}"]   = df.groupby("city")["temperature_2m_mean"].shift(lag)

    # Cumulative monsoon rainfall (resets each year per city)
    df["monsoon_cumrain"] = df.groupby(["city", "year"]).apply(
        lambda g: g["precipitation_sum"].where(g["is_monsoon"] == 1, 0).cumsum()
    ).reset_index(level=[0, 1], drop=True)

    df = df.dropna(subset=["precip_lag_7", "temp_lag_7"]).reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame, test_year: int = 2024):
    """Temporal split: train on years before test_year, test on test_year."""
    train = df[df["year"] < test_year].copy()
    test  = df[df["year"] == test_year].copy()
    return train, test


def scale_features(train: pd.DataFrame, test: pd.DataFrame,
                   feature_cols: list, target_col: str,
                   save_dir: str = "models"):
    """Scale features and target; save scalers."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    feat_scaler   = StandardScaler()
    target_scaler = MinMaxScaler()

    X_train = feat_scaler.fit_transform(train[feature_cols])
    X_test  = feat_scaler.transform(test[feature_cols])
    y_train = target_scaler.fit_transform(train[[target_col]])
    y_test  = target_scaler.transform(test[[target_col]])

    joblib.dump(feat_scaler,   f"{save_dir}/feature_scaler.pkl")
    joblib.dump(target_scaler, f"{save_dir}/target_scaler.pkl")

    return X_train, X_test, y_train.ravel(), y_test.ravel(), feat_scaler, target_scaler


def run_pipeline(raw_path: str = "data/raw/all_cities_daily.csv",
                 save_path: str = "data/processed/features.csv") -> pd.DataFrame:
    print("Loading raw data...")
    df = load_raw(raw_path)
    print(f"  Raw shape: {df.shape}")

    print("Cleaning...")
    df = clean(df)

    print("Engineering features...")
    df = engineer_features(df)
    print(f"  Processed shape: {df.shape}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  Saved to {save_path}")
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(df.describe())
