"""
Machine Learning Models Module
Trains, evaluates, and compares multiple ML models for rainfall/temperature prediction.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")


MODELS = {
    "LinearRegression":    LinearRegression(),
    "Ridge":               Ridge(alpha=1.0),
    "Lasso":               Lasso(alpha=0.01),
    "ElasticNet":          ElasticNet(alpha=0.01, l1_ratio=0.5),
    "DecisionTree":        DecisionTreeRegressor(max_depth=8, random_state=42),
    "RandomForest":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "ExtraTrees":          ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost":            AdaBoostRegressor(n_estimators=100, random_state=42),
    "XGBoost":             XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    "LightGBM":            LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    "CatBoost":            CatBoostRegressor(iterations=100, random_seed=42, verbose=0),
    "SVR":                 SVR(kernel="rbf", C=10, epsilon=0.1),
    "KNN":                 KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "MSE":  round(mse, 4),
        "RMSE": round(np.sqrt(mse), 4),
        "MAPE": round(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100, 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


def train_all(X_train, y_train, X_test, y_test,
              save_dir: str = "models") -> pd.DataFrame:
    """Train all models, evaluate, and save best model."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for name, model in MODELS.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate(y_test, preds)
        metrics["model"] = name
        results.append(metrics)
        joblib.dump(model, f"{save_dir}/{name}.pkl")

    df_results = pd.DataFrame(results).set_index("model").sort_values("RMSE")
    df_results.to_csv(f"{save_dir}/ml_comparison.csv")

    best_name = df_results.index[0]
    print(f"\nBest model: {best_name} (RMSE={df_results.loc[best_name, 'RMSE']})")
    return df_results


def load_best_model(save_dir: str = "models") -> tuple:
    """Load the best model based on saved comparison CSV."""
    results = pd.read_csv(f"{save_dir}/ml_comparison.csv", index_col="model")
    best_name = results.sort_values("RMSE").index[0]
    model = joblib.load(f"{save_dir}/{best_name}.pkl")
    return model, best_name


def predict_with_model(model_name: str, features: np.ndarray,
                       save_dir: str = "models") -> np.ndarray:
    model = joblib.load(f"{save_dir}/{model_name}.pkl")
    return model.predict(features)


if __name__ == "__main__":
    from src.data.preprocessing import run_pipeline, split_train_test, scale_features

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

    df = run_pipeline()
    train, test = split_train_test(df)
    X_train, X_test, y_train, y_test, _, _ = scale_features(train, test, FEATURE_COLS, TARGET)

    results = train_all(X_train, y_train, X_test, y_test)
    print(results)
