"""
Explainable AI Module
SHAP-based feature importance and explanation for trained ML models.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


def explain_model(model, X_train: np.ndarray, X_test: np.ndarray,
                  feature_names: list, model_name: str,
                  save_dir: str = "reports") -> pd.DataFrame:
    """Generate SHAP explanations and save plots."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Use TreeExplainer for tree-based models, KernelExplainer otherwise
    tree_models = {"RandomForest", "ExtraTrees", "GradientBoosting",
                   "XGBoost", "LightGBM", "CatBoost", "DecisionTree", "AdaBoost"}

    if model_name in tree_models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_test[:200])

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:len(shap_values)],
                      feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary — {model_name}")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary_{model_name}.png", dpi=150)
    plt.close()

    # Feature importance from SHAP
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(f"{save_dir}/shap_importance_{model_name}.csv", index=False)

    return importance


def explain_best_model(save_dir_models: str = "models",
                       save_dir_reports: str = "reports",
                       X_train=None, X_test=None,
                       feature_names: list = None):
    results = pd.read_csv(f"{save_dir_models}/ml_comparison.csv", index_col="model")
    best_name = results.sort_values("RMSE").index[0]
    model = joblib.load(f"{save_dir_models}/{best_name}.pkl")
    print(f"Explaining: {best_name}")
    return explain_model(model, X_train, X_test, feature_names, best_name, save_dir_reports)


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

    df = run_pipeline()
    train, test = split_train_test(df)
    X_train, X_test, _, _, _, _ = scale_features(train, test, FEATURE_COLS, "precipitation_sum")

    importance = explain_best_model(X_train=X_train, X_test=X_test, feature_names=FEATURE_COLS)
    print(importance.head(10))
