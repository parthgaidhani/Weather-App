"""
Statistical Time-Series Forecasting Module
ARIMA, SARIMA, Holt-Winters, and Prophet for monsoon rainfall forecasting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mse), 4),
        "R2":   round(r2_score(y_true, y_pred), 4),
    }


def run_arima(train_series: pd.Series, test_series: pd.Series) -> dict:
    from pmdarima import auto_arima
    model = auto_arima(train_series, seasonal=False, stepwise=True,
                       suppress_warnings=True, error_action="ignore")
    preds = model.predict(n_periods=len(test_series))
    metrics = evaluate(test_series.values, preds)
    metrics["model"] = "ARIMA"
    return metrics, model, preds


def run_sarima(train_series: pd.Series, test_series: pd.Series,
               m: int = 12) -> dict:
    from pmdarima import auto_arima
    model = auto_arima(train_series, seasonal=True, m=m, stepwise=True,
                       suppress_warnings=True, error_action="ignore",
                       max_p=2, max_q=2, max_P=1, max_Q=1)
    preds = model.predict(n_periods=len(test_series))
    metrics = evaluate(test_series.values, preds)
    metrics["model"] = "SARIMA"
    return metrics, model, preds


def run_holtwinters(train_series: pd.Series, test_series: pd.Series) -> dict:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(
        train_series, trend="add", seasonal="add", seasonal_periods=12
    ).fit(optimized=True)
    preds = model.forecast(len(test_series))
    metrics = evaluate(test_series.values, preds.values)
    metrics["model"] = "HoltWinters"
    return metrics, model, preds.values


def run_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame,
                date_col: str = "date", target_col: str = "precipitation_sum") -> dict:
    from prophet import Prophet
    prophet_train = train_df[[date_col, target_col]].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="multiplicative")
    model.fit(prophet_train)

    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = model.predict(future)
    preds = forecast.tail(len(test_df))["yhat"].values
    preds = np.clip(preds, 0, None)

    metrics = evaluate(test_df[target_col].values, preds)
    metrics["model"] = "Prophet"
    return metrics, model, preds


def run_all_statistical(city_df: pd.DataFrame,
                        target_col: str = "precipitation_sum",
                        test_year: int = 2024,
                        save_dir: str = "models") -> pd.DataFrame:
    """Run all statistical models for a single city's time series."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    city_df = city_df.sort_values("date").reset_index(drop=True)
    train = city_df[city_df["date"].dt.year < test_year]
    test  = city_df[city_df["date"].dt.year == test_year]

    # Monthly aggregation for ARIMA/SARIMA/HW (daily is too noisy)
    train_monthly = train.set_index("date")[target_col].resample("ME").sum()
    test_monthly  = test.set_index("date")[target_col].resample("ME").sum()

    results = []
    print("  Running ARIMA...")
    m, _, _ = run_arima(train_monthly, test_monthly)
    results.append(m)

    print("  Running SARIMA...")
    m, _, _ = run_sarima(train_monthly, test_monthly)
    results.append(m)

    print("  Running Holt-Winters...")
    m, _, _ = run_holtwinters(train_monthly, test_monthly)
    results.append(m)

    print("  Running Prophet...")
    m, _, _ = run_prophet(train, test, target_col=target_col)
    results.append(m)

    df_results = pd.DataFrame(results).set_index("model").sort_values("RMSE")
    df_results.to_csv(f"{save_dir}/statistical_comparison.csv")
    return df_results


if __name__ == "__main__":
    from src.data.preprocessing import run_pipeline

    df = run_pipeline()
    city_df = df[df["city"] == "Mumbai"].copy()
    results = run_all_statistical(city_df)
    print(results)
