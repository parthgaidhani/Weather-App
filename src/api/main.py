"""
FastAPI Backend
Serves weather data, ML predictions, DL forecasts, and current conditions.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.acquisition import (
    fetch_current_weather, fetch_forecast, INDIAN_CITIES
)

app = FastAPI(
    title="Weather Intelligence API",
    description="AI-powered Southwest Monsoon prediction platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-load models ──────────────────────────────────────────────────────────
_ml_model = None
_feat_scaler = None
_target_scaler = None

FEATURE_COLS = [
    "month_sin", "month_cos", "doy_sin", "doy_cos", "is_monsoon",
    "temp_range", "precip_roll_7d", "precip_roll_30d",
    "temp_roll_7d", "temp_roll_30d",
    "precip_lag_1", "precip_lag_3", "precip_lag_7",
    "temp_lag_1", "temp_lag_3", "temp_lag_7",
    "wind_speed_10m_max", "et0_fao_evapotranspiration",
    "shortwave_radiation_sum"
]


def _load_models():
    global _ml_model, _feat_scaler, _target_scaler
    if _ml_model is None:
        try:
            results = pd.read_csv("models/ml_comparison.csv", index_col="model")
            best = results.sort_values("RMSE").index[0]
            _ml_model = joblib.load(f"models/{best}.pkl")
            _feat_scaler = joblib.load("models/feature_scaler.pkl")
            _target_scaler = joblib.load("models/target_scaler.pkl")
        except FileNotFoundError:
            pass


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    month: int
    day_of_year: int
    is_monsoon: int
    temp_range: float
    precip_roll_7d: float
    precip_roll_30d: float
    temp_roll_7d: float
    temp_roll_30d: float
    precip_lag_1: float
    precip_lag_3: float
    precip_lag_7: float
    temp_lag_1: float
    temp_lag_3: float
    temp_lag_7: float
    wind_speed_10m_max: float
    et0_fao_evapotranspiration: float
    shortwave_radiation_sum: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Weather Intelligence API is running", "version": "1.0.0"}


@app.get("/cities")
def list_cities():
    return {"cities": list(INDIAN_CITIES.keys())}


@app.get("/current/{city}")
def current_weather(city: str):
    try:
        return fetch_current_weather(city)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {e}")


@app.get("/forecast/{city}")
def weather_forecast(city: str, days: int = Query(default=7, ge=1, le=16)):
    try:
        df = fetch_forecast(city, days)
        return df.to_dict(orient="records")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API error: {e}")


@app.post("/predict/rainfall")
def predict_rainfall(req: PredictRequest):
    _load_models()
    if _ml_model is None:
        raise HTTPException(status_code=503, detail="Models not trained yet. Run training pipeline first.")

    month = req.month
    doy   = req.day_of_year
    features = np.array([[
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * doy / 365),
        np.cos(2 * np.pi * doy / 365),
        req.is_monsoon, req.temp_range,
        req.precip_roll_7d, req.precip_roll_30d,
        req.temp_roll_7d, req.temp_roll_30d,
        req.precip_lag_1, req.precip_lag_3, req.precip_lag_7,
        req.temp_lag_1, req.temp_lag_3, req.temp_lag_7,
        req.wind_speed_10m_max, req.et0_fao_evapotranspiration,
        req.shortwave_radiation_sum
    ]])

    scaled = _feat_scaler.transform(features)
    pred_scaled = _ml_model.predict(scaled).reshape(-1, 1)
    pred = _target_scaler.inverse_transform(pred_scaled)[0, 0]

    return {
        "predicted_rainfall_mm": round(max(0.0, float(pred)), 2),
        "is_monsoon": bool(req.is_monsoon),
        "month": month
    }


@app.get("/model/performance")
def model_performance():
    try:
        ml = pd.read_csv("models/ml_comparison.csv", index_col="model")
        return {"ml_models": ml.to_dict(orient="index")}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No trained models found.")


@app.get("/health")
def health():
    _load_models()
    return {
        "status": "ok",
        "models_loaded": _ml_model is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
