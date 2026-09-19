"""
Data Acquisition Module
Fetches historical weather data for Indian cities using Open-Meteo API (free, no key required).
Covers key meteorological variables relevant to the Southwest Monsoon.
"""

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from pathlib import Path
import json

# Indian cities covering different monsoon zones
INDIAN_CITIES = {
    "Mumbai":      {"lat": 19.0760, "lon": 72.8777, "zone": "West Coast"},
    "Chennai":     {"lat": 13.0827, "lon": 80.2707, "zone": "East Coast"},
    "Kolkata":     {"lat": 22.5726, "lon": 88.3639, "zone": "East"},
    "Delhi":       {"lat": 28.6139, "lon": 77.2090, "zone": "North"},
    "Bengaluru":   {"lat": 12.9716, "lon": 77.5946, "zone": "South"},
    "Hyderabad":   {"lat": 17.3850, "lon": 78.4867, "zone": "Central"},
    "Pune":        {"lat": 18.5204, "lon": 73.8567, "zone": "West"},
    "Bhopal":      {"lat": 23.2599, "lon": 77.4126, "zone": "Central"},
    "Patna":       {"lat": 25.5941, "lon": 85.1376, "zone": "East"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "zone": "Kerala"},
}

HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "cloud_cover", "et0_fao_evapotranspiration"
]

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
    "et0_fao_evapotranspiration", "precipitation_hours",
    "shortwave_radiation_sum"
]


def _build_client():
    session = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)


def fetch_historical_daily(city: str, start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Fetch daily historical data for a single city."""
    if city not in INDIAN_CITIES:
        raise ValueError(f"City '{city}' not in supported list: {list(INDIAN_CITIES.keys())}")

    meta = INDIAN_CITIES[city]
    client = _build_client()

    params = {
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "start_date": start,
        "end_date": end,
        "daily": DAILY_VARIABLES,
        "timezone": "Asia/Kolkata"
    }

    responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    r = responses[0]
    daily = r.Daily()

    data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}

    for i, var in enumerate(DAILY_VARIABLES):
        data[var] = daily.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["city"] = city
    df["zone"] = meta["zone"]
    df["latitude"] = meta["lat"]
    df["longitude"] = meta["lon"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def fetch_all_cities(start: str = "2015-01-01", end: str = "2024-12-31",
                     save_path: str = "data/raw") -> pd.DataFrame:
    """Fetch and combine data for all cities, save to CSV."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    frames = []

    for city in INDIAN_CITIES:
        print(f"  Fetching: {city}...")
        try:
            df = fetch_historical_daily(city, start, end)
            frames.append(df)
            df.to_csv(f"{save_path}/{city.lower().replace(' ', '_')}_daily.csv", index=False)
        except Exception as e:
            print(f"  [WARN] Failed for {city}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(f"{save_path}/all_cities_daily.csv", index=False)
    print(f"\nSaved combined dataset: {len(combined)} rows")
    return combined


def fetch_current_weather(city: str) -> dict:
    """Fetch current weather conditions for a city (used by FastAPI)."""
    if city not in INDIAN_CITIES:
        raise ValueError(f"City '{city}' not supported.")

    meta = INDIAN_CITIES[city]
    client = _build_client()

    params = {
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "current": [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "surface_pressure", "wind_speed_10m", "cloud_cover",
            "weather_code"
        ],
        "timezone": "Asia/Kolkata"
    }

    responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    r = responses[0]
    current = r.Current()

    variables = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "cloud_cover", "weather_code"
    ]

    result = {
        "city": city,
        "zone": meta["zone"],
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    for i, var in enumerate(variables):
        val = current.Variables(i).Value()
        result[var] = round(float(val), 2)

    return result


def fetch_forecast(city: str, days: int = 7) -> pd.DataFrame:
    """Fetch weather forecast for next N days."""
    if city not in INDIAN_CITIES:
        raise ValueError(f"City '{city}' not supported.")

    meta = INDIAN_CITIES[city]
    client = _build_client()

    params = {
        "latitude": meta["lat"],
        "longitude": meta["lon"],
        "daily": DAILY_VARIABLES,
        "forecast_days": min(days, 16),
        "timezone": "Asia/Kolkata"
    }

    responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
    r = responses[0]
    daily = r.Daily()

    data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}

    for i, var in enumerate(DAILY_VARIABLES):
        data[var] = daily.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df["city"] = city
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


if __name__ == "__main__":
    print("Fetching historical weather data for Indian cities...")
    df = fetch_all_cities(start="2015-01-01", end="2024-12-31")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")