"""
Streamlit Weather Intelligence Dashboard
Interactive UI for weather data, forecasts, and ML predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.acquisition import (
    fetch_current_weather, fetch_forecast, INDIAN_CITIES
)

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Weather Intelligence Platform",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌧️ Weather Intelligence")
st.sidebar.markdown("*Southwest Monsoon AI Platform*")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📊 EDA & Analysis", "🤖 ML Predictions", "📈 Forecasts", "🛰️ Model Performance"]
)

city = st.sidebar.selectbox("Select City", list(INDIAN_CITIES.keys()))

# ── Dashboard ─────────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.title("🌧️ Weather Intelligence Platform")
    st.markdown("**AI-powered Southwest Monsoon Analysis & Prediction System**")

    col1, col2, col3 = st.columns([1, 1, 1])

    with st.spinner(f"Fetching current weather for {city}..."):
        try:
            weather = fetch_current_weather(city)

            with col1:
                st.metric("🌡️ Temperature", f"{weather['temperature_2m']} °C")
                st.metric("💧 Humidity", f"{weather['relative_humidity_2m']} %")

            with col2:
                st.metric("🌧️ Precipitation", f"{weather['precipitation']} mm")
                st.metric("💨 Wind Speed", f"{weather['wind_speed_10m']} km/h")

            with col3:
                st.metric("🔵 Pressure", f"{weather['surface_pressure']} hPa")
                st.metric("☁️ Cloud Cover", f"{weather['cloud_cover']} %")

            st.info(f"📍 {city} | Zone: {weather['zone']} | "
                    f"Lat: {weather['latitude']}, Lon: {weather['longitude']}")

        except Exception as e:
            st.error(f"Could not fetch weather data: {e}")

    st.divider()

    # 7-day forecast
    st.subheader(f"📅 7-Day Forecast — {city}")
    try:
        forecast_df = fetch_forecast(city, days=7)
        forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.date

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=forecast_df["date"], y=forecast_df["precipitation_sum"],
            name="Rainfall (mm)", marker_color="steelblue"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["temperature_2m_max"],
            name="Max Temp (°C)", yaxis="y2", line=dict(color="red")
        ))
        fig.update_layout(
            yaxis=dict(title="Rainfall (mm)"),
            yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Forecast unavailable: {e}")

# ── EDA ───────────────────────────────────────────────────────────────────────
elif page == "📊 EDA & Analysis":
    st.title("📊 Exploratory Data Analysis")

    processed_path = "data/processed/features.csv"
    if not Path(processed_path).exists():
        st.warning("⚠️ Processed data not found. Run the data pipeline first:")
        st.code("python -m src.data.preprocessing", language="bash")
        st.stop()

    df = pd.read_csv(processed_path, parse_dates=["date"])
    city_df = df[df["city"] == city].sort_values("date")

    tab1, tab2, tab3 = st.tabs(["Rainfall", "Temperature", "Monsoon Analysis"])

    with tab1:
        st.subheader(f"Daily Rainfall — {city}")
        fig = px.line(city_df, x="date", y="precipitation_sum",
                      title=f"Daily Rainfall (mm) — {city}",
                      color_discrete_sequence=["steelblue"])
        st.plotly_chart(fig, use_container_width=True)

        monthly = city_df.set_index("date")["precipitation_sum"].resample("ME").sum().reset_index()
        fig2 = px.bar(monthly, x="date", y="precipitation_sum",
                      title="Monthly Rainfall", color_discrete_sequence=["navy"])
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader(f"Temperature Trends — {city}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=city_df["date"], y=city_df["temperature_2m_max"],
                                  name="Max", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=city_df["date"], y=city_df["temperature_2m_min"],
                                  name="Min", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=city_df["date"], y=city_df["temperature_2m_mean"],
                                  name="Mean", line=dict(color="orange")))
        fig.update_layout(title=f"Temperature (°C) — {city}", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Monsoon Season Analysis")
        monsoon_df = city_df[city_df["is_monsoon"] == 1]
        annual = monsoon_df.groupby("year")["precipitation_sum"].sum().reset_index()
        fig = px.bar(annual, x="year", y="precipitation_sum",
                     title=f"Annual Monsoon Rainfall (Jun–Sep) — {city}",
                     color="precipitation_sum", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Monsoon Rain/Year", f"{annual['precipitation_sum'].mean():.1f} mm")
        col2.metric("Max Year", str(annual.loc[annual['precipitation_sum'].idxmax(), 'year']))
        col3.metric("Min Year", str(annual.loc[annual['precipitation_sum'].idxmin(), 'year']))

# ── ML Predictions ────────────────────────────────────────────────────────────
elif page == "🤖 ML Predictions":
    st.title("🤖 Rainfall Prediction")
    st.markdown("Enter weather parameters to predict rainfall using the best trained ML model.")

    col1, col2 = st.columns(2)
    with col1:
        month       = st.slider("Month", 1, 12, 7)
        doy         = st.slider("Day of Year", 1, 365, 180)
        is_monsoon  = st.selectbox("Monsoon Season?", [1, 0])
        temp_range  = st.number_input("Temp Range (°C)", 0.0, 30.0, 8.0)
        wind_speed  = st.number_input("Max Wind Speed (km/h)", 0.0, 100.0, 20.0)
        et0         = st.number_input("ET0 (mm)", 0.0, 15.0, 4.0)
        radiation   = st.number_input("Shortwave Radiation (MJ/m²)", 0.0, 40.0, 18.0)

    with col2:
        precip_roll_7d  = st.number_input("7-day Avg Rainfall (mm)", 0.0, 100.0, 5.0)
        precip_roll_30d = st.number_input("30-day Avg Rainfall (mm)", 0.0, 100.0, 8.0)
        temp_roll_7d    = st.number_input("7-day Avg Temp (°C)", 10.0, 45.0, 28.0)
        temp_roll_30d   = st.number_input("30-day Avg Temp (°C)", 10.0, 45.0, 27.0)
        precip_lag_1    = st.number_input("Yesterday's Rainfall (mm)", 0.0, 200.0, 2.0)
        precip_lag_3    = st.number_input("3-day Ago Rainfall (mm)", 0.0, 200.0, 3.0)
        precip_lag_7    = st.number_input("7-day Ago Rainfall (mm)", 0.0, 200.0, 4.0)
        temp_lag_1      = st.number_input("Yesterday's Temp (°C)", 10.0, 45.0, 28.0)
        temp_lag_3      = st.number_input("3-day Ago Temp (°C)", 10.0, 45.0, 27.5)
        temp_lag_7      = st.number_input("7-day Ago Temp (°C)", 10.0, 45.0, 27.0)

    if st.button("🔮 Predict Rainfall", type="primary"):
        payload = {
            "month": month, "day_of_year": doy, "is_monsoon": is_monsoon,
            "temp_range": temp_range, "precip_roll_7d": precip_roll_7d,
            "precip_roll_30d": precip_roll_30d, "temp_roll_7d": temp_roll_7d,
            "temp_roll_30d": temp_roll_30d, "precip_lag_1": precip_lag_1,
            "precip_lag_3": precip_lag_3, "precip_lag_7": precip_lag_7,
            "temp_lag_1": temp_lag_1, "temp_lag_3": temp_lag_3,
            "temp_lag_7": temp_lag_7, "wind_speed_10m_max": wind_speed,
            "et0_fao_evapotranspiration": et0, "shortwave_radiation_sum": radiation
        }
        try:
            resp = requests.post(f"{API_BASE}/predict/rainfall", json=payload, timeout=10)
            result = resp.json()
            st.success(f"🌧️ Predicted Rainfall: **{result['predicted_rainfall_mm']} mm**")
        except Exception as e:
            st.error(f"API error: {e}. Make sure the FastAPI server is running.")

# ── Forecasts ─────────────────────────────────────────────────────────────────
elif page == "📈 Forecasts":
    st.title("📈 Extended Weather Forecast")
    days = st.slider("Forecast Days", 1, 16, 7)

    try:
        df = fetch_forecast(city, days)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["date"], y=df["precipitation_sum"],
                              name="Rainfall (mm)", marker_color="steelblue"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["temperature_2m_max"],
                                  name="Max Temp", yaxis="y2", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["temperature_2m_min"],
                                  name="Min Temp", yaxis="y2", line=dict(color="blue")))
        fig.update_layout(
            title=f"{days}-Day Forecast — {city}",
            yaxis=dict(title="Rainfall (mm)"),
            yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Forecast error: {e}")

# ── Model Performance ─────────────────────────────────────────────────────────
elif page == "🛰️ Model Performance":
    st.title("🛰️ Model Evaluation & Comparison")

    ml_path = "models/ml_comparison.csv"
    dl_path = "models/dl_comparison.csv"
    stat_path = "models/statistical_comparison.csv"

    if Path(ml_path).exists():
        st.subheader("Machine Learning Models")
        ml_df = pd.read_csv(ml_path, index_col="model").sort_values("RMSE")
        fig = px.bar(ml_df.reset_index(), x="model", y="RMSE",
                     color="R2", color_continuous_scale="RdYlGn",
                     title="ML Model RMSE Comparison")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ml_df.style.highlight_min(subset=["RMSE"], color="lightgreen")
                              .highlight_max(subset=["R2"], color="lightgreen"),
                     use_container_width=True)
    else:
        st.info("Train ML models first: `python -m src.models.ml_models`")

    if Path(dl_path).exists():
        st.subheader("Deep Learning Models")
        dl_df = pd.read_csv(dl_path, index_col="model").sort_values("RMSE")
        st.dataframe(dl_df, use_container_width=True)

    if Path(stat_path).exists():
        st.subheader("Statistical Models")
        stat_df = pd.read_csv(stat_path, index_col="model").sort_values("RMSE")
        st.dataframe(stat_df, use_container_width=True)

    shap_path = Path("reports")
    shap_imgs = list(shap_path.glob("shap_summary_*.png")) if shap_path.exists() else []
    if shap_imgs:
        st.subheader("SHAP Feature Importance")
        for img in shap_imgs:
            st.image(str(img), caption=img.stem.replace("shap_summary_", ""))
