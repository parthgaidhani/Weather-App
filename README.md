# 🌧️ Weather Intelligence & Climate Decision Support Platform

> **AI-powered Southwest Monsoon Analysis and Prediction System**
> Team GenAI Research 2026 — Phase 02, Part 01

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](.github/workflows/ci.yml)
[![Made with love](https://img.shields.io/badge/made%20with-%E2%9D%A4%EF%B8%8F-red.svg)](https://github.com/parthgaidhani/Weather-App)

End-to-end AI platform for analyzing and forecasting the **Indian Southwest Monsoon** using classical ML, deep learning, statistical forecasting, and explainable AI — served through a FastAPI backend and Streamlit dashboard.

---

## ✨ Features

- 🛰️ **Data acquisition** from Open-Meteo, ERA5, NASA POWER, IMD (free, no API keys)
- 🤖 **15+ ML algorithms** — Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost, SVR, KNN, …
- 🧠 **Deep Learning** — LSTM, Bi-LSTM, GRU, CNN-LSTM (TensorFlow + PyTorch)
- 📈 **Statistical forecasting** — ARIMA, SARIMA, Holt-Winters, Prophet
- 🔍 **Explainability** — SHAP & LIME feature attributions
- ⚡ **FastAPI** REST backend with auto-generated OpenAPI docs
- 📊 **Streamlit** interactive dashboard
- 🐳 **Docker** & docker-compose for one-command deployment

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (8501)                │
│              src/ui/app.py  — interactive charts             │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (8000)                  │
│           src/api/main.py  — /predict /forecast /health      │
└──────────────────────────────────────────────────────────────┘
                              ▲
┌───────────────┬──────────────┬───────────────────────────────┐
│  ML Models    │  DL Models   │   Time-Series Models          │
│  14 algos     │  LSTM/GRU    │   ARIMA/SARIMA/Prophet        │
└───────────────┴──────────────┴───────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────┐
│           Preprocessing  ←  Acquisition (Open-Meteo)        │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
Weather-App/
├── .github/
│   ├── workflows/ci.yml            # GitHub Actions CI
│   ├── ISSUE_TEMPLATE/             # Bug & feature templates
│   └── dependabot.yml              # Weekly dependency updates
├── src/
│   ├── data/
│   │   ├── acquisition.py          # Open-Meteo / ERA5 fetcher
│   │   └── preprocessing.py        # Cleaning & feature engineering
│   ├── models/
│   │   ├── ml_models.py            # 14 classical ML algorithms
│   │   ├── dl_models.py            # LSTM, BiLSTM, GRU, CNN-LSTM
│   │   ├── timeseries_models.py    # ARIMA, SARIMA, HoltWinters, Prophet
│   │   └── explainability.py       # SHAP & LIME
│   ├── api/main.py                 # FastAPI backend
│   └── ui/app.py                   # Streamlit dashboard
├── data/{raw,processed}/           # Local data (gitignored)
├── models/                         # Saved trained models (gitignored)
├── reports/                        # Plots & evaluation artifacts
├── notebooks/                      # EDA notebooks
├── train.py                        # End-to-end training pipeline
├── Dockerfile                      # Production container
├── docker-compose.yml              # One-command stack
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Dev/test/lint tools
├── Makefile                        # Common commands
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Local (Python 3.10+)

```bash
# 1. Clone & enter
git clone https://github.com/parthgaidhani/Weather-App.git
cd Weather-App

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full training pipeline
python train.py --steps data preprocess ml

# 5. Launch the API (terminal 1)
uvicorn src.api.main:app --reload --port 8000
# → http://localhost:8000/docs

# 6. Launch the dashboard (terminal 2)
streamlit run src/ui/app.py
# → http://localhost:8501
```

### Option 2 — Docker (recommended)

```bash
docker-compose up --build
```

This brings up **both** the API (`:8000`) and the dashboard (`:8501`).

---

## 🌆 Cities Covered

Mumbai · Chennai · Kolkata · Delhi · Bengaluru · Hyderabad · Pune · Bhopal · Patna · Thiruvananthapuram

---

## 🔌 API Endpoints

| Method | Endpoint                  | Description                  |
| ------ | ------------------------- | ---------------------------- |
| GET    | `/health`                 | Liveness check               |
| GET    | `/cities`                 | List supported cities        |
| GET    | `/current/{city}`         | Current weather              |
| GET    | `/forecast/{city}?days=7` | N-day forecast               |
| POST   | `/predict/rainfall`       | ML rainfall prediction       |
| GET    | `/model/performance`      | Model comparison metrics     |

Full interactive docs at **`http://localhost:8000/docs`**.

---

## 📊 Evaluation Metrics

`MAE`, `MSE`, `RMSE`, `MAPE`, `R²` — every model written to `models/ml_comparison.csv`.

---

## 🧪 Development

```bash
# Run linters
make lint          # ruff + black --check
make format        # auto-format with black

# Verify imports compile
make test-imports

# Clean caches
make clean
```

---

## 🛣️ Roadmap

- [x] Phase 02 Part 01 — Classical ML + DL + Stats + SHAP
- [ ] Phase 02 Part 02 — RAG pipeline, LangChain agents, LLM integration
- [ ] Phase 03 — Vector DB, MCP tools, autonomous agents
- [ ] Phase 04 — Multi-region expansion & cloud deploy

---

## 📜 License

[MIT](LICENSE) © 2026 parthgaidhani

---

## 🙏 Acknowledgements

Data providers: **Open-Meteo**, **Copernicus ERA5**, **NASA POWER**, **IMD**.
Built as part of **Team GenAI Research 2026**.
