# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Production-grade README with architecture diagram and badges
- GitHub Actions CI (Python 3.10 / 3.11 / 3.12)
- Issue & PR templates, Dependabot
- `requirements-dev.txt` and `Makefile`
- `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- `.dockerignore`, `.editorconfig`, `.gitattributes`

## [0.1.0] — 2026-07-30

### Added
- Data acquisition from Open-Meteo
- 14 classical ML models
- LSTM / BiLSTM / GRU / CNN-LSTM deep learning models
- ARIMA / SARIMA / Holt-Winters / Prophet time-series models
- SHAP & LIME explainability module
- FastAPI backend with `/predict`, `/forecast`, `/current`, `/cities`
- Streamlit dashboard
- Dockerfile + docker-compose
- End-to-end `train.py` pipeline
