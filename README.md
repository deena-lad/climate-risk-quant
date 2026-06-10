---
title: Climate Risk Quantification
emoji: 🌍
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Physical climate risk (flood/heat/cyclone) mapped to financial asset exposure
tags:
  - climate
  - finance
  - ESG
  - risk-modeling
  - geospatial
  - ERA5
  - streamlit
---

# 🌍 Climate Risk Quantification for Financial Assets

> **Map physical climate hazards to asset-level financial exposure — built to the same standard as MSCI ESG and ECB stress-test frameworks.**

[![CI](https://github.com/deena-lad/climate-risk-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/deena-lad/climate-risk-quant/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange)](https://huggingface.co/spaces/deena-lad/climate-risk-quant)

---

## Motivation

Physical climate risk — floods, extreme heat, cyclones — is now a material
financial risk. The TCFD framework requires institutional investors to disclose
exposure, and the ECB has conducted mandatory climate stress tests since 2022.
Yet most public implementations either stop at data visualisation or rely on
proprietary black-box scores.

This project builds a transparent, auditable, end-to-end pipeline that:

1. Ingests 30+ years of ERA5 reanalysis climate data (global, 0.25° resolution)
2. Maps three physical hazards to the exact coordinates of each financial asset
3. Produces normalised, sector-weighted climate risk scores (0–100)
4. Computes risk-adjusted portfolio valuations and Monte Carlo Climate VaR
5. Exposes every parameter to the analyst via an interactive dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                   │
│  ERA5 / CDS API ──► NetCDF grids    Company CSV ──► lat/lon     │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION  (src/pipeline/ingest.py)                            │
│  cdsapi download │ xarray lazy load │ pydantic asset validation │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  PREPROCESSING  (src/pipeline/preprocess.py)                    │
│  Clip → monthly max → annual max → unit convert → tidy parquet  │
│  30-yr climatology: μ, σ, p95 per grid cell → z-score          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING  (src/pipeline/features.py)                │
│  KD-tree snap to ERA5 grid │ spatial join │ compound flag       │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODEL  (src/model/)                                            │
│  HazardScorer: RobustScaler + percentile rank → 0-100           │
│  CompositeRiskModel: sector-weighted average + PCA diagnostic   │
│  AnomalyDetector: IsolationForest tail-risk flagging            │
│  Valuation: linear/convex haircut + Monte Carlo Climate VaR     │
│  Evaluate: Gini, KS test, tail concentration, weight sensitivity│
│  MLflow: experiment tracking + artifact store                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  DASHBOARD  (app/streamlit_app.py)                              │
│  Risk Map │ Score Analysis │ Valuation │ Scorecard │ Diagnostics│
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (local, 4 commands)

```bash
git clone https://github.com/deena-lad/climate-risk-quant.git && cd climate-risk-quant
pip install -r requirements.txt && pip install -e .
cp .env.example .env          # add your CDS_API_KEY (optional for demo mode)
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 — the dashboard loads with synthetic data instantly,
no ERA5 download required.

---

## Live Demo

🔗 **[https://huggingface.co/spaces/deena-lad/climate-risk-quant](https://huggingface.co/spaces/deena-lad/climate-risk-quant)**

---

## Results / Key Findings

All numbers below are from the 200-asset synthetic demo run (`seed=42`, `α=0.30`):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Portfolio climate haircut | **14.8%** | ~15 cents of every dollar at risk from physical climate exposure |
| 95% Climate VaR | **~292 USD M** | Tail downside at 95th-percentile scenario (Monte Carlo, 10k paths) |
| Expected Shortfall (CVaR) | **~363 USD M** | Average loss beyond the VaR threshold |
| Composite score Gini | **0.21–0.35** | Meaningful differentiation across portfolio (target > 0.30) |
| PCA PC1 variance | **~47%** | No single hazard dominates — composite is balanced |
| Compound-risk assets | **~12%** | Multi-hazard exposure warranting priority review |
| Critical-tier assets | **~6%** | Require immediate ESG disclosure review |
| High + Critical NAV share | **~40%** | Portfolio concentration in elevated-risk tiers |

**Sector findings:** Real Estate and Utilities carry the highest absolute
haircuts due to fixed asset locations in coastal/tropical exposure zones.
Energy shows highest cyclone sensitivity from offshore infrastructure.

**Weight sensitivity:** Gini is stable (± 0.05) across 200 random weight
combinations — the composite score is robust to reasonable weight uncertainty,
a key requirement for model validation sign-off.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ERA5](https://img.shields.io/badge/Data-ERA5%2FcdsAPI-green)
![xarray](https://img.shields.io/badge/xarray-NetCDF-lightblue)
![GeoPandas](https://img.shields.io/badge/GeoPandas-Geospatial-darkgreen)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Plotly](https://img.shields.io/badge/Plotly-Viz-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Docker](https://img.shields.io/badge/Docker-Deploy-blue)
![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)

| Layer | Tool | Reason |
|-------|------|--------|
| Climate data | `cdsapi` + `xarray` | Official ERA5 client; lazy NetCDF loading for multi-year global grids |
| Geospatial ops | `geopandas` + `rioxarray` + `shapely` | Industry standard; GEOS-backed correctness; CRS-aware raster clip |
| Spatial indexing | `scipy.cKDTree` | O(n log n) nearest-neighbour for snapping assets to ERA5 grid |
| Risk scoring | `scikit-learn` RobustScaler + PCA + IsolationForest | Interpretable, auditable; no black-box regression on unlabelled data |
| Experiment tracking | `MLflow` | Lightweight local/remote; logs params, metrics, artifacts; HF-compatible |
| Visualisation | `plotly` + `folium` | Interactive charts; geo scatter globe; zero JS required |
| Frontend | `Streamlit` | Python-native interactive apps; first-class HF Spaces support |
| Validation | `pydantic-settings` | Type-safe config; weight-sum and year-range checked at startup |
| Deployment | Docker (multi-stage) + HF Spaces | Reproducible environment; free public hosting |

---

## Skills Demonstrated

`ERA5 / NetCDF` · `xarray` · `GeoPandas` · `rioxarray` · `Spatial joins` ·
`KD-tree indexing` · `ESG analytics` · `Physical risk modelling` · `TCFD` ·
`Climate VaR` · `Monte Carlo simulation` · `scikit-learn` · `IsolationForest` ·
`PCA diagnostics` · `MLflow experiment tracking` · `Plotly` · `Streamlit` ·
`Docker multi-stage builds` · `HuggingFace Spaces` · `pydantic` ·
`pytest` · `GitHub Actions CI/CD` · `Conventional Commits`

---

## Repo Structure

```
climate-risk-quant/
├── README.md
├── requirements.txt          # 40 pinned packages
├── pyproject.toml            # ruff + mypy + pytest config
├── .env.example              # all env vars documented
├── Dockerfile                # multi-stage: builder → runtime
├── .dockerignore
├── .streamlit/
│   └── config.toml           # port 7860, theme, fast reruns
│
├── src/
│   ├── config.py             # pydantic-settings singleton
│   ├── pipeline/
│   │   ├── ingest.py         # ERA5 download + asset loader + synthetic gen
│   │   ├── preprocess.py     # clip → resample → unit convert → climatology
│   │   ├── features.py       # spatial join → hazard scores → ESG tier
│   │   └── validate.py       # 10 data quality checks
│   ├── model/
│   │   ├── scorer.py         # HazardScorer, CompositeRiskModel, AnomalyDetector
│   │   ├── valuation.py      # haircut, portfolio NAV, Climate VaR, sector attr
│   │   ├── evaluate.py       # Gini, KS, tail conc., MLflow logging, sensitivity
│   │   ├── artifacts.py      # joblib save/load, run metadata
│   │   └── pipeline.py       # run_pipeline() end-to-end orchestrator
│   └── viz/
│       └── charts.py         # 10 Plotly figure builders
│
├── app/
│   ├── streamlit_app.py      # 5-tab dashboard, 10+ interactive elements
│   └── components/
│       ├── sidebar.py        # SidebarConfig dataclass + all widgets
│       └── metrics_row.py    # 6 KPI metric cards
│
├── notebooks/
│   └── data_exploration.ipynb  # 7 EDA charts with interpretation notes
│
├── tests/
│   ├── test_pipeline.py      # 30+ pipeline unit tests
│   └── test_model.py         # 40+ model unit tests, full integration test
│
└── .github/
    └── workflows/
        └── ci.yml            # lint → test → docker build → (deploy to HF)
```

---

## Future Improvements

- **Scenario analysis:** Extend beyond historical ERA5 to CMIP6 SSP2-4.5 and
  SSP5-8.5 climate projections (2050/2100 horizons), enabling forward-looking
  TCFD Scope 3 disclosures.

- **Asset-level granularity:** Replace company centroids with building-level
  footprint polygons (OpenStreetMap) for real-estate portfolios, enabling
  flood inundation depth estimates rather than proximity scores.

- **Transition risk overlay:** Add a carbon-price sensitivity module using
  NGFS transition scenarios so users can see physical + transition risk on
  a single dashboard — the dual lens required by TCFD.

- **Live ERA5 updates:** Schedule a monthly CDS API pull via GitHub Actions
  so the dashboard always reflects the most recent completed climate year
  without manual intervention.

- **Regulatory report export:** Auto-generate a TCFD-aligned PDF risk report
  (asset table + heatmap + VaR) using `reportlab`, reducing analyst
  copy-paste time from dashboard to board presentation.

---

## References and Data Sources

| Source | Description | Link |
|--------|-------------|------|
| ERA5 Reanalysis | Hourly global climate data, 1940–present, 0.25° resolution | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) |
| TCFD Framework | Task Force on Climate-related Financial Disclosures | [TCFD](https://www.fsb-tcfd.org/) |
| ECB Climate Stress Test 2022 | ECB methodology for physical risk VaR | [ECB](https://www.ecb.europa.eu/pub/pdf/scpops/ecb.op293~bc175b6b86.en.pdf) |
| IPCC AR6 WG1 Ch.11 | Compound climate events science basis | [IPCC](https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-11/) |
| MSCI ESG Physical Risk | Industry benchmark scoring methodology | [MSCI](https://www.msci.com/our-solutions/esg-investing/esg-ratings) |
| GADM v4.1 | Global administrative boundaries | [GADM](https://gadm.org/) |
| GeoPandas docs | Spatial operations reference | [GeoPandas](https://geopandas.org/) |
| MLflow docs | Experiment tracking reference | [MLflow](https://mlflow.org/) |

---

## Setup & Contributing

See [SETUP.md](SETUP.md) for full local setup instructions.
See [DEPLOY.md](DEPLOY.md) for HuggingFace Spaces deployment.

Contributions welcome — please follow [Conventional Commits](https://www.conventionalcommits.org/)
and ensure `pytest` passes before opening a PR.