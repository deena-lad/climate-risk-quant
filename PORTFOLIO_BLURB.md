# Portfolio Blurb — Climate Risk Quantification

## ~150-word version (paste directly into portfolio website)

Built an end-to-end physical climate risk quantification system for financial
portfolios, replicating the methodology used by MSCI ESG, S&P Trucost, and
the ECB's 2022 mandatory climate stress tests. The pipeline ingests 30+ years
of ERA5 reanalysis climate data (flood, heat, cyclone) via the Copernicus CDS
API, performs geospatial joins using GeoPandas and a SciPy KD-tree to snap
200+ assets to a 0.25° global grid, and produces normalised sector-weighted
risk scores (0–100) via a scikit-learn RobustScaler pipeline. A Monte Carlo
Climate VaR engine (10,000 paths) quantifies tail exposure: on a synthetic
200-asset portfolio, the model identifies a 14.8% portfolio-wide haircut and
a 95% Climate VaR of ~292 USD M. An IsolationForest anomaly detector flags
compound multi-hazard tail-risk assets missed by linear scoring. All
experiments are tracked in MLflow; the interactive Streamlit dashboard
(deployed on HuggingFace Spaces via Docker) exposes α, hazard weights, and
sector filters in real time. Demonstrates ERA5/NetCDF, GeoPandas, ESG
analytics, risk modelling, and full-stack Python deployment.

---

## Shorter variant (100 words — for CV project bullet or LinkedIn featured section)

Engineered a TCFD-aligned physical climate risk platform mapping ERA5 flood,
heat, and cyclone hazards to financial asset exposure. Pipeline: CDS API →
xarray/NetCDF preprocessing → GeoPandas spatial join → scikit-learn
percentile scoring → Monte Carlo Climate VaR. Key results on a 200-asset
portfolio: 14.8% climate haircut, 95% VaR of 292 USD M, Gini = 0.30+ across
sector-weighted composite scores. IsolationForest anomaly detection flags
multi-hazard tail-risk assets; all runs tracked in MLflow. Deployed as an
interactive Streamlit dashboard on HuggingFace Spaces (Docker). Stack: Python,
ERA5, GeoPandas, scikit-learn, MLflow, Plotly, Streamlit, GitHub Actions CI.

---

## One-liner (for project card title/subtitle)

**Climate Risk Quantification** — Physical hazard exposure (flood · heat ·
cyclone) mapped to asset-level ESG scores and risk-adjusted NAV using ERA5,
GeoPandas, and Monte Carlo VaR. Deployed on HuggingFace Spaces.
