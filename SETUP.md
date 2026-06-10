# Setup Instructions

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | 3.11 recommended |
| Git | any | |
| pip | ≥ 23 | comes with Python |
| (optional) conda/mamba | any | alternative to venv |

---

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/climate-risk-quant.git
cd climate-risk-quant
```

---

## 2. Create and activate a virtual environment

**Using venv (all platforms)**
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (cmd.exe)
.\.venv\Scripts\activate.bat
```

**Using conda (alternative)**
```bash
conda create -n climate-risk python=3.11 -y
conda activate climate-risk
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .        # installs src/ as editable package
```

> **Note for Windows users**: `fiona` and `gdal` can be tricky.
> If pip fails, install GDAL first via [OSGeo4W](https://trac.osgeo.org/osgeo4w/)
> or use the conda route: `conda install -c conda-forge geopandas`.

---

## 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in your editor and set at minimum:

```
CDS_API_KEY=your-uid:your-api-key-here
```

**How to get a CDS API key:**
1. Register at https://cds.climate.copernicus.eu/
2. Log in → click your name (top right) → "Your profile"
3. Copy your UID and API key
4. Paste as `UID:API_KEY` (e.g. `123456:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)

The CDS API also requires a `~/.cdsapirc` file. The ingestion script
creates this automatically from your `.env`, but you can also create it manually:

```
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
verify: 1
```

---

## 5. Create data directories

```bash
python -c "from src.config import settings; settings.ensure_dirs()"
```

This creates `data/raw/`, `data/interim/`, and `data/processed/`.

---

## 6. Verify the installation

```bash
python -c "
import xarray, geopandas, rioxarray, mlflow, streamlit
print('All core imports OK')
from src.config import settings
print(f'Config loaded. ERA5 years: {settings.era5_year_start}–{settings.era5_year_end}')
"
```

Expected output:
```
All core imports OK
Config loaded. ERA5 years: 1990–2023
```

---

## 7. Run the test suite

```bash
pytest
```

---

## 8. Launch the Streamlit app (local dev)

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 9. Launch JupyterLab (for notebooks)

```bash
jupyter lab notebooks/data_exploration.ipynb
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run `pip install -e .` from the repo root |
| `eccodes not found` | `conda install -c conda-forge eccodes` or `brew install eccodes` (macOS) |
| CDS API 403 error | Ensure you accepted the ERA5 licence at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels |
| `fiona` GDAL mismatch | Use conda: `conda install -c conda-forge geopandas fiona` |
| Weights don't sum to 1.0 | Edit `WEIGHT_FLOOD`, `WEIGHT_HEAT`, `WEIGHT_CYCLONE` in `.env` so they sum to exactly 1.0 |
