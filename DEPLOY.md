# Deployment Guide — Hugging Face Spaces

This guide covers three deployment paths:

| Path | Use case |
|---|---|
| [A. HuggingFace Spaces (recommended)](#a-huggingface-spaces) | Public demo, free hosting, zero infra |
| [B. Local Docker](#b-local-docker) | Reproducible local environment, pre-deploy testing |
| [C. Direct HF Spaces Git push](#c-direct-git-push-workflow) | Faster iteration without Docker build locally |

---

## A. HuggingFace Spaces

### Prerequisites
- GitHub account with this repo pushed to `main`
- Hugging Face account: https://huggingface.co/join
- (Optional) CDS API key for live ERA5 data: https://cds.climate.copernicus.eu/

---

### Step 1 — Create a new Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `climate-risk-quant` (or your preferred name)
   - **License**: MIT
   - **SDK**: **Docker** ← critical; do NOT choose Streamlit SDK
   - **Hardware**: CPU Basic (free tier) is sufficient for the demo
3. Click **Create Space**

Your Space URL will be:
`https://huggingface.co/spaces/YOUR_USERNAME/climate-risk-quant`

---

### Step 2 — Link your GitHub repo to the Space

**Option A — HF Git remote (simplest)**

```bash
# Clone the HF Space repo locally
git clone https://huggingface.co/spaces/YOUR_USERNAME/climate-risk-quant hf-space
cd hf-space

# Copy project files into it
cp -r /path/to/climate-risk-quant/. .

# Commit and push — HF auto-builds on every push to main
git add .
git commit -m "feat: initial deployment"
git push
```

**Option B — GitHub Actions auto-sync (recommended for ongoing dev)**

Add this job to `.github/workflows/ci.yml` (after the existing `docker` job):

yaml
  deploy-hf:
    name: Deploy to HuggingFace Spaces
    runs-on: ubuntu-latest
    needs: [test, docker]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true

      - name: Push to HuggingFace Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git config --global user.email "ci@github.com"
          git config --global user.name "GitHub Actions"
          git remote add hf https://deena-lad:${HF_TOKEN}@huggingface.co/spaces/deena-lad/climate-risk-quant
          git push hf main --force


Then add `HF_TOKEN` to your GitHub repo secrets:
`GitHub repo → Settings → Secrets → Actions → New repository secret`

Get your HF token at: https://huggingface.co/settings/tokens
(Role: **write**)

---

### Step 3 — Set environment variables securely

**Never commit your `.env` file.** Set secrets in the HF Spaces UI:

1. Go to your Space → **Settings** tab → **Repository secrets**
2. Add each variable from `.env.example`:

| Variable | Value | Notes |
|---|---|---|
| `CDS_API_KEY` | `YOUR_UID:YOUR_API_KEY` | From https://cds.climate.copernicus.eu/ |
| `RISK_HAIRCUT_ALPHA` | `0.30` | Adjust per your model spec |
| `WEIGHT_FLOOD` | `0.40` | Must sum to 1.0 with HEAT + CYCLONE |
| `WEIGHT_HEAT` | `0.35` | |
| `WEIGHT_CYCLONE` | `0.25` | |
| `LOG_LEVEL` | `INFO` | Use `WARNING` to reduce noise |

> **Important:** HF Spaces injects these as environment variables at runtime,
> identical to a local `.env` file. The `pydantic-settings` config module
> picks them up automatically — no code changes needed.

---

### Step 4 — Monitor the build

1. Go to your Space → **Build** tab
2. Watch the Docker build log (first build: ~5–8 min; cached rebuilds: ~1–2 min)
3. When status turns **Running** (green), click **App** tab to see the dashboard

**Common build errors and fixes:**

| Error | Fix |
|---|---|
| `libgdal32: package not found` | Check the exact package name for the Debian version in your base image. Use `apt-cache search gdal` in the build log. |
| `No space left on device` | Upgrade to a larger HF hardware tier, or prune unused layers from the Dockerfile. |
| `Permission denied: /home/appuser` | Ensure `COPY --chown=appuser:appuser` is present for all app files. |
| `Port 7860 already in use` | HF Spaces always expects port 7860 — verify your `CMD` and `config.toml`. |
| `ModuleNotFoundError: src` | Confirm `pip install -e .` is in the Dockerfile and `pyproject.toml` is copied. |

---

### Step 5 — Verify the live deployment

```bash
# Check the Space is healthy (replace with your URL)
curl -f https://YOUR_USERNAME-climate-risk-quant.hf.space/_stcore/health
# Expected: {"status":"ok"}

# Optionally smoke-test the API
curl https://YOUR_USERNAME-climate-risk-quant.hf.space/
# Expected: 200 HTML response (the Streamlit app)
```

---

## B. Local Docker

Use this to test the exact container that will run on HuggingFace before pushing.

### Build

```bash
# From repo root
docker build -t climate-risk-quant:local .

# Check image size (target < 2 GB)
docker images climate-risk-quant:local
```

### Run

```bash
docker run --rm \
  -p 7860:7860 \
  --env-file .env \
  -v $(pwd)/data:/home/appuser/app/data \
  -v $(pwd)/mlruns:/home/appuser/app/mlruns \
  climate-risk-quant:local
```

Open http://localhost:7860 in your browser.

**Flags explained:**
- `--env-file .env` — injects all secrets from your local `.env`
- `-v $(pwd)/data:...` — mounts your downloaded ERA5 data so the container can read it without re-downloading
- `-v $(pwd)/mlruns:...` — persists MLflow runs between container restarts

### Inspect the container

```bash
# Open a shell inside the running container
docker exec -it $(docker ps -q --filter ancestor=climate-risk-quant:local) /bin/bash

# Check that imports work
python -c "from src.model.pipeline import run_pipeline; print('OK')"

# Check environment variables were injected
env | grep WEIGHT_
```

---

## C. Direct Git Push Workflow

If you don't want to build Docker locally and just want fast iteration:

```bash
# 1. Clone your HF Space as a second remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/climate-risk-quant

# 2. Push a branch to HF (triggers a build)
git push hf main

# 3. Watch the build
#    Go to: https://huggingface.co/spaces/YOUR_USERNAME/climate-risk-quant/logs
```

HuggingFace will detect the `Dockerfile` and use the Docker SDK automatically.

---

## Updating the deployment

```bash
# Make code changes locally, then:
git add .
git commit -m "feat: improve valuation waterfall chart"
git push origin main        # triggers GitHub Actions CI
git push hf main            # triggers HF rebuild (or let Actions do it)
```

With the GitHub Actions auto-sync job, a single `git push origin main` triggers CI → tests → HF deploy automatically.

---

## Rollback

```bash
# Find the last good commit
git log --oneline -10

# Force-push that commit to HF
git push hf <COMMIT_SHA>:main --force
```

---

## Cost reference (as of 2025)

| Resource | Cost |
|---|---|
| HF Spaces CPU Basic (2 vCPU, 16 GB RAM) | Free |
| HF Spaces CPU Upgrade (8 vCPU, 32 GB RAM) | ~$0.03/hr |
| HF Spaces persistent storage (for ERA5 data) | $0.018/GB/month |
| ERA5 download via CDS API | Free (rate-limited) |

For a portfolio demo with synthetic data, the **free tier is sufficient**.
For live ERA5 data, attach a HF persistent storage volume to `/home/appuser/app/data`.
