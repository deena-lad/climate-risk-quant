# ─────────────────────────────────────────────────────────────────────────────
# Climate Risk Quantification — Dockerfile
# Target: Hugging Face Spaces (Streamlit SDK)
#
# Build stages:
#   builder  — installs all Python deps into a venv (cached layer)
#   runtime  — copies venv + app code only (no build tools in final image)
#
# HuggingFace Spaces requirements:
#   - Must expose port 7860
#   - Entry process must not run as root (HF runs as UID 1000)
#   - $HOME must be writable (for ~/.cdsapirc and MLflow local store)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile geo libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        libeccodes-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first — Docker layer caching means this layer
# is only rebuilt when requirements.txt changes, not on code changes.
COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libraries only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal32 \
        libgeos3.11.1 \
        libproj25 \
        libspatialindex6 \
        libeccodes2 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user (UID 1000 matches HuggingFace Spaces default)
RUN useradd -m -u 1000 appuser
WORKDIR /home/appuser/app

# Copy application code
COPY --chown=appuser:appuser src/       ./src/
COPY --chown=appuser:appuser app/       ./app/
COPY --chown=appuser:appuser pyproject.toml .

# Create writable data directories and MLflow store
RUN mkdir -p \
        data/raw data/interim data/processed \
        mlruns \
    && chown -R appuser:appuser /home/appuser/app

# Streamlit config — must live at ~/.streamlit/config.toml
USER appuser
RUN mkdir -p /home/appuser/.streamlit
COPY --chown=appuser:appuser .streamlit/config.toml /home/appuser/.streamlit/config.toml

# Install the src package in editable mode (so imports resolve)
RUN pip install --no-cache-dir -e . --no-deps

# Health check — confirms Streamlit is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

EXPOSE 7860

# HuggingFace Spaces expects the server on 0.0.0.0:7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
