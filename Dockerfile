# RadFrac - fracture screening (Streamlit UI + FastAPI) in one container.
#
#   docker build -t radfrac:latest .
#   docker run -p 8501:8501 -p 8000:8000 radfrac:latest
#
# UI  -> http://localhost:8501      API docs -> http://localhost:8000/docs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    MPLBACKEND=Agg \
    RADFRAC_DATA_ROOT=/data \
    RADFRAC_SPLIT_DIR=/data/splits_grouped \
    RADFRAC_DB_PATH=/app/runtime/radfrac.db \
    RADFRAC_CASE_STORE=/app/runtime/cases

# libgomp: OpenMP runtime needed by LightGBM / XGBoost
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --upgrade pip && pip install -r requirements-docker.txt

# Application code
COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/
COPY .streamlit/ .streamlit/
COPY pytest.ini docker/entrypoint.sh ./

# Trained artifacts + metrics/figures the UI reads (no raw data, no patient cases)
COPY outputs/models/ outputs/models/
COPY outputs/metrics/ outputs/metrics/
COPY outputs/figures/ outputs/figures/

RUN sed -i 's/\r$//' entrypoint.sh && chmod +x entrypoint.sh \
 && useradd --create-home --uid 1000 radfrac \
 && mkdir -p /app/runtime/cases /data/splits_grouped \
 && chown -R radfrac:radfrac /app /data
USER radfrac

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fs http://localhost:8000/health || exit 1

# "serve" (default) runs UI + API; any other command is executed as-is,
# e.g.  docker run radfrac python -m scripts.train_boosting
ENTRYPOINT ["./entrypoint.sh"]
CMD ["serve"]
