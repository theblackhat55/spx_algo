FROM python:3.11-slim

# System dependencies for LightGBM, XGBoost, hmmlearn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 libopenblas-dev make \
    && rm -rf /var/lib/apt/lists/*

# Timezone
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# ── Security: run as a non-root user ─────────────────────────────────────────
# Never run untrusted code or a long-lived service as root inside a container.
RUN groupadd --gid 1001 algo \
 && useradd  --uid 1001 --gid algo --shell /bin/bash --create-home algo

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create directories for persistent data and outputs, owned by algo user
RUN mkdir -p data/raw data/processed data/external output/signals output/models output/trades \
 && chown -R algo:algo /app

# Drop to non-root for all subsequent operations
USER algo

# Health check: verify latest_signal.json was updated within 25 hours
HEALTHCHECK --interval=5m --timeout=10s --retries=3 \
    CMD python -c "\
import os, time, sys; \
f = 'output/signals/latest_signal.json'; \
sys.exit(0 if os.path.exists(f) and (time.time() - os.path.getmtime(f)) < 90000 else 1)"

# Default entrypoint: run the scheduler (pipeline fires at 4:05 PM ET)
CMD ["python", "-m", "src.pipeline.scheduler"]
