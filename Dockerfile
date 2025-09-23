# Stage 1: Builder
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel AS builder
WORKDIR /app

# System build deps (builder only) - Combined into single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ---- Dependencies Layer (cached if unchanged) ----
# Copy requirements first for better caching
COPY pyproject.toml ./

# Pre-install common heavy dependencies that rarely change
# This creates a separate cache layer for the most stable dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps \
    torch torchvision torchaudio \
    numpy pandas scikit-learn \
    streamlit plotly \
    && pip cache purge

# Copy source and build project wheels
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/setuptools \
    pip wheel --no-cache-dir --wheel-dir /app/wheels . \
    && pip cache purge

# Copy remaining files (do this last to avoid cache invalidation)
COPY . .

# Stage 2: Runtime - Use slimmer base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS runtime
WORKDIR /app

# Combined system setup in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && useradd -m -u 1000 appuser \
    && mkdir -p /app/results \
    && chown -R appuser:appuser /app /home/appuser

# Install wheels efficiently
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/* \
    && rm -rf /wheels /root/.cache/pip

# Copy application code and set permissions in one step
COPY --from=builder --chown=appuser:appuser /app /app

# Set up DVC directories with proper permissions
RUN mkdir -p /app/.dvc/tmp /app/.dvc/cache /app/.dvc/plots \
    && chown -R appuser:appuser /app/.dvc \
    && chmod -R 755 /app/.dvc

# Environment setup
ENV PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]