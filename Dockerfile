# Stage 1: Builder
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and install dependencies
COPY pyproject.toml ./
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels .

# Stage 2: Runtime (smaller, faster)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 1000 appuser \
    && mkdir -p /home/appuser /app/results /app/.dvc/cache \
    && chown -R appuser /app /home/appuser \
    && chmod -R 777 /app/results /app/.dvc/cache

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY pyproject.toml ./
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment
ENV PYTHONPATH=/app
ENV PATH="/usr/bin:${PATH}"

USER appuser
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]