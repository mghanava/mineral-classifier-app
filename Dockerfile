# Stage 1: Builder
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel AS builder

WORKDIR /app

# System build deps (builder only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies Layer (cached if unchanged) ----
# Copy only the files needed to build the wheels to leverage caching
COPY pyproject.toml ./
COPY src ./src

# Build wheels for the project and ALL its dependencies.
# This is the slowest step. It is only re-run if pyproject.toml or src/ changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-cache-dir --wheel-dir /app/wheels .

# ---- Source code (not cached for deps) ----
# Copy the rest of the project. Changes here won't invalidate the wheel cache.
COPY . .

# Stage 2: Runtime
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# System runtime deps (only what’s needed at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/results /app/.dvc/cache \
    && chown -R appuser:appuser /app /home/appuser

# Copy wheels from builder and install them.
# Using --no-deps is fast because all dependencies are already in /wheels.
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/* \
    && rm -rf /wheels

# Copy application code from the final state of the builder stage
COPY --from=builder --chown=appuser:appuser /app /app

# Env
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
# Make sure pip-installed CLI tools are found
ENV PATH="/home/appuser/.local/bin:${PATH}"

USER appuser
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
