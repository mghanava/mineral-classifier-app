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
    git gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a default non-root user. The entrypoint will adjust its UID/GID at runtime.
# A GID of 1000 already exists in the base image, so we use 1001 to avoid conflicts.
RUN groupadd -g 1001 appgroup \
    && useradd -m -u 1001 -g appgroup -s /bin/bash appuser \
    && mkdir -p /app/results /app/.dvc/cache \
    && chown -R appuser:appgroup /app /home/appuser

# Copy wheels from builder and install them.
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/* \
    && rm -rf /wheels

# Copy application code from the final state of the builder stage
COPY --from=builder --chown=appuser:appgroup /app /app

# Copy and set up the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Env
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
# Make sure pip-installed CLI tools are found
ENV PATH="/home/appuser/.local/bin:${PATH}"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
