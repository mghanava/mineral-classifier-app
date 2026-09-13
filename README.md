# Mineral Deposit Classification

A web-based dashboard for classifying mineral deposits using graph neural networks with an online learning pipeline managed by DVC.

## Demo

[![Demo Video](https://img.youtube.com/vi/TZHFRYZLA9k/maxresdefault.jpg)](https://youtu.be/TZHFRYZLA9k)

## How It Works

1. **Bootstrap**: Generate initial data samples and construct a graph (KNN, distance threshold, or percentile). The graph is split for cross-validation ensuring no data leakage.
2. **Train & Evaluate**: Train GNN models (GCN/GAT) and calibrate output probabilities using techniques like temperature scaling, isotonic regression, or Platt scaling.
3. **Online Learning Cycles**: Introduce new unlabeled data → predict with calibrated model → analyze drift (MMD, Energy Distance, Wasserstein) → integrate into training set → repeat.
4. **Performance Analysis**: Compare model performance across learning cycles.

## Quick Start

**Requirements:**
- Docker >= 23.0
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA Driver >= 560.28 (check with `nvidia-smi`, must support CUDA >= 12.6)

```bash
git clone https://github.com/mghanava/mineral-classifier-app.git
cd mineral-classifier-app
make run
```

Open [http://localhost:8501](http://localhost:8501)

### Using the Dashboard

1. **Parameters tab** - configure pipeline settings
2. **Generate dvc.yaml** - creates pipeline from your config
3. **Run Pipeline** - executes dvc repro (full or per-stage)

### Stop

```bash
make stop          # stop containers
make clean         # remove everything (containers, volumes, images)
```

## Developer Setup

**Additional requirement:** [uv](https://docs.astral.sh/uv/)

```bash
make install           # create venv + install all dependencies
make hooks             # install pre-commit hooks
make dev               # start in dev mode (bind mount, hot reload)
```

### Workflow

| Change | Action |
|---|---|
| Edit .py / params.yaml | Save - browser auto-refreshes |
| Add Python dependency | make rebuild |
| Change Dockerfile/compose | make rebuild |
| Something broken | make clean and make dev |

### Reset Results

```bash
make reset   # wipe generated results, keep DVC cache
```

`make reset` deletes everything under `results/` (preserving `.gitkeep`) but **keeps the DVC object cache**, so a subsequent run re-materializes cached outputs in seconds instead of retraining.

Use it to remove stale artifacts when lowering the cycle count — e.g. after running 5 cycles, change `cycles:` to 3 and run `make reset` to drop the now-unused cycle_4/cycle_5 folders. Then run the pipeline again; the bootstrap stage and cycles 1–3 are restored from cache, so nothing is retrained and cycle_0 is not regenerated.

This is intentionally distinct from `make clean`, which also deletes the DVC cache, volumes, and images for a true from-scratch state.

### Abort a Pipeline Run

```bash
make abort   # stop a running DVC pipeline run (keep the dashboard up)
```

Realized you made a mistake while a full or per-stage run is in progress? `make abort` stops the running pipeline (SIGINT for a clean stop, SIGKILL as a fallback after a few seconds) while leaving the Streamlit dashboard up, so you can fix `params.yaml` or a stage setting and re-run right away.

The dashboard will show "Command failed" for the aborted run. A stage killed mid-way leaves partial outputs — DVC detects the mismatch and re-runs that stage on the next `dvc repro`. To drop the partial outputs too, run `make reset` after `make abort`. No image rebuild is needed: the helper is copied into the running container on the fly.

### Track Experiments

```bash
make track
git add results.dvc .gitignore
git commit -m "Experiment: description"
```

## Make Commands

```
make run       Start the app
make stop      Stop the app
make logs      Follow container logs
make dev       Start in development mode
make rebuild   Rebuild without cleaning volumes
make shell     Open shell in container
make lint      Run ruff linter
make fmt       Run ruff formatter
make test      Run tests
make hooks     Install pre-commit hooks
make check     Run all pre-commit hooks
make track     Track results with DVC
make reset     Wipe generated results, keep DVC cache
make abort     Stop a running DVC pipeline run
make clean     Remove containers, volumes, artifacts, and build cache
make help      Show all commands
```

## Project Structure

```
├── app.py                       # Streamlit dashboard
├── scripts/
│   ├── setup_dvc.py             # DVC pipeline generator
│   └── abort_pipeline.py        # Stop a running pipeline (`make abort`)
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yaml          # App user config
├── docker-compose.dev.yaml      # Developer overlay
├── Makefile                     # Convenience commands
├── pyproject.toml               # Dependencies and tooling config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── dvc.yaml / dvc.lock          # DVC pipeline definition and state
├── params.yaml                  # Pipeline parameters
├── results/                     # Pipeline outputs (DVC tracked)
└── src/
    ├── models/                  # GNN architectures (GCN, GAT)
    ├── stages/                  # Pipeline stages
    └── utilities/               # Shared utilities
```

## Tech Stack

[Streamlit](https://streamlit.io/) - [PyTorch](https://pytorch.org/) - [PyTorch Geometric](https://pyg.org/) - [DVC](https://dvc.org/) - [uv](https://docs.astral.sh/uv/) - [Ruff](https://docs.astral.sh/ruff/) - [Docker](https://www.docker.com/)
