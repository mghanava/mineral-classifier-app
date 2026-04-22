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
uv sync                # install dev dependencies locally
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
make clean     Remove containers and volumes
make help      Show all commands
```

## Project Structure

```
├── app.py                       # Streamlit dashboard
├── setup_dvc.py                 # DVC pipeline generator
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

