# Mineral Deposit Classification

A web-based dashboard for synthetic mineral classification using graph neural networks and a DVC-managed learning pipeline. The current development version is `0.2.0.dev2`.

## Demo

[![Demo Video](https://img.youtube.com/vi/TZHFRYZLA9k/maxresdefault.jpg)](https://youtu.be/TZHFRYZLA9k)

## How It Works

1. **Bootstrap**: Generate labeled synthetic observations, make stratified partitions, and construct independent graphs for each partition using KNN, a distance threshold, or a percentile.
2. **Train & Evaluate**: Train GNN models (GCN/GAT) and calibrate output probabilities using techniques like temperature scaling, isotonic regression, or Platt scaling.
3. **Learning Cycles**: Generate synthetic batches, predict with the calibrated model, analyze drift (MMD, energy distance, Wasserstein), and append the simulator-labeled observations for the next cycle.
4. **Performance Analysis**: Compare model performance across learning cycles.

## Container Quick Start

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

1. **Home** shows the applied configuration and any existing workspace results. **Configure Quick demo** highlights the reduced CPU preset. **Configure an experiment** opens the same editor for a custom experiment.
2. In **Configure**, select a preset and click **Load preset into draft**, or customize the current draft. **Basic** shows model, device, cycle count and the two observation targets. **Advanced** groups scientific settings by purpose. **Expert** edits the same draft as complete YAML. Use **Validate & Preview** to review findings and differences, then **Apply configuration** to update `params.yaml`. Preview again if the draft or active file changes. The draft and selected preset survive navigation within the session.
3. In **Run**, review the applied settings and choose **Run full experiment** or open **Advanced stage execution** for a targeted action. The dashboard validates the applied file and regenerates DVC definitions before reproduction. Follow the observed phase and stage. **Technical logs** contains the bounded command-output tail and the latest session log download.
4. In **Results**, use **Refresh result evidence** to check the applied configuration and workspace artifacts. **Experiment overview** shows the latest complete matching learning cycle, population-specific trajectories and realized base growth. Use **Inspect cycle** to open the shared **Cycle explorer**, or switch modes directly to browse the existing scientific sections. The Data section includes the input base, incoming batch and final post-append snapshot. Historical diagnostics retain their evidence labels.

Quick demo and Standard completed their two-cycle CPU pipelines in 74.31 s and 101.66 s on the verification host. Both retain their approved budgets. Presets set computational budgets, and predictive performance depends on the data and evaluation protocol. Research resolves to the full canonical defaults and has been checked through static validation and DVC generation. [Verification report](docs/configuration_verification.md)

### Configuration CLI

Run these commands from the repository with the project environment installed.

```fish
uv run --frozen --no-sync python -m scripts.configure list
uv run --frozen --no-sync python -m scripts.configure validate params.yaml
uv run --frozen --no-sync python -m scripts.configure resolve quick_demo --against params.yaml
uv run --frozen --no-sync python -m scripts.configure resolve standard --output experiment.yaml
uv run --frozen --no-sync python -m scripts.configure import legacy.yaml --output resolved.yaml
```

Resolution and import preview to stdout by default. Use `--overrides` with a partial YAML mapping for custom settings. Use `--output` for an explicit export, and add `--overwrite` when replacing an existing file. For direct DVC use, regenerate with `uv run --frozen --no-sync python -m scripts.setup_dvc` after applying topology changes. Keep the active file fixed during a direct DVC run. Dashboard reproduction blocks cooperating configuration writers in its workspace until the run exits. [Configuration design](docs/configuration_design.md#cli-and-streamlit)

### Current scope

The verified workflow uses synthetic data and random stratified validation. Generalization to unseen regions, full-budget Research execution and container deployment remain unverified. The dashboard supports guided configuration, DVC-backed result correspondence and selected-diagnostic browsing. Cross-cycle curves describe changing evaluation populations. User data ingestion and independent inference are planned. [Roadmap](ROADMAP.md)

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
| Edit Python source | Save to trigger a development-mode reload |
| Change an experiment | Preview and apply it in Configure |
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

Run shows **Failed** with the exit status when an aborted command returns a nonzero status. A stage killed mid-way leaves partial outputs — DVC detects the mismatch and re-runs that stage on the next `dvc repro`. To drop the partial outputs too, run `make reset` after `make abort`. No image rebuild is needed: the helper is copied into the running container on the fly.

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
├── app.py                       # Streamlit entry point and native page registry
├── dashboard/                   # Views, session state, workspace readers and execution
├── scripts/
│   ├── configure.py             # Configuration CLI
│   ├── setup_dvc.py             # DVC pipeline generator
│   └── abort_pipeline.py        # Stop a running pipeline (`make abort`)
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yaml          # App user config
├── docker-compose.dev.yaml      # Developer overlay
├── Makefile                     # Convenience commands
├── pyproject.toml               # Dependencies and tooling config
├── .pre-commit-config.yaml      # Pre-commit hooks
├── dvc.yaml / dvc.lock          # DVC pipeline definition and state
├── config/                      # Canonical defaults and versioned presets
├── params.yaml                  # Applied pipeline configuration
├── results/                     # Pipeline outputs (DVC tracked)
└── src/
    ├── models/                  # GNN architectures (GCN, GAT)
    ├── stages/                  # Pipeline stages
    └── utilities/               # Shared utilities
```

## Tech Stack

[Streamlit](https://streamlit.io/) - [PyTorch](https://pytorch.org/) - [PyTorch Geometric](https://pyg.org/) - [DVC](https://dvc.org/) - [uv](https://docs.astral.sh/uv/) - [Ruff](https://docs.astral.sh/ruff/) - [Docker](https://www.docker.com/)
