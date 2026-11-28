# Mineral Deposit Classification

A Streamlit workbench for synthetic spatial mineral classification with GCN and GAT models. Use Home, Configure, Run and Results to define a DVC-managed experiment and inspect its graphs, calibrated predictions and drift evidence. The current development version is `0.2.0.dev3`.

## Historical demo

[![Original prototype video](https://img.youtube.com/vi/TZHFRYZLA9k/maxresdefault.jpg)](https://youtu.be/TZHFRYZLA9k)

## How It Works

1. **Bootstrap**: Generate labeled synthetic observations, make stratified partitions, and construct independent graphs for each partition using KNN, a distance threshold, or a percentile.
2. **Train & Evaluate**: Train GNN models (GCN/GAT) and calibrate output probabilities using techniques like temperature scaling, isotonic regression, or Platt scaling.
3. **Learning Cycles**: Generate synthetic batches, predict with the calibrated model, analyze drift (MMD, energy distance, Wasserstein), and append the simulator-labeled observations for the next cycle.
4. **Performance Analysis**: Compare model performance across learning cycles.

## Install and start

### From a source checkout

Use [uv](https://docs.astral.sh/uv/) with a supported Python version (the project declares Python 3.11 or newer). Run these commands from the repository root so Streamlit loads the project theme and the dashboard uses the intended workspace.

```fish
git clone https://github.com/mghanava/mineral-classifier-app.git
cd mineral-classifier-app
git switch develop/v0.2.0
uv sync --locked
uv run --frozen --no-sync streamlit run app.py
```

Open [localhost:8501](http://localhost:8501). The local Quick demo preset requests CPU execution, including on hosts without a dedicated GPU. The lock still uses the declared CUDA 12.6 PyTorch source where applicable.

### Optional container startup

The supplied Compose service uses the NVIDIA runtime. It requires Docker with Compose, an NVIDIA Container Toolkit setup and a compatible NVIDIA driver. Container deployment was outside the dev3 verification.

```fish
make run
make logs
make stop
```

`make run` builds and starts the service at [localhost:8501](http://localhost:8501). Results and the DVC cache use separate named volumes. `make stop` stops the service while keeping those volumes.

## Quick demo journey

1. **Home** shows the applied configuration and any existing workspace results. **Configure Quick demo** highlights the reduced CPU preset. **Configure an experiment** opens the same editor for a custom experiment.
2. In **Configure**, select a preset and click **Load preset into draft**, or customize the current draft. **Basic** shows model, device, cycle count and the two observation targets. **Advanced** groups scientific settings by purpose. **Expert** edits the same draft as complete YAML. Use **Validate & Preview** to review findings and differences, then **Apply configuration** to update `params.yaml`. Preview again if the draft or active file changes. The draft and selected preset survive navigation within the session.
3. In **Run**, review the applied settings and choose **Run full experiment** or open **Advanced stage execution** for a targeted action. The dashboard validates the applied file and regenerates DVC definitions before reproduction. Follow the observed phase and stage. **Technical logs** contains the bounded command-output tail and the latest session log download.
4. In **Results**, use **Refresh result evidence** to check the applied configuration and workspace artifacts. **Experiment overview** shows the latest complete matching learning cycle, population-specific trajectories and realized base growth. Use **Inspect cycle** to open the **Cycle explorer**. In **Data and spatial graphs**, select an input base, partition, fold, incoming batch or appended snapshot. In **Spatial predictions**, explore verified class predictions, probabilities, entropy and correctness. Both sections offer equal-scale XY and physical-aspect 3D views, an observation ID picker and display filters. Choose **Original diagnostics** for stored figures and the standalone prediction CSV. Open **Prediction downloads** for full and filtered enriched CSVs after alignment verifies.
5. In **Model performance and training**, compare published final-test scores before and after calibration or inspect incoming-batch counts by probability family. Use **Training** for the CV fold figures and deployed refit schedule. **Probability calibration** compares ECE and MCE on the same final-test cohort and gives access to the saved reliability and optimizer diagnostics.
6. In **Distribution drift**, review per-method permutation findings for the verified input base and incoming batch. Select a raw feature, saved scientific figure, technical settings or the original report. Use **Compare two cycles** to view paired classification scores, calibration errors or realized dataset growth. Each result retains its population and evidence context. Cross-cycle differences do not establish learning improvement.

The unmodified Quick demo and Standard reference presets completed their two-cycle CPU pipelines in 74.31 s and 101.66 s on the verification host. Both retain their approved budgets. Presets set computational budgets, and predictive performance depends on the data and evaluation protocol. Research resolves to the full canonical defaults and has been checked through static validation and DVC generation. [Verification report](docs/configuration_verification.md)

The dashboard follows the system theme by default. Use the Streamlit menu to select Light or Dark. Results reports experiment correspondence, selected diagnostic evidence and verified spatial prediction alignment at their own scopes. [Dashboard review and legacy diagnostic routes](docs/ui_ux_design.md#m36-application-wide-visual-refinement)

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

## Current scope and limitations

- The verified workflow uses synthetic data and random stratified validation. Predictive performance in unseen geological regions remains unverified. Evaluation populations change across cycles, so the comparison views provide descriptive context. Fixed-holdout analysis, user data ingestion and independent inference remain planned.
- Spatial dependence limits interpretation of drift permutation p-values. Quick demo budgets 19 permutations per test, giving coarse p-value increments of about 0.0526. Drift detection alone does not establish predictive degradation.
- Full-budget Research execution and container deployment remain unverified. The sampled Chromium review found no horizontal overflow at 1920, 1280 and 390 px. Native tables may still scroll internally, and saved scientific figures retain their original labels and colors. [Dev3 limitations](CHANGELOG.md#dev3-limitations) · [Roadmap](ROADMAP.md)

## Developer setup

```fish
make install  # sync the lock with development tools
make hooks    # install pre-commit hooks
make lint
make test
```

Use the source startup command above for local development, or `make dev` for the bind-mounted Compose service. Use `make rebuild` when the development image needs rebuilding.

### Reset Results

```bash
make reset   # wipe generated results, keep DVC cache
```

`make reset` deletes generated files under `results/` and keeps the DVC object cache. DVC can restore matching cached outputs and may rerun stages when a match is unavailable. `make clean` also removes cached data and container volumes. Preserve any outputs you need before running either command.

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
make install   Sync the locked local environment with development tools
make run       Build and start the Compose service
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
├── .streamlit/config.toml       # Native light and dark theme
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
├── results/                     # Generated pipeline outputs excluded from Git
└── src/
    ├── models/                  # GNN architectures (GCN, GAT)
    ├── stages/                  # Pipeline stages
    └── utilities/               # Shared utilities
```

## Tech Stack

[Streamlit](https://streamlit.io/) - [PyTorch](https://pytorch.org/) - [PyTorch Geometric](https://pyg.org/) - [DVC](https://dvc.org/) - [uv](https://docs.astral.sh/uv/) - [Ruff](https://docs.astral.sh/ruff/) - [Docker](https://www.docker.com/)
