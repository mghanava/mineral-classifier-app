# Changelog

All notable changes to the Mineral Classifier App will be documented here. The format follows the principles of Keep a Changelog; version numbers use PEP 440-compatible Python development and release identifiers.

Record actual changes only. Use Added, Changed, Deprecated, Removed, Fixed and Security headings as applicable. Omit empty categories. Keep upcoming plans in `ROADMAP.md`, not here. Move Unreleased entries into a dated version section when the project owner approves a milestone or release. Do not invent release dates, test results or completed features.

## [Unreleased]

### Added

- Deterministic CPU regression tests for GCN and GAT forward passes, graph edges, dataset partition disjointness, feature and label alignment, and induced subgraphs.
- Isolated DVC pipeline generation checks and a Streamlit rendering test for missing artifacts.
- Standalone fold-local preprocessing for RobustScaler, StandardScaler, MinMaxScaler and disabled scaling, with immutable raw data and partition contracts, validated model-bound JSON and deterministic CPU coverage.
- Standalone independent spatial graph construction with KNN, fixed-radius and per-node percentile rules, bounded or unit edge affinities, explicit self-loops, unique-edge statistics and CPU regression coverage.

### Changed

- Store canonical float64 synthetic observations and stable sample IDs across cycles. Training publishes the winning fold's scaler with its weights, and evaluation and prediction verify and reuse that scaler. Calibration records its fitted split and exact bundle digests.
- Restored drift analysis using raw marginal features and the same frozen model transform for both populations' multivariate statistics and plots. Drift records dataset, checkpoint and representation provenance.
- Updated preprocessing-related DVC dependencies and verified invalidation and sidecar restoration with a temporary fixture pipeline. Legacy generated data and checkpoints require regeneration.
- Added a scientific stabilization milestone with seven work items before configuration management.
- Removed executable permissions from `AGENTS.md`, `ROADMAP.md` and `CHANGELOG.md`.

### Fixed

- Use the selected Wasserstein statistic for both the observed comparison and its permutations.

- Initialize a fresh model for each cross-validation fold alongside its own optimizer, learning-rate scheduler and early stopper.
- Copy best-epoch weights independently of live CPU parameters and include the stopping epoch in fold diagnostics. The existing `model.pt` now contains the restored weights of the independently trained fold with the lowest validation loss.
- Restore training mode before every epoch so dropout and batch normalization use training behavior after each validation pass. A three-epoch CPU test checks the model mode and gradient state of every forward pass.

## [0.1.0]

Original research-oriented application; baseline summary based on existing project documentation and prior repository review. The `v0.1.0` tag points to baseline commit `d00c2cd`.

### Added

- Synthetic mineral classification data generation and iterative learning workflows.
- Graph neural network classification using GCN and GAT architectures.
- Configurable spatial graph construction, model training and evaluation.
- Probability calibration and statistical data drift analysis.
- Streamlit interface for configuration, data, training, evaluation, prediction and diagnostics.
- DVC-based pipeline orchestration and Docker-based deployment configuration.

### Known limitations at baseline

- User workflows expose technical pipeline configuration and execution steps.
- General-purpose import of user-provided geological datasets is not yet established.
- CPU-only deployment and end-to-end reproducibility require verification.

See the [baseline report](docs/baseline_v0.1.0.md) for verified checks and limitations.
