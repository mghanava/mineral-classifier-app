# Changelog

All notable changes to the Mineral Classifier App will be documented here. The format follows the principles of Keep a Changelog; version numbers use PEP 440-compatible Python development and release identifiers.

Record actual changes only. Use Added, Changed, Deprecated, Removed, Fixed and Security headings as applicable. Omit empty categories. Keep upcoming plans in `ROADMAP.md`, not here. Move Unreleased entries into a dated version section when the project owner approves a milestone or release. Do not invent release dates, test results or completed features.

## [Unreleased]

### Added

- Verify production configuration workflows for Quick demo and Standard on CPU, Research default equivalence, and legacy CLI and dashboard compatibility. Record the measured runtimes and artifact checks in the [configuration verification report](docs/configuration_verification.md).
- Add shared preset and legacy configuration resolution, scientific YAML diffs and atomic export through `python -m scripts.configure`.
- Distribute version-1 Quick demo, Standard and Research presets with a read-only application API and shared static validation. Quick demo matches the verified M1.7 Experiment A settings, and Research inherits the complete canonical configuration.
- Pin the v0.2.0.dev1 configuration in `config/defaults.yaml` and add a shared static validator for the complete resolved synthetic configuration. Return path-specific errors, warnings and informational findings while preserving active YAML, scientific defaults and existing pipeline behavior.

### Changed

- Make Parameters use an editable draft, validation and explicit Apply. Full and targeted dashboard runs regenerate DVC definitions from the applied configuration before reproduction.

### Fixed

- Prevent another dashboard session or cooperative CLI export from changing the applied configuration during a dashboard reproduction.
- Apply each fitted isotonic map to its matching multiclass logit column. Normalize positive rows by their actual sum and report undefined zero-mass rows. Existing fitted isotonic artifacts remain readable, but affected reports and predictions need regeneration.

## [0.2.0.dev1]

Scientific stabilization is complete for the verified reduced synthetic workflow. The full suite has 147 passing regression tests. Production-path checks covered CPU GCN and GAT, host CUDA GCN, CPU loading of the GPU-trained bundle, seeded recomputation, byte-for-byte restoration of 124 DVC artifacts, and all seven dashboard tabs. See the [end-to-end verification report](docs/end_to_end_verification.md).

### Added

- Verified reduced production DVC runs for CPU GCN, CPU GAT and CUDA GCN. Recorded real cache restoration, seeded CPU recomputation, populated dashboard rendering and CPU loading of the CUDA-trained bundle in the [end-to-end report](docs/end_to_end_verification.md).

- Deterministic CPU regression tests for GCN and GAT forward passes, graph edges, dataset partition disjointness, feature and label alignment, and induced subgraphs.
- Isolated DVC pipeline generation checks and a Streamlit rendering test for missing artifacts.
- Standalone fold-local preprocessing for RobustScaler, StandardScaler, MinMaxScaler and disabled scaling, with immutable raw data and partition contracts, validated model-bound JSON and deterministic CPU coverage.
- Standalone independent spatial graph construction with KNN, fixed-radius and per-node percentile rules, bounded or unit edge affinities, explicit self-loops, unique-edge statistics and CPU regression coverage.

### Changed

- Seed synthetic prediction by default and record reproducible model, calibration and drift streams in artifact metadata. Preserve the configured partition seed and explicit unseeded mode.
- Use the locked project Ruff for pre-commit and project checks. Register an opt-in Pytest filter for faster local runs while keeping integration coverage in the default suite.

- Deploy a fresh model and scaler trained on the complete development population after independent CV. Record fold best epochs, stopping epochs, learning rates and final refit provenance, and verify the new bundle in evaluation, calibration, prediction and drift.

- Build independent training, validation, calibration, test and prediction graphs from canonical coordinates. Keep the fifth-percentile default, use bounded affinities with a 460-unit synthetic scale and bind the protocol to model and calibration artifacts. Legacy graph settings and bundles require explicit migration.

- Store canonical float64 synthetic observations and stable sample IDs across cycles. Evaluation and prediction verify and reuse model-bound preprocessing. Calibration records its fitted split and exact bundle digests.
- Restored drift analysis using raw marginal features and the same frozen model transform for both populations' multivariate statistics and plots. Drift records dataset, checkpoint and representation provenance.
- Updated preprocessing-related DVC dependencies and verified invalidation and sidecar restoration with a temporary fixture pipeline. Legacy generated data and checkpoints require regeneration.
- Added a scientific stabilization milestone with seven work items before configuration management.
- Removed executable permissions from `AGENTS.md`, `ROADMAP.md` and `CHANGELOG.md`.

### Fixed

- Support explicit CPU or visible CUDA selection, and load CUDA-tagged graph and checkpoint storage on CPU before moving model inputs to the selected device.
- Report unsuccessful dashboard commands with their context and stop full reproduction when pipeline generation fails.

- Include base-cycle t-SNE diagnostics in DVC outputs, track shared calibration and model source dependencies, and keep performance aggregation within the declared cycle range. Verify selective invalidation and fixture cache restoration across two cycles.

- Render stored graph edges once per physical link and report unique-edge diagnostics, including for partitions with non-contiguous class labels.

- Use the selected Wasserstein statistic for both the observed comparison and its permutations.

- Initialize a fresh model for each cross-validation fold alongside its own optimizer, learning-rate scheduler and early stopper.
- Copy best-epoch weights independently of live CPU parameters and include the stopping epoch in fold diagnostics.
- Restore training mode before every epoch so dropout and batch normalization use training behavior after each validation pass. A three-epoch CPU test checks the model mode and gradient state of every forward pass.

### Removed

- Removed the obsolete tracked `dvc.lock`. The same file is preserved at the `v0.1.0` tag. DVC reproduction generates a new lockfile for the configuration it runs. Reduced production configurations were verified, and full default-budget reproduction is pending. Use a lock generated from the corresponding configuration. A smoke-test lock is valid only for its smoke configuration.

### Known limitations

- Validation uses random stratified splits, so generalization to unseen regions is unverified. Evaluation populations also change across learning cycles, limiting longitudinal comparisons.
- Percentile-based graph degree depends on batch size. The CV-informed final training duration remains a heuristic.
- Spatial dependence limits interpretation of drift permutation p-values. No accuracy improvement has been established by benchmarking.
- Full default-budget production reproduction and container deployment are unverified. User-uploaded data workflows and UI simplification remain future milestones.

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
