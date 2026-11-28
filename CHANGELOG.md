# Changelog

All notable changes to the Mineral Classifier App will be documented here. The format follows the principles of Keep a Changelog; version numbers use PEP 440-compatible Python development and release identifiers.

Record actual changes only. Use Added, Changed, Deprecated, Removed, Fixed and Security headings as applicable. Omit empty categories. Keep upcoming plans in `ROADMAP.md`, not here. Move Unreleased entries into a dated version section when the project owner approves a milestone or release. Do not invent release dates, test results or completed features.

## [Unreleased]

### Added

- View verified drift findings for each method, explore a raw feature with shared histogram bins, and open the original diagnostics, report and recorded permutation settings.
- Compare two selected learning cycles using verified classification scores, same-cohort calibration errors or realized dataset growth. Show missing and historical evidence, gate charts on compatible schemas and metric definitions, and preserve each cycle's drift method context.
- Compare verified final-test classification and calibration scores before and after calibration, and view incoming-batch count matrices and per-class rates for either verified probability family. Inspect CV fold progress and the recorded schedule for the fresh full-development refit.
- Explore one selected population in a shared XY or 3D view with stored links, display filters, observation details and verified prediction overlays. Download the original producer CSV and full or filtered enriched results, and open original figures through the diagnostic selector.
- Read selected raw spatial populations and verified membership with immutable float64 observations and stored topology. Add a fresh, cycle-scoped evidence gate for positional prediction joins, display filters and enriched prediction exports.
- Add an Experiment overview with verified headline metrics, an interactive learning trajectory, realized base growth and a bounded cycle summary. Inspect a cycle in the existing diagnostic explorer and retain mode, cycle and section across navigation.
- Add an explicit Results evidence refresh that verifies scoped DVC records, local content and scientific bindings. Distinguish availability, lock-matching bytes and correspondence with applied settings and workspace sources.
- Add typed scientific summary readers for realized populations, final-test and incoming-batch metrics, model and calibration metadata and per-cycle completeness.
- Show Run phases, scientific stage observations and scoped execution outcomes, including explicit unchanged skips. Keep a bounded technical log and download for the latest session action.
- Guide configuration through five Basic settings and 28 Advanced fields across seven groups. Keep complete Expert YAML and all three views bound to the same session draft.
- Add a Home page with the Quick demo entry point, the measured M2.5 runtime reference and local workspace availability.
- Retain the configuration draft, preset, preview, selected Results cycle and section, and last execution outcome within the dashboard session.

### Changed

- Use a native scientific-blue light and dark theme. Clarify applied settings, draft review, observed Run phases and the scope of Results evidence.
- Improve 3D framing and align spatial probability formats across colorbars, hover and the inspector. Move the batch probability selector above both derived views and give calibration optimizer figures a wider layout.
- Make raw drift histograms legible with paired bars and a pattern for incoming observations. Show numerical comparison-bar labels, warm orange before calibration and blue after calibration.
- Emphasize Results coverage and headline metrics, report same-cohort ECE changes in percentage points, and add a verified partition composition view alongside base totals.
- Show the applied and draft status, preset provenance and measured CPU references beside explicit validation and Apply. Show separate finding counts and keep blocking errors visible.
- Replace seven eager tabs with native top navigation for Home, Configure, Run and Results. Configure retains explicit validation and Apply. Run owns all full and targeted execution actions.
- Browse one existing diagnostic at a time across six scientific sections. Identify bootstrap, input base, incoming batch and post-append snapshots. Label the selected scope with its checked correspondence or unverified status.

### Fixed

- Present one Results evidence and execution disclosure and use singular validation finding labels when appropriate.
- Label validation and across-cycle figures with Macro F0.5. Place overconfidence below the identity diagonal and underconfidence above it in reliability diagrams, and label the secondary axis with the sum of sample weights per bin. Metric calculations, serialized keys and artifact filenames are preserved.

## [0.2.0.dev2] - 2026-11-25

Configuration management is complete for the synthetic workflow, with 336 passing regression tests. Quick demo and Standard completed all 14 stages of their two-cycle CPU pipelines in 74.31 s and 101.66 s, respectively. Quick demo matched the retained M1.7 configuration and scientific outputs exactly for both cycles. Research matched the canonical defaults and generated an equivalent DVC definition. All three preset budgets and versions were retained after verification. Preset budgets describe computational effort. Predictive performance depends on the data and evaluation protocol. See the [configuration verification report](docs/configuration_verification.md).

### Added

- Inventory and classify all 88 configurable parameters by proposed Basic, Advanced, Expert and Internal visibility. Document their stage consumers, defaults and compatibility. See the [configuration design](docs/configuration_design.md).
- Pin the v0.2.0.dev1 scientific defaults in `config/defaults.yaml`, separate from the active experiment. Centralize static validation with path-specific errors, warnings and informational findings.
- Distribute version-1 Quick demo, Standard and Research presets. Resolve new experiments with explicit overrides and import legacy YAML with documented runtime fallbacks.
- Add the `python -m scripts.configure` CLI for listing presets, validating, previewing, comparing and exporting configuration. Publish validated YAML atomically and protect against stale previews.

### Changed

- Add an expert YAML draft, read-only **Validate & Preview**, and explicit **Apply configuration** to Parameters. Full and targeted dashboard runs regenerate DVC definitions from the applied configuration. Equivalent configurations generate equivalent definitions.

### Fixed

- Prevent another dashboard session or cooperative CLI export from changing the applied configuration during a dashboard reproduction.
- Apply each fitted isotonic map to its matching multiclass logit column. Normalize positive rows by their actual sum and report undefined zero-mass rows. Existing fitted isotonic artifacts remain readable, but affected reports and predictions need regeneration.

### Known limitations

- The workflow uses synthetic data and random stratified validation. Generalization to unseen regions remains unverified.
- Full-budget Research execution and container deployment remain unverified.
- Direct filesystem edits and direct DVC commands require a stable active configuration during execution. The dashboard lock protects cooperating writers in the shared workspace.
- The dashboard retains seven tabs and an expert YAML editor. The [roadmap](ROADMAP.md) tracks guided Basic, Advanced and Expert configuration for `0.2.0.dev3` and user-uploaded geological data for `0.2.0.dev4`.

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
