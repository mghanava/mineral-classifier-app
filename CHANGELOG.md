# Changelog

All notable changes to the Mineral Classifier App will be documented here. The format follows the principles of Keep a Changelog; version numbers use PEP 440-compatible Python development and release identifiers.

Record actual changes only. Use Added, Changed, Deprecated, Removed, Fixed and Security headings as applicable. Omit empty categories. Keep upcoming plans in `ROADMAP.md`, not here. Move Unreleased entries into a dated version section when the project owner approves a milestone or release. Do not invent release dates, test results or completed features.

## [Unreleased]

No implementation changes recorded yet.

## [0.1.0]

Original research-oriented application; baseline summary based on existing project documentation and prior repository review. Release date and exact baseline commit are pending verification in Milestone 0.

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

These notes describe the starting point, not independently verified test outcomes. See the planned baseline report at `docs/baseline_v0.1.0.md` once it has been created.
