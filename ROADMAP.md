# Mineral Classifier App — Development roadmap

## Vision

Transform the v0.1.0 research prototype into an accessible spatial machine learning application for geoscientists, data scientists and ML researchers. Preserve GCN/GAT, reproducible DVC workflows, synthetic experiments, calibration and drift analysis while separating ordinary user workflows from expert controls.

## Release strategy

The `v0.1.0` tag preserves the original application at `d00c2cd`. The development foundation is recorded at `55aeacf`. Develop on `develop/v0.2.0`, keep `main` stable, and use the existing `pyproject.toml` and `uv.lock`. Development version labels below identify milestones. The current project version is `0.2.0.dev2`. The owner approves version changes, commits, tags and merges. Check off milestones and work items after verification.

## v0.2.0 — Accessible classification application

### Milestone 0 — Development foundation (`0.2.0.dev0`)

- [x] Verify Git status, branch, tags and current version; preserve original baseline with owner approval.
- [x] Inspect architecture, entry points, configuration, DVC dependencies, Docker, and existing test tooling.
- [x] Run lightweight existing tests and static checks; record failures or unavailable tooling.
- [x] Identify required artifacts, CPU compatibility obstacles and reproducibility risks.
- [x] Create `docs/baseline_v0.1.0.md` with execution instructions, verified functionality, test outcomes and limitations.
- [x] Establish project instructions, roadmap and changelog; update development version and lockfile if authorized.

**Acceptance:** Baseline is documented, Git preservation is verified, and no scientific pipeline or application behavior has changed. Full training is not required without approval; unverified behavior is clearly identified.

### Milestone 1 — Scientific correctness and regression tests (`0.2.0.dev1`)

- [x] **M1.1 — Automated regression tests.** Added deterministic CPU checks for GCN and GAT forward passes, edge validity, partition disjointness, feature and label alignment, and induced subgraphs. Added DVC generation tests for one and two cycles and a Streamlit empty-results test using temporary directories.
- [x] **M1.2 — Cross-validation fold independence.** Each fold starts with a fresh model, optimizer, scheduler and early stopper. CPU regression coverage checks fold-specific metrics and checkpoint selection.
- [x] **M1.2a — Correct training and evaluation mode transitions.** Restore training mode at the start of every epoch. A three-epoch CPU regression test verifies that training and gradient-free validation alternate in the correct modes.
- [x] **M1.3 — Preprocessing and inference consistency.**
  - [x] **M1.3a — Design audit.** Traced scaling, graph and artifact interfaces and documented the proposed cutover in [preprocessing design](docs/preprocessing_design.md).
  - [x] **M1.3b — Implementation.**
    - [x] **M1.3b-i — Preprocessing utilities and artifact contracts.** Add validated raw and partition contracts, fold-local scaler fitting, frozen transforms and versioned JSON bound to model weights.
    - [x] **M1.3b-ii — Raw data, training and inference integration.** Preserve raw observations and IDs across cycles, fit per fold, publish the winning scaler with its model, and reuse it for calibration, test and prediction. Track the new artifact dependencies and suspend drift scoring until M1.3b-iii.
    - [x] **M1.3b-iii — Drift, DVC integration and lifecycle verification.** Restored drift with raw marginals and a shared frozen transform for multivariate diagnostics. CPU fixtures verified drift routing, DVC invalidation and sidecar restoration.
- [x] **M1.4 — Graph construction and leakage review.**
  - [x] **Graph design audit.** Traced connectivity, normalization, model inputs and visualization and documented the independent inductive protocol in [graph design](docs/graph_design.md).
  - [x] **M1.4-i — Standalone graph utilities.** Implemented and tested partition-local connectivity, explicit loop ownership, bounded or unit affinities, and unique-edge diagnostics.
  - [x] **M1.4-ii — Production graph integration.** Built and verified independent production graphs, explicit model loop weights, versioned graph bindings, stored-edge visualizations and DVC dependencies.
- [x] **M1.4a — Full development-pool retraining.** Refit a fresh model and scaler on the verified development graph for the rounded median CV best epoch. Replay a representative fold's learning rates without validation. Bind the deployed weights and preprocessing to the development population and refit diagnostics.
- [x] **M1.5 — DVC dependencies and artifact declarations.** Audited stage inputs and outputs, corrected missing diagnostics and source dependencies, and scoped performance aggregation to declared cycles. Verified selective invalidation and cache restoration in isolated fixtures. See the [DVC integrity audit](docs/dvc_integrity.md) for production verification requirements.
- [x] **M1.6 — Reproducibility, device portability and execution reliability.** Preserve split membership, record reproducible stage seeds, support explicit CPU and portable checkpoint loading, propagate command failures, and align Ruff and pre-commit. See the [reproducibility guide](docs/reproducibility.md) for verification and remaining platform restrictions.
- [x] **M1.7 — Small end-to-end pipeline verification.** Verified two-cycle GCN and one-cycle GAT on CPU, byte-for-byte DVC restoration, independent seeded replay and all seven dashboard tabs. A separate host run verified CUDA GCN and CPU portability. See the [verification report](docs/end_to_end_verification.md).

**Final checkpoint policy.** Independent CV folds retain best-epoch and early-stopping diagnostics. The stage then saves the fresh full-development model after its fixed CV-informed duration. The separate calibration and test graphs remain reserved. See the [final-model strategy](docs/graph_design.md#m14a-development-pool-final-model).

**Acceptance met.** All 147 regression tests passed. Reduced production runs verified CPU GCN and GAT, host CUDA GCN, seeded recomputation, DVC cache restoration, and the populated dashboard. See the [verification report](docs/end_to_end_verification.md).

**Release boundary.** The obsolete tracked DVC lockfile was removed and remains available at `v0.1.0`. Generate a new lock by reproducing the corresponding configuration. The full default-budget run and container deployment remain unverified. A reduced-run lock applies only to the reduced configuration. The [changelog](CHANGELOG.md) records the remaining scientific and workflow limitations.

### Milestone 2 — Configuration management (`0.2.0.dev2`)

- [x] **M2.1 — Configuration inventory and design.** Inventory and classify all 88 configurable parameters. Document their consumers, conditional behavior and compatibility in the [configuration design](docs/configuration_design.md).
- [x] **M2.1a — Multiclass isotonic calibration.** Verified and corrected per-class inference, row normalization and handling of undefined rows. Confirmed existing artifact compatibility.
- [x] **M2.2 — Centralized configuration validation.** Separate canonical defaults from the active configuration and validate the resolved configuration with structured errors, warnings and informational findings. See the [validation contract](docs/configuration_design.md#m22-static-validation).
- [x] **M2.3 — Experiment presets.** Distribute and validate version-1 overlays for Quick demo, Standard and Research against the canonical defaults. See the [preset contract](docs/configuration_design.md#m23-experiment-presets).
- [x] **M2.4 — Configuration resolution and integration.** Add preset resolution, explicit overrides, legacy YAML imports and a dedicated CLI. Integrate read-only previews and explicit dashboard Apply with atomic writes, stale-preview protection and DVC regeneration. Verify equivalent definitions for identical settings. See the [configuration integration contract](docs/configuration_design.md#m24-configuration-resolution-and-integration).
- [x] **M2.5 — Configuration integration and production verification.** Verified CLI and dashboard workflows, Research equivalence, and complete two-cycle CPU reproductions for Quick demo (74.31 s) and Standard (101.66 s). Prevented concurrent dashboard runs and configuration writes in the same workspace. See the [configuration verification report](docs/configuration_verification.md).

**Acceptance met.** All 336 regression tests passed. Users can configure a synthetic run by loading, previewing and applying a preset without editing YAML. Existing configurations can be imported, and static validation errors block application and dashboard execution. Quick demo matched M1.7 scientifically, Research matched the canonical defaults, and the preset budgets were retained.

**Release boundary.** The seven-tab dashboard and expert YAML editor remain in use. Guided configuration belongs to `0.2.0.dev3`, and geological data uploads belong to `0.2.0.dev4`. The verified workflow uses synthetic data and random stratified validation. Full-budget Research and container deployment remain unverified. Keep the active file stable during direct DVC runs. See the [release limitations](CHANGELOG.md#known-limitations).

### Milestone 3 — User interface redesign (`0.2.0.dev3`)

- [x] **M3.1 — Design and feasibility audit.** Audited the seven-tab dashboard, captured screens, scientific output contracts, spatial alignment, DVC provenance and locked Streamlit APIs. See the [UI/UX design and feasibility audit](docs/ui_ux_design.md).
- [ ] **M3.2 — Navigation and dashboard architecture.** Extract Home, Configure, Run and Results with shared state and artifact access.
- [ ] **M3.3 — Guided configuration.** Integrate presets, five Basic controls, grouped Advanced options, complete Expert YAML, structured validation and explicit Apply.
- [ ] **M3.4 — Execution experience.** Move full and targeted run actions to Run with applied settings, observed stage status, logs and the existing locking and failure handling.
- [ ] **M3.5a — Experiment overview and cycle navigation.** Show result correspondence, separate test and batch trajectories, realized dataset growth and compact cycle drill-through.
- [ ] **M3.5b — Spatial data and prediction explorer.** Inspect selected populations and graphs, with verified prediction overlays, sample details and downloads.
- [ ] **M3.5c — Classification, calibration and training results.** Present population-specific metrics, calibration diagnostics, selected CV fold plots and final refit context.
- [ ] **M3.5d — Drift and cross-cycle comparison.** Preserve drift diagnostics and support descriptive two-cycle inspection with cohort and method context.
- [ ] **M3.6 — UI verification and visual refinement.** Verify the supported user journeys, responsive layout, accessibility, dark and light themes and legacy diagnostic coverage.

**Acceptance:** A first-time visitor can configure and run the synthetic workflow without manual DVC or YAML operations and explore provenance-aware Results. Page length stays bounded as cycle and fold counts grow. Existing scientific diagnostics and expert execution controls remain accessible.

### Milestone 4 — User data ingestion (`0.2.0.dev4`)

- [ ] Define a canonical dataset contract and column mapping for sample ID, coordinates, numerical predictors, label, optional group/batch metadata and CRS/units when applicable.
- [ ] Separate synthetic data acquisition from shared preparation and graph construction.
- [ ] Implement CSV import, preview, schema mapping and explicit validation.
- [ ] Handle missing values, duplicate IDs, repeated coordinates, invalid numeric values, class imbalance and geological measurement conventions without silently changing data.
- [ ] Infer feature count and class vocabulary from eligible training data; preserve mappings and provenance.
- [ ] Adapt DVC dependencies to track imported data and preprocessing configuration.
- [ ] Test synthetic and small uploaded datasets, malformed input and optional unlabeled prediction data.

**Acceptance:** Both sources feed a common downstream representation; a valid CSV can be prepared without source-code changes; invalid data receives actionable errors; synthetic generation still works. Uploaded data is not transmitted externally by default.

### Milestone 5 — Training and prediction workflows (`0.2.0.dev5`)

- [ ] Support independent training and prediction operations; never require retraining for each new batch.
- [ ] Persist model, preprocessing, feature ordering, label mappings and inference protocol.
- [ ] Implement appropriate grouped/spatial validation for real data before presenting generalization claims; retain random validation for suitable synthetic experiments.
- [ ] Verify that the preprocessing, graph, tuning and calibration safeguards apply to user data.
- [ ] Support unlabeled prediction batches with compatible schemas; reject incompatible features/classes appropriately.
- [ ] Export stable sample IDs, predictions and class probabilities, with clear model provenance.
- [ ] Add lightweight end-to-end tests for training and inference on both data sources.

**Acceptance:** A user can train on labeled data, classify compatible new samples without retraining and export results. Evaluation protocols are documented and tested. No unverified labels enter supervised training. If a required scientific safeguard is incomplete, do not present the workflow as validated.

### Milestone 6 — Results and accessibility (`0.2.0.dev6`)

- [ ] Present interpretable performance summaries, confusion matrix, class metrics, calibration and uncertainty.
- [ ] Provide interactive spatial visualization and filtering plus accessible tabular alternatives.
- [ ] Introduce a simple tabular baseline on identical splits to contextualize GNN performance; expand model comparisons later.
- [ ] Improve CPU support and automatic device selection; retain GPU acceleration. Verify packaging and actual CPU execution.
- [ ] Package a small precomputed demo that does not require training.
- [ ] Document installation, data templates, usage and limitations; investigate a read-only hosted demo with resource/privacy constraints.

**Acceptance:** Users can explore example results quickly, run a small supported workflow without a dedicated GPU, and interpret/export predictions. Proposed target: first precomputed results within 30 seconds under documented conditions; measure rather than assume.

### Release candidate (`0.2.0rc1`) and stable release (`0.2.0`)

- [ ] Run agreed regression and end-to-end checks, including the original synthetic workflow, uploaded training, inference and CPU execution.
- [ ] Verify DVC dependencies, artifact reproducibility, error handling and installation instructions.
- [ ] Review data privacy, upload safety, scientific caveats and accessibility.
- [ ] Obtain owner review of the application and resolve release-blocking issues.
- [ ] Update README, changelog, version and lockfile; tag and merge only with approval.

**Release gate:** Basic configuration has approximately six or fewer primary controls; no mandatory YAML editing or dedicated GPU; user data can be imported without Python; predictions do not require retraining; evaluation is defensible; limitations are documented. A hosted demo is a desirable distribution goal, subject to hosting feasibility, not a claim of existing deployment.

## Later releases (provisional; not authorized by this roadmap)

| Version | Focus |
|---|---|
| 0.3.0 | Independent workspaces/runs, run recovery/cancellation, model registry, fixed longitudinal benchmark, experiment comparison, verified-label batch updates and drift-guided human review |
| 0.4.0 | Real-world licensed geological case study, expanded baselines, advanced spatial validation and graph diagnostics, additional file formats, active learning research |
| 1.0.0 | Mature documented and tested application with stable interfaces and deployment workflows |

Existing research-oriented learning cycles and diagnostics should remain available during v0.2.0 development. Real-data continual learning is deferred until verified-label handling and model comparison are established. Essential leakage safeguards and defensible validation are not deferred.

## Working agreement

Complete one milestone or smaller task at a time. Agree on scope and acceptance criteria, implement, test, review the diff, update the changelog, and obtain owner approval before committing or advancing. Record unresolved blockers in the milestone or baseline report. Do not change scientific behavior, dependency versions or architectural scope merely to satisfy a development version label.
