# Mineral Classifier App — Development roadmap

## Vision

Transform the v0.1.0 research prototype into an accessible spatial machine learning application for geoscientists, data scientists and ML researchers. Preserve GCN/GAT, reproducible DVC workflows, synthetic experiments, calibration and drift analysis while separating ordinary user workflows from expert controls.

## Release strategy

The `v0.1.0` tag preserves the original application at `d00c2cd`. The development foundation is recorded at `55aeacf`. Develop on `develop/v0.2.0`, keep `main` stable, and use the existing `pyproject.toml` and `uv.lock`. Development version labels below identify milestones. The current project version is `0.2.0.dev0`. The owner approves version changes, commits, tags and merges. Check off milestones and work items after verification.

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
- [ ] **M1.7 — Small end-to-end pipeline verification.** Run an isolated CPU synthetic cycle and verify training, evaluation, calibration, prediction and drift artifacts.

**Final checkpoint policy.** Independent CV folds retain best-epoch and early-stopping diagnostics. The stage then saves the fresh full-development model after its fixed CV-informed duration. The separate calibration and test graphs remain reserved. See the [final-model strategy](docs/graph_design.md#m14a-development-pool-final-model).

**Acceptance:** The CPU regression suite passes, scientific protocol changes are documented and tested, and a small synthetic run verifies the path from data generation through prediction and drift analysis.

### Milestone 2 — Configuration management (`0.2.0.dev2`)

- [ ] Audit existing YAML keys, defaults, consumers and valid combinations.
- [ ] Introduce a typed, validated configuration layer (use an existing dependency if adequate; propose Pydantic if needed).
- [ ] Add Quick, Balanced and Research presets; choose actual values after examining current defaults and feasibility.
- [ ] Provide basic settings, grouped advanced settings and expert YAML import/export.
- [ ] Validate model-specific and data-source-specific parameters; avoid exposing irrelevant controls.
- [ ] Preserve legacy YAML compatibility and save resolved configuration snapshots.
- [ ] Test defaults, invalid combinations, preset resolution and YAML round trips.

**Acceptance:** A user can configure a synthetic run without editing YAML; the legacy configuration remains usable; invalid settings fail before computation; scientific defaults are preserved or changes are explicitly documented.

### Milestone 3 — User interface redesign (`0.2.0.dev3`)

- [ ] Extract Streamlit pages incrementally from the existing entry point.
- [ ] Add an overview with Explore Demo and Use My Own Data entry points; clearly label unavailable workflows until implemented.
- [ ] Organize navigation around Data, Train, Results and Predict, with advanced diagnostics separate.
- [ ] Integrate configuration forms and presets; retain expert access to existing pipeline stages.
- [ ] Hide routine DVC setup behind application orchestration where feasible.
- [ ] Improve empty states, errors, progress feedback and contextual help.

**Acceptance:** A first-time visitor can understand and navigate the existing synthetic workflow; no manual DVC/YAML operations are needed for the supported basic path; existing expert functionality remains accessible.

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
