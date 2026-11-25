# Mineral Classifier App — Development roadmap

## Vision

Transform the v0.1.0 research prototype into an accessible spatial machine learning application for geoscientists, data scientists and ML researchers. Preserve GCN/GAT, reproducible DVC workflows, synthetic experiments, calibration and drift analysis while separating ordinary user workflows from expert controls.

## Release strategy

Preserve the existing release using Git tag `v0.1.0` (verify the tag and baseline commit first). Develop on `develop/v0.2.0`, keep `main` stable, and use the existing `pyproject.toml` and `uv.lock`. Development version labels below are proposed milestone identifiers, not released versions. The owner approves version changes, commits, tags and merges. All milestones start pending until verified; do not mark them complete based on plans alone.

## v0.2.0 — Accessible classification application

### Milestone 0 — Development foundation (`0.2.0.dev0`)

- [ ] Verify Git status, branch, tags and current version; preserve original baseline with owner approval.
- [ ] Inspect architecture, entry points, configuration, DVC dependencies, Docker, and existing test tooling.
- [ ] Run lightweight existing tests and static checks; record failures or unavailable tooling.
- [ ] Identify required artifacts, CPU compatibility obstacles and reproducibility risks.
- [ ] Create `docs/baseline_v0.1.0.md` with execution instructions, verified functionality, test outcomes and limitations.
- [ ] Establish project instructions, roadmap and changelog; update development version and lockfile if authorized.

**Acceptance:** Baseline is documented, Git preservation is verified, and no scientific pipeline or application behavior has changed. Full training is not required without approval; unverified behavior is clearly identified.

### Milestone 1 — Configuration management (`0.2.0.dev1`)

- [ ] Audit existing YAML keys, defaults, consumers and valid combinations.
- [ ] Introduce a typed, validated configuration layer (use an existing dependency if adequate; propose Pydantic if needed).
- [ ] Add Quick, Balanced and Research presets; choose actual values after examining current defaults and feasibility.
- [ ] Provide basic settings, grouped advanced settings and expert YAML import/export.
- [ ] Validate model-specific and data-source-specific parameters; avoid exposing irrelevant controls.
- [ ] Preserve legacy YAML compatibility and save resolved configuration snapshots.
- [ ] Test defaults, invalid combinations, preset resolution and YAML round trips.

**Acceptance:** A user can configure a synthetic run without editing YAML; the legacy configuration remains usable; invalid settings fail before computation; scientific defaults are preserved or changes are explicitly documented.

### Milestone 2 — User interface redesign (`0.2.0.dev2`)

- [ ] Extract Streamlit pages incrementally from the existing entry point.
- [ ] Add an overview with Explore Demo and Use My Own Data entry points; clearly label unavailable workflows until implemented.
- [ ] Organize navigation around Data, Train, Results and Predict, with advanced diagnostics separate.
- [ ] Integrate configuration forms and presets; retain expert access to existing pipeline stages.
- [ ] Hide routine DVC setup behind application orchestration where feasible.
- [ ] Improve empty states, errors, progress feedback and contextual help.

**Acceptance:** A first-time visitor can understand and navigate the existing synthetic workflow; no manual DVC/YAML operations are needed for the supported basic path; existing expert functionality remains accessible.

### Milestone 3 — Bring your own data (`0.2.0.dev3`)

- [ ] Define a canonical dataset contract and column mapping for sample ID, coordinates, numerical predictors, label, optional group/batch metadata and CRS/units when applicable.
- [ ] Separate synthetic data acquisition from shared preparation and graph construction.
- [ ] Implement CSV import, preview, schema mapping and explicit validation.
- [ ] Handle missing values, duplicate IDs, repeated coordinates, invalid numeric values, class imbalance and geological measurement conventions without silently changing data.
- [ ] Infer feature count and class vocabulary from eligible training data; preserve mappings and provenance.
- [ ] Adapt DVC dependencies to track imported data and preprocessing configuration.
- [ ] Test synthetic and small uploaded datasets, malformed input and optional unlabeled prediction data.

**Acceptance:** Both sources feed a common downstream representation; a valid CSV can be prepared without source-code changes; invalid data receives actionable errors; synthetic generation still works. Uploaded data is not transmitted externally by default.

### Milestone 4 — Training, prediction and scientific safeguards (`0.2.0.dev4`)

- [ ] Support independent training and prediction operations; never require retraining for each new batch.
- [ ] Persist model, preprocessing, feature ordering, label mappings and inference protocol.
- [ ] Implement appropriate grouped/spatial validation for real data before presenting generalization claims; retain random validation for suitable synthetic experiments.
- [ ] Audit scaler fitting, graph edges/message passing, tuning and calibration for leakage.
- [ ] Support unlabeled prediction batches with compatible schemas; reject incompatible features/classes appropriately.
- [ ] Export stable sample IDs, predictions and class probabilities, with clear model provenance.
- [ ] Add lightweight end-to-end tests for training and inference on both data sources.

**Acceptance:** A user can train on labeled data, classify compatible new samples without retraining and export results. Evaluation protocols are documented and tested. No unverified labels enter supervised training. If a required scientific safeguard is incomplete, do not present the workflow as validated.

### Milestone 5 — Results and accessibility (`0.2.0.dev5`)

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
