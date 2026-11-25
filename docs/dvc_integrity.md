# M1.5 DVC dependency and artifact integrity

The audit compares the generated DAG to the production entry points, their project imports and the Streamlit paths. The tensor and bundle contracts are described in [preprocessing design](preprocessing_design.md) and [graph and final-refit design](graph_design.md#m14a-development-pool-final-model). The earlier defects are recorded in the [v0.1.0 baseline](baseline_v0.1.0.md).

## Stage inventory

For learning cycle **N**, paths in this table are relative to `results/`. `B(N−1)` means `data/base/cycle_{N−1}`, and `P(N)` means `data/prediction/cycle_N/pred_data.pt`. Every stage tracks its CLI file in `src/stages/` and `general_utils.py`. The source groups below refer to files in `src/utilities/` unless a path is given.

- Raw/graph sources are `data_utils.py`, `raw_graphs.py`, `preprocessing.py`, `graph_protocol.py` and `spatial_graph.py`.
- Bundle sources add `model_bundle.py` and `training_schedule.py`, including validation of the saved refit schedule.
- Model sources are the factory `src/models/__init__.py` and its imports `src/models/gat.py` and `src/models/gcn.py`.
- Schema keys are `data.base.n_features`, `data.base.n_classes` and `data.base.class_names`. Graph keys are `add_self_loops`, `data.edge_affinity` and the three `data.base` connectivity settings (`k_nearest`, `connection_radius`, `distance_percentile`). The latter are covered by `data.base` in the graph producer stages.

| Stage and CLI | Material artifact inputs | Configuration and additional sources | Outputs and dashboard use |
| --- | --- | --- | --- |
| `bootstrap` · `generate_base_data.py` | None | `data.base` for synthetic population, schema, partitions and graph keys. Raw/graph sources. | `B(0)` contains the canonical `base_data.pt`, `fold_data.pt`, `calib_data.pt`, `test_data.pt`, `hotspots.npy`, `prototypes.npy`, base graph diagnostics and `tsne_comparison.png`. The Data tab reads the plots. |
| `train_cycle_N` · `train.py` | The four tensors in `B(N−1)`, including calibration and test for verification and exclusion from the development graph | `train`, selected `models.<model>`, schema and graph keys, `data.scaler_type`, `data.scaler_params`. Bundle and model sources, `train_utils.py`, `early_stopping.py`, `calibration_utils.py` for fold ECE. | `trained/cycle_N/` owns `model.pt`, `preprocessing.json`, `graph_binding.json` with final-refit metadata and each `training_plots_dataset_*.png` shown in the Train tab. |
| `evaluate_cycle_N` · `evaluate.py` | The four tensors in `B(N−1)` and the three exact model bundle files from cycle N | `evaluate`, selected model, schema and graph keys. Bundle and model sources, `eval_utils.py`, `calibration_utils.py`, `early_stopping.py`. | `evaluation/cycle_N/` owns `calibrator.pt`, `calibration_binding.json`, `metrics.json`, `reliability_diagram.png`, `confusion_matrix.png` and method-dependent `optimum_params.txt` and `calibration_*_loss_and_gradient.png`. The Evaluation tab reads metrics and figures. |
| `generate_pred_data_cycle_N` · `generate_pred_data.py` | Prior `base_data.pt`, always the cycle-zero hotspot and prototype arrays | `data.pred`, `data.base` for synthetic generation and parent verification, graph keys. Raw/graph sources. | `P(N)` includes the canonical incoming observations and source-parent digest. The Data tab reads its companion `prediction_data_graph.html` and `.json`. |
| `predict_cycle_N` · `predict.py` | `P(N)`, three model bundle files and both calibration bundle files for N | Selected model, schema and graph keys. Bundle and model sources, `pred_utils.py`, `eval_utils.py` for the confusion matrix, `calibration_utils.py`, `early_stopping.py` via the calibration module. | `prediction/cycle_N/` owns `predictions.csv`, `metrics.json`, `histograms.png` and `confussion_matrix_.png` (existing spelling). The Prediction tab reads the CSV and figures. |
| `analyze_drift_cycle_N` · `analyze_drift.py` | Prior `base_data.pt`, `P(N)` and the three model bundle files for N. The weights are verified by digest. | `analyze_drift`, schema and graph keys. Bundle sources and `drift_detection_utils.py`. | `drift_analysis/cycle_N/` owns `drift_metadata.json`, `drift_results.txt` with statistics and p-values, and `distance_distributions.png`, `marginal_distributions.png`, `mutual_information.png`, `pca_kpca_grid.png`. The Drift tab reads the text report and figures. |
| `prepare_next_cycle_data_N` · `prepare_next_cycle_data.py` | Prior `base_data.pt` and `P(N)` including synthetic labels | `data.base` for schema and split/graph settings, graph keys. Raw/graph sources. | `B(N)` owns the four tensors, graph diagnostics and `tsne_comparison.png`. This feeds cycle N+1 and the Data tab. |
| `analyze_cycles_performance` · `analyze_cycles_performance.py` | `evaluation/cycle_N/metrics.json` (calibrated metrics) and `prediction/cycle_N/metrics.json` for every configured N | The generated `--cycles` command argument selects the exact declared range. `cycles_performance_utils.py`. Generated for multiple cycles. | `cycles_performance_analysis/cycles_performance.png`, shown in the Performance tab. |

Base graph diagnostics are HTML and Plotly JSON pairs for `base_data_graph`, `test_graph`, `calib_graph`, and each `train_graph_<fold>` and `val_graph_<fold>`. The existing names and locations let Streamlit render JSON first with HTML as a fallback. The next base consumes the incoming tensor with synthetic truth. Scoring, calibration, drift, and their reports have no edge back to raw generation or next-cycle preparation. Metrics JSON files are ordinary cached artifacts inside their producing directories. DVC `metrics` or `plots` declarations would add no required cache coverage.

## Corrections and ownership

- Added `tsne_comparison.png` to the explicit file outputs of bootstrap and every next-cycle base producer. The existing pipeline already declared the Plotly pairs, the bootstrap arrays and the consumers of those arrays.
- Added the calibration metrics implementation to training's source dependencies and the calibration module's early-stopping dependency to evaluation and prediction. Enumerated model source files explicitly so unrelated files and interpreter bytecode in the model directory do not enter its hash.
- Bound performance aggregation to cycles 1 through the generated cycle count. The prior directory scan could read undeclared metrics from obsolete runs, making the plot depend on leftover files and allowing incompatible legacy JSON to fail the stage. The direct CLI defaults to the configured count when `--cycles` is omitted.
- Confirmed the raw parent and split edges, model weights, fitted preprocessing, final-refit provenance, calibration pairing and prediction and drift dependencies. Each result directory has a single owner, and consumers depend on exact files within those directories. The base and prediction-data stages own explicit files. No directory and descendant file share output ownership.

Regenerate the pipeline after changing `cycles`, `default_model` or `data.base.n_splits`, which control stage count, commands and the declared fold plot filenames. Regenerating these declarations is distinct from running the stages.

## Verification and selective invalidation

DVC 3.67.1 parsed one-cycle GCN and two-cycle GAT definitions with different fold counts, plus the checked-in three-cycle definition. Regression tests checked unique stage names, output ownership, exact cross-cycle artifact paths and byte-for-byte generator consistency for the checked-in three-cycle pipeline. Existing scientific tests cover the raw schema, independent graphs, fitted preprocessing and bundle validation. The extended DVC fixture uses the generated dependency, parameter and output declarations, including source files copied into an isolated repository, with deterministic replacement commands. The aggregate CLI also has a regression test that exercises the actual metric loader against stale, incompatible JSON outside the selected range.

The complete Pytest suite passed 130 tests in 40.28 seconds. The two-cycle DVC test took 33.44 seconds. Ruff lint passed on modified Python files, repository-wide formatting passed, applicable pre-commit hooks passed without edits, and Git whitespace checks passed. Repository-wide Ruff lint reports four pre-existing docstring findings, down from six findings on the starting revision.

| Isolated two-cycle change | Stages rerun |
| --- | --- |
| No changes | None |
| Scaler quantile range, training learning rate or selected model width | Train, evaluate, predict, drift and dependent performance aggregation. All raw producers reused. |
| Evaluation bin count | Evaluate, predict and performance aggregation. Training, drift and raw producers reused. |
| Drift permutation count | Drift only |
| Edge affinity scale | All stages, starting with graph producers |
| Training utility source | Train and its downstream model consumers and performance aggregation |
| Evaluation utility source | Evaluate and predict, then performance aggregation |
| Shared calibration utility source | Train for fold diagnostics, evaluate and predict directly, then downstream consumers whose fixture inputs changed |
| Drift utility source | Drift only |
| Graph protocol source | All stages |

The fixture makes declared input changes visible in output hashes so it can exercise the DAG. In a real run a downstream stage may be reused if the exact bytes it consumes stay the same. A calibration-setting change reruns the combined calibration and evaluation stage. Graph-setting edits rerun synthetic generation where graph assembly and observation generation share a command. Graph, preprocessing and data code are tracked at file granularity. The existing `data.base` mapping is tracked conservatively by raw producers, so an edit to a key used only by another producer can still invalidate that mapping. These boundaries require no stage split for M1.5.

## Artifact restoration

The same isolated two-cycle fixture removed its entire disposable `results/` tree, ran `dvc checkout` from the temporary local cache, and compared the restored paths and bytes against the pre-checkout snapshot. The expected collection contains 124 files, covering bootstrap arrays, raw and split tensors, all Data tab diagnostics, trained and calibration bundles including refit and binding sidecars, training and evaluation plots, prediction CSVs, reports and figures, drift metadata, statistics and plots, and the performance image. A subsequent reproduction ran zero commands. Fixture files are deterministic stand-ins. This verifies declared storage and restore edges. Production serialization, conditional reports from other calibration methods and production cache completeness still require real artifacts.

## Legacy lockfile and M1.7 procedure

The historical `dvc.lock` used schema 2.0 and had 122 records. All 20 current names are present. Another 102 records belong to cycles 4–20, six per cycle. Comparing commands, dependency paths, tracked parameter keys and output paths found zero exact matches among the 20 overlapping records. For example, bootstrap has hashes for four tensor outputs, evaluation's first-cycle record depends on the test graph and weights without the calibration graph or binding files, and the performance command predates the explicit cycle argument. There are no explicit hashes for the bootstrap arrays, t-SNE diagnostics or binding sidecars. Directory hashes alone cannot establish the exact contents or provenance of those historical bundles. These records predate canonical raw data, independent split graphs, bound preprocessing and the full development-pool refit. Treat the lockfile and any matching artifacts as historical.

For the authorized M1.7 run:

1. Start a disposable checkout or worktree with separate results and a fresh local DVC cache. Preserve the production workspace and its legacy data. Remove the inherited lockfile only from the disposable checkout before reproduction so retired records cannot be carried forward. Use the existing locked environment and no remote.
2. Choose the small approved CPU configuration and record it, including seeds and the supported sample counts. The current t-SNE diagnostics require more than 100 base observations. Regenerate with `uv run --frozen --no-sync python -m scripts.setup_dvc`, then parse the DAG with `uv run --frozen --no-sync python -m dvc dag`. Verify that generated stages and declared paths match the chosen configuration.
3. In that isolated environment, run `uv run --frozen --no-sync python -m dvc repro` using the production commands. Compare the generated lock with that DAG and require exactly the active stage records. Verify real tensors, model and refit bindings, fitted preprocessing, calibration and test separation, prediction CSV and metrics, and drift metadata and figures against the consumer contracts. Exercise the dashboard's populated paths. A one-cycle run has no performance stage. Verify production aggregation with compatible multi-cycle metrics if that stage is in the authorized run.
4. In the disposable run, snapshot the result paths and hashes, remove the disposable results, restore from that run's local cache and verify the real bundle consumers again. Confirm a subsequent reproduction is current. Review the clean lockfile together with the exact generated definition and configuration. A reduced CPU test lock cannot serve as the default production lock. Any checked-in replacement needs an approved reproduction of the matching configuration.

M1.5 leaves the production lockfile, cache and generated results in place.

Release preparation for `0.2.0.dev1` removed the obsolete tracked lockfile. The identical file remains available at `v0.1.0:dvc.lock`. [M1.7](end_to_end_verification.md) verified reduced production configurations. Reproducing a corresponding configuration generates its new lockfile. The default-budget configuration still needs its own reproduction and lockfile.
