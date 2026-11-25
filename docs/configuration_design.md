# M2.1 Configuration inventory and design

This is a design for the existing synthetic workflow. The inventory covers all 88 configurable leaves in [params.yaml](../params.yaml). Checked-in values serve as the Research baseline. The categories describe how users would encounter settings, independent of how DVC tracks them.

## Inventory

**Basic** controls the experiment's scope and execution. **Advanced** controls scientific choices that materially affect the experiment. **Expert** exposes detailed simulator, model, optimizer and statistical options. **Internal** retains derived or implementation-specific fields in the resolved file.

**R** means the active consumer requires a key or a specific value. **O** means the source has an omission fallback or accepts an inactive setting. **D** means the UI should derive or preserve the setting. A key indexed by the stage remains required even when its value is inactive. Model blocks pass settings to constructors, so **O** there denotes constructor defaults rather than permission to omit them from the resolved output. Null is permitted only where listed or explicitly noted. **Enforced** describes project code or the called library. **Proposed** describes a future validation rule. Numeric options with no listed enforced range are passed to numerical libraries or used directly in formulas.

Consumer abbreviations describe actual calls. **Bootstrap** is `generate_base_data.prepare_base_data`, **Batch** is `generate_pred_data.prepare_pred_data`, **Merge** is `prepare_next_cycle_data.combine_split_data`, **Split** is `raw_graphs.build_cycle_graphs → data_utils.partition_indices`. **Graph** is `graph_protocol.resolve_graph_protocol → spatial_graph.build_spatial_graph`, or protocol verification in the model and drift consumers. **Schema** is `raw_graphs.synthetic_schema` and, for dimensions, `models.get_model`. **Train** is `train.run_training → train_utils.train/train_final`. **Eval** is `evaluate.run_evaluation → eval_utils.evaluate_with_calibration → CalibratedModel`. **Drift** is `analyze_drift.run_analyze_drift → AnalyzeDrift`. Stage functions live under [src/stages](../src/stages) and numerical consumers under [src/utilities](../src/utilities).

### Experiment and preprocessing

| YAML path | Checked-in default | Type and null meaning | Status and category | Actual use and constraints |
| --- | --- | --- | --- | --- |
| `cycles` | `3` | int | R · Basic | Pipeline generator expands learning cycles; dashboard bounds selectors; performance CLI defaults to this count. Generator can emit bootstrap alone for zero; proposed minimum 1 for experiment presets. Performance stage exists only for more than one cycle. |
| `default_model` | `gcn` | str | R · Basic | Generator puts the model in train/evaluate/predict CLI commands; graph settings enforce `gcn` or `gat`. Direct stage calls select their model explicitly. |
| `device` | `auto` | str | O · Basic | Train/Eval/predict call `select_device`. `auto` uses first visible CUDA or CPU, `cpu` forces CPU, `cuda` means index 0, `cuda:N` selects the visible index. Explicit unavailable device fails. Omission is auto, explicit null fails. |
| `add_self_loops` | `true` | bool | R · Advanced | Graph owns self-loops for both models. Exact boolean enforced. Per-layer automatic insertion is disabled internally. |
| `data.scaler_type` | `RobustScaler` | str or null for identity | R · Advanced | Train calls `fit_preprocessing`. Enforced names are `RobustScaler`, `StandardScaler`, `MinMaxScaler` or null with empty options. Eval/prediction/drift read fitted state. |
| `data.scaler_params.quantile_range` | `[25.0, 75.0]` | pair of numbers | O · Expert | Train passes to sklearn RobustScaler, converting YAML list to tuple. Installed sklearn checks ordered endpoints within 0–100; proposed strict lower < upper for new inputs. Keep only for RobustScaler. |
| `data.edge_affinity.method` | `inverse_quadratic` | str | R · Advanced | Graph requires `inverse_quadratic` or `unit`, with explicit scale key. GCN uses affinity in convolution. GAT remains topology-only; weights are still stored and verified. |
| `data.edge_affinity.scale` | `460.0` | real or null for unit weights | R · Advanced | Graph requires finite positive scale in coordinate units for inverse quadratic, null for unit. Does not select neighbors. |

`data.scaler_params` is an open mapping passed to the selected sklearn constructor. Omission, null and an empty map mean no options in `fit_preprocessing`, subject to the resolved DVC contract. RobustScaler also accepts `with_centering`, `with_scaling`, `unit_variance`, and `copy`; StandardScaler accepts `with_mean`, `with_std`, and `copy`; MinMaxScaler accepts `feature_range`, `clip`, and `copy`. These are existing supported options, not additional checked-in leaves. Fit-time validation is delegated to sklearn. Use a scaler-specific form with constructor defaults when selecting a different scaler. Keep `copy` in expert YAML and preserve it on round-trip. Disable scaling with null plus empty options. Never carry RobustScaler's quantiles into another scaler implicitly.

### Synthetic generation, schema and partitions

| YAML path | Checked-in default | Type and null meaning | Status and category | Actual use and constraints |
| --- | --- | --- | --- | --- |
| `data.base.radius` | `2500` | real | R · Advanced | Bootstrap/Batch coordinate cylinder radius. Proposed >0. |
| `data.base.depth` | `-500` | real | O · Advanced | Bootstrap/Batch sample z from depth to zero. Proposed negative. |
| `data.base.n_samples` | `1000` | int | O · Basic | Bootstrap requested rows. Proposed positive plus preflight feasibility checks. Current base t-SNE requires realized rows >100. |
| `data.base.spacing` | `10` | real | O · Advanced | Bootstrap/Batch initial XY placement threshold. Proposed ≥0; class top-up has its own placement path. |
| `data.base.n_features` | `5` | int | R · Advanced | Bootstrap/Batch feature generation; Schema derives ordered names and model input width. Proposed ≥1 plus diagnostic feasibility. |
| `data.base.n_classes` | `4` | int | R · Advanced | Bootstrap/Batch label count; Schema and Train use it for class mapping/model outputs/loss. Proposed ≥2 plus actual class coverage checks. |
| `data.base.class_names` | `null` | ordered list of strings or null for generated `Class N` | R/D · Advanced | Schema/Bootstrap/Batch/Merge. Truthy list must match class count and raw validation requires unique nonblank names. Current empty list also uses generated names. |
| `data.base.threshold_binary` | `0.5` | real | O · Expert | Bootstrap/Batch `_assign_labels` only with two classes. Proposed bounded to simulator's normalized signal range [0, 1]. |
| `data.base.min_samples_per_class` | `50` | int or null to skip top-up | O · Advanced | Bootstrap `_change_label_distribution` appends deficits per class. Proposed nonnegative integer with the default retained. |
| `data.base.k_nearest` | `null` | int or null for inactive | D · Advanced | Graph. Enforced ≥1 when selected. Quota capped at partition size minus one; symmetric union can raise node degree. |
| `data.base.connection_radius` | `null` | real or null for inactive | D · Advanced | Graph. Enforced finite >0; edge distance is strictly less than radius. |
| `data.base.distance_percentile` | `5` | real or null for inactive | D · Advanced | Graph. Enforced finite [0, 100]; each node uses an inclusive cutoff based on its partition's non-self distances. Degree depends on batch size. |
| `data.base.n_splits` | `3` | int | R · Advanced | Split and generator fold output names. sklearn requires ≥2 and constrains membership by class counts. Null fails. |
| `data.base.test_size` | `0.3` | fraction, int count, or null for sklearn 25% default | O · Advanced | Split reserves the entire holdout from the full base. sklearn enforces fraction (0, 1) or positive count below population size, subject to stratification. |
| `data.base.calib_size` | `0.3` | fraction, int count, or null for sklearn 25% default | O · Advanced | Split takes calibration from the holdout. Same constraints against the holdout size; remainder is the final test set. |
| `data.base.n_hotspots_max` | `5` | int | O · Expert | Bootstrap/Batch `_create_hotspots` samples 1 through this bound inclusive; needs ≥1 for the RNG draw. |
| `data.base.hotspot_drift` | `0.1` | real or null to regenerate hotspots | O · Expert | Batch only, `_create_hotspots` perturbs retained bootstrap hotspots. Gaussian standard deviation must be ≥0. Zero retains positions for kept hotspots. |
| `data.base.mineral_noise_level` | `0.5` | real | O · Expert | Bootstrap/Batch `_calculate_gold_values` noise; proposed nonnegative. |
| `data.base.exp_decay_factor` | `0.05` | real | O · Expert | Bootstrap/Batch `_calculate_gold_values` distance decay; proposed nonnegative. |
| `data.base.feature_noise_level` | `0.3` | real | O · Expert | Bootstrap/Batch `_generate_features` noise and prototype evolution scale. Gaussian scale needs ≥0. |
| `data.base.prototype_evolution_rate` | `0.25` | real or null to regenerate prototypes | O · Expert | Batch only, `_generate_features` evolves bootstrap prototypes when non-null. Proposed ≥0; zero keeps prototypes fixed. |
| `data.base.class_separation` | `2.0` | real | O · Expert | Bootstrap/Batch `_generate_features` when prototypes are created. Inactive for batches reusing or evolving prototypes. Proposed nonnegative. |
| `data.base.class_influence` | `0.3` | real | O · Expert | Bootstrap/Batch `_generate_features` scales prototype adjustment. Proposed finite; no project bound. |
| `data.base.spatial_weight` | `0.5` | real | O · Expert | Bootstrap/Batch `_generate_features` blends columns at indices 2+ with spatial proximity. Proposed [0, 1] for a blend. |
| `data.base.interaction_strength` | `0.25` | real | O · Expert | Bootstrap/Batch `_generate_features` weights paired interactions when there are ≥2 features. Proposed finite. |
| `data.base.correlation_strength` | `0.25` | real | O · Expert | Bootstrap/Batch `_generate_features` sets class noise covariance. Proposed [0, 1] for a positive semidefinite blend. |
| `data.base.feature_weight_range` | `[0.7, 1.7]` | pair of reals | O · Expert | Bootstrap/Batch `_generate_features` uniform feature weights. RNG requires upper ≥ lower; proposed finite endpoints. |
| `data.base.feature_offset_std` | `0.2` | real | O · Expert | Bootstrap/Batch `_generate_features` normal offset scale; requires ≥0 for the RNG draw. |
| `data.base.seed` | `110` | int or null for unseeded | O · Advanced | Bootstrap and Split share the seed, including Merge partitions. Enforced unsigned 32-bit integer or null. Omission at runtime yields 42. |
| `data.pred.n_samples` | `200` | int | O · Basic | Batch target per cycle. Proposed positive plus placement feasibility. |
| `data.pred.min_samples_per_class` | `null` | int or null to skip top-up | O · Expert | Batch `_change_label_distribution` can append rare classes. Independent of the bootstrap setting; proposed nonnegative integer. |
| `data.pred.seed` | `42` | int or null for entropy | O · Advanced | Batch uses nonnegative NumPy seed directly each cycle, without an upper bound. Omission yields 42; changing the parent affects placement and batch provenance. |

### Model and training

The selected model block goes through `model_graph_settings` to `get_model` and the relevant constructor for training, evaluation and prediction. Hide inactive model controls while preserving both YAML blocks. Feature and class dimensions are derived from the schema. When batch normalization is enabled, training rejects singleton training folds. Show graph affinity as shared generation/provenance metadata for GAT, whose attention uses topology without a configured edge-feature projection.

| YAML path | Checked-in default | Type and null meaning | Status and category | Actual use and constraints |
| --- | --- | --- | --- | --- |
| `models.gat.hidden_channels` | `64` | int | O · Expert | GAT constructor, channels per head before final attention layer. Proposed ≥1. |
| `models.gat.n_layers` | `3` | int | O · Expert | GAT constructor always appends first and last convolutions. Proposed ≥2 so count matches label. |
| `models.gat.n_heads` | `5` | int | O · Expert | GAT constructor, intermediate multi-head attention; last layer always one head. Proposed ≥1. |
| `models.gat.dropout` | `0.1` | real | O · Expert | GAT constructor, attention and activation/head dropout. Torch enforces [0, 1]; proposed <1 for meaningful training. |
| `models.gat.batch_norm` | `true` | bool | O · Expert | GAT constructor enables normalization after convolutions; Train checks singleton folds. |
| `models.gat.negative_slope` | `0.2` | real | O · Expert | GAT constructor, LeakyReLU attention slope. Proposed finite ≥0. |
| `models.gat.bias` | `true` | bool | O · Expert | GAT constructor enables attention-layer biases. |
| `models.gat.residual` | `true` | bool | O · Expert | GAT constructor enables attention-layer residual projections. Omitted constructor value is false. |
| `models.gcn.hidden_channels` | `64` | int | O · Expert | GCN constructor, hidden width. Proposed ≥1. |
| `models.gcn.n_layers` | `3` | int | O · Expert | GCN constructor always appends first convolution. Proposed ≥1 so count matches label. |
| `models.gcn.dropout` | `0.3` | real | O · Expert | GCN constructor, activation/head dropout. Torch enforces [0, 1]; proposed <1. |
| `models.gcn.batch_norm` | `true` | bool | O · Expert | GCN constructor enables normalization after convolutions; Train checks singleton folds. |
| `models.gcn.improved` | `true` | bool | O · Expert | GCN constructor and graph settings. Changes model loop weight from 1 to 2 when explicit loops are enabled. Omission resolves false. |
| `models.gcn.normalize` | `true` | bool | R/D · Internal | Graph settings enforce true for GCN adjacency normalization. Preserve in resolved file. |
| `train.seed` | `42` | int or null for unseeded | O · Advanced | Train derives independent Torch streams for each cycle/fold and final refit. Enforced unsigned 32-bit; omission 42. |
| `train.n_epochs` | `1500` | int | R · Advanced | Train caps each CV fold; final plan requires positive budget. Final duration is derived from the rounded median fold best epoch. |
| `train.lr` | `0.0005` | real | R · Expert | CV Adam initial learning rate; refit replays observed CV rates. Proposed finite >0. |
| `train.max_grad_norm` | `1.0` | real | R · Expert | `_training_step` clips gradients in CV and final refit. Proposed finite >0. |
| `train.weight_decay_adam_optimizer` | `5.0e-4` | real | R · Expert | Adam in CV and refit. Nonnegative required by Adam. |
| `train.factor_learning_rate_scheduler` | `0.5` | real | R · Expert | CV ReduceLROnPlateau on validation loss; library requires factor <1; proposed (0, 1). |
| `train.patience_learning_rate_scheduler` | `20` | int | R · Expert | CV scheduler patience in epochs. Proposed ≥0. |
| `train.patience_early_stopping` | `50` | int | R · Expert | CV `EarlyStopping` on validation loss; no early stop during refit. Proposed ≥1. |
| `train.min_delta_early_stopping` | `1.0e-3` | real | R · Expert | CV `EarlyStopping` improvement tolerance. Proposed ≥0. |

### Evaluation and calibration

Eval indexes its configuration even when the method ignores certain options. Parametric methods are temperature, Platt, beta and Dirichlet. They use a fixed internal stratified 75/25 split of the reserved calibration set for fitting and monitoring. Isotonic fits the reserved calibration set directly. Both evaluate on the separate final test set. Dirichlet expects two-dimensional logits and rejects the single-logit binary branch; two-class GCN/GAT logits have two columns and do not select that branch. Method menus should carry support status as well as method availability.

| YAML path | Checked-in default | Type and null meaning | Status and category | Actual use and constraints |
| --- | --- | --- | --- | --- |
| `evaluate.initial_temperature` | `1.2` | real | R · Expert | `TemperatureScaling` initialization only. Proposed positive finite. |
| `evaluate.n_bins` | `10` | int | R · Advanced | `CalibrationMetrics` for test ECE/MCE and reliability diagnostics enforces ≥1. Does not change the calibrator's internal stopping metric, which uses 15 bins. |
| `evaluate.calibration_method` | `dirichlet` | str | R · Advanced | `CalibratedModel` enforces `temperature`, `isotonic`, `platt`, `beta`, `dirichlet`. |
| `evaluate.n_epochs` | `1000` | int | R · Advanced | Parametric fitting loop only. Proposed ≥1. |
| `evaluate.lr` | `1.0e-4` | real | R · Expert | Parametric Adam only. Proposed finite >0. |
| `evaluate.weight_decay_adam_optimizer` | `5.0e-4` | real | R · Expert | Parametric Adam only; nonnegative required by Adam. Separate from classifier decay and calibration loss penalties. |
| `evaluate.verbose` | `true` | bool | R · Internal | Parametric `_train_model` enables calibration console output and optimizer diagnostics plots/text. Preserve in presets and expert YAML. |
| `evaluate.factor_learning_rate_scheduler` | `0.4` | real | R · Expert | Parametric ReduceLROnPlateau on calibration-validation ECE; factor <1, proposed (0, 1). |
| `evaluate.patience_learning_rate_scheduler` | `20` | int | R · Expert | Parametric calibration scheduler patience. Proposed ≥0. |
| `evaluate.patience_early_stopping` | `100` | int | R · Expert | Parametric `EarlyStopping` on calibration-validation ECE. Proposed ≥1. |
| `evaluate.min_delta_early_stopping` | `1.0e-3` | real | R · Expert | Parametric minimum monitored improvement tolerance. Proposed ≥0. |
| `evaluate.reg_lambda` | `1.0e-2` | real | R · Expert | `_compute_regularization` penalizes parametric weights, including temperature. Dirichlet penalizes off-diagonals only. Proposed ≥0. |
| `evaluate.reg_mu` | `1.0e-2` | real | R · Expert | `_compute_regularization` penalizes Dirichlet bias only. Proposed ≥0. |
| `evaluate.eps` | `1.0e-8` | real | R · Internal | Probability clamp for Platt/beta logs. Stored in calibrator state. Multiclass isotonic divides by actual positive row mass; Dirichlet/temperature do not use this option for transformation. Proposed finite (0, 0.5), with numerical suitability checked for the working dtype. |
| `evaluate.seed` | `42` | int or null for unseeded parametric fitting | O · Advanced | Parametric split uses exact configured seed; initialization uses derived cycle stream. Enforced unsigned 32-bit; omission 42. Isotonic keeps provenance but performs neither seeded operation. |

### Drift

| YAML path | Checked-in default | Type and null meaning | Status and category | Actual use and constraints |
| --- | --- | --- | --- | --- |
| `analyze_drift.seed` | `42` | int or null for unseeded | O · Advanced | Drift derives cycle and method streams for permutations, projections and stochastic diagnostics. Enforced unsigned 32-bit; omission 42. |
| `analyze_drift.n_permutations` | `1000` | int | R · Advanced | `_perform_permutation_test` cap per statistic. Requires ≥1 to avoid empty denominator; proposed validate before run. |
| `analyze_drift.use_max_sliced_wasserstein` | `null` | bool or null for off | R/D · Internal | Truthy selects max-sliced transport, taking precedence over Sinkhorn. Encode from one future method selector. |
| `analyze_drift.n_projections` | `100` | int | R · Expert | Max-sliced random directions per multivariate statistic only, skipped for one feature. Proposed ≥1. |
| `analyze_drift.use_sinkhorn_wasserstein` | `null` | bool or null for off | R/D · Internal | Truthy selects Sinkhorn only if max-sliced is off. Both off selects exact Wasserstein. |
| `analyze_drift.sinkhorn_wasserstein_reg` | `0.1` | real | R · Expert | Sinkhorn regularization in squared transformed feature-space units, skipped for one feature. Proposed finite >0. |
| `analyze_drift.gamma` | `null` | real or null for pooled median heuristic | R · Expert | MMD RBF and kernel PCA use the same effective gamma. Proposed finite positive explicit value. |
| `analyze_drift.early_stopping.enabled` | `true` | bool | R · Advanced | `_perform_permutation_test` enables the one-sided running p-value stop. Missing, null or empty **mapping** uses an internal disabled fallback; a nonempty partial mapping is not merged by the consumer. |
| `analyze_drift.early_stopping.confidence_threshold` | `0.01` | real | R · Expert | Active only with early stopping. Stop if current p-value is strictly below this threshold. Proposed (0, 1). This is not the reporting significance cutoff. |
| `analyze_drift.early_stopping.min_permutations` | `100` | int | R · Expert | Active only with early stopping. Proposed ≥1. If greater than total budget, early stopping is unreachable and all permutations run. |

## Usability and conditional controls

- **Current dashboard.** `app.params_tab` previews YAML, shows a textual diff, and writes after a syntax check when the three-second autosave threshold is met on a rerun. It lacks semantic and cross-field checks. An explicit **Validate and preview** followed by **Apply** would make it clear which configuration runs. Preserve expert YAML access and report the full resolved file before execution.
- **Model, graph and scaler.** Use a model selector and show its relevant settings. Show one connectivity selector with one numeric input, mapping inactive slots to null. Graph construction enforces exactly one active rule. Avoid competing loop controls and keep GCN normalization internal. Provide a separate affinity selector (Unit or Inverse quadratic) and show scale only for the latter. Use a scaler selector with its own option set. Preserve current fit populations and graph locality.
- **Calibration.** Show diagnostics bins for every method. Reveal epoch, optimizer, scheduler and stop controls only for parametric methods, initial temperature only for temperature, and the two penalty controls together only for Dirichlet. Keep inactive legacy values in the resolved file. Multiclass isotonic uses one fitted map per class column and raises an explicit error if normalization is undefined. See [M2.1a verification](#m21a-multiclass-isotonic-verification). M1.7 verified Dirichlet.
- **Drift.** Offer Exact Wasserstein, Max sliced and Sinkhorn as one selector. Map choices to the two existing flags. For legacy imports with both flags true, surface the current max-sliced precedence and preserve the explicit values unless the user changes selection. Show projection count or regularization only for the corresponding method and multivariate input. Show maximum and minimum permutations together. Warn when the stop is unreachable. Reporting uses an existing fixed p-value cutoff of 0.05; the stop threshold is a different setting. Low budgets give coarse p-value resolution and spatial dependence still limits inference.
- **Device and seeds.** Offer Auto, CPU and CUDA with optional visible-device index. Record requested and effective devices separately; `auto` can resolve differently without changing the DVC key. Leave CPU-only generation and drift intact. Distinguish *use preset seed*, *use this integer* and *run unseeded*. Missing and explicit null must remain distinct. Maintain each seed's current scope and replay rules, including the shared base partition seed and fixed t-SNE seed.
- **Redundancy and inactive settings.** Every checked-in leaf has a production consumer or conditional role. Current Research leaves such as `threshold_binary`, `initial_temperature`, `eps`, and transport-specific options can be inactive for the selected workflow. DVC may still track their containing group. Legacy `data_utils.scaler_setup` is outside production fitting and does not define the current scaler contract. Keep compatibility keys and avoid presenting internal options such as normalization and transport flags as independent toggles.

### Requested and realized sample counts

Label the inputs **Requested bootstrap observations** and **Requested observations per prediction batch**. Display the bootstrap class minimum as a separate, intentional setting. It appends observations for classes below the floor. M1.7 requested 400 bootstrap observations and realized 440 because of the 50-per-class floor. Before generation, show the target and floor and explain that the realized count is determined by generation. Afterward show realized total, class histogram and rows added for representation when available from the run. Derive the realized count from artifacts and keep it out of the configured target. Coordinate placement has an attempt budget and can fall short in crowded geometries. Treat that as a feasibility/runtime issue rather than promising target fulfillment. Optional batch top-up likewise can increase realized batch size. Top-up positions use a separate path from spacing-enforced initial placement.

### Partition labels and summaries

Use **Holdout share of full base** for `test_size`, **Calibration share of holdout** for `calib_size`, and **Cross-validation folds within development** for `n_splits`. For the two default fractions, the approximate total shares are 70% development, 9% calibration, and 21% final test, before rounding. In M1.7 base cycle zero, the realized 440 rows split into 308 development, 40 calibration and 92 final test. Per-fold validation is drawn within the development pool. The final model refits on that full pool.

Predict counts using the realized base size where available. For fractional holdout `h` and calibration share `c`, sklearn first rounds up `N × h` and then rounds up `holdout × c`. For integers use exact requested counts, and for null retain sklearn's 25% fallback at the relevant split. Present integer/count and null choices in expert editing. Check class feasibility using realized labels, including the parametric calibrator's internal split and per-fold training support. Stratification can fail for small classes or partitions even when the total size appears sufficient. Preserve split order, seed behavior and changing populations across cycles.

## Proposed presets

Presets are overlays on the full checked-in configuration. Values below are recommendations for later implementation, with all unlisted settings inherited from Research. A preset controls effort and workflow scope; it is not an accuracy ranking.

| YAML path | Quick demo | Standard | Research |
| --- | ---: | ---: | ---: |
| `default_model` | gcn | gcn | gcn |
| `device` | cpu | auto | auto |
| `cycles` | 2 | 2 | 3 |
| `data.base.n_samples` | 400 | 700 | 1000 |
| `data.pred.n_samples` | 100 | 150 | 200 |
| `data.base.n_splits` | 3 | 3 | 3 |
| `train.n_epochs` | 15 | 100 | 1500 |
| `train.patience_early_stopping` | 5 | 20 | 50 |
| `evaluate.n_epochs` | 20 | 100 | 1000 |
| `analyze_drift.n_permutations` | 19 | 199 | 1000 |
| `analyze_drift.early_stopping.enabled` | true | true | true |
| `analyze_drift.early_stopping.min_permutations` | 100 | 100 | 100 |

Quick demo is the verified two-cycle CPU GCN Experiment A configuration. Its 19 permutations all run because the early-stop minimum is 100. Training scheduler patience 20 and calibration early-stop patience 100 also remain inherited, beyond the respective 15- and 20-epoch budgets. Retaining those settings preserves the verified configuration. Two cycles expose performance aggregation. The verified one-cycle GAT CPU and GCN CUDA variants are possible explicit overrides, with no aggregation plot for one cycle. The bootstrap floor remains 50, the scaler remains RobustScaler, graph connectivity/affinity remain unchanged, and calibration remains Dirichlet. Seeds stay 110 for bootstrap and partitioning and 42 elsewhere. [M1.7 verification report](end_to_end_verification.md)

Standard is an unverified intermediate budget. It keeps the existing scheduler patience and allows the original drift stop rule to activate after 100 of up to 199 permutations. Review the training stop/scheduler interaction and actual calibration populations during M2.5. Changing calibration method changes whether an epoch budget is relevant. Larger populations and exact transport can materially increase runtime. Research preserves the complete checked-in configuration, whose full default-budget reproduction is still pending.

## Resolver architecture and compatibility

Use a small Python module with typed inputs and path-specific validators, sharing the existing YAML dependencies. The core should be independent of Streamlit, DVC execution and filesystem writes. Conceptually, return a fresh resolved mapping plus validation issues and a preview summary from a baseline mapping, optional preset and explicit overrides.

1. Load a canonical baseline snapshot. Apply the named preset, then explicit overrides. For a legacy full YAML import, use a separate compatibility path with no implicit preset. Use a missing-value sentinel so null is an actual override. Replace sequences as units. Treat scaler options as a method-owned map so an empty map can clear options and switching methods cannot inherit invalid keys.
2. Validate shape and types without coercing booleans into integers or silently replacing unfamiliar values. Validate rule combinations and method-specific options. Put hard runtime requirements in errors, return warnings for inactive settings and unreachable budgets, and identify proposed restrictions explicitly before making them errors. Separate static validation from device checks and realized-data feasibility checks. Preserve ambiguous but currently accepted legacy combinations until an explicit compatibility policy is approved.
3. Return the complete existing `params.yaml` structure with both model blocks and required inactive keys. Keep friendly selectors, preset identity, effective devices and realized counts in the draft or run summary, outside DVC-tracked YAML. Preserve explicit user seed and null semantics. Warn about compile-time changes to cycle count, selected model and fold count.
4. Share the resolver across future Streamlit preview/apply and CLI validate/resolve entry points. Keep original input intact. Output a preview or explicitly named file by default. Replacing the active file should be an explicit operation, with validation before an atomic write. DVC and production stages continue to consume a resolved `params.yaml` in their working directory.

### DVC boundary

[scripts/setup_dvc.py](../scripts/setup_dvc.py) embeds `cycles` and `default_model` in stage commands and topology, and `data.base.n_splits` controls the enumerated fold visualization outputs. Regenerate definitions when those values change. Bootstrap and Merge track `data.base` and graph settings. Batch tracks those groups plus `data.pred`. Train, Eval and predict track selected model, graph, schema and device. Train additionally tracks `train` and scaler settings; Eval tracks `evaluate`. Drift tracks its block, graph settings and schema. Performance reads declared cycle metrics and uses its generated cycle argument. Retain this tracking and output contract. Group-level tracking can invalidate a stage even when a changed member is inactive for the selected method. See the [DVC integrity audit](dvc_integrity.md).

### Regression plan for later milestones

- Compare the inventory and serializable schema against every checked-in YAML leaf. Research must resolve to the same values and preserve distinctions between null, false, zero, empty mapping and omission.
- Test precedence, unknown and misplaced keys, separate method blocks, scalar types, round-trip behavior and preservation of the original file on validation failure. Cover accepted existing constructor options and library null/count split behavior.
- Test model/device combinations, exclusive graph rules, affine scale compatibility, scaler option isolation, method-specific calibration and drift fields, unreachable budgets and dual-flag compatibility. Check class feasibility using realized labels and ensure the class floor remains active.
- Generate candidate DVC YAML in temporary directories and compare topology, commands and tracked groups for equal resolved input. Check stage invalidation boundaries and saved seed provenance.
- In M2.5, smoke-test the resolved Research fixture for scientific equivalence and use reduced production runs for Quick and a budget-appropriate Standard check, with actual CPU/CUDA claims limited to tested configurations.

### Open decisions

- Decide where the canonical baseline should live if users can overwrite their active `params.yaml`, and how to version presets without changing old experiment meaning.
- Confirm Standard's runtime target and preset budgets after measurement. Decide whether unreachable stopping controls should warn only or be visually marked inactive in new drafts.
- Define expert import rules for currently tolerated values, unknown keys and dual Wasserstein flags. Decide which new validation constraints can be enforced without breaking existing experiments.
- Decide how a future guided selector should explain isotonic zero-mass rows. Selecting a fallback distribution would require a separate methodological decision.
- Decide when to promote generated class names, seeds, and simulation geometry to the guided workflow, and whether to support a distinct bootstrap-only mode.

### M2.1a multiclass isotonic verification

The fitter stores an ordered list of per-class threshold maps. Previously every multiclass column went through the binary helper, which always took map zero. A deterministic regression with three distinct fitted maps and an independent NumPy interpolation reference failed before the fix, with 8 of 9 probabilities differing. Inference now pairs each class column with its own map and normalizes by the row's actual positive sum. Positive sums below the previous epsilon floor still yield a normalized row. Zero-mass and non-finite rows raise a clear error, and a mismatch between map count and logit width is rejected. The single-logit binary path continues to use the fitted positive-class score and its complement.

Existing trusted artifacts already store all fitted maps and remain readable through the existing CPU-mapped loader. Tests cover CUDA-tagged storage loaded on CPU and legacy state without seed provenance. Corrected inference can reuse existing valid, bound thresholds without refitting the classifier or the calibrator. Previously published multiclass isotonic evaluation metrics and plots, predictions, and performance summaries derived from them need regeneration for that correction to be reflected on disk. Existing provenance and binding checks still apply. DVC already tracks `calibration_utils.py` in evaluation and prediction, and also in training for shared calibration diagnostics. Those file-level dependencies can invalidate stages beyond the affected method without any change to the DAG.

The regression verifies class-specific interpolation and normalization. It does not establish a fallback for all-zero isotonic scores or benchmark calibration quality.
