# Streamlit UI/UX design and feasibility audit

M3.1 records the design for the synthetic spatial ML workbench planned for `0.2.0.dev3`. The audited application is `0.2.0.dev2` on `develop/v0.2.0` at revision `1042656`. Streamlit, Plotly, DVC and the scientific backend are the foundation. Home, Configure, Run and Results are the four primary destinations. Results has an experiment overview and a cycle explorer. Each page has one main task and each results view loads the selected evidence.

## Existing interface audit

The seven-tab audit describes the pre-M3.2 version of `app.py`. The local capture index is `/tmp/mineral-ui-review.RL7qaf/REVIEW.md`, with 196 overlapping Chromium viewport PNGs at 1920 × 1080 across the seven tabs. Representative sequences from all seven tabs were inspected, including the active configuration and Quick demo draft, bootstrap graphs and the separate cycle 1 prediction graph, each cycle 1 diagnostic family, the prediction table scroll and the performance overview. The index covers three populated learning cycles and a 200-node cycle 1 prediction graph. It records a live session and does not establish which configuration produced the displayed outputs. The Quick demo preview in the captures was a draft. The captures are local review evidence and may expire.

The repository workspace contained `results/.gitkeep` and no `dvc.lock`. The retained [M2.5 verification workspaces](configuration_verification.md) under `/tmp/mineral-m25.SkdwUb/a` and `b` supplied the artifact samples and DVC records. These are separate two-cycle Quick demo and Standard CPU runs. Each contains 124 result files. M3.1 inspected their schemas, selected dimensions, relationships, locks, and declared output hashes and directory manifests without reproducing an experiment. The dimensions below distinguish contractual shapes from illustrative dimensions in Quick demo.

| Current tab | All dashboard controls and user-facing outputs | Audit and destination |
| --- | --- | --- |
| Parameters | Preset selector (Quick demo, Standard, Research), explicit **Load preset into draft**, full **Draft YAML** textarea for every setting, **Validate & Preview**, severity/path messages or YAML errors, active-comparison warnings, DVC regeneration notice, expanded resolved YAML and semantic diff (or no-change message), and **Apply configuration** for current valid previews. Separate **Generate & Run Full Pipeline** button, spinner, streamed command output and completion/failure feedback. | Long YAML and consecutive alerts dominate the first visit. Move configuration controls and findings to Configure. Move execution to Run. |
| Data | Independent `0…N` view selector, **Run Bootstrap Stage** and full-pipeline guidance. Base t-SNE/silhouette PNG and each full-base, test, calibration, training-fold and validation-fold 3D graph. For `c > 0`, an additional prediction-data graph. Filename expanders, Plotly hover/zoom/rotation and class legend, plus missing data/graph messages. JSON is preferred per stem, HTML fallback uses a scrolling embed. | Graphs with dense edges and long titles compete for attention. Put population and fold selectors above one canvas in Cycle explorer. Place Bootstrap action on Run. |
| Train | Independent `1…N` cycle selector, **Run Training Stage**, one four-panel PNG per CV fold and missing-results warning. | Put the fold summary, selected fold and full-development refit details in Results → Model. Put the stage action on Run. |
| Evaluation | Independent cycle selector, **Run Evaluation Stage**, calibrated test confusion matrix, pre/post reliability and confidence distribution, optional calibration optimizer loss and gradient PNG, and raw test metrics JSON. | Put named metrics, matrix and reliability diagram in Model and Calibration. Put the stage action on Run. |
| Prediction | Independent cycle selector, **Run Prediction Stage**, calibrated batch confusion matrix, score/label/entropy histogram grid and sortable/scrollable predictions dataframe. Batch MCC appears in the matrix title. The batch metrics JSON exists on disk and feeds Performance. | Lead with spatial predictions and a compact sample table. Place batch classification diagnostics and named metrics in Model. Put the stage action on Run. |
| Drift Analysis | Independent cycle selector, **Run Drift Analysis Stage**, four PNGs covering pairwise distances, per-feature marginals, mutual information and PCA/Kernel PCA, plus text of three permutation tests and their interpretations. | Lead with reference/batch sizes, method, statistic and p-value. Select a single diagnostic in Drift. Put the stage action on Run. |
| Performance | **Run Performance Analysis Stage**, one two-panel metrics PNG and missing-plot message. Action disabled below two learning cycles. | Put the trajectory in Experiment overview and the legacy figure on demand. Put the stage action on Run. |

At zero learning cycles, Bootstrap remains available and the learning-stage and aggregation controls are disabled. Each command exposes merged stdout/stderr and start or exit errors. Run buttons use the applied file and regenerate DVC definitions. The current seven `st.tabs` use eager rendering, so changing a widget can also read and prepare plots on hidden tabs. Collapsing filename expanders leaves the current plot loading in place. The selected cycle is independent on each tab. Data's current “Cycle to view” refers to a base snapshot, while Evaluation and Prediction use the same label number for a learning cycle. The redesign makes that relationship explicit. The existing capture's stacked folds and tiny full-width labels motivate selecting a comparable population and a single chart first.

## Scientific artifact and diagnostic inventory

Paths below are relative to `results/`. For learning cycle `c ≥ 1`, **B(b)** is `data/base/cycle_{b}/`, **I(c)** is `data/prediction/cycle_{c}/`, **T(c)** is `trained/cycle_{c}/`, **E(c)** is `evaluation/cycle_{c}/`, **P(c)** is `prediction/cycle_{c}/`, and **D(c)** is `drift_analysis/cycle_{c}/`. Bootstrap produces B(0), merge `prepare_next_cycle_data_{c}` produces B(c). Train, evaluate and drift for cycle c consume B(c−1). Data Prep opens the next base population using incoming simulator labels from I(c). P(c) records the model's predictions separately. Stage identifiers and output granularity are declared by [scripts/setup_dvc.py](../scripts/setup_dvc.py).

`N` denotes rows in the selected population, `F` features, `C` classes, `K` CV folds and `E` stored directed edges including any loops. **Direct** means the artifact supports the proposed view as saved or through a read-only presentation adapter, with any verification gate indicated. **Limited** means the supported view has less information than a fully interactive or sample-level counterpart. Paths with `.{json,html}` are two exports of the **same** figure.

### Data, graph and simulator outputs

| Source and format | Producer and scope | Payload, dimensions and scientific purpose | Proposed location and support |
| --- | --- | --- | --- |
| B(b)`base_data.pt` | Bootstrap or Merge, per base snapshot | Versioned PyG overview graph. Float64 raw X `N×F` and coordinates `N×3`; integer labels, stable IDs, ordered names, generation/partition seeds where recorded, dataset and graph digests and policy. Descriptive view of all rows. | Overview growth and Data population browser. **Direct** through trusted raw reader. Overview edges span partitions by design and are descriptive. |
| B(b)`fold_data.pt` | Bootstrap or Merge, per base snapshot and fold | K pairs of independent PyG train/validation graphs. Each carries local IDs, coordinates, labels, raw features, parent row indices and dataset/split/graph digests, role and zero-based fold. Supports disjoint CV folds inside development. | Data → Fold → Train or Validation. **Direct** with parent checks. UI labels folds starting at one. |
| B(b)`test_data.pt`, `calib_data.pt` | Bootstrap or Merge, per base snapshot and held-out role | Two independent PyG graphs. Same membership, shape and binding convention as folds, with no fold index. Test is reserved for reporting and calibration for calibrator fitting. | Data → Final test or Calibration. **Direct** for observations and stored edges. Test and calibration score maps lack row-level evaluation output. |
| B(b)`base_data_graph.{json,html}`, `test_graph.{json,html}`, `calib_graph.{json,html}`, `train_graph_{fold}.{json,html}`, `val_graph_{fold}.{json,html}` | Bootstrap or Merge, graph family repeated per base snapshot, role and fold | Plotly 3D node/edge figure, class legends with counts, node and directed-edge totals, unique links, self-loops, isolates, density, non-loop degree and lengths (min/median/mean/max), effective KNN quota when applicable. Displays each physical link once. | Data → Stored graph and graph statistics. **Direct** via JSON, one role/fold at a time. Export lacks stable IDs and full feature arrays. |
| B(b)`tsne_comparison.png` | Bootstrap or Merge, per base snapshot | Four raw-feature 2D t-SNE projections (perplexity 5, 30, 50, 100; fixed seed 42), colored by observed class. Silhouette is computed in raw feature space before embedding. Quick demo bootstrap PNG 4499×3543. | Data → Feature separation diagnostic. **Direct** image with explanatory caption. **Limited** for interactive embeddings or exact silhouette extraction. |
| B(0)`hotspots.npy`, `prototypes.npy` | Bootstrap, experiment seed inputs reused by Batch | NumPy `H×3` hotspot positions and `C×F` class prototypes. Quick demo shapes 3×3 and 4×5. Synthetic generator context; batch-evolved hotspots and prototypes are not exported. | Data → Simulator metadata, optional technical disclosure of bootstrap shapes and role. **Direct** with trusted numeric readers. |
| I(c)`pred_data.pt` | `generate_pred_data_cycle_{c}`, per incoming batch | Versioned raw PyG graph with X `N×F`, coordinates `N×3`, integer simulator labels, stable IDs, generation seed, ordered schema and parent dataset digest. On every raw graph, COO edge index is `2×E`, physical distances and weights are `E`, affinity attribute `E×1`. | Spatial predictions and Data → Incoming batch. **Direct** for coordinates, raw attributes and topology. Joining model scores has a separate verification gate. |
| I(c)`prediction_data_graph.{json,html}` | Batch, per incoming batch | Same 3D diagnostic and graph statistics as base exports, with roles and colors for simulator labels. | Spatial predictions → Stored graph. **Direct** for viewing, **limited** for score overlays. |

Three folds yield nine distinct graph instances per base snapshot across five roles (full base, test, calibration, fold train, fold validation). Each base or batch graph export has two file formats. Increasing fold or cycle count adds instances of these same diagnostic types.

### Training, test evaluation and calibration

| Source and format | Producer and scope | Payload, dimensions and scientific purpose | Proposed location and support |
| --- | --- | --- | --- |
| T(c)`training_plots_dataset_{fold}.png` | `train_cycle_{c}`, per cycle/fold | 2×2 panels over completed CV epochs: train/validation loss, validation macro Fβ with β=0.5/accuracy/MCC, training gradient norm, validation ECE/MCE. Quick demo 3638×3321. Inspection of optimization, classification and calibration over epochs. | Model → Training → one fold. **Direct** image. **Limited** for structured per-epoch curves, which are not persisted as a table. |
| T(c)`graph_binding.json` | Train, per learning cycle | Graph/model settings, deployed development-graph digest, file digests and training record: dataset and development partition, observation count, epoch budget, fold best/stopped information and CV learning-rate histories, selected refit schedule and effective seed/device. | Model → Training summary and Provenance. **Direct** versioned JSON. Separate CV diagnostics from the deployed fresh full-development refit. |
| T(c)`preprocessing.json`, `model.pt` | Train, per learning cycle | Fitted transform with ordered `F` feature names, class mapping, scaler parameters and fitted state, dataset/partition IDs, checkpoint digest. Model is a variable-shaped PyTorch state dict for GCN or GAT. | Model/Provenance → bundle metadata. **Direct** JSON and byte checks. Dashboard charting needs no deserialized weights. |
| E(c)`metrics.json` | `evaluate_cycle_{c}`, per final-test population and calibration state | Two objects (`uncalibrated`, `calibrated`), each with `acc`, `f1`, `mcc`, `ece`, `mce`. Class-balanced sample weights feed classification and calibration metrics. `f1` stores macro Fβ with β=0.5. | Overview test headline/trend, Model and Calibration before/after tables. **Direct**. Label metrics by method and population, and use lower-is-better labels for ECE/MCE. |
| E(c)`confusion_matrix.png` | Evaluate, per final-test population, after calibration | Actual vs calibrated argmax counts over represented labels, with weighted MCC in title. Four-class sample 2341×2100. | Model → Final test → Confusion matrix. **Direct** static image. **Limited** for per-class tabular counts and test error brushing because source counts and rows are not saved separately. |
| E(c)`reliability_diagram.png` | Evaluate, per final-test population, before and after calibration | Accuracy vs confidence by occupied bins, identity diagonal, ECE/MCE in legend, paired bin-weight bars. Quick demo 2069×2067. M3.1a labels the secondary axis “Sum of Sample Weights per Bin” and places overconfidence below the diagonal and underconfidence above it. | Calibration → Reliability. **Direct** image with a caption explaining weighted bars and the correct reading (below diagonal means overconfidence). **Limited** for exact bin tables, responsive native redraw or per-class reliability. |
| E(c)`calibration_{method}_loss_and_gradient.png`, `optimum_params.txt` | Evaluate, per learning cycle, verbose parametric methods only | Two log-scale panels over optimizer iterations; textual fitted temperature, Platt/beta coefficients or Dirichlet weights/biases. Dirichlet sample 2000×600 plot. Fit diagnostics on the reserved calibration workflow. | Calibration → Optimizer details and parameters. **Direct** when produced. Isotonic and verbosity off make these inapplicable. |
| E(c)`calibration_binding.json`, `calibrator.pt` | Evaluate, per learning cycle | JSON binds development bundle, source dataset, calibration split and calibrator bytes. PyTorch state is method-specific and may carry threshold maps, learned parameters and seed metadata. Calibrator operates on the deployed model's outputs. | Calibration/Provenance → method, binding and fit population. **Direct** for binding and digest-based integrity, with method from verified settings. Dashboard need not load the calibrator. |

Parametric calibration fits and monitors on an internal stratified split of the calibration partition. Isotonic fits on that reserved partition directly. The separate test partition is used for the evaluation metrics and reliability figure. The helper `plot_roc_curve` and older graph/scaler helpers are not current UI outputs or DVC result artifacts.

### Prediction, drift and cross-cycle outputs

| Source and format | Producer and scope | Payload, dimensions and scientific purpose | Proposed location and support |
| --- | --- | --- | --- |
| P(c)`predictions.csv` | `predict_cycle_{c}`, per labeled synthetic incoming batch | `N×(2C+4)`: `calib_prob_class_i` and `uncalib_prob_class_i` for each class, calibrated `predicted_label`, `true_label`, `calibrated_entropy`, `uncalibrated_entropy`. Four classes yield 12 fields. No IDs or coordinates in CSV. | Spatial predictions → score view, sample table and download. **Direct** standalone; **limited** to verified same-production-order graph joins for spatial overlays. |
| P(c)`confussion_matrix_.png`, `histograms.png` | Predict, per labeled synthetic incoming batch | Confusion counts with weighted MCC in title, and histogram grid of numeric CSV fields (both probability sets, predicted and true labels, both entropies). Quick demo 2341×2100 and 2000×1000. | Model → Prediction batch → Confusion matrix and Spatial predictions → Score distributions. **Direct** originals; CSV allows accessible confusion counts, support and selected histograms as display derivations. |
| P(c)`metrics.json` | Predict, per labeled synthetic incoming batch | `acc`, `f1` (macro Fβ, β=0.5), `mcc` with class-balanced sample weights, after calibration. | Overview batch headline/trend and Model → Prediction batch. **Direct**. Batch ECE/MCE is unavailable. |
| D(c)`drift_results.txt`, `drift_metadata.json` | `analyze_drift_cycle_{c}`, per comparison of **all previous base rows** and incoming batch | Text provides statistic, permutation p-value and threshold interpretation for MMD, energy and selected Wasserstein method. JSON identifies reference/incoming digests, model, fitted representation, ordered fields, configuration and seed/effective gamma and stopping policy. Realized permutation counts or null distributions are not stored here. | Drift → tests table and Provenance. **Direct** through a strict text parser with raw fallback and effective policy from JSON. The 0.05 report cutoff and early-stop threshold are distinct. |
| D(c)`distance_distributions.png` | Drift, per cycle pair | Two panels: within-reference, within-incoming and cross-set Euclidean distance densities and box plots in frozen model feature space (sampled diagnostic). Quick demo 3553×1468. | Drift → Distances. **Direct** static diagnostic. |
| D(c)`marginal_distributions.png` | Drift, per cycle pair, repeated across F features | Raw-feature density histogram, mean lines and KS/Mann–Whitney p-value annotations per feature. Code obtains the annotations from transformed columns. Quick demo 4471×2370. | Drift → Feature distributions. **Direct** with representation caption. Selected-feature raw summaries are feasible from graphs, while exact published annotation values are only embedded in the image. |
| D(c)`mutual_information.png` | Drift, per cycle pair | Three `F×F` normalized mutual-information matrices: reference, incoming, reference minus incoming in the model feature-space calculation. Quick demo 4427×1267. | Drift → Feature relationships. **Direct** static diagnostic; structured matrix values are not exported. |
| D(c)`pca_kpca_grid.png` | Drift, per cycle pair | Combined-fit feature-space projections, up to three paired 2D views each for PCA and RBF Kernel PCA, with PCA explained variance ratios, Kernel PCA eigenvalue-derived values and a descriptive panel. Panels depend on dimensions. Quick demo 5370×4470. | Drift → Projections. **Direct** static diagnostic. Projection coordinates and axes are not persisted as comparable numerical outputs. |
| `cycles_performance_analysis/cycles_performance.png` | `analyze_cycles_performance`, experiment-wide, emitted for more than one cycle | Two-panel calibrated final-test and calibrated prediction-batch trajectories for accuracy, macro F0.5 and MCC over declared learning cycles. Quick demo 4467×1764. | Overview → Original performance figure. **Direct** image. Rebuild interactive trajectories from individual JSON files to expose exact values, counts, missing-cycle gaps and pre/post test metrics. |

### Configuration and execution evidence

| Source and format | Producer and scope | Available evidence and use |
| --- | --- | --- |
| `config/defaults.yaml`, `config/presets/{quick_demo,standard,research}.yaml`, `params.yaml` | Distributed configuration and explicit Apply/CLI export, experiment-wide | 88 canonical scientific leaves with conditional options. Presets carry version, identifier, name, description and overrides. Active YAML provides complete settings without preset origin or run identity. Configure, Run and Results can show these directly with their respective labels. |
| `dvc.yaml`, `dvc.lock` at workspace root | Setup generator and DVC reproduction, experiment-wide with stage records | Definition supplies topology, commands, tracked params and paths. Lock v2 records commands, tracked parameter values, dependency and output hashes (including directory hashes) and sizes. It does not record a full immutable experiment snapshot, app version, global run ID, user, timestamp, or selected preset. Use with content checks for per-stage currentness. |
| Command stream and exit status | Dashboard subprocess during setup and reproduction, per launch | Merged stdout/stderr, command, explicit DVC run/skip lines, occasional stage diagnostics/timing and exit status. Current dashboard has no durable run record. Isolated M2.5 verification logs are separately retained evidence. Use on Run for status and technical details. |

## User journeys and page hierarchy

| Journey | Path and outcome |
| --- | --- |
| First synthetic exploration | Home → **Explore demo** → view available evidence in Results or go to Configure with Quick demo highlighted when results are missing. Explicitly load the preset, validate and apply, then run on Run. Existing outputs can be explored with their provenance state displayed. |
| Routine experiment | Home or Configure → choose Standard or Quick demo → edit five Basic controls → validate and apply → Run → inspect the Overview and open a cycle. |
| Research and diagnosis | Configure → Advanced or Expert → review findings and semantic diff → Apply → Run full or target stage → Results → choose cycle, population or fold → open the specific technical diagnostic. |
| Inspect a spatial prediction | Results → open a cycle → Spatial predictions → color the batch by predicted class or uncertainty → inspect a sample in map/table → download the supported view or original CSV. |
| User-managed data | Home explains training with user data and independent inference as separate planned workflows. The available demo uses synthetic labels. Preserve the supported synthetic entry point and keep future workflows visibly unavailable. |

### Annotated page wireframes

These wireframes describe reading order. The native navigation appears the same on all four pages and highlights the active destination.

```text
  Home     Configure     Run     Results

  Mineral Classifier
  Explore synthetic mineral observations, graph models and calibrated predictions.

  [Explore demo]             [Configure experiment]
   Opens results if present   Starts with the current applied settings.
   or offers Quick demo.

  Workspace
  Applied settings summary | Results status with reason | [Open Results]
  Synthetic workflow and evaluation protocol ▸
  Use my own data · Planned     Predict with an existing model · Planned
```

```text
  Configure
  Applied settings summary      Draft status / unresolved changes
  Experiment preset [Quick demo ▾] [Load preset into draft]
  Basic
  Model [GCN ▾]   Device [CPU ▾]   Learning cycles [2]
  Requested bootstrap observations [400]   Requested batch observations [100]
  Advanced ▸  Scientific groups, one expanded at a time
  Expert YAML ▸  Full draft, editable
  [Validate & Preview]   Findings counts and blocking issue summary
  Warnings and information ▸    Scientific differences ▸    Resolved YAML ▸
  [Apply configuration]          Opens Run only through an explicit next action
```

```text
  Run
  Applied configuration · Synthetic · model, device request, cycles and sample targets
  [Configure]   Draft differs from applied settings (when applicable)
  [Run full experiment]
  Stage status · Validating → Preparing DVC pipeline → Reproducing
  Running / Completed / Failed, with observed stage and actionable message
  Technical logs ▸
  Advanced stage execution ▸ [Stage ▾] [Cycle ▾ when applicable] [Run stage]
  [Open Results]  Artifact availability is evaluated separately from command status
```

```text
  Results
  Applied configuration · Results correspondence · [Provenance details ▸]
  [Experiment overview | Cycle explorer]

  Overview                                 Cycle explorer
  Final test · Cycle c, n                  [Cycle c ▾]    [Back to overview]
  Accuracy | ECE after (before below)      Base used b=c−1 · Batch c · Appended base b=c
  Prediction batch · Cycle c, n           [Section ▾]  Data / Model / Calibration /
  Batch accuracy                                           Spatial predictions / Drift
  Changing evaluation populations         [Population / fold / diagnostic ▾ as needed]
  [Trend metric ▾]  Fixed-height chart       One selected chart and textual summary
  Dataset growth and cycle summary         Technical detail ▸   Accessible data/table ▸
  Bounded-height table [Cycle ▾] [Inspect]   [Start two-cycle comparison ▸ when eligible]
```

## Results architecture

### Experiment overview

- Put applied model, requested device, cycle count and a short configuration fingerprint beside the result status. Expand Provenance for source-cycle identities, actual device when recorded, the relevant tracked settings and output verification. Identify an exact preset-equivalent configuration only after a value-and-type match and label it “Matches Quick demo v1”, for example. Preset selection history is unavailable after reload.
- Use the latest **complete matching learning cycle** by default when available. Display the cycle number and sample count with each headline. Lead with calibrated class-weighted test accuracy and test ECE after calibration, with the corresponding before-calibration values on the same test cohort. The separate prediction-batch section shows calibrated class-weighted accuracy. Offer macro Fβ (β=0.5), MCC, and test MCE through a metric selector or secondary table. Missing populations and metrics get explicit labels. Cycle 0 has data diagnostics and no model headline. If only historical or unverified evidence exists, open the artifact browser with that state shown and avoid presenting those numbers as current.
- Draw one bounded-height trend panel with distinct series for final test and incoming-batch results where the metric exists for both. For classification, select Accuracy, Macro Fβ (β=0.5), or MCC. State that classification scores use class-balanced sample weights. For ECE/MCE, show only the test series, with paired before/after marks and bin policy context. Use integer cycle ticks and hover values with available counts. Leave gaps for missing or incompatible artifacts. One cycle renders as a point with its value, even when DVC omits the aggregate PNG. Allow the original aggregation PNG on demand.
- Give dataset growth its own view or compact sparkline from verified populations. Use a bounded-height cycle table with one row per learning cycle and a bootstrap row. Summarize starting base, development, calibration, test, incoming and appended population counts, artifact state and selected metrics. Counts and metrics come from graph and metric artifacts. A missing post-merge file leaves appended count unknown. A selected table row or explicit **Inspect cycle** selector updates the shared cycle state. For very long runs, a bounded table with scrolling or pagination retains the same vertical footprint.
- Put this caveat beside the trajectory and in comparisons: “Each learning cycle splits a growing base population again and scores a new prediction batch. These results describe different evaluation populations.” Report observed cohort sizes; optionally show verified test overlap using sample IDs. Explain the simulator supplies the labels for append. Describe the test protocol as synthetic random stratified evaluation and avoid claims about generalization to unseen regions. Avoid cross-cycle delta arrows in headline cards.

### Cycle explorer

One selector governs all sections. Use available and declared cycles inside the current experiment scope, with explicit states for absent or partial cycles. Out-of-scope directories may be examined in Provenance as historical evidence and stay out of current trajectories. In Cycle 0, default to Data. Show the bootstrap label and omit or disable learning-stage views with a short explanation. In a learning cycle, an input lineage line distinguishes **base used** (B(c−1)), **incoming prediction batch** (I(c)), and **base after append** (B(c)), including a last-cycle snapshot even when no later model consumes it.

| Section | Default view | Progressive disclosure |
| --- | --- | --- |
| Data | Selected base snapshot with class counts, assignment summary and spatial plot. For a learning cycle start with the input base. | Choose full base/overview, development membership, final test, calibration, a train/validation role and one fold, or switch to post-append snapshot. Choose stored graph or t-SNE. Show graph statistics with definitions. Development membership is derivable; the deployed development graph geometry is not exported as a separate artifact. |
| Model | Selected population (final test or prediction batch), named metric row and calibrated count confusion matrix. | Test before/after metric table; incoming-batch per-class counts and support from verified CSV; Training view with CV summary and refit details, then one fold panel. Make weighted MCC and raw confusion counts distinct. |
| Calibration | Test reliability figure with pre/post ECE/MCE, method and split summary. | Calibration fit and evaluation population details, optimizer history and parameter text when produced, paired classification metrics. No batch reliability claim. |
| Spatial predictions | Incoming batch map with selected score or class color, then bounded sample table. | Toggle edges, raw/adjusted probabilities, a specific class or feature, entropy, labelled correctness, stored diagnostic and distributions. Offer original CSV and a clearly labeled joined export when verified. |
| Drift | Reference/incoming sizes and representation, three method rows with statistic and p-value. | One of raw marginals, within/cross distances, mutual information or projections. Offer settings, effective gamma/stopping policy and original text. Display “Detected at report threshold” or “Not detected at report threshold” with p-values; neither state establishes performance change or absence of shift. Spatial dependence and low permutation budgets limit inference. |

Select the view *before* opening files, constructing figures or sending payloads. Keep one main plot or existing composite PNG at a time. Mount the sample table only when requested or when it is the selected chart alternative. Build the overview from small per-cycle summaries with cached reads and gaps. Read raw graph data only when required for counts or the selected view. The verified split reader may need to open the combined fold artifact to check membership, even when only one fold figure is displayed. Switch diagnostic variants in place. Collapse logs and secondary reports behind controls that actually gate work. Preserve selected section and applicable plot controls across navigation, and reset invalid population/fold/color choices with an explanation. Keep browser history and deep links scoped to validated cycle and mode values.

### Optional comparison

Offer **Compare two cycles** from Overview or Cycle explorer. Default to one paired metric table and one chosen chart, with cycle identifiers, evidence states, counts, methods and cohort context on both sides. Meaningful initial comparisons include growth, class proportions, metric values, and drift reference/batch summaries for the same feature schema and method. Expose observation-level intersection sizes for verified base/test sets when useful. Do not imply points from different prediction batches represent paired observations. Permit side-by-side maps only for compatible coordinate systems, feature representations and class mappings, with shared bounds and color scales clearly indicated. Compare raw feature marginals with common bin edges as a presentation view. Avoid overlaying independently fitted PCA or Kernel PCA coordinates, and avoid presenting drift statistics as directly commensurate when transforms, populations or permutation settings differ. A same-cycle test pre/post calibration comparison uses the same cohort. A fixed longitudinal benchmark or controlled comparative improvement claim requires a separate scientific design.

## Spatial exploration feasibility

### Alignment and display contract

[Raw graphs](../src/utilities/raw_graphs.py), [graph protocol](../src/utilities/graph_protocol.py) and [prediction exporter](../src/utilities/pred_utils.py) establish different levels of alignment. Stable IDs are derived from synthetic batch content and parent digest plus the sample's batch index. They survive append and folds preserve them in local graph order. They are scoped to verified artifact lineage and can change on regeneration with changed data or parent. Local edge indices refer to rows in their own graph. Resolve indices to IDs before joining a derived view. Fold/test/calibration graph readers verify parent dataset, sample IDs, original indices, coordinates, labels, raw features and split role. The full graph export has coordinate traces and labels but no ID field or customdata. Do not treat coordinates, chart marker indices, or plot export trace order as a cross-artifact key.

| Overlay target | Existing support and proposed behavior |
| --- | --- |
| Spatial coordinates, actual class, individual features | Direct in verified raw graphs, keyed by sample ID. Use stored feature order and class mapping. Show x/y/z or Easting/Northing/Depth in synthetic coordinate units. No supplied CRS or physical unit conversion. |
| Graph connectivity | Use the selected raw graph's stored directed COO and local row mapping. Draw one physical link per undirected pair, omit self-loop glyphs, retain loop and topology statistics. With point filters, draw edges only between displayed endpoints and describe the filter as a view. Do not present filtered density as the original graph's density. Keep the descriptive full-base graph visibly distinct from the independent model partition graphs. |
| Dataset membership | Direct from verified roles, fold indices and parent indices. Derive development as base minus calibration and test. Show train/validation memberships for the selected fold and existing/incoming membership using exact ID sets. A membership view can show points with edges off when the matching graph is unavailable. |
| Predicted class and per-class probabilities | CSV writes calibrated argmax and two probability matrices in `pred_data.pt` node order. `model_graph()` clones the verified graph, transforms features in that order and retains topology; prediction and calibration return per-node outputs. The exporter constructs the CSV straight from those arrays and writes without a sort or index. This is a producer-order contract for the paired original files. CSV supplies no standalone key, digest or binding. |
| Maximum probability and predictive entropy | Derive max across the selected probability family from the verified join. Use the corresponding persisted natural-log entropy; bounds are approximately 0 to ln(C) with roundoff. Optional normalized entropy needs an explicit label and the class count; keep native values available. Probabilities and entropy summarize model uncertainty and cannot identify novel classes reliably. |
| Correct/incorrect | Compare calibrated argmax with the verified simulator label of each batch row. For pre-calibration color modes, compute argmax from the matching uncalibrated columns and label the probability state. Labeled correctness is available for the matched synthetic batch. For base/test/calibration/CV, current artifacts support observations but lack saved per-row model scores or flags. |

**Join gate.** For a prediction overlay, require a trusted, workspace-local pair bound to the same predict stage: DVC must tie the exact `pred_data.pt` dependency to the exact prediction directory containing the CSV, with matching model/evaluation dependencies and their binding identities. Compare the incoming graph parent with the matching source base/deployed model, and verify graph/version, shape, schema and unique IDs. Validate CSV headers, row count, ordered label vector, both probability families and finite normalized rows, and configured argmax/entropy interpretation. Only then associate each positional CSV row with its graph sample ID and use that ID for all filters, chart selections and exports. Equal lengths and agreement of labels are valuable checks but cannot detect a reorder within a class. If the pair fails verification, show coordinates/labels and the standalone CSV with their own evidence states and disable joined views.

M3.1 read the original graphs and CSVs in both cycles of both M2.5 workspaces. The production reader verified held-out role and fold membership. Batch parent and append identities matched; the CSV had 12 columns and respectively 100 or 150 rows, labels matched in order, calibrated argmax matched stored classes, both probability families normalized and their persisted entropies matched. For example Quick demo cycle 1 has a 100×5 batch with a 2×714 directed edge index. The capture's displayed 200-node batch is separate evidence.

### Explorer interaction

Use a Plotly XY plan view as the accessible first map with equal axis scales, a depth filter, and a 3D view for depth and rotation. Provide **Population**, **Color by**, **Class/feature** when applicable, and **Show graph edges** with edges off by default. Default to **Observed class** in Data and **Predicted class (calibrated)** for a verified Spatial predictions batch. Offer class probability, max probability, entropy, raw feature, membership and correctness when supported. Hover includes sample ID, coordinates, active color value and relevant labels. Put full scores and features in a bounded table and sample details, with an ID selector that works without hover or 3D gestures. Start with native 2D selection where demonstrated and keep table/ID selection as the cross-view control. Use Plotly `uirevision` and stable keys for compatible view changes, with a deliberate reset on cycle/population changes. Benchmark edges separately; for large plots require edge opt-in and use explicit limits or a selected spatial filter with displayed counts. Full exports and metrics retain their stated populations.

**Presentation work with current contracts** includes a verified incoming-batch map, selected raw feature maps, existing graph/statistic views, display-only count/score summaries, and enriched sample CSV derived for download. An enriched download includes stable ID, coordinates, label/probability state, verified cycle and relevant schema, distinct from the unchanged producer CSV. **New scientific artifacts or contracts** would be needed for portable separately copied prediction rows with stable IDs and source bindings, per-sample test/calibration/CV scores, exact structured test confusion/reliability bins, interactive exact per-epoch training curves and exact saved drift diagnostic arrays or realized permutation counts. The stored development graph digest does not supply inspectable edges without reconstruction. Serving a rebuilt graph or replayed inference would be a separate decision. Keep PNGs with textual summaries for available diagnostics and defer new scientific outputs.

## Configure

The approved [configuration inventory](configuration_design.md) is the field-level source of truth, including classes of settings, fallback semantics and inactive values. Present one resolved draft across Basic, Advanced and Expert. Offer Quick demo, Standard and Research v1, with brief compute-budget descriptions. Choosing a preset is a selection and **Load preset into draft** is explicit replacement. Indicate modifications to a preset-derived draft without writing preset metadata into scientific YAML. The first visit starts with the applied YAML when readable. With none, offer an explicit preset load. Display the applied summary and draft state independently of any result state.

| Level | Layout and behavior |
| --- | --- |
| Basic | Exactly five parameters: Model (`default_model`), Device (`device`), Learning cycles (`cycles`), Requested bootstrap observations (`data.base.n_samples`), Requested observations per prediction batch (`data.pred.n_samples`). Offer Auto, CPU and CUDA with conditional visible CUDA index under the same Device field. Zero cycles stays valid with a bootstrap-only hint. Explain targets versus class floors and placement feasibility. |
| Advanced → Synthetic data and schema | Generator radius, depth, spacing, feature and class counts, ordered class names, bootstrap class floor, bootstrap and batch seeds. Explain null/blank distinctions and generated class names. |
| Advanced → Partitions and validation | CV fold count, holdout size of the full base, calibration size of the holdout. Each size preserves fraction, integer count and explicit default/null. Outline development, calibration and test populations; the partition seed is shared with the base seed. Use realized values in Results. |
| Advanced → Graph and preprocessing | One connectivity rule with its applicable numeric control, plus graph-owned self-loops and Unit/Inverse quadratic affinities with conditional scale. Explain independent graphs, percentile size dependence and GAT's topology-only use. One scaler or Identity. Show relevant option details under Expert and reuse the resolver's scaler transition behavior. |
| Advanced → Model training | Selected model summary and navigation to its expert block, training epoch limit and seed. Show full development refit versus CV in contextual help. Keep GAT and GCN constructor settings Expert and retain both in YAML. |
| Advanced → Probability calibration | Method, ECE/MCE reporting bins, parametric fit budget/seed when applicable, and links to method-specific Expert options. Preserve fields while inactive. |
| Advanced → Drift | Permutation budget, seed, early-stop enable and a single Wasserstein method choice mapped to existing Internal flags. Expose details for max-sliced projections or Sinkhorn regularization through Expert when multivariate and applicable. Distinguish reporting significance and an early-stop threshold. |
| Expert and Internal | Complete editable YAML with precise keys and stable values. Contextual help shows selected model block, scaler-specific options, parametric optimizer/stopping controls, temperature initialization, Platt/beta clamp, Dirichlet bias penalty and drift-specific options only where relevant. Internal keys remain present in the full YAML (including GCN normalization, verbosity, epsilon and transport flags) without general-purpose independent form switches. Do not strip inactive model/optimizer/seed fields. |

Use the existing new-preset and compatibility-import resolver paths and safe YAML reader. Keep one authoritative draft and stable widget state. Apply Basic/Advanced edits as typed changes to the authoritative mapping through the resolver. A still-invalid Expert buffer stays intact while users fix the syntax in Expert, with guided edits unavailable until an explicit successful conversion. Re-entering or leaving the Expert view preserves unsent edits. Preserve explicit null, empty values, sequence replacement and semantic type differences; formatting/comments may change when reserializing. A scaler change without explicit options clears inherited options; only the resolver should reconcile this transition. Connectivity menu changes set unused selectors to null. Dual legacy Wasserstein flags retain current precedence and values until the user explicitly changes the method.

Keep **Validate & Preview** separate from **Apply**. Summary displays errors/warnings/info counts and blocking issue paths/messages near affected groups. Collapse nonblocking warnings and information by default, with stable finding code and path available. Expand semantic differences, regeneration notes for cycles, model and fold count, and full resolved YAML on demand. A preview is valid for the exact draft and active byte snapshot. Active or draft edits invalidate it. Apply allows warnings, uses atomic persistence and existing workspace locking, and handles busy/stale cases without losing draft edits. Keep validation static and share runtime constraint reporting with Run.

## Run

The applied summary shows model, requested device, cycles, sample targets, calibration and configuration identity. Add a link to Configure and a visible draft-difference message when applicable. Keep one primary **Run full experiment** action. The execution module reuses the existing sequence and POSIX directory reservation: acquire the cooperative lock, snapshot and strictly validate the applied file, regenerate the DVC definition, confirm the active bytes are unchanged, then run DVC with the same interpreter PATH and merged output. Release the lock on all exits. A setup failure stops the run before reproduction. Expose concurrent/busy behavior in both Run and Configure and preserve current stale-preview handling. External editors and direct DVC commands remain outside the cooperative lock.

Use named phases (Checking applied configuration, Preparing DVC pipeline, Reproducing) and run states (Ready, Running, Completed, Failed). The retained logs show DVC emits stage command boundaries and explicit unchanged-skip messages. Display an observed stage label and optional checklist for detected starts and skips. Require an explicit subsequent completion signal or successful exit for a completion claim. With unclear or buffered output, show “DVC is running” and the last observed event. Stage durations, optional fold progress and log timestamps have diagnostic value; they give no reliable overall percentage or time remaining. Say “Up to date” when all expected stages are explicitly skipped and the command succeeds. A targeted success is scoped to its target and reproduced dependencies.

Use one live status summary with collapsible technical logs, a bounded rendered tail and session state for the last command, result and exit information. Keep a bounded memory buffer plus optional technical log download if available to that session. Distinguish launch error, early validation/setup failure and DVC nonzero exit. The failure state shows the command and observed stage when known, points to its log, and allows retry through the existing actions after addressing the issue. On completion or failure, refresh result evidence. A session state record describes the current browser session and does not establish a durable workspace job history. Cross-session live progress, cancellation, reconnection and background workers are deferred. The existing external Makefile/CLI behavior is independent.

Keep Bootstrap (`bootstrap`), Train, Evaluate, Predict, Drift (the corresponding `*_cycle_{c}` stage names), and Aggregate cycle performance (`analyze_cycles_performance`) behind **Advanced stage execution** with one stage selector, applicable cycle selector, and one run button. Bootstrap targets cycle 0. Train, Evaluate, Predict and Drift require learning cycles; aggregate requires at least two. Build command targets from approved stage identifiers and the applied topology, and explain DVC may run prerequisite stages. Results has a link to Run and does not issue execution commands.

## Result provenance policy

Display applied configuration, run status, on-disk coverage and correspondence as separate facts. The workspace root is the result scope. The active file may change independently of a session draft or existing outputs. The current versioned model/preprocessing, graph, calibration and drift sidecars provide strong *local lineage*: graph protocol, source dataset and partition/split identities; model/calibrator byte digests; training scope, seeds and actual device where recorded; and drift cohort/representation. They do not bind every output file to a full applied YAML. In particular, evaluation scalar metrics and the prediction CSV lack standalone bindings. Use outputs and dependency hashes from the matching DVC stage records to make that connection.

Verification for M3 should:

1. Safely read the current valid applied YAML, generated definition and DVC lock. Scope expected stages to active cycle count, model and fold-dependent outputs. A stale generated definition can otherwise report a self-consistent older DAG. Check the active topology against the generated definition before trusting DVC status. Extra old cycle directories stay outside the applied scope.
2. Verify relevant stage commands, tracked parameters, dependency chain and content of declared outputs, including directory members for train/eval/predict/drift. Use DVC-equivalent status/hash and value comparison for the locked version. In M3.1 both retained runs had 14 matching topology/command records, DVC-decoded parameter snapshots corresponding to active settings, and matching content for 54 distinct dependencies, 78 single-file outputs and eight output directory manifests per run. A plain PyYAML load of the DVC lock decoded `evaluate.eps` as string `1e-08` when DVC's loader recovered the numeric value. Use DVC-compatible parsing for DVC evidence and the existing scientific reader for configuration semantics.
3. Prefer a scoped adapter around local DVC status and structured lock data. The installed DVC has `status --json`; prototype tests must confirm field meanings for missing lock, absent outputs, changed dependencies, attached caches and stale topology. Verify selected output bytes and bindings when necessary (especially the positional prediction join). Status is a local check and can acquire DVC's lock or update transient bookkeeping. Use an explicit evidence refresh or a safe post-run snapshot, with no remote checks, regeneration or reproduction from Results. Compare active/lock snapshot again before publishing a verification result and treat a busy or changing workspace as unverified/check pending. Avoid trusting exit code alone or recursing through DVC internals during every UI rerun.
4. Show result identities only where the artifacts support them. Expose a short digest with the full identifier and evidence on demand. Verified means “matches applied DVC inputs and outputs for this scope at time checked”. It is weaker than a historical record of the full active-YAML text, preset origin, app revision, timestamp or deterministic execution environment. Group-level tracking can invalidate inactive settings; conversely, if a value is outside the relevant DVC tracking, do not claim agreement for it. `device: auto` can resolve on different hardware under the same DVC key; report the producer device recorded by training separately. Never infer a preset from directory presence or a capture. An immutable run record would require separately approved persistent metadata.

| UI state | Evidence and presentation |
| --- | --- |
| **Verified matching** | Required artifacts for displayed scope are readable, expected topology and DVC inputs/outputs match the applied settings, and applicable bindings agree. Show checked scope and time, with provenance details. The overview is “complete” only when required cycle outputs agree across the declared scope. |
| **Unverified correspondence** | Files exist with missing/incomplete lock or scope evidence, an unsupported/legacy schema, unreadable verification metadata or a busy workspace. Let the user explicitly browse readable artifacts with a persistent label. Withhold current headline claims and unproved positional joins. A legacy Plotly export can still be viewed if locally trusted. |
| **Outdated** | Observed evidence of a different tracked setting, topology, source dependency or a superseded snapshot. Keep the proven old scope available with the reason and timestamp of the check; keep it out of the current trend. A content or binding conflict may also indicate modification or mixed artifacts, so describe the conflict and disable the affected join. |
| **Partial** | Some expected outputs are missing or invalid after a targeted run, interruption, storage loss or other cause. Show availability and currentness per cycle/section. Mark a latest complete cycle where eligible and keep gaps in trends. Optional method-disabled diagnostics carry “Not applicable”. |
| **Missing** | No readable result artifacts in the active scope. Show the applied state and a single link to Run or Configure as applicable. Bootstrap-only output at zero cycles is a valid result and model scores are inapplicable. |
| **Draft differs from applied** | Compare resolved scientific values when possible and flag unresolved edits separately. This is an independent banner, including when the applied results verify. Show which settings Run uses, and link to Configure for preview. |

While a run is writing, avoid cross-stage joins and current-run completion claims based on transient file presence. Display existing evidence with its scope and recorded verification boundary. A failed run can leave readable older or partial files, so retain failure in Run and evaluate artifact availability independently on Results.

## Visual design system

The stored graph and t-SNE plots use Matplotlib Set1 for ordered class IDs. Four classes use red, purple, brown and gray; existing graph edges are gray and thin. Current confusion matrices use Blues with integers, training panels use blue/red/green, reliability uses red before and blue after, and drift comparisons typically use blue reference and red incoming. The original performance series use green accuracy, blue Fβ and red MCC. Preserve emitted artifacts with their legends and place them on a neutral light image surface in both themes.

| Area | Proposed design |
| --- | --- |
| Navigation and layout | Native top navigation for four destinations, ordinary page links for next actions, a consistent compact applied/run/result line and one page heading. Results keep the selection row above the plot. Put interpretation and controls adjacent to the evidence. Prefer a readable single-column flow at narrow widths, with no wide essential table column required to see the selected sample. Use native containers and spacing. |
| Typography and spacing | Use the native Streamlit sans-serif for body/interface and monospace for YAML, logs and digests. Start with 16 px-equivalent body and 13–14 px chart/secondary text on desktop, with larger page titles and clear section grouping. Keep preamble and status short, then move explanatory details below the summary. Use consistent labels and precision with full values in accessible tables. |
| Chart sizing | Target 320–420 px for single metric charts, 480–600 px for the primary spatial canvas, and a bounded table around 300–360 px tall on desktop. Fit square charts at a readable maximum width. Keep composite diagnostic PNGs legible with an original-size option. Stack planned two-cycle panels when horizontal space is limited. Verify at desktop and narrow widths before settling sizes. |
| Mineral classes | For new interactive charts use a stable qualitative codebook indexed by the persisted class mapping, with alternate marker shapes and visible labels. Provisional first four swatches are blue `#0072B2`, orange `#E69F00`, green `#009E73`, purple `#CC79A7`, with theme-appropriate outlines and more distinguishable markers/filters for larger class counts. Review the change from Set1 and label legacy figures with their own legends. Keep the same mapping across cycles of a compatible experiment. |
| Probability, uncertainty and features | Use perceptually ordered Cividis for probability (absolute 0–1) and Viridis for entropy (0–ln C in nats). Annotate which end represents higher uncertainty and the probability family. Use a sequential scale for continuous raw features with documented bounds. If chosen for a justified signed quantity, use a diverging scale centered at the known reference. Reserve a gray/outlined encoding for unavailable values and use shape plus legend for correctness. |
| Calibration and drift comparisons | For native pre/post charts use warm orange dashed/open markers before and blue solid/filled after, with neutral dashed identity reference and explicit labels. Keep before/after semantically distinct from class colors through context. Use blue/reference and warm/incoming with marker and line variants in comparable drift views. Keep probability error and count units visible. |
| Accessibility, status and themes | Use native labels, keyboard-focusable navigation and controls, readable status text with icons optional, and named table alternatives to chart interactions. Target 4.5:1 normal-text contrast and 3:1 for large text and necessary non-text indicators. Test actual swatches and keyboard/focus order in both themes. Match new Plotly backgrounds, axes, labels, edges and color bars to the theme. Never blend fixed white PNGs into a dark canvas; border and identify the original figure surface. Empty, stale, partial, unverified and failed messages show reason and the appropriate action or detail, with status conveyed in words. |

## Dashboard architecture and locked APIs

The installed environment has Streamlit 1.59.0, Plotly 6.8.0 and DVC 3.67.1, in agreement with `uv.lock`. This audit inspected the installed signatures and docs and compared the current dashboard source and captures. The implementation checkpoint below records how M3.2 uses these APIs.

- Use `st.Page` with `st.navigation(..., position="top")` from the entry point and call only the chosen page. Use `st.page_link` or `st.switch_page` for internal actions. In-page mode/section selection can use a native segmented control for the two modes and a labelled selectbox/radio for the diagnostic. Validate any `st.query_params` cycle/mode against available scope before loading. A page switch can pass intentional query parameters with the installed API. Keep domain state in stable session keys apart from transient, page-local widget keys.
- Installed `st.tabs` and `st.expander` are eager by default. Both support `on_change="rerun"` plus `.open` for explicit conditional work where the control benefits from lazy loading. Prefer a selector and branch at the top for the main diagnostic. Use a native status panel for live progress and bound the log contents since a collapsed status panel can still send data. Standard synchronous subprocess streaming preserves existing behavior.
- Use `st.plotly_chart(width="stretch", height=...)`, stable chart keys and a theme-aware layout. The installed selection API supports point/box/lasso events, but delivery depends on trace type and chart behavior. Prove selection for the chosen 2D traces and keep ID/table inspection available for 3D. Installed `st.dataframe` supports bounded height, column formatting and selection; `st.download_button` supports a download without triggering a rerun via `on_click="ignore"`. Test browser behavior before relying on responsive layout, theme reporting via `st.context.theme`, URL switching or Plotly view-state retention.
- Cache small, read-only parsed data keyed by workspace plus file identity, preferably content digest when correctness depends on it. Recheck after Apply/run and avoid sharing unverified snapshots across sessions. Keep lazy readers and restyling separate. Accept trusted locally generated artifacts only where PyG `.pt` loading requires `weights_only=False`; use existing version and integrity checks after loading, and treat local origin trust as a prerequisite. Prefer JSON/PNG/CSV for display. Existing HTML fallback is script-bearing and references a CDN; keep trusted legacy HTML available as an explicit original artifact with dependency context, with native JSON rendering as the regular path. Do not embed arbitrary HTML or deserialize externally supplied graphs.

```text
app.py                        Native page registry and shared page config
dashboard/
  state.py                    Draft/preview bindings, current selection and session run state
  execution.py                Existing lock, setup/repro, typed command progress and errors
  artifacts.py                Workspace paths, evidence checks and selected artifact reads
  components.py               Status, metrics, provenance, figures and table primitives
  views/
    home.py                   Entry points and workspace summary
    configure.py              Basic/Advanced/Expert draft editing and Apply
    run.py                    Applied summary, execution and stage/log status
    results.py                Overview and cycle routing, drill-through and shared filters
```

Extract spatial view code when the Plotly/table implementation warrants a separate module. Keep scientific math and fit/inference in the existing `src` pipeline and use existing validators, resolver, bindings and readers. Thin display calculations such as counts, max probability and membership joins can be tested as pure presentation transforms. Keep Streamlit and DVC CLI entry points, configuration schema, graph protocol and output contracts. The UI introduces no new dependencies.

## Proposed implementation sequence and acceptance

| Milestone | Priority and work | Acceptance |
| --- | --- | --- |
| **M3.1 Design and feasibility audit** | Inventory, retained evidence, flows, artifact gates, wireframes and roadmap. | Document reviewed against the current dashboard and stage outputs. |
| **M3.1a Scientific visualization labeling corrections** | Correct the validation and cross-cycle metric legends and the reliability diagram labels. | Verify labels, numerical data, published filenames and schemas with synthetic fixtures. |
| **M3.2 Navigation and dashboard architecture** | Extract four pages and common workspace, state and reader foundations. Migrate existing diagnostics through accessible routes during extraction. | Native navigation and links work on rerun and page transitions. Existing app entry point, CLI, locks and DVC workflow still work. Only the selected destination executes its view. |
| **M3.3 Guided configuration** | Build three preset choices, five Basic parameters, Advanced groups, conditional Expert guidance, validation, preview and Apply. | Guided path requires no YAML editing. Editing across levels retains a single draft, resolver parity and lossless scientific types/values. Invalid, stale and busy cases have actionable messages. Full YAML, diff and findings are available. |
| **M3.4 Execution experience** | Add observed stage status and a dedicated log presentation to Run. | Full and each existing targeted action use the applied configuration and preserve setup/exit/lock behavior. Test zero cycles, prerequisite runs, skipped stages, a run with incomplete output, and failure. Visible stage progress reflects observed signals. |
| **M3.5a-i Result provenance and scientific summaries** | Establish read-only DVC evidence, scientific binding and summary readers and an explicit refresh. | Declared, partial, historical and matching scopes distinguish local availability from correspondence. |
| **M3.5a-ii Experiment overview and cycle navigation** | Present verified headlines, separate test and batch trends, dataset growth and cycle drill-through. | A bounded summary and one selected cycle keep page height effectively constant. |
| **M3.5b-i Verified spatial data readers and prediction alignment** | Verify selected raw cohorts, exact same-stage graph and CSV bytes, display links, filtered views and enriched exports. | Both retained Quick demo cycles verify. Same-class row swaps, mixed-cycle artifacts and stale evidence cannot authorize a join. |
| **M3.5b-ii Interactive spatial visualization and observation explorer** | Build the interactive spatial views, accessible sample details, controls and downloads. | Verified colors and edges agree with sources. Offer an edge-free plot, keyboard access to sample data and supported fallback for legacy files. |
| **M3.5c Classification, calibration and training results** | Rehome original figures, published metrics, batch class counts and fold/refit diagnostics. | Calibration states and cohorts are clearly identified, legacy figure caveats are accessible, and inactive diagnostics have a defined state. CV and final refit remain distinct. |
| **M3.5d Drift and cross-cycle comparison** | Refine the drift and cross-cycle views and implement one optional paired assessment. | Reference cohorts, raw versus model space, permutation method and caveats remain visible. Selected feature/diagnostic only; comparable views share scales and show cohort changes. |
| **M3.6 UI verification and visual refinement** | Browser verification in addition to focused AppTest and fixture regression checks. | Test forward/back and deep links, keyboard journey, layout reflow, browser contrast, figure resolution, theme changes and supported baseline workflows. Inventory each legacy output against its new route. |

For the interactive views, validate desktop at representative 1280 and 1920 px widths and narrow use around 390 px. Home and first-run flows should present a clear action without scrolling past technical text. A 1-, 3- and many-cycle fixture should mount the same number of principal plots and bounded tables. Assert reader calls for hidden diagnostics are zero. Test state restoration, stale-draft scenarios, named metric labels, undefined/unsupported values, cycle 0 semantics, and per-cycle scope. Use isolated retained or fixture outputs for visual checks. Record rendering performance for plots with edges both enabled and disabled before choosing a data-size cutoff. Verify numerical display against retained outputs and ensure scientific files remain unaffected by navigation, Apply previews and Results exploration. Browser usability and appearance are future acceptance checks.

## M3.1 owner decisions

- The owner accepted the audit and approved four separate primary pages (Home, Configure, Run and Results) with native top navigation. Results has an experiment overview and a cycle explorer with one shared cycle selector.
- The spatial explorer defaults to an XY plan view with a prominent 3D switch and edges off. The owner approved the provisional qualitative class palette, Cividis for probability and Viridis for entropy.
- Positional prediction joins are approved for existing synthetic artifacts only when the DVC-bound checks described above verify the producing scope, source graph and exact prediction output.
- Persistent experiment identity, saved experiments and new scientific artifact contracts are deferred. The design uses the existing single workspace and DVC evidence for currentness.

Fixed holdouts, external user data, independent uploaded-model inference, a packaged demo, new baselines, background jobs and cancellation remain deferred to applicable later work. Current cycle comparisons remain descriptive.

## M3.1a scientific visualization labels

Training legends use “Validation Macro F0.5” and cross-cycle legends use “Macro F0.5”. The score is macro F-beta with beta 0.5 and class-balanced sample weights. Evaluation and prediction JSON still store the score under `f1`. The reliability diagram plots weighted observed accuracy against weighted mean confidence. Below the identity diagonal, confidence exceeds accuracy (overconfidence). Above it, accuracy exceeds confidence (underconfidence). The secondary axis is “Sum of Sample Weights per Bin” because each bar is the sum of the input sample weights in that confidence bin, before any division by total weight.

The future dashboard should flag original PNGs with legacy labels using verified producer dependency and output hashes when available. A source hash mismatch shows that a dependency differs, but does not by itself identify which labels the image contains. The visible “Validation F1 scores (beta=0.5)” or “F1” legend, or a reliability axis marked “Sample Count” with inverted annotations, identifies the affected legacy rendering. Filenames and metric keys are shared across versions. When provenance is insufficient, mark label revision as unverified and show the interpretation beside the image. Caption legacy F1 legends as macro F0.5. For legacy reliability diagrams, explain both the correct sides of the diagonal and the summed-weight bars. Use Macro F0.5 in native metrics built from `f1`. Keep original images intact.

The existing DVC definition and its generator track `train_utils.py` for training, `eval_utils.py` for evaluation and prediction, `pred_utils.py` for prediction, and `cycles_performance_utils.py` for aggregation. Changing those files invalidates the corresponding stages on the next reproduction. DVC tracks training and evaluation as whole stages, so refreshing their figures through DVC can refit models or calibrators and invalidate dependent stages. Aggregation also consumes the evaluation and prediction metric JSON files. No DVC declarations or existing figures changed in M3.1a.

The full regression suite passed 340 tests, including four focused plotting and artifact checks and the disposable DVC invalidation test. Ruff lint and formatting, applicable pre-commit hooks, Git whitespace checks and offline lock consistency passed. The plotting fixtures exposed dependency deprecation warnings. No production experiment or figure regeneration was performed.

## M3.1 verification

The design was checked against source paths, local screenshots, the capture index, retained graphs/CSV/JSON and DVC evidence, and the locked Streamlit APIs. The full project test suite passed all 336 tests, including the disposable DVC integration fixture. Existing dependency deprecation warnings remain. No new scientific experiment was executed. M3.1 changes this document and the roadmap.


## M3.2 implemented dashboard

[app.py](../app.py) registers four native `st.Page` scripts in `st.navigation(position="top")` and runs the selected page. The shared page configuration uses the wide layout and native themes. Explicit workflow actions use page links or page switching. Configure Quick demo selects the preset in session state and opens Configure. Loading the preset, previewing, applying and running remain separate actions.

| Module | Responsibility |
| --- | --- |
| `dashboard/state.py` | Keep the exact draft, selected preset, active-file-bound preview, applied-draft comparison, Results cycle and preferred section, and last session execution outcome. |
| `dashboard/execution.py` | Reserve the existing directory lock, strictly validate the applied file, regenerate DVC definitions, recheck the snapshot and stream reproduction with the current interpreter on PATH. |
| `dashboard/artifacts.py` | Resolve workspace paths and discover display files using filesystem metadata. Open only a selected local artifact and check for common read and format failures. |
| `dashboard/components.py` | Render applied settings, validation findings and original figure or diagnostic formats. |
| `dashboard/views/` | Implement Home, Configure, Run and Results as separate scripts. |

Durable values live in one session-owned state object. Page-local widgets can be discarded by Streamlit on navigation. Returning to Configure recreates the editor from the saved draft and retains a valid preview. Editing the draft clears the preview. A changed active snapshot disables Apply and requires a fresh preview. The existing atomic writer rechecks the snapshot under the cooperative lock. Results restores the cycle and section when valid. If a cycle disappears, Results chooses an available or declared cycle. Bootstrap shows data and can link to an existing experiment-wide aggregate. A temporarily unavailable section falls back to Data. The last Run outcome describes only the most recent session command.

Results is a transitional single-diagnostic browser. The six sections expose the existing exports listed in the artifact inventory, including per-fold graphs and PNGs, test and batch metrics and confusion matrices, calibration fit and reliability diagnostics, predictions CSV and histograms, drift plots and reports, and the cross-cycle PNG. For learning cycle c the population selector exposes input snapshot c−1, incoming batch c and post-append snapshot c, including the final append. Cycles above the applied count remain selectable when display files exist and are labelled as existing files only. Bootstrap is available separately as cycle 0. Spatial graphs prefer Plotly JSON by filename stem. A legacy HTML-only graph requires explicit opt-in for the trusted local export and may fetch Plotly from a CDN. Paths resolving outside the workspace results directory are excluded. PNGs can be downloaded in their original form. Partially written or unreadable selected files can report a display error.

Home and Results report file availability separately from evidence of correspondence. All existing result sets are labelled unverified against the applied configuration and source revision. The browser does not deserialize model, calibrator or raw graph objects. The original stored 3D graph views retain their traces and legends. The graph title and statistics appear as wrapping text below the chart. Legacy training, metrics, reliability and cross-cycle figures retain the interpretation guidance above.

The approved experiment overview, DVC evidence checks, native metric summaries, joined spatial overlays and interactive trend charts remain in M3.5. M3.2 uses the original figures and standalone tables. At M3.2, Configure exposed expert YAML and Run retained synchronous command streaming. The dashboard stores session state only. Container packaging uses the existing whole-source copy, and the source checkout launches through the same `app.py` entry point. The scientific wheel still packages `src`.

Initial Chromium review covered 1920 × 1080, 1280 × 900 and 390 × 844 in native light and dark themes. The captures use isolated workspaces, one with empty results and one with copies of the historical M2.5 Quick demo exports. The local capture index is `/tmp/mineral-m32-review/INDEX.md`. The owner accepted M3.2 following visual review.

Engineering checks passed with 361 tests in the complete suite and 360 in the non-integration suite. Repository-wide Ruff lint and formatting, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed. The container build and deployment were not exercised.


## M3.3 guided configuration

Configure starts in Basic with the applied YAML as the initial draft, when available. The compact editor is paired with Review and apply on desktop and stacked on a narrow display. The five Basic parameters are model, requested device, learning-cycle count and the bootstrap and batch observation targets. The CUDA index appears under Computing device only when applicable. Zero cycles is labelled bootstrap only. The observation targets can differ from realized counts because of placement and class top-up.

Preset selection updates the description and measured reference for the selected preset. Quick demo and Standard use the M2.5 host measurements of 74.31 s and 101.66 s. Loading is an explicit replacement. Draft status identifies the loaded preset and whether its resolved values have been customized.

Advanced renders one of seven groups. Its 28 inventory fields include three mutually exclusive graph keys represented by one rule selector and one parameter input. Seeds live in Reproducibility, with the calibration seed hidden for isotonic. Isotonic also hides the calibration fit budget. Graph affinity, scaler, and drift controls show only applicable inputs or Expert guidance. The Wasserstein selector represents the two internal flags and leaves a legacy pair intact until the user changes the method. The full Expert YAML remains editable and includes all inactive fields.

[Guided field bindings](../dashboard/guided_configuration.py) apply typed overrides through the existing import resolver. The session stores one draft document, with the configuration level, group and preset selection stored separately. Switching views or pages leaves the exact YAML buffer intact. A malformed or semantically invalid Expert edit blocks guided editing until corrected or explicitly replaced. Actual guided edits reserialize the draft and may normalize YAML formatting and comments.

Review and apply remain explicit. Validation shows separate counts for errors, warnings and information. Blocking findings and regeneration requirements are visible. Expanders hold nonblocking details, the semantic diff and resolved YAML. The existing active snapshot and workspace lock still guard atomic persistence. The successful Apply action offers a link to Run.

Chromium review covered desktop widths of 1920 and 1280 px and a 390 px layout in native light and dark themes. At narrow widths, Review and apply follows the stacked editor fields. The local capture index at `/tmp/mineral-m33-review/INDEX.md` records 11 captures from a disposable workspace. Browser testing also preserved an edited malformed Expert buffer through level and page navigation. The owner accepted M3.3 following engineering and visual review.

Engineering checks passed with 375 tests in the complete suite and 374 in the non-integration suite. Repository-wide Ruff lint and formatting, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed.


## M3.4 execution experience

Run shows requested model, device and observation targets, cycles, calibration and a 12-character SHA-256 prefix of the exact applied configuration bytes. Draft changes are shown separately. The full action and advanced targeted controls use the existing cooperative directory lock through strict validation, definition generation, the active-byte recheck and reproduction. The child interpreter and PATH handoff stay the same. Session state holds the latest attempt and its configuration identity, scope, commands, exit statuses and stage observations.

Ready describes a session before execution. Running highlights Checking applied configuration, Preparing DVC pipeline or Reproducing. Completed requires a successful reproduction exit and describes the scope shown, including a specific target when selected. Failed identifies where the attempt stopped, including contention, invalid settings, preparation, launch or nonzero reproduction exit. Up to date requires a successful reproduction exit, DVC's no-op summary and explicit unchanged-skip evidence. A full run requires every generated stage to have been observed as skipped. A targeted run requires the target's skip and DVC's no-op summary for that traversal. Cache checkout is separate from an unchanged skip.

The parser was checked against the retained M2.5 Quick demo reproduction and no-op logs and the locked DVC 3.67.1 source. It recognizes complete stage starts, unchanged skips and cache-checkout lines for this pipeline. The sequential fail-fast walk supplies completion evidence at the next recognized stage boundary or a successful exit. A generic lock-file update alone does not establish completion. Partial, unknown and buffered output can leave the last event or a stage outcome unconfirmed. Run keeps the latest learning cycle's observations visible and summarizes other cycles. It reports observations without estimating elapsed work or time remaining.

Technical logs use merged stdout and stderr and keep the last 65,536 characters in the session. The expander renders at most 4,096 characters of output, with separate command and exit-status details. The download includes the retained tail, run scope and an omission count. On navigation away, the synchronous execution drains the subprocess and keeps the workspace lock through its exit. The latest outcome remains available when Run rerenders. File availability refreshes after a terminal outcome, independently of result correspondence verification.


Engineering checks passed with 392 tests in the complete suite and 391 in the non-integration suite. Repository-wide Ruff lint and formatting, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed.

Chromium review recorded eight simulated execution states at 1920, 1280 and 390 px widths in native light and dark themes. The capture index is `/tmp/mineral-m34-review/INDEX.md`. A paused simulated command also finished under its existing lock after navigation away from Run, and its outcome appeared on return. The owner accepted M3.4 following visual and engineering review.


## M3.5a-i result provenance and scientific summaries

**Refresh result evidence** checks the declared local synthetic pipeline. Results displays the selected scope, availability, correspondence, check time, applied configuration fingerprint and expandable reasons. An unrelated draft and the last execution outcome are independent of verification. Home and Run continue to report file availability without performing a DVC check. The existing diagnostic browser can display historical and unverified artifacts with their original interpretation guidance.

### Verification contract

1. Reserve the existing cooperative configuration lock. Capture the applied bytes, generated definition, DVC lock, local DVC controls and metadata for declared dependencies and outputs. Directory metadata includes membership and each regular file's identity, size, modification time and change time. The lock and active definition are read-only inputs to this operation.
2. Strictly validate the applied scientific YAML with the existing validator. Use the same in-memory definition builder as setup to compare cycle count, model commands, fold-dependent paths, dependencies, outputs and tracked-parameter declarations. The builder performs no workspace writes.
3. Parse schema 2.0 locks and tracked values with the locked DVC YAML reader. Compare scalar types recursively, including explicit null and integer versus float. Require the recorded command and paths to agree with their applied declarations. Follow producer ownership, including dependencies on members of declared output directories, and propagate upstream conflicts or missing evidence.
4. Run local `dvc status --json dvc.yaml` where the valid definition and lock permit it. Corroborate status with fresh DVC hashes of declared dependencies and outputs. The adapter uses DVC 3.67.1 and its locked dvc-data 3.18.3 file and directory codecs without cached hashes or a writable artifact ODB. The directory builder is a version-sensitive dvc-data API and needs rechecking on dependency upgrades. Local status can take DVC locks and update transient files in `.dvc/tmp`. Site cache is directed to a disposable temporary directory and analytics is disabled. Cache absence alone does not prevent verification of matching workspace bytes. No cache restore or remote status is used.
5. Check applicable scientific contracts. Verify raw schema and graph roles, fold membership, input/batch lineage and ordered append values. Reconstruct the development graph for the binding check. Verify the fitted preprocessing and model bundle digests, the calibration fitting split and calibrator bytes, and the drift populations, representation, model and settings. Use the trusted local raw graph reader for hash-matched dataset artifacts. Model and calibrator files are hashed as bytes. A DVC-matching report can still be partial when its scientific fields or bindings fail validation.
6. Recheck controls and the scoped manifest before publishing. Busy workspaces, changing files, missing records, legacy schemas, unsupported paths and unrecognized status evidence fail closed. The UI retains at most one session snapshot, clears it on Apply or execution, and drops its matching badge on a relevant metadata change. Selector reruns perform metadata rechecks and load only the selected diagnostic. They neither rerun DVC nor reopen graphs and summaries from the verification snapshot.

**Verified matching** applies to the checked scope at check time under the trusted local workspace assumption. It establishes agreement between the applied definition and tracked settings, workspace source and dependency content, locked outputs and applicable scientific bindings. It does not establish independent replay, immutable experiment identity, execution time or preset origin. Scoped metadata invalidates cached display evidence under ordinary filesystem changes. A future consumer that reads new bytes, especially a spatial join, must verify those exact bytes and bindings again. No metadata guard makes an uncooperative filesystem writer transactional. This adapter accepts confined regular workspace files and directories and conservatively excludes symlinks and special files.

### Summary schemas and historical results

- Applied configuration has a SHA-256 of the exact YAML bytes and typed model, requested device, cycles, folds, requested sample targets and calibration method. The declared learning range keeps gaps for missing cycles. Discovered out-of-range files remain available for historical browsing.
- Bootstrap and per-cycle base summaries contain realized total, development, calibration and final-test counts. Learning cycle c relates input base c−1, its own incoming batch, and post-append base c. It never derives realized counts from configuration targets.
- Final-test metrics carry separate before- and after-calibration records with accuracy, Macro F0.5, MCC, ECE and MCE. The batch summary carries calibrated accuracy, Macro F0.5 and MCC. Classification metrics use class-balanced sample weights. The key `f1` maps to `macro_f05`. Invalid, nonfinite and absent values remain absent. Model and calibration metadata include checkpoint and graph digests, fitting split, calibration method from tracked settings, and recorded device where present.
- Per-stage and per-cycle evidence tracks availability, correspondence and whether local declared output bytes match the lock. The local summary can contain historical observations. Current comparisons must use the cycle's matching summary, which is available only for complete matching scope. An aggregate is inapplicable below two learning cycles. A historical lock can match all its output bytes even if producer sources have changed.

### Verification and performance

A disposable one-stage DVC fixture generated its own lock and checked actual status and hash behavior with missing locks and outputs, direct file and directory-member changes, added members, source changes and tracked parameter type changes. Copies of retained M2.5 Quick demo and Standard workspaces verified against their own historical sources. Rechecking the same Quick demo artifacts with the M3.1a producer substitutions preserved output-integrity evidence but excluded both learning cycles from current comparisons. Tests also perturb mixed-cycle files, scientific bindings and mid-check snapshots. Retained integration cases skip when the local M2.5 workspace is unavailable. A complete scientific run established from the current producer revision was unavailable for this milestone. No new scientific run or lock rewrite was used to create one.

Quick demo had 124 result files and 34.3 MB of output. On the review host, its first explicit refresh took 1.33 s and its warm median was 0.70 s. The current-source mismatch copy also took 0.70 s warm. Standard took 1.20 s on its first refresh and 0.83 s warm. Metadata-only rechecks took approximately 5 ms median across 30 repetitions. These are local observations. Explicit refresh scales with declared files, bytes hashed and graph verification cost. Repeated views scale with the scoped manifest and skip DVC and scientific rereads.

M3.5a-ii owns headline cards, trajectories, bounded growth and cycle navigation. M3.5b-i owns exact prediction CSV-to-graph binding and aligned rows. The overview evidence alone grants no positional join.

Engineering verification passed with 410 tests in the full suite and 405 in the non-integration suite, with five integration tests deselected in the latter. Repository-wide Ruff lint and formatting, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed. Pre-commit modified no files.

## M3.5a-ii experiment overview and cycle navigation

**Experiment overview** opens on the first Results visit. The mode, selected cycle and scientific section survive navigation in the current session. **Inspect cycle** opens the selected cycle in **Cycle explorer**. Choosing Bootstrap temporarily limits the available sections. Returning to a learning cycle restores the remembered scientific section. The Data population selector still opens input base c−1, incoming batch c and post-append base c, including the final appended snapshot. Cross-cycle performance remains available as an original diagnostic.

The applied YAML fingerprint identifies exact file bytes. The Results header separates the applied settings, requested and recorded training device, verified cycle coverage and correspondence. A separate draft notice shows unapplied edits. **Evidence and execution details** provides the full fingerprint, the latest session action when available, verification time and stage evidence. **Refresh result evidence** performs the local DVC and scientific check. Ordinary widget reruns use the existing workspace stamp to invalidate cached evidence after changes. Historical files stay available in the explorer without joining the applied overview.

The latest complete matching cycle supplies class-weighted accuracy for final test and prediction batch and the final-test ECE before and after calibration. Accuracy, Macro F0.5 and MCC compare calibrated classifications across the two separately labeled cohorts. ECE and MCE compare the two probability states on the same final-test cohort. The trajectory leaves a gap for each missing or nonmatching cycle and reports metric values and cohort sizes on hover. Changing evaluation populations limit longitudinal interpretation. The growth chart uses verified bootstrap and append producers independently of model diagnostics. Appends add incoming observations with simulator labels. The bounded table uses compact count and metric columns by default and makes the development, calibration and final-test partition counts available through **Show partition counts**.

### Current-source Quick demo verification

One isolated two-cycle CPU reproduction used the distributed Quick demo preset at revision `05a9a6dd85c150dfa94a11f3f4c361c78bc9fb8e` (scientific source tree `d3ace04b5c810dfc5320f736bdad758558df460c`). The locked project interpreter executed all 14 stages in **73.527 s** in `/tmp/mineral-m35aii-quick.JeKROY`, starting with no DVC lock or generated results and an independent cache. All stages and both learning cycles verified as complete and matching. The workspace contains 124 result files (34,263,685 bytes). The corrected training, reliability and cross-cycle figures were generated. Visual inspection confirmed Macro F0.5 legends, the reliability diagram's sample-weight axis and confidence annotations. The applied `params.yaml` SHA-256 is `8657a4533b5b0c80adad9dadbdc2c9a6800da0389514d3d37e9eba35cae0ade1`. The recorded training device in both cycles was CPU.

| Cycle | Input base | Development | Calibration | Final test | Incoming | Post-append | Calibrated final-test accuracy | Final-test ECE before / after | Calibrated batch accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 440 | 308 | 40 | 92 | 100 | 540 | 36.30% | 0.071203 / 0.090677 | 43.82% |
| 2 | 540 | 378 | 49 | 113 | 100 | 640 | 30.83% | 0.028284 / 0.040227 | 42.67% |

The final appended base has 448 development, 58 calibration and 134 final-test observations if used as a subsequent input. The check took 1.17 s. Across 30 subsequent workspace stamp checks, the median was 5.15 ms. These are measurements on the review host.

The browser capture index and measurement log are in `/tmp/mineral-m35aii-review`. The index identifies genuine results, disposable altered workspace fixtures and simulated presentation values for the 40-cycle layout.


### Owner visual review refinement

The Results header leads with verified coverage and correspondence. The applied model and device remain visible, and the full fingerprint, check time and stage evidence are available in **Evidence and execution details**. The three headline groups identify the same latest complete matching cycle. Final-test ECE uses percent units in the headline, trajectory and selected table metric, while the underlying artifact values retain their fractional units. The headline reports the before-calibration ECE and the direction and magnitude of change in percentage points on the same final-test cohort.

**Dataset evolution** shows the initial and latest verified population sizes and switches between **Totals** and **Partitions** in one chart slot. The partition view stacks verified development, calibration and final-test counts with distinct patterns. Both views retain unavailable snapshot positions as gaps, distinguish the final append and keep label density bounded. The final append describes available data for future use. The cycle summary retains its bounded height, optional partition counts and **Inspect cycle** action.

The updated review uses the retained Quick demo artifacts and the original DVC lock. Its capture archive and measurements are in `/tmp/mineral-m35aii-polish.vGt2dt`. The owner accepted M3.5a-ii for engineering and visual quality after reviewing the eight refined browser captures.

## M3.5b-i verified spatial data

### Selected populations and topology

`dashboard.spatial_data.read_population` reads one trusted local cohort. Snapshot indices start at 0 for bootstrap. For learning cycle c, the input base and its holdout and CV populations come from snapshot c−1. CV fold indices are zero-based. Development membership is the verified complement of final test and calibration, cross-checked against all CV folds and returned in parent order. The standalone incoming batch is checked against its input base. The final append verifies the ordered IDs, coordinates, raw features and labels of the two source populations before exposing their row memberships.

Each population carries its selected role, cycle or snapshot, fold when applicable, stable sample IDs, canonical float64 XYZ and ordered raw float64 features, observed class indices and names, schema version and dataset identity. For subsets, the dataset digest identifies the source dataset and the selected graph carries its split digest. Coordinates are in synthetic units. There is no supplied CRS or depth conversion. Full-base topology is descriptive. Final-test, calibration, fold and prediction topology is independently constructed by the scientific pipeline and read from its stored graph. Derived development membership has no topology. The reader checks raw structure and local bindings. Correspondence to the applied run remains a separate evidence decision.

The representation retains stored COO, weights and distances, and the original unique-edge, degree, density, distance and self-loop statistics. Display links collapse both directions of an edge into one physical link and omit self-loops. The view resolves endpoints to stable IDs and includes a link only if both endpoints are visible. Inclusive stored Z, observed class and stable ID filters preserve source row order. The view count describes visible observations. Original population counts and graph statistics remain associated with the full selected cohort.

### Exact prediction join

`dashboard.spatial_predictions.verify_prediction_join` checks one declared learning cycle under the cooperative configuration lock. It uses fresh DVC status, generated topology, lock records, typed tracked settings and fresh content hashes for the selected predict stage and its recursive producers. The status CLI can inspect the whole generated pipeline. Applied correspondence is required for the selected ancestry, which includes the incoming dataset producer, input-base lineage, training and evaluation. Drift and later prediction outputs are independent of this join. Overview evidence is insufficient for authorization.

- The verified stage must declare the exact `pred_data.pt` dependency and prediction output directory. The captured incoming graph and CSV bytes must match the file hash and the CSV entry in the matching DVC directory manifest. Changed directory members, including extra ignored files within the prediction output, cause rejection. The join uses the captured graph and CSV bytes for all subsequent reads.
- The verified input base and independent split artifacts establish the raw schema, observed labels and membership. Existing model and preprocessing readers check the weights and sidecar bytes, the model graph settings, source dataset identity, ordered development partition identity and fitted scaler type. Existing calibration readers check the exact model bundle, calibration split and calibrator bytes. The training graph digest stays bound to the checked bundle and lock. The spatial reader does not reconstruct the training graph to rederive that digest. The incoming graph is checked against the same base and the applied schema, including its raw and graph protocol versions, ordered unique IDs and tensor shapes.
- The reviewed [prediction stage](../src/stages/predict.py) loads the verified incoming graph and applies the frozen preprocessing row-wise on a clone. GCN and GAT emit a row per input node in input order. The calibration path acts on the per-node logits and the uncalibrated path applies class-wise softmax. The [exporter](../src/utilities/pred_utils.py) uses those rows and the original `y` vector directly and writes an index-free CSV without sorting or filtering. The gate checks the reviewed producer source fingerprints as well as DVC lineage. A new producer revision requires a fresh review of the row-order contract before its outputs can be joined. Row counts and ordered labels alone cannot detect a row exchange within a class.
- The parser requires the exact ordered class-index columns and the expected label and entropy fields. The row count and ordered observed labels must match the graph. Both probability families must be finite and bounded by 0 and 1. Row sums must be within eight float32 epsilons of 1 (approximately 9.54 × 10⁻⁷ absolute). The stored calibrated label must match the first maximum. Entropies are checked against the producer's `-sum(p * ln(p + 1e-8))` convention with absolute tolerance `8 * eps(float32) * (1 + ln(class_count))`. These allowances cover float32 arithmetic, probability serialization and the scale of the entropy reduction. The persisted entropy value is retained without rounding, clipping or renormalization.

The lock and filesystem stamp are checked after DVC evidence collection and after loading and scientific checks. Both probability matrices are then associated with the same stable IDs in verified row order. The snapshot holds immutable numeric storage. Returned arrays have their own metadata and read-only buffers, and a returned table is detached. Repeated snapshot access performs no filesystem reads or DVC calls. `still_observed` checks the original workspace identity, exact control bytes and metadata for the selected files and directory members. It is an invalidation check for normal local changes. A new file read or a new join requires fresh verification. A snapshot can still describe its original point-in-time data after an invalidation. The cooperative lock covers dashboard runs and configuration changes. Metadata and content checks reject ordinary concurrent edits from other tools, while non-cooperative writers remain outside a filesystem transaction.

These readers are for trusted local production artifacts. PyG graph serialization requires the existing pickle-capable Torch loading mode on CPU. DVC hashes and scientific bindings establish correspondence within the trusted workspace and do not make an uploaded graph or fabricated lock trustworthy. Model and calibrator payloads are checked as bytes. No model or calibrator is deserialized for inference. Missing, partial, stale, conflicting and busy evidence can withhold the joined result with reasons. The existing original graph and standalone CSV diagnostics keep their independent evidence. Historical outputs with unsupported source, schema or correspondence remain unjoined.

### Display values and exports

The joined snapshot supplies observed class and raw features from the graph, plus either family's predicted class, maximum probability, selected-class probability, persisted entropy in nats and correctness where an observation has a label. No per-observation scores are available for test, calibration or CV populations. The data layer performs no inference, calibration, graph construction or training.

An enriched export is generated in memory with stable IDs, XYZ, observed and both predicted labels, both class-indexed probability families and their entropy values. Cycle, schema version, ordered class names, membership, source digests and original population count travel with the export. An export is explicitly `full_population` or `filtered_view`. Empty filtered exports keep their headers and the export object retains the scope and counts. The scientific producer CSV stays independently accessible and unchanged.

### Verification and measurements

The retained current-source workspace `/tmp/mineral-m35aii-quick.JeKROY` was unavailable during the initial engineering review. Disposable two-cycle fixtures verified the gate, including an integration check using actual local DVC status, and supplied the failure tests. Fixture locks and scores exercise the contracts without representing trained results.

The following warm medians were measured locally on an Intel Core i9-12900H. Raw loading and separate graph edge preparation used 30 repetitions. Joins included a fresh local DVC status subprocess and used seven repetitions per cycle. Repeated in-memory reads used 2,000 repetitions. The edge timings include original graph statistics, deduplication and immutable copies of stored topology, using prepared COO fixture arrays.

| Operation | Fixture size | Median |
| --- | ---: | ---: |
| Full base load | 128 observations | 4.55 ms |
| CV validation load, including parent and split verification | 26 selected observations | 5.97 ms |
| Incoming load, including parent verification | 16 incoming observations | 3.96 ms |
| Fresh prediction join, cycle 1 | 16 incoming observations | 493.75 ms |
| Fresh prediction join, cycle 2 | 16 incoming observations | 528.78 ms |
| Immutable coordinate access and calibrated argmax | 16 incoming observations | 1.8 µs |
| Cycle 2 metadata recheck | Selected producer scope | 2.90 ms |
| Edge preparation | 100 nodes, 700 COO entries | 0.70 ms |
| Edge preparation | 640 nodes, 12,160 COO entries | 12.29 ms |
| Edge preparation | 5,000 nodes, 55,000 COO entries | 60.24 ms |

These are fixture timings. Fresh verification scales with DVC status and the bytes and graph partitions in the selected producer ancestry. Edge preparation scales with the stored COO population. Snapshot reuse incurs only in-memory access and derivation, while metadata checks remain separate.

The initial full suite reported **461 passed and 7 skipped**. The non-integration suite reported **459 passed and 9 deselected**. Repository-wide Ruff lint and formatting, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed. Pre-commit modified no files.

### Retained Quick demo acceptance

The original temporary workspace could not be recovered from retained local files. The owner authorized one replacement full CPU reproduction, which completed in **63.647 s** in `~/Projects/DataScience/review-workspaces/mineral-classifier/v0.2.0.dev3/quick-demo/`. The distributed Quick demo v1 preset resolved to the same applied YAML SHA-256 as M3.5a-ii. The scientific source tree was `d3ace04b5c810dfc5320f736bdad758558df460c` at revision `20135e154cde6dba11623002fe4b0e9017661e3c`. The workspace retains its independent cache, source identity and hashes, generated DVC definition, DVC lock, scientific outputs and logs under `verification/`. Production configuration, DVC files, results and cache agreed with the before-run manifest.

All 14 expected stages executed, and the existing evidence layer verified all 14 stages and both learning cycles as matching. Both genuine spatial integration tests passed on disposable copies of the retained artifacts. Each checked 100 rows for exact sample IDs, float64 coordinates and raw features, observed labels, both probability families, persisted entropy and correctness. They also checked source membership, append order, and full and filtered enriched exports. Across both cycles and probability families, the maximum absolute entropy difference from the producer's formula was 1.86 × 10⁻⁷ nats.

All counts and published metrics matched the M3.5a-ii record at its reported precision.

| Cycle | Input base | Development | Calibration | Final test | Incoming | Post-append | Calibrated final-test accuracy | Final-test ECE before / after | Calibrated batch accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 440 | 308 | 40 | 92 | 100 | 540 | 36.30% | 0.071203 / 0.090677 | 43.82% |
| 2 | 540 | 378 | 49 | 113 | 100 | 640 | 30.83% | 0.028284 / 0.040227 | 42.67% |

The final base contained 448 development, 58 calibration and 134 final-test observations, as previously reported. Both runs produced 124 files. The replacement totaled 34,262,219 bytes, 1,466 bytes fewer than the prior reported total. The older files were unavailable for per-file comparison.

| Genuine retained-workspace operation | Cycle 1 median | Cycle 2 median |
| --- | ---: | ---: |
| Input base load | 15.00 ms | 29.02 ms |
| CV validation load, first fold | 7.48 ms | 10.03 ms |
| Incoming batch load | 5.03 ms | 6.29 ms |
| Fresh prediction join | 593.31 ms | 626.59 ms |
| Immutable coordinates and calibrated argmax | 1.99 µs | 2.73 µs |
| Workspace metadata recheck | 2.71 ms | 4.27 ms |

The 640-row final append loaded in 45.36 ms median. Measurements used 30 repetitions per raw load and metadata recheck, seven fresh joins per cycle, and 3,000 repeated in-memory reads per cycle. The full evidence refresh took 863 ms.

Set `MINERAL_QUICK_DEMO_WORKSPACE` to the persistent path for the genuine spatial checks and overview headline test. Without a configured path, the tests fall back to their original local fixtures and can skip if those are absent. The separate historical evidence tests accept `MINERAL_HISTORICAL_QUICK_DEMO_WORKSPACE` for the M2.5-era artifacts. The current-source replacement supports the current headline and spatial checks. The historical tests still require the historical source and lock.

Of the seven initial skips, two were the genuine spatial joins and one was the overview headline comparison. All three passed against the replacement. The other four require the absent M2.5 workspace and cover historical-source invalidation, mixed-cycle artifacts and bindings, and two workspace-change cases. The final full suite reported **464 passed and 4 skipped**. The fast suite reported **459 passed and 9 deselected**. Repository-wide Ruff, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed. Pre-commit modified no files.

## M3.5b-ii interactive spatial explorer

Data and Spatial predictions share a 520 px spatial canvas and observation inspector in Cycle explorer. One map or original diagnostic is mounted at a time. The shared learning-cycle selector retains input base c−1, incoming observations c and appended snapshot c. Data includes bootstrap, full input bases, each verified holdout and CV partition, and development membership. Appended base coloring uses verified source membership and includes the final snapshot. Full-base links are descriptive. Partition and incoming links come from their own stored graphs. The development selection exposes membership without topology.

The default plan view locks Y to X at 1:1. The 3D view uses Plotly's data aspect mode with the original XYZ values, and greater Z is higher. The generator samples Z from a typically negative lower bound to zero. Existing graph documentation uses synthetic coordinate units, so the labels are X, Y and Z and the caption identifies the simulated local Cartesian coordinates. There is no CRS or unit conversion.

Observed and predicted labels share colors and symbols indexed by class ID. The first four classes use blue circles, orange diamonds, green squares and purple crosses, in persisted mapping order. The legend leaves filtering to the explicit filters so hiding a class cannot silently alter map counts or exports. Probability color uses Cividis on the fixed 0–1 domain. Entropy uses Viridis on 0–ln(C) in nats. Features use their full-population bounds, with a single color and a labeled value when constant. Correctness uses contrasting shapes and colors. Score coloring and the inspector switch together when the probability family changes. Calibrated scores are the default.

Observed-class and signed Z filters are display-only. The context line shows the visible count and full-population class counts. With edges enabled, the map uses the stored deduplicated links that connect visible endpoints. The limit is **10,000 visible observations** and **8,000 visible links**. Above the observation limit the map asks for a narrower filter. Above the link limit it keeps the observations and omits the entire edge layer, with the actual visible link count and an explanation. Graph statistics still describe the original selected graph, including stored self-loops.

The inspector uses a searchable stable ID selector in both perspectives and highlights that observation on the map. It shows coordinates, observed class and membership plus matched prediction, correctness, max probability, entropy and a probability bar chart when scores are available. Raw feature detail and the observation table are mounted on request. Keeping the ID when a filter excludes it makes the exclusion explicit. Changing populations resets incompatible selections. Plotly keys and `uirevision` are scoped to the population and perspective. The installed Streamlit version can remount charts when the figure specification changes, so retained zoom or camera across every interaction is not guaranteed.

The map opens one fresh spatial join for the selected batch. Ordinary edits reuse the immutable result after the workspace stamp and cooperative lock checks. Changed evidence triggers a fresh verification, and an unsuccessful join withholds scores and enriched downloads. It keeps readable raw graph observations and offers verification details, explicit retry and original diagnostics. Historical cycles outside the applied scope use the original diagnostic route. The original CSV, enriched full-population CSV and enriched filtered-view CSV are clearly distinct. Verified download content is prepared on demand under the workspace lock and is checked again against the snapshot before delivery. Native browser content already drawn remains visible until a rerun.

Original diagnostics remain selectable, including stored graph figures, t-SNE, prediction histograms and the standalone CSV. The display does not regenerate scientific artifacts or scores.

### Browser review and performance

The 14-image review is in `/tmp/mineral-m35bii-review.ruyRWq/`. Its index identifies genuine Quick demo states, the deliberate failed-join case on a disposable copy, and the separate synthetic stress benchmark. Chromium verified a native XY pointer click through Streamlit 1.59.0. The returned point's custom data selected its stable ID, the map ring highlighted that exact row and the inspector displayed its probability chart. The 3D view uses the ID picker. Both genuine learning cycles also passed the Results AppTest integration checks against copies of the retained output.

The initial 1920×1080 view placed the 520 px map from y=542 to y=1062, with the inspector beside it. At 1280×900 the native dark view retained two columns and required scrolling for the bottom. At 390 px the 348 px plot and inspector stacked, with the map below the initial viewport. Horizontal overflow measured zero at all three widths.

| Measurement | Observed time |
| --- | ---: |
| Python input base load, 540 genuine observations | 23.24 ms |
| Python fresh join, genuine cycle 2 | 602.88 ms |
| Python full-base XY figure construction | 8.85 ms |
| Python genuine 3D figure, edges off / on | 8.75 / 10.08 ms |
| AppTest color / filter rerun | 37.4 / 38.6 ms |
| Chromium first genuine prediction view | 915 ms |
| Chromium color / filter rerender | 512 / 389 ms |
| Chromium genuine 3D edge toggle, off / on | 717 / 621 ms |
| Chromium isolated stress render, edges off / on | 311.6 / 593.0 ms |

Python and AppTest timings exclude the browser. Chromium app timings include the Streamlit rerun and frame delivery. The three genuine 3D toggles per state varied by several hundred milliseconds on the headless software renderer. The isolated stress render uses 5,000 synthetic UI observations and 7,500 predefined links, timing Plotly drawing and two animation frames over five runs per state. It excludes Streamlit and represents no scientific result.

Retained checksums still match all 124 result files, the scientific sources, applied configuration, DVC definition and lock. The retained cache still contains 146 files. The production before and after manifests match. No new experiment ran.

The full regression suite reported **473 passed and 4 skipped**, with skips confined to the unavailable historical M2.5 fixture. The fast suite reported **466 passed and 11 deselected**. Repository-wide Ruff, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed. Pre-commit modified no files. This initial review preceded owner visual acceptance.

### Focused visual refinement

The revised XY map fits an equal-span square around the full selected population with 5% padding on each side of its longest dimension. Plotly constrains the axis domains and preserves equal units per pixel. Display filters leave the initial framing alone. The centered desktop layout gives the map and inspector a combined content width of about 1,120 px. Filters are grouped in a popover alongside perspective, color and links, and the chart is 550 px high.

Probability and entropy start with the approved absolute domains, Cividis 0–1 and Viridis 0–ln(C). **Population range** optionally uses the exact minimum and maximum across the full selected population for the active value and probability family. The caption identifies the bounds and limits color comparisons to that population. Filters leave the domain fixed. A collapsed range uses one marker color and a constant-value caption. Stronger outlines keep low-contrast markers readable against both plot backgrounds. The 3D scene starts in an angled orthographic view with physical data aspect and signed Z. Stored edges remain opt-in with the existing limits. Plotly can clip some outer 3D axis ticks at the captured width. Signed Z remains available through hover and observation details.

The inspector defaults to a compact class summary with total and shown counts and each class's share of the full population. Verified prediction modes offer a context-specific predicted-class or correctness breakdown and keep zero-count categories visible. The class filter continues to use observed labels. Selecting an observation replaces the summary with labels, a correctness badge, max probability, entropy and probability bars as available. Display ordinals such as **Observation 001** identify choices within the current population. The picker, map events and exports still use the exact stable IDs. **Stable ID lookup and details** accepts an exact full ID and exposes the selected row's original ID, coordinates, membership and raw features.

Chromium confirmed native XY selection and exact full-ID selection in 3D on the genuine cycle 2 batch. The observed relative range for calibrated maximum probability was 0.2570238–0.2896868. At 1920×1080 the chart began at y=480, about 62 px earlier than the initial review. It ended at y=1030. The 467×467 px data pane used equal scales of 11.5713 synthetic units per pixel. The 1280×900 dark view retained side-by-side panels, with the chart bottom below the fold. At 390×850 the plot was 348 px wide and the inspector stacked below. There was no horizontal overflow at any of those widths.

The eight updated captures and measurement record are in `/tmp/mineral-m35bii-polish.0KlRFW/`. They cover the full-base frame, default prediction summary, selected sample, absolute and relative probability scales, entropy, stored links in 3D and the dark theme. The retained and production manifests were compared before and after the review. The owner accepted M3.5b-ii for visual quality and engineering after reviewing all eight refined screenshots.

The refined full suite reported **479 passed and 4 historical-fixture skips**. The fast suite reported **472 passed and 11 deselected**. Ruff lint and formatting, applicable pre-commit hooks, Git whitespace checks and offline uv lock consistency passed. Pre-commit modified no files. All 341 retained-file digests and modification timestamps matched the pre-review manifest, and the production manifest matched the earlier review baseline.

M3.6 will address the remaining 3D axis tick clipping, clarify verification-scope wording and refine probability formatting.
