# M2.5 Configuration integration and production verification

## Source, environment and isolation

The two production runs used revision `a0ff590753d3c491b11af1d13e588e78a9e3c6e8` on `develop/v0.2.0`. Dashboard verification also exercised the concurrency correction described below. The locked project environment supplied Python 3.13.13, PyTorch 2.12.1+cu126, PyTorch Geometric 2.8.0, DVC 3.67.1, NumPy 2.5.1, scikit-learn 1.9.0, POT 0.9.6.post1 and Streamlit 1.59.0. The host used an Intel Core i9-12900H, x86-64 Linux 7.3.5-2-cachyos and glibc 2.44. Both runs forced CPU, used Matplotlib Agg, and set OpenMP, MKL, OpenBLAS and NumExpr thread counts to one. Wall times are single observations under ordinary host activity.

Following [M1.7](end_to_end_verification.md), scoped Git archives populated separate directories under `/tmp/mineral-m25.SkdwUb/`. Each production workspace began without generated results or a DVC lock, initialized DVC with `--no-scm`, set a separate local cache and workspace-scoped DVC config, site cache and Matplotlib cache, and confirmed an empty cache and remote list. Reproduction used `python -m dvc repro --no-run-cache` through the existing project interpreter. Fully resolved YAML, timing and stage logs, generated definitions, DVC locks, compatibility checks and outputs remain in that temporary directory while it exists. The production files and artifacts stayed in the repository workspace. Hash checks covered production `params.yaml`, `dvc.yaml` and `uv.lock`.

## Effective configurations

The complete effective documents are [canonical defaults](../config/defaults.yaml) with the following overrides applied. Both retain the complete GCN and GAT blocks, RobustScaler with quantiles 25–75, fifth-percentile connectivity, explicit self-loops, inverse-quadratic affinity with scale 460, five features and four classes, test and calibration settings of 0.3 each, Dirichlet calibration, and exact multivariate Wasserstein. The bootstrap and partition seed is 110. Prediction generation, training, evaluation and drift use seed 42.

| Setting | A Quick demo v1 | B Standard v1 on CPU |
| --- | --- | --- |
| Model, device, cycles | GCN, CPU, 2 | GCN, CPU, 2 |
| Base request, prediction request per cycle | 400, 100 | 700, 150 |
| CV folds | 3 | 3 |
| CV epoch limit, early-stopping patience | 15, 5 | 100, 20 |
| Calibration iterations budget | 20 | 100 |
| Drift permutations budget per statistic | 19 | 199 |

In each workspace `python -m scripts.configure resolve <preset> --against params.yaml` produced complete scientific YAML on stdout and findings and diffs on stderr. Preview preserved the initial active bytes. Explicit `--output params.yaml --overwrite` published the same bytes, and the exported files passed strict `validate`. The effective YAML contains only scientific settings. Standard resolved with an override file containing only `device: cpu`. Comparing against a regular Standard resolution confirmed that device was the sole difference.

Quick demo's resolved YAML also matched the retained M1.7 Experiment A configuration in every value and scalar type. Its four nonblocking budget warnings covered the training scheduler, calibration scheduler and early stop, and drift early-stop minimum. The generator produced the same 14-stage DVC definition as M1.7 A. Both current production locks contained exactly the stages, commands and declared outputs of their generated definitions.

## Production results

Both experiments executed all 14 stage commands successfully. Quick demo took **74.31 s**, compared with 82.55 s in M1.7. An unchanged Quick demo reproduction skipped all 14 stages and issued zero stage commands in **0.74 s**. Standard took **101.66 s**, below the proposed 3–5 minute CPU target. Neither preset budget changed.

| Stage duration in seconds | Quick cycle 1 | Quick cycle 2 | Standard cycle 1 | Standard cycle 2 |
| --- | ---: | ---: | ---: | ---: |
| Bootstrap | 5.311 | | 8.631 | |
| Train | 2.533 | 2.213 | 4.558 | 5.773 |
| Evaluate and calibrate | 0.829 | 0.879 | 0.758 | 0.856 |
| Generate prediction batch | 0.175 | 0.183 | 0.212 | 0.211 |
| Predict | 0.736 | 0.727 | 0.740 | 0.733 |
| Analyze drift | 2.720 | 2.606 | 4.794 | 5.850 |
| Prepare next-cycle data | 6.242 | 8.004 | 12.329 | 14.220 |
| Aggregate cycle performance | Completed | | 1.545 | |

The table uses the stages' internal timing messages. Performance aggregation had no built-in timer. A workspace-only wrapper measured its Standard process from Python startup instrumentation to exit. Quick demo aggregation is included in the DVC wall time. DVC wall times also include interpreter startup, hashing and cache writes. Standard used workspace-only timing wrappers around the unchanged training and refit calls and counted calibration iterations from the existing per-iteration loss history.

| Preset and cycle | Starting base | Batch | Appended base | Development / calibration / test | CV best / completed epochs by fold | Refit epochs | Calibration iterations |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Quick 1 | 440 | 100 | 540 | 308 / 40 / 92 | 15/15, 15/15, 15/15 | 15 | 20 |
| Quick 2 | 540 | 100 | 640 | 378 / 49 / 113 | 15/15, 15/15, 15/15 | 15 | 20 |
| Standard 1 | 736 | 150 | 886 | 515 / 67 / 154 | 52/71, 42/62, 29/48 | 42 | 100 |
| Standard 2 | 886 | 150 | 1,036 | 620 / 80 / 186 | 39/57, 40/60, 83/96 | 40 | 100 |

The bootstrap class floor added 40 Quick demo rows and 36 Standard rows. Both prediction batches per experiment met their requested sizes. Quick demo's calibration iterations followed the configured bound and its early-stopping patience. Standard CV training calls took 1.668, 1.136 and 0.926 s in cycle 1 (3.731 s total), then 1.727, 1.350 and 1.657 s in cycle 2 (4.734 s total). These calls include fold diagnostic plotting. The final-refit training calls took 0.746 and 0.929 s, respectively. The full train stage also includes graph and preprocessing preparation and bundle publication.

| Preset and cycle | Test accuracy before / after calibration | Test ECE before / after calibration | Batch prediction accuracy | Drift p-values for MMD / energy / Wasserstein |
| --- | --- | --- | ---: | --- |
| Quick 1 | 0.3451 / 0.3630 | 0.0712 / 0.0907 | 0.4382 | 0.5789 / 0.7368 / 0.0526 |
| Quick 2 | 0.3012 / 0.3083 | 0.0283 / 0.0402 | 0.4267 | 1.0000 / 1.0000 / 0.4211 |
| Standard 1 | 0.4837 / 0.4670 | 0.1210 / 0.1024 | 0.4607 | 0.8543 / 0.9146 / 0.6131 |
| Standard 2 | 0.3051 / 0.3137 | 0.0838 / 0.0786 | 0.4048 | 0.9799 / 0.9799 / 0.4271 |

Each Quick demo statistic exhausted 19 permutations, below the 100-permutation early-stop minimum. Standard exhausted 199 for each of the three statistics in each cycle. Its logs show progress through 150 with no early stop, and the reported p-values are consistent with the full count.

Both workspaces produced 124 result files. The production readers loaded the raw data, partitions, CPU models, fitted preprocessing and bound calibrators. Both cycles preserved the source sample IDs through append and matched the drift input and model bindings. Evaluation and prediction metrics were finite. Both stored probability representations and recomputed calibrated test probabilities were finite, in `[0, 1]`, and summed to one per row within `1e-6`. Recomputed batch probabilities matched their CSV values within numerical tolerance, and predicted classes matched calibrated argmax. In both Quick demo cycles, model tensors matched the retained M1.7 tensors exactly. Training and preprocessing sidecars, evaluation and prediction metrics, prediction CSVs, and drift reports and provenance matched byte for byte. This verifies scientific equivalence for the tested Quick demo configuration. The metrics describe the synthetic runs.

## Research, CLI and DVC compatibility

- Research passed strict validation and matched the complete canonical YAML recursively, including scalar types, explicit nulls and both model blocks. Generating the pipeline separately from Research and canonical defaults produced identical stage mappings. Research training was skipped as scoped.
- The CLI resolved GAT with one cycle and four folds, reported all three topology changes, and produced seven stages with GAT commands and fourth-fold graph outputs. Importing the exported result preserved all settings and generated the same definition.
- A StandardScaler change cleared inherited robust quantile options. A MinMaxScaler override retained the explicit feature range and clipping option. Scaler options belonging to the wrong scaler failed validation.
- The exported override preserved explicit null seeds in all five scopes, an integer holdout count and null calibration size. A separate legacy import retained null holdout and integer calibration size and resolved an omitted bootstrap seed to its runtime fallback of 42. Both enabled legacy Wasserstein flags survived and the CLI reported max-sliced precedence.
- Malformed YAML, duplicate mapping keys, out-of-range cycles and invalid scaler options returned exit status 1 without publishing output. Diagnostics were separate from scientific stdout.

## Dashboard and concurrency

Streamlit AppTest exercised the actual Parameters tab. It showed the active YAML initially. Preset selection, preset load, expert edits and validation left the active bytes unchanged. Preview displayed the complete resolved YAML, semantic differences and topology regeneration guidance. Invalid YAML or settings could not be applied. A valid preview with warnings applied explicitly without spawning subprocesses. An external change made an earlier preview stale and the dashboard preserved the newer bytes. Targeted execution tests verified actual regeneration before DVC with applied model settings and ignored unapplied edits.

A second AppTest pass opened the actual Quick demo results in the seven-tab dashboard. It discovered base and batch charts, training diagnostics, evaluation metrics and figures, 100-row prediction tables for both cycles, drift figures and statistics, and the performance chart. With a Standard draft loaded and previewed, the actual dashboard full-run control regenerated from the applied Quick demo YAML and DVC skipped all 14 stages.

**Confirmed defect and correction.** A second dashboard session could apply a changed cycle count while the first session's reproduction subprocess was still live. The pre-fix AppTest regression observed the changed active bytes. The write lock previously covered only the atomic write. Dashboard reproduction now reserves the same directory lock from before reading the applied configuration through completion of the subprocess. Competing dashboard runs and cooperative configuration exports fail promptly while it is held. A failed Apply keeps its preview available for retry. The regression exercises a separate AppTest process during reproduction, checks that Apply reports the busy state and preserves the active file, then applies successfully after the run exits. A CLI export attempted while the directory reservation was held also failed with the busy message and preserved the file.

Direct file edits, direct stage commands and direct `dvc repro` do not acquire the dashboard lock. These workflows still require a fixed active configuration for the duration of the run. Dashboard runs share the applied configuration and results between sessions, and the lock is cooperative on the supported local POSIX filesystem. Paths in the same workspace directory share the reservation. The dashboard still has Streamlit `use_container_width` deprecation messages. Static validation still leaves data-dependent constraints to execution.

## Final regression checks

The full locked Pytest suite passed 336 tests. The non-integration suite passed 335 tests with one deselected. Repository-wide Ruff lint and formatting, the applicable pre-commit hooks, Git whitespace checks and `uv lock --check --offline` passed. The initial pre-commit pass reformatted `app.py`. The final pass made no edits. No M2.5 release blocker remained after the dashboard concurrency correction.

## Preset and UI recommendations

Keep the approved Quick demo and Standard budgets. Standard completed below the proposed target, and next-cycle data preparation was its largest timed stage. Communicate that displayed runtime and realized sample counts depend on the host and class floor. In the UI redesign, keep the resolved YAML and configuration diff accessible, show the applied preset or effective settings alongside the run status, and make the shared-workspace busy state visible before users attempt Apply. Benchmark in the deployment environment before publishing a runtime estimate. Full Research runtime, container deployment and broader scientific validation remain outside M2.5.
