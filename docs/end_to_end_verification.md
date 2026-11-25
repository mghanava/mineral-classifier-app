# M1.7 End-to-end production pipeline verification

## Source, environment and isolation

The runs used clean source revision `81e98c80fe9fde39e1938333cc675118a4e293ec` on `develop/v0.2.0`. The existing project `.venv/bin/python` provided Python 3.13.13, PyTorch 2.12.1+cu126, PyTorch Geometric 2.8.0, DVC 3.67.1, NumPy 2.5.1, scikit-learn 1.9.0, POT 0.9.6.post1 and Streamlit 1.59.0. The host was x86-64 Linux 7.3.5 with glibc 2.44. Runs used Matplotlib's Agg backend and set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1. The CPU runs used the sandbox. Experiment C used approved host execution on an NVIDIA GeForce RTX 3070 Ti Laptop GPU and resolved CUDA to `cuda:0`.

Source and baseline parameters were copied from Git into four separate directories under `/tmp/mineral-m17.eBxR6e/` using a scoped archive that omitted the production DVC files, results, cache and Git metadata. Each run used its own `PYTHONPATH`, DVC configuration and site cache, Matplotlib cache and workspace-local `.dvc/cache`. Before reproduction, the workspace location, effective cache path, empty cache, absence of results and lock records, and empty remote list were checked. Experiment A's result tree was rechecked before removal. Logs, effective parameter files, generated DAGs and lockfiles, validation scripts and the SHA-256 manifest are retained in that temporary directory for local inspection while it exists.

## Effective configurations

| Setting | A and CPU replay | B | C |
| --- | --- | --- | --- |
| Model and requested device | GCN, `cpu` | GAT, `cpu` | GCN, `cuda` |
| Configured cycles | 2, with replay targeted to cycle 1 | 1 | 1 |
| Bootstrap and prediction sample requests | 400 and 100 per batch | Same | Same |
| Features, classes, and CV folds | 5, 4, 3 | Same | Same |
| CV epoch budget and early-stopping patience | 15 and 5 | Same | Same |
| Calibration method and epoch budget | Dirichlet, 20 | Same | Same |
| Drift permutations | 19 for MMD, energy and exact multivariate Wasserstein | Same | Same |

All other parameters came from the revision's [params.yaml](../params.yaml). This includes RobustScaler with quantiles 25–75, the fifth-percentile graph rule with explicit self-loops and inverse-quadratic affinities at scale 460, stratified test and calibration settings of 0.3 each, and seeds 110 for base generation and partitioning and 42 for prediction generation, training, evaluation, and drift. The drift early-stop minimum remained 100, so each test executed all 19 permutations. The bootstrap generator appended 40 rows to meet the existing 50-per-class setting, yielding 440 rows rather than the 400 requested. Each appended batch contained 100 rows, yielding base populations of 540 and 640. No sample or epoch budget adjustments were needed after the initial smoke configuration.

## Commands and timing

Using the project interpreter in each temporary directory, setup ran `python -m dvc init --no-scm`, `python -m dvc cache dir --local <absolute-workspace>/.dvc/cache`, `python -m scripts.setup_dvc` and `python -m dvc dag`. Experiments A, B and C used `python -m dvc repro --no-run-cache`. The separate CPU replay used `python -m dvc repro --no-run-cache predict_cycle_1 analyze_drift_cycle_1 prepare_next_cycle_data_1`. DVC's logs recorded the actual commands shown below. `N` was 1 and 2 for A, and 1 for all other runs. `MODEL` was `gat` for B and `gcn` otherwise.

| Command emitted by DVC | A cycle 1 | A cycle 2 | B | C | CPU replay |
| --- | ---: | ---: | ---: | ---: | ---: |
| `python src/stages/generate_base_data.py --cycle 0` | 6.103 s | | 6.099 s | 6.162 s | 6.156 s |
| `python src/stages/train.py --cycle N --model MODEL` | 2.895 s | 2.597 s | 4.046 s | 2.974 s | 2.850 s |
| `python src/stages/evaluate.py --cycle N --model MODEL` | 0.977 s | 0.823 s | 1.009 s | 1.427 s | 1.006 s |
| `python src/stages/generate_pred_data.py --cycle N` | 0.212 s | 0.182 s | 0.209 s | 0.217 s | 0.213 s |
| `python src/stages/predict.py --cycle N --model MODEL` | 0.961 s | 0.817 s | 0.994 s | 1.193 s | 0.957 s |
| `python src/stages/analyze_drift.py --cycle N` | 3.224 s | 2.822 s | 3.162 s | 3.421 s | 3.252 s |
| `python src/stages/prepare_next_cycle_data.py --cycle N` | 7.938 s | 7.702 s | 7.950 s | 7.023 s | 8.105 s |
| `python src/stages/analyze_cycles_performance.py --cycles 2` | Passed, included in A wall time | | | | |

Stage durations are the production timing messages. Totals below include interpreter startup, DVC hashing and cache writes, and concurrent smoke-run activity. They are smoke timings rather than performance comparisons.

| Experiment | Outcome | Reproduction wall time | Verified files |
| --- | --- | ---: | ---: |
| A, two-cycle CPU GCN | 14 stages ran, including performance aggregation | 82.55 s | 124 |
| B, one-cycle CPU GAT | All 7 stages ran. The attention layers had no edge-feature projection. | 48.19 s | 74 |
| C, one-cycle host CUDA GCN | All 7 stages ran. Training, final refit, evaluation, calibration, and scoring used `cuda:0`. | 48.90 s | 74 |
| Independent CPU replay of A cycle 1 | All 7 targeted production stages ran from a fresh cache | 47.53 s | 74 |

## Scientific artifact checks

- Loaded the bootstrap hotspots with shape `(3, 3)` and prototypes with shape `(4, 5)`. Checked finite canonical float64 features and coordinates, valid four-class labels, and unique sample IDs. Next-cycle preparation preserved the exact raw row order, feature values, labels, coordinates and IDs of both source populations.
- Verified fold, calibration and test membership with the production readers. Rebuilt these graphs independently from their own coordinates and matched their topology digests, edge indices and affinities. The cycle 1 development, calibration and test populations contained 308, 40 and 92 rows. Cycle 2 contained 378, 49 and 113 respectively. The lock confirmed that later cycles consumed the previous base and that next-cycle preparation depended on raw inputs, with no dependency on scored predictions.
- Reconstructed each full-development graph and fitted RobustScaler on the verified development rows. Its coefficients matched the frozen preprocessing sidecar exactly. Checked the model and graph bindings, finite weight tensors, and all CV and refit schedule metadata. All runs selected a 15-epoch refit. A separate read-only call trace of the real cycle 1 training path observed three fold-local scaler fits and transforms, one development fit, and four distinct models and optimizers. The traced run wrote to a separate output directory, reproduced the same binding, and took 4.35 s.
- Verified that the calibration binding pointed to the reserved calibration graph and the matching model bundle. Recomputed probabilities on the independent test and prediction graphs using the bound weights, scaler and calibrator. Every prediction CSV had 100 rows and four probabilities per representation. Checked finite metrics and probabilities in `[0, 1]` with row sums within `1e-6`, and matched the stored probabilities and labels against production inference. The predicted labels and test metrics were checked for compatibility, with no accuracy or significance threshold.
- Validated drift's input and fitted-transform provenance and finite MMD, energy and Wasserstein statistics. The p-values fell on the expected 19-permutation grid. Decoded the Plotly JSON and PNGs, confirmed the HTML graph render payloads, and checked that performance aggregation read evaluation and prediction metrics for exactly cycles 1 and 2.

## Cache restoration, seeded recomputation and dashboard

For A, an unchanged reproduction skipped all 14 stages and executed zero stage commands in 0.62 s. After verifying isolation, the experiment's declared outputs were inventoried and SHA-256 hashed, and its results tree was removed. `python -m dvc checkout` restored all 124 files from the workspace's cache in 0.45 s. The restored path list and every hash matched, including all split, model, calibration and prediction files, 28 PNGs and 29 HTML files. The production readers and scientific artifact checks passed again. A second unchanged reproduction executed zero stage commands in 0.71 s. The lockfile hash matched before and after.

The independent CPU workspace recomputed cycle one with the exact A configuration, an empty cache, and run-cache lookup disabled. Raw arrays, IDs, bootstrap arrays, partition indices, topology and affinities matched exactly. CV diagnostics, final duration, learning-rate schedule, fitted preprocessing, state-dict tensors, evaluation and prediction metrics, prediction CSVs and drift reports also matched exactly. These comparisons apply to the tested environment and seed settings.

Streamlit AppTest rendered all seven tabs from the restored A tree in 6.72 s, including bootstrap graphs and both prediction cycles. It found the base and batch Plotly charts, t-SNE figure, all three training diagnostics per cycle, evaluation figures and metrics, each 100-row prediction table, drift figures and statistics, and the performance chart. No tab raised an unhandled exception or reported missing results.

## CUDA bundle portability

Experiment C's GPU-trained bundle was loaded and evaluated on CPU through the production evaluator. CPU recalibration and test evaluation wrote to `c/cpu_portability/evaluation/` and took 0.80 s. Two CPU prediction passes wrote into separate directories and took 1.28 s together, one with the original CUDA-fitted calibrator and one with the CPU-refitted calibrator. The SHA-256 hashes of all 17 original training, evaluation and prediction files matched before and after.

For a comparison with identical weights, preprocessing and calibrator, CPU prediction was compared to the saved CUDA predictions using absolute tolerance `2e-6` and relative tolerance `1e-5`. Maximum absolute differences were `7e-8` for calibrated and `4e-8` for uncalibrated probabilities, and predicted labels matched. Separately refitting calibration on CPU also yielded finite metrics and valid probabilities. That fit uses a device-specific random stream and had different calibrated test metrics, so it was excluded from the same-calibrator equivalence comparison.

## Defects, checks and release status

All production commands succeeded. Setup initially stopped before reproduction because the sandbox lacks the external `/usr/bin/time` utility. The smoke runner used Bash timing for the actual execution. The temporary dashboard checker initially queried the wrong AppTest element name and passed after it checked the actual `image` elements. Agg and Streamlit emitted nonfatal plotting and API deprecation warnings. The probes required no production code corrections.

The production `params.yaml`, `dvc.yaml`, `dvc.lock` and `uv.lock` retained their original Git blob hashes. The temporary locks contained exactly 14, 7 and 7 active stages for A, B and C. The checked-in production lock remains the historical 122-record file described in the [DVC integrity audit](dvc_integrity.md). Full default reproduction still needs separate authorization, a clean isolated run with the matching production configuration, and review of its replacement lock. A smoke-test lock corresponds only to its smoke configuration. The verified scope is the reduced synthetic workflow in the existing environment. Default budgets, container deployment, broader hardware and spatial generalization remain outside this verification.

The complete Pytest suite passed all 147 tests in 58.39 s. Repository-wide Ruff lint and format, applicable pre-commit hooks, Git whitespace checks and the offline uv lock consistency check passed. Pre-commit made no edits.
