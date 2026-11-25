# Mineral Classifier App v0.1.0 baseline

Milestone 0 inspected the existing synthetic workflow and ran lightweight checks without starting training. The candidate baseline is commit `d00c2cd0c01566f930c51b63aa66c7bbbda9fa97` (commit date 2026-09-13). The release date and intended release commit still require owner confirmation.

## Git and environment

- The checkout was on `main`, aligned with the local `origin/main` ref. The index and tracked working tree were clean. The local and remote-tracking refs showed no `develop/v0.2.0` branch, and there were no local tags. Remote state was not queried.
- `AGENTS.md`, `ROADMAP.md`, and `CHANGELOG.md` were already present as untracked files when the audit began. They were read and preserved. This report is the sole addition.
- Both `pyproject.toml` and `uv.lock` identify the project as `0.1.0` and require Python 3.11 or newer. Docker uses Python 3.13. The proposed `0.2.0.dev0` version, development branch, and preservation tag are pending owner direction. The candidate commit remains in Git history and is not protected by a release tag.
- Checks used the existing Linux virtual environment with Python 3.13.13, uv 0.13.10, Streamlit 1.59.0, DVC 3.67.1, PyTorch 2.12.1+cu126, and PyTorch Geometric 2.8.0. No dependency sync was performed.

## Entry points and execution

Run from the repository root. For an existing compatible uv environment, the local dashboard and a read-only DVC graph check are available through module entry points.

```fish
uv run --frozen --no-sync python -m streamlit run app.py
uv run --frozen --no-sync python -m dvc dag --dot
```

On an environment provisioned from the lockfile, `uv sync --locked --extra dev` installs the project and development extra. It can require large CUDA downloads. The existing environment here has some stale executable shebangs after relocation, including `dvc` and `streamlit`. Module invocation works around them for local inspection. Dashboard stage buttons invoke `python` and `dvc` as subprocesses, so normal command resolution still matters.

The README's container entry point is `make run`, which builds and starts Compose and exposes the dashboard on port 8501. The multi-stage Dockerfile installs from the uv lockfile and sets `PYTHONPATH=/app`. The base Compose file requires the NVIDIA runtime and uses named results and DVC cache volumes. The development overlay bind-mounts the source, results and cache. Both Compose configurations validate, but container build and startup were not exercised.

`app.py` owns the Parameters, Data, Train, Evaluation, Prediction, Drift Analysis and Performance tabs. It edits the shared `params.yaml`, runs DVC targets and reads results from disk. `scripts/setup_dvc.py` regenerates `dvc.yaml` using `cycles` and `default_model`. On a working installation, the Data tab can run bootstrap, and the Parameters tab can regenerate and run the full pipeline. The CLI equivalents from the root are `uv run --frozen --no-sync python -m dvc repro bootstrap` and, for a full run, `uv run --frozen --no-sync python -m scripts.setup_dvc` followed by `uv run --frozen --no-sync python -m dvc repro`. The setup command rewrites `dvc.yaml`, and reproduction can execute upstream stages and update the lockfile and results. The full default run is substantial, with three cycles, three folds, up to 1,500 epochs per training fold, 1,000 calibration epochs and 1,000 drift permutations.

## Pipeline and required artifacts

`src/stages/` contains CLI stages, `src/models/` contains GCN and GAT, and `src/utilities/` implements data generation, spatial graphs, training, calibration, prediction, metrics and drift. Defaults include GCN, 1,000 initial samples, 200 prediction samples per cycle, five features, four classes, RobustScaler, percentile-based graph connectivity and Dirichlet calibration. The calibration implementation also supports temperature, isotonic, Platt and beta methods.

| Step | Inputs and outputs |
| --- | --- |
| Bootstrap and next-cycle preparation | Bootstrap creates base, fold, test and calibration graph tensors under `results/data/base/cycle_0`, plus synthetic hotspots, prototypes and diagnostics. Next-cycle preparation combines base data with labeled synthetic prediction data and rebuilds the splits for the next cycle. |
| Train and evaluate | Training reads prior-cycle fold graphs and writes `results/trained/cycle_N/model.pt` with diagnostics. Evaluation reads the checkpoint and prior-cycle test and calibration graphs and writes plots, metrics and `calibrator.pt` under `results/evaluation/cycle_N`. |
| Generate prediction data and predict | Generation reads the prior base coordinates and the bootstrap hotspot/prototype arrays. Prediction consumes the resulting graph, checkpoint and calibrator and produces a prediction CSV, metrics and plots under `results/prediction/cycle_N`. |
| Drift and performance | Drift compares base and prediction data and writes diagnostics under `results/drift_analysis/cycle_N`. The performance stage reads per-cycle evaluation and prediction metrics to produce the aggregate plot. |

The current pipeline has 20 stages. The checked-in DVC lock contains 122 stage records, including 102 from cycles 4 through 20 that are absent from the current DAG. The checkout has only `results/.gitkeep`, an empty local DVC cache and no DVC remote configuration under `.dvc`. Artifact availability in Docker volumes or another checkout was not checked. A fresh checkout has no bundled results to explore.

## Check results

| Check | Outcome |
| --- | --- |
| `uv --no-cache lock --check --offline` | Passed. The uv project lock is current. |
| Python compilation in memory | Passed for all 26 tracked Python files. |
| Pytest 9.1.1, `pytest -p no:cacheprovider` | Exit 5. No tests collected and no files under the configured `tests` path. |
| Ruff 0.15.20, `ruff check --no-cache --statistics app.py scripts src` | Exit 1. Ten existing findings across D104, D417, RUF003, B007, D103 and RUF059. Ruff also reported incompatible docstring rule pairs D203/D211 and D212/D213. |
| `ruff format --check --no-cache app.py scripts src` | Passed. All 26 files already formatted, with the same rule warnings. |
| Streamlit AppTest | Passed. All seven tabs rendered without an app exception. Six missing-artifact messages appeared on the empty results tree. No action buttons were invoked. |
| In-memory CPU graph and model smoke | Passed with CUDA hidden. An 80-node, five-feature, four-class fixture produced three folds with disjoint train, validation, test and calibration node sets per fold. Each split's edges matched its induced subset of the full graph. GCN and GAT returned finite `(80, 4)` tensors. The initial smoke invocation passed tensors individually and was corrected to use the models' `forward(data)` interface. |
| CLI and DVC | Training CLI `--help` worked. DVC parsed the 20-stage graph using module invocation and a temporary site cache. Regeneration in a temporary directory matched the checked-in pipeline as parsed YAML. No stages were run. |
| Compose configuration and runtime | Both Compose configurations passed `config -q`. Docker daemon access was denied at the socket. Build, GPU access and container startup remain unverified. |
| Other static tooling | No type checker is configured. Hadolint was unavailable as a standalone command. Pre-commit hooks were not run because the configuration includes auto-fix and file-rewriting hooks. |
| Git checks | Tracked and staged diff whitespace checks passed. |

## Regression and reproducibility risks

- **Split preprocessing and graph construction.** `construct_graph` fits the scaler and constructs the graph before splitting. The subgraphs remove cross-partition edges, but scaling and adaptive graph statistics can still incorporate held-out inputs. Prediction generation fits a new scaler on the prediction batch. Splits are stratified random and rebuilt each cycle. These scores do not establish generalization to unseen regions or track performance against a fixed holdout.
- **Fold independence.** `train.py` creates one model before the fold loop, and the training utility updates it in place. Successive folds can start with weights learned on previous folds, where some of the new validation samples could have been training samples. Review this before treating fold results as independent.
- **DVC cache completeness and invalidation.** Bootstrap writes `hotspots.npy` and `prototypes.npy` that prediction generation reads, but they are undeclared. Graph diagnostics also fall outside declared outputs. Evaluation reads `calib_data.pt` without declaring it as a dependency. Bootstrap reads scaler and edge settings outside its tracked parameter keys, and calibration utilities are absent from several consumers' declared code dependencies. Cache restoration or changes to these inputs can leave required files missing or results stale.
- **Repeatability.** Base generation and splitting have a configured seed, but prediction seed is null and the generator passes it to NumPy's RNG. The training entry point does not explicitly seed model initialization or training. No repeated full runs were compared.
- **CPU deployment and artifact portability.** Training, evaluation and prediction select CPU when CUDA is unavailable. The lock still selects the CUDA 12.6 PyTorch distribution, Compose requests NVIDIA, and dataset and model loads lack device remapping. A CUDA-saved artifact may fail on a CPU-only host. The current CPU forward smoke does not establish CPU-only installation or portability. Graph and calibrator deserialization uses full-object loading in trusted local workflows.
- **Dashboard contracts.** Tabs depend on shared paths and artifact filenames. The full-run button ignores subprocess return codes and can display success after a failed step. Preserve stage and artifact contracts during extraction and test both missing and populated results states.

## Owner follow-up

Confirm whether the candidate commit is the intended v0.1.0 release and provide direction for preserving it with a tag and setting up `develop/v0.2.0`. Git history is intact, but the milestone's tag-based preservation criterion remains open. Before using this baseline to judge later changes, a focused follow-up can verify trusted artifacts in a supported environment and cover the identified DVC and scientific regression risks. Full training, calibration, prediction, drift, cache restoration and end-to-end reproduction remain unverified.
