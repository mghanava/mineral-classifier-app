# M1.6 Reproducibility and execution

## Seeds and provenance

Configure random seeds in `params.yaml`. An omitted seed uses 42. The checked-in bootstrap uses 110. Explicit `null` opts into unseeded execution for that setting. The configuration accepts integers from 0 through 4294967295, with any nonnegative integer accepted for the NumPy-only prediction generator.

| Setting | Effective behavior | Recorded in |
| --- | --- | --- |
| `data.base.seed` | Still passed directly to NumPy's bootstrap generator and to the existing stratified holdout, calibration and fold splitters for every base cycle. The same seed on the same ordered rows preserves partition membership. | Bootstrap `base_data.pt` records the generation and partition seeds. Later base artifacts record their partition seed. |
| `data.pred.seed` | The checked-in value is now 42. A numeric seed still goes directly to NumPy for every prediction batch, using that cycle's parent as the exclusion and provenance input. `null` retains the previous entropy-backed generation behavior when the stage executes. | `pred_data.pt` records the generation seed alongside its source-parent digest. |
| `train.seed` | Derive one Torch stream per cycle and zero-based CV fold and a separate stream for the final development-pool refit. Each stream covers model initialization and stochastic training operations such as dropout. | The `training.seed_provenance` field in `graph_binding.json` records the configured seed, cycle, resolved device and fold and refit seeds. |
| `evaluate.seed` | Still passed directly to the internal stratified calibration split for parametric calibration. The initialization RNG gets a derived per-cycle seed. Isotonic fitting uses neither draw. | The metadata within `calibrator.pt` records the split seed, initialization seed, cycle and resolved device. The existing calibration binding hashes the complete file. |
| `analyze_drift.seed` | Derive an effective seed per cycle and independent streams for each permutation statistic, max-sliced transport projections, mutual information of each population, and PCA or KernelPCA when their solvers use randomness. Max-sliced transport draws fresh unit vectors per statistic, including the observed statistic and every permutation. | `drift_metadata.json` records both the configured and effective seeds. |

The `sha256-v1` scheme takes the first four bytes of SHA-256 as an unsigned big-endian integer. Its input is the compact JSON array containing `"sha256-v1"`, the configured seed and an ordered scope. Training uses scopes `["train", cycle, "cv", fold_index]` and `["train", cycle, "final_refit"]`. Evaluation uses `["calibration", cycle, "init"]`. Drift first uses `["drift", cycle]`, then derives from the effective seed with the scopes `["permutations", method]`, `["max_sliced_projections"]`, `["mutual_information", "reference"]`, `["mutual_information", "incoming"]`, `["pca"]` and `["kernel_pca"]`. The existing t-SNE diagnostics keep their fixed seed of 42. Prediction scoring reuses the fitted classifier and calibrator in evaluation mode.

Seeded runs are intended to repeat with the same inputs, configuration, device, software versions and execution environment. CPU tests compare raw digests and indices, initialization and training draws, calibrated probabilities, and drift results. Hardware, operating system, thread settings and library versions can affect numerical results and serialized bytes. Torch RNG state is scoped to the seeded operation and strict deterministic GPU algorithms are not forced. With `null`, metadata records the unseeded choice rather than the generated entropy. DVC can reuse cached artifacts for unchanged inputs, including an explicitly unseeded stage.

Seed keys are already covered by the stage parameter groups `data.base`, `data.pred`, `train`, `evaluate` and `analyze_drift`. Device settings are tracked only by training, evaluation and prediction. The isolated DVC fixture checks that drift seed edits rerun only drift, calibration seed edits reuse training, and training seed or device edits reuse raw generation. The DVC artifact paths and bindings are described in the [M1.5 integrity audit](dvc_integrity.md).

## Device and artifact portability

Set the top-level `device` key to `auto`, `cpu`, `cuda` or `cuda:N`, where N is a zero-based index among visible devices. `auto` is the default and selects the first visible CUDA device, falling back to CPU when CUDA is unavailable. `cpu` runs training, evaluation and prediction on CPU even when CUDA is visible. An explicitly requested unavailable CUDA device fails with guidance to check CUDA visibility and drivers or select CPU. Use an explicit device when the execution device must be part of the DVC configuration. Availability changes under `auto` do not change the configured value or by themselves invalidate DVC cache keys. Drift and canonical graph production run on CPU.

PyG graph artifacts are built and stored on CPU, and trusted tensor object loads map storage to CPU before schema and graph validation. Model state dictionaries use weights-only loading with CPU mapping. Each prepared graph and model is moved to the selected compute device for the forward pass. Calibrator loading maps to CPU before restoring parameters on the inference device. Isotonic thresholds move to that device on use. Remapping leaves the original bytes and their bound digests intact. Graph and calibrator files use trusted-local object deserialization and are only for artifacts produced or verified within the trusted workflow. A digest does not authenticate the file's origin.

The image and uv lock still use the existing CUDA 12.6 PyTorch distribution. Compose still requests the NVIDIA runtime, including through the development overlay. An app-level CPU selection does not remove the container runtime requirement. CPU execution was exercised locally, and automated tests simulate CUDA availability and CUDA-tagged storage. CPU-only container deployment remains unverified. The Linux `make abort` process scan still depends on `/proc`.

### CUDA diagnostic

Both probes used the project's `.venv/bin/python` with PyTorch `2.12.1+cu126` and `torch.version.cuda == "12.6"`.

| Execution context | GPU visibility |
| --- | --- |
| Sandbox | `torch.cuda.is_available()` returned `False` and PyTorch counted zero devices. Opening `/dev/nvidia0`, `/dev/nvidiactl` and `/dev/nvidia-uvm` returned “No such file or directory.” `nvidia-smi` could not communicate with the driver. |
| Host outside the sandbox | `torch.cuda.is_available()` returned `True` and PyTorch counted one device, an NVIDIA GeForce RTX 3070 Ti Laptop GPU. The same device nodes opened successfully. |

On the host, the production graph builder and fold-local preprocessing prepared a 37-node training fold with 131 stored edges. The probe used the configured fifth-percentile connectivity, inverse-quadratic affinities with scale 460, and explicit self-loops. GCN and GAT each passed one CUDA forward and backward pass followed by synchronization. Both produced finite logits of shape `(37, 4)`, finite cross-entropy loss, and finite CUDA parameter gradients, with a nonzero gradient in each model. This was an in-memory smoke test without optimizer steps or artifact writes. Production pipeline and container verification remain pending.

## Command failures

The dashboard starts DVC and pipeline generation with its own Python interpreter and adds that interpreter's directory to child `PATH` for DVC's stage commands. The log includes stdout and stderr, and the dashboard reports the failing command and exit status or launch error. Full reproduction starts only if generation succeeds. The full-run success message appears only after both commands return zero. CLI errors propagate to the caller with a nonzero exit status, and stage timing distinguishes completion from failure.

## Development checks

Set up the locked development environment with `make install`. Use the same environment for manual and pre-commit Ruff checks. Both use the project's locked Ruff version, currently 0.15.20. The docstring rules prefer D211 and D212.

```fish
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync pytest -q -m 'not integration'
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync ruff format --check .
uv lock --check --offline
```

The complete suite includes the slower isolated DVC invalidation and cache restoration test, marked `integration`. The fast suite deselects it. Production-path reproduction and verification with real cached bundles remain part of M1.7.

The complete suite passed 147 tests in 55.28 seconds, with the DVC integration test taking 45.36 seconds. The fast suite passed 146 tests in 9.92 seconds and deselected one test. Repository-wide Ruff lint and format, applicable pre-commit hooks, Git whitespace and the offline uv lock consistency check passed.
