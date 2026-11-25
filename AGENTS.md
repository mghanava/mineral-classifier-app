# Agent instructions — Mineral Classifier App

## Purpose and scope

Develop the existing research-oriented mineral classification application into an accessible, scientifically trustworthy spatial ML workbench. The current development target is v0.2.0. Preserve the v0.1.0 synthetic workflow and its scientific capabilities. Follow `ROADMAP.md` for priorities and `CHANGELOG.md` for completed changes. Implement only the milestone or task explicitly requested; do not treat the roadmap as authorization to implement everything.

## Repository and version control

- Work in the existing repository; inspect `git status --short` and the active branch before editing.
- Expected development branch: `develop/v0.2.0`. Verify rather than assume it exists. If on an unexpected branch, report it and request guidance before editing tracked files.
- Preserve the v0.1.0 release through Git history and a tag; do not move or overwrite tags.
- Use the existing `pyproject.toml` and `uv.lock`. Do not create parallel copies of the app for each release.
- Use PEP 440 development versions (`0.2.0.dev0`, etc.) and update versions only when explicitly requested. Synchronize the lockfile when the project version or dependencies change.
- Never commit, push, merge, tag, switch branches, reset, stash, or rewrite history without explicit approval.
- Preserve user changes, including uncommitted work. Never overwrite unrelated files or revert changes you did not make.
- Do not delete datasets, trained models, DVC caches, or generated experiment artifacts without explicit approval.

## Architecture and dependencies

- Retain Streamlit, PyTorch, PyTorch Geometric, DVC, Plotly, uv and Docker where practical. Verify the actual stack in the repository before assuming details.
- Prefer incremental extraction of UI, configuration, data preparation and orchestration logic over a rewrite.
- Keep scientific computation independent of Streamlit. Keep UI pages focused on input, presentation and calls to application services.
- Reuse existing models, stages and utilities. Avoid parallel synthetic and uploaded-data training pipelines.
- Preserve CLI and DVC workflows unless an explicitly approved migration requires otherwise.
- Do not add frameworks, services or dependencies without justification and approval. Do not upgrade unrelated packages.
- Use the project's declared Python version, dependency groups and tooling. Prefer `uv run` for project commands; do not modify global Python environments.

## Coding conventions

- Follow existing repository conventions; use type hints for public interfaces, descriptive names, focused functions and concise docstrings.
- Use `pathlib` for filesystem operations where appropriate and explicit inputs/outputs rather than hidden global state.
- Use Ruff formatting/linting and existing type-checking tools if configured. Do not introduce broad formatting churn or suppress errors just to pass checks.
- Use fish-compatible commands when presenting commands for the project owner's Linux terminal.
- Avoid speculative abstractions and unrelated cleanup. Comment non-obvious scientific assumptions, not obvious code.

## User experience

- Make simple tasks easy and advanced tasks possible. Default to sensible settings and disclose complexity progressively.
- Separate demonstration, training on user data and prediction with an existing model.
- Do not require ordinary users to edit YAML, understand DVC or own a dedicated GPU.
- Keep full validated configuration and technical diagnostics available in advanced/expert mode.
- Provide actionable validation messages, clear status and understandable labels; do not rely on color alone.
- Prefer interactive spatial results and downloadable sample-level predictions over raw diagnostic output as the primary experience.

## Scientific integrity

- Do not silently alter model architecture, scientific assumptions, preprocessing, graph construction, data splits, calibration or drift methods. Explain and test any approved change to scientific behavior.
- Prevent leakage: fit preprocessing and feature selection only on the permitted training data; respect validation, calibration and test boundaries.
- Define the intended inductive/transductive graph protocol. Verify cross-partition edges and message passing against that protocol.
- For real geological data, support appropriate grouped or spatial validation; never present random-split performance as evidence of generalization to unseen regions without qualification.
- Preserve sample identifiers, feature names/order, class mappings, units, coordinate reference systems and relevant provenance. Do not silently impute, drop or reinterpret measurements.
- Require valid known labels for supervised training. Unlabeled prediction data must not automatically become verified training data; pseudo-labeling is out of scope unless explicitly approved.
- Keep model inference independent of retraining. Persist fitted preprocessing and graph/inference requirements with model artifacts.
- Keep probability calibration and drift detection available. Drift is not proof of performance degradation; classifier probabilities do not guarantee recognition of unknown classes.
- Compare models on identical appropriate splits; preserve a fixed holdout for longitudinal comparisons when implemented.
- Maintain deterministic seeds where feasible and record configuration, data identity, model version and evaluation protocol. Do not fabricate results or claim reproducibility without verification.

## Data and operational safety

- Treat uploaded data and file contents as untrusted. Validate schema, types, sizes and paths; do not execute content from uploaded files.
- Do not send private geological data, credentials, environment files or artifacts to external services without explicit approval.
- Do not commit sensitive or large generated data. Respect `.gitignore`, DVC configuration and existing artifact ownership.
- Do not load untrusted serialized model objects or use unsafe deserialization for user uploads.
- Avoid destructive DVC commands, broad cache cleanup, Docker pruning, system package changes, or GPU/system configuration changes without approval.

## Task workflow

1. Read these instructions and the relevant roadmap milestone; inspect Git state and relevant code.
2. Identify the smallest coherent implementation, affected interfaces and regression risks. State a brief plan for nontrivial tasks.
3. Implement only the requested scope. Ask before major architectural, dependency, scientific or destructive changes; otherwise proceed with reasonable local decisions and disclose them.
4. Add or update focused tests for changed behavior, including relevant failure cases.
5. Run targeted tests, existing lint/type checks and appropriate lightweight smoke tests. Expand verification when shared interfaces change.
6. Inspect the diff for unrelated changes, data leakage and compatibility regressions.
7. Update documentation and the Unreleased changelog only for actual changes, when applicable.
8. Report modified files, checks and outcomes, remaining risks, and next recommended action. Stop at the task boundary; do not commit or begin the next milestone automatically.

## Resource efficiency

- Use targeted searches and file reads; do not repeatedly scan the whole repository. Initial baseline inspection may examine the repository structure and key entry points.
- Exclude `.git`, `.venv`, `.dvc/cache`, caches, generated datasets and large binary/model artifacts from general exploration.
- Prefer small fixtures, mocks and targeted tests over full training or pipeline reproduction during routine development.
- Obtain approval before expensive model training, full DVC runs, large downloads or environment rebuilds.
- Avoid repeated failed commands, unnecessary tool calls, dependency installations and subagents. If blocked, diagnose with focused checks and report the blocker rather than retry indefinitely.
- Keep plans, summaries and code changes concise; do not repeat the entire roadmap in each task.
- Do not compromise scientific correctness or necessary verification solely to reduce token usage.

## Definition of done

- Requested acceptance criteria are met, or outstanding items are explicitly documented.
- Existing supported workflows are preserved or approved changes are documented.
- Relevant checks pass; distinguish passed, failed, skipped and not-run checks. Never claim a test ran if it did not.
- Documentation and changelog reflect actual behavior. No unrelated changes, secrets or generated artifacts are included.
- The project owner can inspect the diff and approve the work before a Git commit.
- Run pre-commit checks on modified Python files before reporting task completion.
  Local Ruff checks alone are insufficient because pre-commit may use a different Ruff version.
- Do not bypass Git hooks or use --no-verify.
- Report any files automatically modified by pre-commit.
