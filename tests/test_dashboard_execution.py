"""Verify session execution evidence with retained DVC line forms and fixtures."""

import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.execution import (
    CHECKING,
    DISPLAY_LIMIT,
    LOG_LIMIT,
    PREPARING,
    REPRODUCING,
    RunAttempt,
    StageProgress,
    reproduce,
    stage_details,
)
from src.configuration_files import configuration_operation

ROOT = Path(__file__).resolve().parents[1]

# The stage names and literals are from the retained M2.5 Quick demo repro/noop logs.
# The cache-checkout record is from DVC 3.67.1's StageCache.restore logger.
TARGET_NOOP = (
    "Stage 'bootstrap' didn't change, skipping\n"
    "Stage 'train_cycle_1' didn't change, skipping\n"
    "Data and pipelines are up to date.\n"
)


class Process:
    """Mimic subprocess streaming with a deterministic exit status."""

    def __init__(self, output: str, status: int = 0) -> None:
        """Hold merged output as a bounded-read stream."""
        self.stdout = StringIO(output)
        self.status = status

    def __enter__(self):
        """Expose the open process stream."""
        return self

    def __exit__(self, *exc):
        """Close the fixture stream."""
        self.stdout.close()

    def wait(self) -> int:
        """Return the exit status after the fixture is drained."""
        return self.status


@pytest.fixture
def params(tmp_path):
    """Create a writable copy of params.yaml in an isolated workspace."""
    path = tmp_path / "params.yaml"
    path.write_bytes((ROOT / "params.yaml").read_bytes())
    return path


def test_dvc_progress_observes_starts_skips_and_cache_boundaries():
    """Confirm a prior stage on a recognized later boundary, including cache restores."""
    progress = StageProgress(cycles=24)
    assert progress.observe("Running stage 'bootstrap':\n")
    assert not progress.observe("> python src/stages/generate_base_data.py --cycle 0\n")
    assert not progress.observe("Updating lock file 'dvc.lock'\n")
    assert progress.stages == {"bootstrap": "Started"}
    assert progress.observe("Stage 'train_cycle_1' didn't change, skipping\n")
    assert progress.stages["bootstrap"] == "Completed"
    assert progress.active is None
    assert progress.observe(
        "Stage 'evaluate_cycle_1' is cached - skipping run, checking out outputs\n"
    )
    assert progress.stages["evaluate_cycle_1"] == "Cache checkout observed"
    assert progress.observe("\x1b[32mRunning stage 'train_cycle_24':\x1b[0m\n")
    assert progress.stages["evaluate_cycle_1"] == "Restored from cache"
    assert progress.active == "train_cycle_24"
    assert progress.last_learning_cycle == 24
    assert stage_details("prepare_next_cycle_data_24") == (
        "Append and prepare next base",
        24,
    )
    assert stage_details("analyze_cycles_performance") == (
        "Cross-cycle performance aggregation",
        None,
    )


@pytest.mark.parametrize(
    ("cycles", "target", "output", "exit_code", "expected"),
    [
        (
            0,
            None,
            "Stage 'bootstrap' didn't change, skipping\nData and pipelines are up to date.\n",
            0,
            "Up to date",
        ),
        (1, "train_cycle_1", TARGET_NOOP, 0, "Up to date"),
        (1, None, TARGET_NOOP, 0, "Completed"),
        (1, "evaluate_cycle_1", TARGET_NOOP, 0, "Completed"),
        (1, "train_cycle_1", TARGET_NOOP, 2, "Failed"),
        (1, "train_cycle_1", "Data and pipelines are up to date.\n", 0, "Completed"),
        (
            1,
            "train_cycle_1",
            "Stage 'train_cycle_1' didn't change, skipping\n",
            0,
            "Completed",
        ),
        (
            1,
            "train_cycle_1",
            TARGET_NOOP + "Running stage 'unknown_stage':\n",
            0,
            "Completed",
        ),
        (
            1,
            "train_cycle_1",
            TARGET_NOOP + "Stage 'train_cycle_1' didn't change, skipping\n",
            0,
            "Completed",
        ),
        (
            1,
            "train_cycle_1",
            "Stage 'train_cycle_1' is cached - skipping run, checking out outputs\n",
            0,
            "Completed",
        ),
    ],
)
def test_outcome_requires_explicit_scope_and_exit_evidence(
    params, monkeypatch, cycles, target, output, exit_code, expected
):
    """Check no-op versus completion for full and targeted reproduction."""
    params.write_text(params.read_text().replace("cycles: 3", f"cycles: {cycles}"))

    def process(command, **kwargs):
        return Process(output, exit_code) if command[2] == "dvc" else Process("")

    monkeypatch.setattr(subprocess, "Popen", process)
    attempt = RunAttempt(target)
    reproduce(params, attempt, lambda: None)
    assert attempt.status == expected
    assert attempt.commands[-1].argv == (
        sys.executable,
        "-m",
        "dvc",
        "repro",
        *((target,) if target else ()),
    )
    assert attempt.commands[-1].exit_status == exit_code
    assert ("Targeted" in attempt.scope) == bool(target)
    with configuration_operation(params):
        pass


def test_full_multi_cycle_noop_covers_every_generated_stage(params, monkeypatch):
    """Use all 14 unchanged stage records from the retained two-cycle Quick demo."""
    params.write_text(params.read_text().replace("cycles: 3", "cycles: 2"))
    stems = (
        "train_cycle_",
        "evaluate_cycle_",
        "generate_pred_data_cycle_",
        "predict_cycle_",
        "analyze_drift_cycle_",
        "prepare_next_cycle_data_",
    )
    stages = [
        "bootstrap",
        *(f"{stem}{cycle}" for cycle in (1, 2) for stem in stems),
        "analyze_cycles_performance",
    ]
    lines = [f"Stage '{stage}' didn't change, skipping\n" for stage in stages]
    output = "".join(lines) + "Data and pipelines are up to date.\n"

    def process(command, **kwargs):
        return Process(output) if command[2] == "dvc" else Process("")

    monkeypatch.setattr(subprocess, "Popen", process)
    attempt = RunAttempt()
    reproduce(params, attempt, lambda: None)
    assert attempt.status == "Up to date"
    assert len(attempt.progress.stages) == 14


def test_bounded_stream_and_no_inferred_progress_from_partial_records(
    params, monkeypatch
):
    """Expose all named phases while preserving only bounded text from a noisy failure."""
    stage = "Running stage 'train_cycle_1':"
    phases = []

    def process(command, **kwargs):
        if command[2] == "scripts.setup_dvc":
            return Process(
                "Running stage 'bootstrap':\n"
            )  # setup text provides no repro evidence
        return Process(
            "x" * (LOG_LIMIT + 8192) + stage + "\n" + "ERROR: sample fixture\n" + stage,
            4,
        )

    attempt = RunAttempt()
    monkeypatch.setattr(subprocess, "Popen", process)

    def notify():
        phases.append((attempt.status, attempt.phase))

    reproduce(params, attempt, notify)
    assert phases[0] == ("Running", CHECKING)
    assert ("Running", PREPARING) in phases
    assert ("Running", REPRODUCING) in phases
    assert attempt.progress.stages == {}
    assert attempt.status == "Failed"
    assert len(attempt.log) == LOG_LIMIT and attempt.discarded > 0
    assert "ERROR: sample fixture" in attempt.log[-DISPLAY_LIMIT:]
    assert "Earlier output characters omitted:" in attempt.log_download()
    assert "Exit status 4" in attempt.log_download()


def test_running_stage_is_not_marked_complete_on_nonzero_exit(params, monkeypatch):
    """Keep observed start evidence when a process exits before a confirming boundary."""
    seen = []

    def process(command, **kwargs):
        return (
            Process("Running stage 'train_cycle_1':\nunknown line\n", 2)
            if command[2] == "dvc"
            else Process("")
        )

    monkeypatch.setattr(subprocess, "Popen", process)
    attempt = RunAttempt("train_cycle_1")

    def notify():
        seen.append((attempt.status, attempt.progress.active))

    reproduce(params, attempt, notify)
    assert ("Running", "train_cycle_1") in seen
    assert attempt.status == "Failed"
    assert attempt.progress.stages["train_cycle_1"] == "Started"


def test_lock_contention_and_external_change_before_repro(params, monkeypatch):
    """Reject competing execution and stop after changes by a non-cooperating editor."""
    with configuration_operation(params):
        blocked = RunAttempt()
        reproduce(params, blocked, lambda: None)
        assert blocked.status == "Failed"
        assert "Workspace busy" in blocked.error
        assert not blocked.commands

    def mutate_during_setup(command, **kwargs):
        assert command[2] == "scripts.setup_dvc"
        params.write_text(params.read_text() + "\n# external editor\n")
        return Process("setup completed\n")

    monkeypatch.setattr(subprocess, "Popen", mutate_during_setup)
    changed = RunAttempt()
    reproduce(params, changed, lambda: None)
    assert changed.status == "Failed" and changed.phase == PREPARING
    assert "changed during DVC generation" in changed.error
    with configuration_operation(params):
        pass


def test_run_many_cycles_shows_bounded_cycle_view_and_refreshed_availability(
    params, monkeypatch
):
    """Render one cycle of evidence and refresh file availability after reproduction."""
    monkeypatch.chdir(params.parent)
    params.write_text(params.read_text().replace("cycles: 3", "cycles: 24"))

    def process(command, **kwargs):
        if command[2] == "scripts.setup_dvc":
            return Process("")
        artifact = params.parent / "results/trained/cycle_24/graph_binding.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("{}")
        output = "".join(
            f"Stage 'train_cycle_{n}' didn't change, skipping\n" for n in range(1, 24)
        )
        return Process(output + "Running stage 'train_cycle_24':\n")

    monkeypatch.setattr(subprocess, "Popen", process)
    app = (
        AppTest.from_file(str(ROOT / "app.py"))
        .switch_page("dashboard/views/run.py")
        .run()
    )
    assert app.info[0].value == "Ready"
    assert any("Applied configuration ID" in caption.value for caption in app.caption)
    assert any("Calibration** Dirichlet" in text.value for text in app.markdown)
    assert any("No display files found" in caption.value for caption in app.caption)
    next(
        button for button in app.button if button.label == "Run full experiment"
    ).click().run()
    assert not app.exception
    assert app.success[0].value == "Completed · Full experiment"
    assert any(text.value == "**Learning cycle 24**" for text in app.markdown)
    assert all(text.value != "**Learning cycle 1**" for text in app.markdown)
    assert any(
        "Other learning cycles observed · 23" in caption.value
        for caption in app.caption
    )
    assert any(
        "source revision is unverified" in caption.value for caption in app.caption
    )
