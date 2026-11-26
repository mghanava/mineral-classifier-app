"""Run the locked DVC workflow and retain bounded, observable session evidence."""

import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import Literal

from src.configuration import ValidationResult, validate_configuration
from src.configuration_files import configuration_operation, snapshot_configuration
from src.configuration_yaml import parse_configuration_yaml

CHECKING = "Checking applied configuration"
PREPARING = "Preparing DVC pipeline"
REPRODUCING = "Reproducing"
PHASES = (CHECKING, PREPARING, REPRODUCING)

_CYCLE_STAGES = {
    "train_cycle_": "Model training",
    "evaluate_cycle_": "Final-test evaluation and calibration",
    "generate_pred_data_cycle_": "Incoming batch generation",
    "predict_cycle_": "Batch prediction",
    "analyze_drift_cycle_": "Distribution drift analysis",
    "prepare_next_cycle_data_": "Append and prepare next base",
}
_AGGREGATE = "analyze_cycles_performance"

_EVENT_PATTERNS = (
    (re.compile(r"Running stage '([^']+)':"), "Started"),
    (re.compile(r"Stage '([^']+)' didn't change, skipping"), "Skipped unchanged"),
    (
        re.compile(r"Stage '([^']+)' is cached - skipping run, checking out outputs"),
        "Cache checkout observed",
    ),
)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_IN_PROGRESS = {"Started", "Cache checkout observed"}
LOG_LIMIT = 65_536
DISPLAY_LIMIT = 4_096
_READ_LIMIT = 4_096


def configuration_id(contents: bytes) -> str:
    """Identify the exact applied YAML bytes for display."""
    return sha256(contents).hexdigest()[:12]


def stage_details(stage: str) -> tuple[str, int | None] | None:
    """Translate generated DVC stage identifiers to scientific names and cycles."""
    if stage == "bootstrap":
        return "Bootstrap", 0
    if stage == _AGGREGATE:
        return "Cross-cycle performance aggregation", None
    for prefix, label in _CYCLE_STAGES.items():
        number = stage.removeprefix(prefix)
        if number != stage and re.fullmatch(r"[1-9][0-9]*", number):
            return label, int(number)
    return None


def stage_label(stage: str) -> str:
    """Format a stage label with its cycle when available."""
    details = stage_details(stage)
    if details is None:
        return stage
    label, cycle = details
    return f"{label} · cycle {cycle}" if cycle is not None else label


@dataclass
class StageProgress:
    """Track only starts, unchanged skips and evidence from the fail-fast walk."""

    cycles: int = 0
    stages: dict[str, str] = field(default_factory=dict)
    active: str | None = None
    last_stage: str | None = None
    last_event: str | None = None
    last_learning_cycle: int | None = None
    noop_summary: bool = False
    ambiguous_boundary: bool = False

    def in_scope(self, stage: str) -> bool:
        """Check the identifier against the generated synthetic topology."""
        details = stage_details(stage)
        if details is None:
            return False
        _, cycle = details
        return self.cycles >= 2 if cycle is None else cycle <= self.cycles

    def observe(self, line: str) -> bool:
        """Accept complete DVC event records and ignore scientific stage stdout."""
        record = _ANSI.sub("", line.rstrip("\r\n"))
        if record == "Data and pipelines are up to date.":
            self.noop_summary = True
            return False
        for pattern, event in _EVENT_PATTERNS:
            match = pattern.fullmatch(record)
            if match is None:
                continue
            stage = match[1]
            if not self.in_scope(stage) or stage in self.stages:
                self.ambiguous_boundary = True
                self.active = None
                self.last_event = (
                    "DVC reported an unrecognized or repeated stage boundary."
                )
                return True
            # The generated command uses DVC's fail-fast sequential walk. A subsequent
            # stage boundary or a successful command exit confirms the previous step.
            self.confirm_active()
            self.stages[stage] = event
            self.last_stage = stage
            self.last_event = f"{event} · {stage_label(stage)}"
            self.active = stage if event in _IN_PROGRESS else None
            _, cycle = stage_details(stage)
            if cycle:
                self.last_learning_cycle = cycle
            return True
        return False

    def confirm_active(self) -> None:
        """Record completion when the DVC walk has confirmed the active step."""
        if self.active is not None:
            prior = self.stages[self.active]
            self.stages[self.active] = (
                "Restored from cache"
                if prior == "Cache checkout observed"
                else "Completed"
            )
            self.active = None

    def all_skipped(self, target: str | None) -> bool:
        """Require explicit skip evidence as well as DVC's successful no-op summary."""
        if not self.noop_summary or self.ambiguous_boundary or not self.stages:
            return False
        if any(value != "Skipped unchanged" for value in self.stages.values()):
            return False
        if target:
            # The DVC summary covers the selected traversal, including prerequisites.
            return target in self.stages
        expected = 1 + len(_CYCLE_STAGES) * self.cycles + int(self.cycles >= 2)
        return len(self.stages) == expected


@dataclass
class CommandRecord:
    """Keep the invoked command and its process exit code when available."""

    argv: tuple[str, ...]
    exit_status: int | None = None

    @property
    def display(self) -> str:
        """Quote command arguments for display only."""
        return shlex.join(self.argv)


@dataclass
class RunAttempt:
    """Describe the most recent execution action within one dashboard session."""

    target: str | None = None
    status: Literal["Running", "Completed", "Failed", "Up to date"] = "Running"
    phase: str = CHECKING
    config_id: str | None = None
    progress: StageProgress = field(default_factory=StageProgress)
    commands: list[CommandRecord] = field(default_factory=list)
    validation: ValidationResult | None = None
    error: str | None = None
    log: str = ""
    discarded: int = 0

    @property
    def scope(self) -> str:
        """Name the action without implying a full run for a stage target."""
        return (
            f"Targeted · {stage_label(self.target)}"
            if self.target
            else "Full experiment"
        )

    def append_log(self, text: str) -> None:
        """Retain the latest bounded text and count removed characters."""
        combined = self.log + text
        excess = max(0, len(combined) - LOG_LIMIT)
        self.discarded += excess
        self.log = combined[excess:]

    def fail(self, message: str) -> None:
        """Save an actionable failure and prevent a success state."""
        self.status = "Failed"
        self.error = message

    def log_download(self) -> str:
        """Return the retained output with its session and command scope."""
        command_lines = [
            f"{record.display} (exit {'unavailable' if record.exit_status is None else record.exit_status})"
            for record in self.commands
        ]
        return "\n".join(
            [
                f"Session action: {self.scope}",
                f"Status: {self.status} | Phase: {self.phase}",
                *command_lines,
                f"Last observed event: {self.progress.last_event or 'unavailable'}",
                f"Earlier output characters omitted: {self.discarded}",
                self.error or "",
                "",
                self.log,
            ]
        )


def run_command(
    command: list[str], attempt: RunAttempt, notify: Callable[[], None]
) -> int | None:
    """Run one command with merged, bounded streaming output."""
    record = CommandRecord(tuple(command))
    attempt.commands.append(record)
    attempt.append_log(f"\n$ {record.display}\n")
    notify()
    # DVC stage commands resolve python through PATH.
    path = os.pathsep.join(
        (str(Path(sys.executable).parent), os.environ.get("PATH", os.defpath))
    )
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            env={**os.environ, "PATH": path},
        )
    except OSError as error:
        attempt.fail(
            f"Could not start {command[2]} during {attempt.phase.lower()}. {error}"
        )
        attempt.append_log(f"{error}\n")
        return None

    with process:
        if process.stdout is not None:
            last_update = monotonic()
            incomplete_line = False
            # Read bounded chunks even when a stage emits a very long line.
            while chunk := process.stdout.readline(_READ_LIMIT):
                attempt.append_log(chunk)
                complete = chunk.endswith("\n")
                event = False
                if attempt.phase == REPRODUCING and complete and not incomplete_line:
                    event = attempt.progress.observe(chunk)
                incomplete_line = not complete
                now = monotonic()
                if event or now - last_update >= 0.5:
                    notify()
                    last_update = now
        record.exit_status = process.wait()
    attempt.append_log(f"\n[Exit status {record.exit_status}]\n")
    return record.exit_status


def _reproduce_locked(
    params_file: Path, attempt: RunAttempt, notify: Callable[[], None]
) -> None:
    try:
        active = snapshot_configuration(params_file)
        if active.contents is None:
            raise ValueError(
                "Apply a configuration in Configure to create params.yaml."
            )
        attempt.config_id = configuration_id(active.contents)
        settings = parse_configuration_yaml(
            active.contents, source="active params.yaml"
        )
        attempt.validation = validate_configuration(settings)
    except (OSError, ValueError) as error:
        attempt.fail(f"Applied configuration could not be checked. {error}")
        return
    if not attempt.validation.is_valid:
        attempt.fail(
            "Applied configuration is invalid. Review and apply corrections in Configure."
        )
        return

    attempt.progress.cycles = settings["cycles"]
    if attempt.target and not attempt.progress.in_scope(attempt.target):
        attempt.fail(
            "The selected stage is outside the applied cycle range. Review the selection and try again."
        )
        return

    attempt.phase = PREPARING
    status = run_command([sys.executable, "-m", "scripts.setup_dvc"], attempt, notify)
    if status is None:
        return
    if status:
        attempt.fail(
            f"Pipeline preparation (scripts.setup_dvc) exited with status {status}. See Technical logs."
        )
        return
    try:
        if snapshot_configuration(params_file) != active:
            raise ValueError(
                "The active configuration changed during DVC generation. Start the run again."
            )
    except (OSError, ValueError) as error:
        attempt.fail(str(error))
        return

    attempt.phase = REPRODUCING
    targets = [attempt.target] if attempt.target else []
    status = run_command(
        [sys.executable, "-m", "dvc", "repro", *targets], attempt, notify
    )
    if status is None:
        return
    if status:
        attempt.fail(
            f"DVC reproduction exited with status {status}. See Technical logs."
        )
        return
    attempt.progress.confirm_active()
    attempt.status = (
        "Up to date" if attempt.progress.all_skipped(attempt.target) else "Completed"
    )


def reproduce(
    params_file: Path, attempt: RunAttempt, notify: Callable[[], None]
) -> None:
    """Own the workspace lock across validation, pipeline generation and reproduction."""
    notify()
    try:
        with ExitStack() as lock:
            try:
                lock.enter_context(configuration_operation(params_file))
            except ValueError as error:
                attempt.fail(f"Workspace busy. {error}")
                return
            except OSError as error:
                attempt.fail(f"Could not reserve the workspace configuration. {error}")
                return
            _reproduce_locked(params_file, attempt, notify)
    except OSError as error:
        attempt.fail(
            f"Could not read subprocess output during {attempt.phase.lower()}. {error}"
        )
    finally:
        if attempt.status == "Running":
            attempt.fail(
                "The execution was interrupted before a command outcome was observed."
            )
