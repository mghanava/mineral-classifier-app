"""Stream the existing locked DVC execution workflow."""

import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

from dashboard.components import show_validation
from src.configuration import validate_configuration
from src.configuration_files import configuration_operation, snapshot_configuration
from src.configuration_yaml import parse_configuration_yaml


def run_command(command: list[str], *, announce_success: bool = True) -> int:
    """Stream a command's output and report its exit status in the dashboard."""
    display = " ".join(command)
    st.info(f"Running `{display}`")
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
    except OSError as exc:
        st.error(f"Could not start `{display}`. {exc}")
        return 1

    with process:
        output = ""
        output_placeholder = st.empty()
        if process.stdout is not None:
            for line in process.stdout:
                output += line
                output_placeholder.code(output)
        status = process.wait()
    if status:
        st.error(f"`{display}` exited with status {status}. See the output above.")
    elif announce_success:
        st.success("Command completed successfully.")
    return status


def reproduce(params_file: Path, *targets: str, announce_success: bool = True) -> int:
    """Regenerate DVC definitions from the active file before reproduction."""
    try:
        with configuration_operation(params_file):
            return _reproduce_active(params_file, targets, announce_success)
    except (OSError, ValueError) as error:
        st.error(str(error))
        return 1


def _reproduce_active(
    params_file: Path, targets: tuple[str, ...], announce_success: bool
) -> int:
    active = snapshot_configuration(params_file)
    if active.contents is None:
        raise ValueError("Apply a configuration in Configure to create params.yaml.")
    validation = validate_configuration(
        parse_configuration_yaml(active.contents, source="active params.yaml")
    )
    if not validation.is_valid:
        st.error(
            "Resolve and apply a valid configuration in Configure before running DVC."
        )
        show_validation(validation)
        return 1
    status = run_command(
        [sys.executable, "-m", "scripts.setup_dvc"], announce_success=False
    )
    if status:
        return status
    if snapshot_configuration(params_file) != active:
        raise ValueError(
            "The active configuration changed during DVC generation. Start the run again."
        )
    return run_command(
        [sys.executable, "-m", "dvc", "repro", *targets],
        announce_success=announce_success,
    )
