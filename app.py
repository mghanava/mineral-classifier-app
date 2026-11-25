"""DVC Pipeline Dashboard interface for ML workflow management.

This module provides a Streamlit-based web interface to manage and monitor
a DVC pipeline for machine learning workflows, including data generation,
model training, evaluation, prediction, and drift analysis.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from src.configuration import ValidationResult, validate_configuration
from src.configuration_files import (
    configuration_operation,
    persist_configuration,
    preview_configuration,
    snapshot_configuration,
)
from src.configuration_yaml import (
    dump_configuration_yaml,
    load_configuration_yaml,
    parse_configuration_yaml,
)
from src.presets import (
    apply_preset,
    list_presets,
    load_preset,
    resolve_imported_configuration,
)


def get_max_cycles():
    """Read the number of cycles from params.yaml."""
    try:
        cycles = load_configuration_yaml(Path("params.yaml")).get("cycles")
        return cycles if type(cycles) is int and cycles >= 0 else 10
    except ValueError:
        return 10


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


def reproduce(*targets: str, announce_success: bool = True) -> int:
    """Regenerate DVC definitions from the active file before reproduction."""
    params_file = Path("params.yaml")
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
        raise ValueError("Apply a configuration in Parameters to create params.yaml.")
    validation = validate_configuration(
        parse_configuration_yaml(active.contents, source="active params.yaml")
    )
    if not validation.is_valid:
        st.error(
            "Resolve and apply a valid configuration in Parameters before running DVC."
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


def show_html_files(path: Path):
    """Display Plotly graphs from a given directory, preferring JSON over HTML."""
    json_files = list(path.glob("*.json"))
    html_files = list(path.glob("*.html"))

    if not json_files and not html_files:
        st.info("No graph files found in this directory.")
        return

    # Prefer JSON files
    for json_file in json_files:
        with st.expander(f"View {json_file.name}"):
            with open(json_file) as f:
                fig = pio.from_json(f.read())
            st.plotly_chart(fig, use_container_width=True)

    # Fall back to HTML if no JSON
    for html_file in html_files:
        json_version = html_file.with_suffix(".json")
        if json_version.exists():
            continue  # already handled
        with st.expander(f"View {html_file.name}"):
            with open(html_file, encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=800, scrolling=True)


def show_validation(validation: ValidationResult) -> None:
    """Display all static findings with their paths."""
    display = {"error": st.error, "warning": st.warning, "info": st.info}
    for issue in validation.issues:
        display[issue.severity](f"{issue.path} — {issue.message}")


def params_tab():
    """Edit, preview and explicitly apply a scientific configuration draft."""
    st.header("Parameters")
    st.write(
        "Edit a draft or load a preset. Apply the preview to update the active `params.yaml` used by the pipeline."
    )
    params_file = Path("params.yaml")
    state = st.session_state
    try:
        active = snapshot_configuration(params_file)
    except (OSError, ValueError) as error:
        st.error(f"Cannot read the active configuration. {error}")
        return

    if "yaml_text_editor" not in state:
        contents = active.contents if active.contents is not None else b""
        state.yaml_text_editor = contents.decode("utf-8", errors="replace")

    try:
        presets = [load_preset(preset_id) for preset_id in list_presets()]
        labels = {preset.id: preset.display_name for preset in presets}
        selected = st.selectbox(
            "Experiment preset", labels, format_func=labels.__getitem__
        )
        if st.button("Load preset into draft"):
            resolved = apply_preset(
                next(preset for preset in presets if preset.id == selected)
            )
            state.yaml_text_editor = dump_configuration_yaml(resolved.configuration)
            state.pop("configuration_preview", None)
    except ValueError as error:
        st.error(str(error))

    editor, results = st.columns(2)
    with editor:
        draft = st.text_area("Draft YAML", height=500, key="yaml_text_editor")
        if st.button("Validate & Preview"):
            state.pop("configuration_preview", None)
            try:
                resolved = resolve_imported_configuration(
                    parse_configuration_yaml(draft, source="draft YAML")
                )
                state.configuration_preview = (
                    draft,
                    preview_configuration(resolved, active),
                )
            except ValueError as error:
                st.error(str(error))

    with results:
        st.subheader("Preview")
        saved = state.get("configuration_preview")
        if saved is not None and saved[0] == draft:
            preview = saved[1]
            stale = preview.active != active
            if stale:
                st.warning(
                    "The active configuration changed since preview. Preview again before applying."
                )
            show_validation(preview.validation)
            if preview.comparison_warning:
                st.warning(preview.comparison_warning)
            if preview.regeneration_paths:
                paths = ", ".join(preview.regeneration_paths)
                st.warning(
                    f"DVC pipeline regeneration is required for changes to {paths}. Dashboard run controls regenerate before reproduction."
                )
            with st.expander("Resolved scientific YAML", expanded=True):
                st.code(preview.yaml, language="yaml")
            with st.expander("Differences from active configuration", expanded=True):
                if preview.diff:
                    st.code(preview.diff, language="diff")
                else:
                    st.info("No scientific configuration changes.")
            if (
                st.button(
                    "Apply configuration",
                    disabled=stale or not preview.validation.is_valid,
                )
                and not stale
                and preview.validation.is_valid
            ):
                try:
                    persist_configuration(preview, active, overwrite=True)
                except (OSError, ValueError) as error:
                    st.error(f"Could not apply the configuration. {error}")
                else:
                    state.pop("configuration_preview", None)
                    st.success("Configuration applied to params.yaml.")
        else:
            st.info(
                "Validate the current draft to preview the resolved settings and changes."
            )

    st.subheader("Run Pipeline")
    st.caption("Pipeline controls use the applied params.yaml.")
    if st.button("Generate & Run Full Pipeline"):
        with st.spinner("Running pipeline..."):
            if reproduce(announce_success=False) == 0:
                st.success("Pipeline completed successfully!")


def data_tab():
    """Display and manage data generation for different cycles in the DVC pipeline dashboard.

    This tab allows users to view generated base data for each cycle and run the bootstrap stage
    to generate initial data (cycle 0).
    """
    st.header("Data Generation")
    st.write("View the generated base and prediction data for each cycle.")

    max_cycles = get_max_cycles()
    cycle_num = st.number_input(
        "Cycle to view", min_value=0, value=0, max_value=max_cycles, key="data_cycle"
    )

    if st.button("Run Bootstrap Stage (Generates Cycle 0 Data)"):
        reproduce("bootstrap")
    st.info(
        "Data for subsequent cycles is generated by running the full pipeline from the 'Parameters' tab."
    )

    st.subheader("Generated Data")
    data_path = Path(f"results/data/base/cycle_{cycle_num}")
    if data_path.exists():
        st.markdown("### Base Data (Training, Evaluation, Test, Calibration)")
        plot_files = list(data_path.glob("*.png"))
        for plot in plot_files:
            st.image(str(plot))
        show_html_files(data_path)
    else:
        st.warning(f"Base data for cycle {cycle_num} not found.")
    if cycle_num > 0:
        pred_data_path = Path(f"results/data/prediction/cycle_{cycle_num}")
        if pred_data_path.exists():
            st.markdown("### Prediction Data")
            show_html_files(pred_data_path)
        else:
            st.warning(f"Prediction data for cycle {cycle_num} not found.")


def train_tab():
    """Display and manage model training for different cycles in the DVC pipeline dashboard.

    This tab allows users to run the training stage for each cycle and view the training results,
    including any generated plots.
    """
    st.header("Training")
    st.write("This stage trains the model for each cycle.")

    max_cycles = get_max_cycles()
    cycle_num = st.number_input(
        "Cycle to run",
        min_value=1,
        value=1,
        max_value=max(1, max_cycles),
        key="train_cycle",
    )
    if st.button("Run Training Stage", disabled=max_cycles == 0):
        reproduce(f"train_cycle_{cycle_num}")

    st.subheader("Training Results")
    train_path = Path(f"results/trained/cycle_{cycle_num}")
    if train_path.exists():
        plot_files = list(train_path.glob("*.png"))
        for plot in plot_files:
            st.image(str(plot))
    else:
        st.warning(f"Training results for cycle {cycle_num} not found.")


def evaluation_tab():
    """Display and manage model evaluation for different cycles in the DVC pipeline dashboard.

    This tab allows users to run the evaluation stage for each cycle and view the evaluation results,
    including metrics and generated plots.
    """
    st.header("Evaluation")
    st.write("This stage evaluates the trained model for each cycle.")

    max_cycles = get_max_cycles()
    cycle_num = st.number_input(
        "Cycle to evaluate",
        min_value=1,
        value=1,
        max_value=max(1, max_cycles),
        key="eval_cycle",
    )
    if st.button("Run Evaluation Stage", disabled=max_cycles == 0):
        reproduce(f"evaluate_cycle_{cycle_num}")

    st.subheader("Evaluation Results")
    eval_path = Path(f"results/evaluation/cycle_{cycle_num}")
    if eval_path.exists():
        for img_file in eval_path.glob("*.png"):
            st.image(str(img_file))

        metrics_file = eval_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                st.json(json.load(f))
    else:
        st.warning(f"Evaluation results for cycle {cycle_num} not found.")


def prediction_tab():
    """Display and manage predictions for different cycles in the DVC pipeline dashboard.

    This tab allows users to run the prediction stage for each cycle and view the results,
    including visualizations and predicted values.
    """
    st.header("Prediction")
    st.write("This stage runs predictions on new data for each cycle.")

    max_cycles = get_max_cycles()
    cycle_num = st.number_input(
        "Cycle for prediction",
        min_value=1,
        value=1,
        max_value=max(1, max_cycles),
        key="pred_cycle",
    )
    if st.button("Run Prediction Stage", disabled=max_cycles == 0):
        reproduce(f"predict_cycle_{cycle_num}")

    st.subheader("Prediction Results")
    pred_path = Path(f"results/prediction/cycle_{cycle_num}")
    if pred_path.exists():
        img_files = sorted(
            pred_path.glob("*.png"),
            key=lambda x: "0" if x.name == "confusion_matrix.png" else "1",
        )
        for img_file in img_files:
            st.image(str(img_file))
        csv_file = pred_path / "predictions.csv"
        if csv_file.exists():
            st.dataframe(pd.read_csv(csv_file))
    else:
        st.warning(f"Prediction results for cycle {cycle_num} not found.")


def drift_analysis_tab():
    """Display and manage drift analysis for different cycles in the DVC pipeline dashboard.

    This tab allows users to run the drift analysis stage for each cycle and view the results,
    including visualizations and statistical measures of data drift between training and prediction data.
    """
    st.header("Drift Analysis")
    st.write(
        "This stage analyzes data drift between the base data used to train model and the unseen prediction data for each cycle."
    )

    max_cycles = get_max_cycles()
    cycle_num = st.number_input(
        "Cycle for drift analysis",
        min_value=1,
        value=1,
        max_value=max(1, max_cycles),
        key="drift_cycle",
    )
    if st.button("Run Drift Analysis Stage", disabled=max_cycles == 0):
        reproduce(f"analyze_drift_cycle_{cycle_num}")

    st.subheader("Drift Analysis Results")
    drift_path = Path(f"results/drift_analysis/cycle_{cycle_num}")
    if drift_path.exists():
        for img_file in drift_path.glob("*.png"):
            st.image(str(img_file))

        results_file = drift_path / "drift_results.txt"
        if results_file.exists():
            with open(results_file) as f:
                st.text(f.read())
    else:
        st.warning(f"Drift analysis results for cycle {cycle_num} not found.")


def performance_tab():
    """Display and manage performance analysis across all cycles in the DVC pipeline dashboard.

    This tab allows users to run the performance analysis stage and view the results,
    including visualizations of model performance across different cycles.
    """
    st.header("Cycles Performance Analysis")
    st.write("This stage analyzes the performance across all cycles.")

    if st.button("Run Performance Analysis Stage", disabled=get_max_cycles() < 2):
        reproduce("analyze_cycles_performance")

    st.subheader("Performance Plot")
    perf_img = Path("results/cycles_performance_analysis/cycles_performance.png")
    if perf_img.exists():
        st.image(str(perf_img))
    else:
        st.warning("Performance analysis plot not found.")


def app():
    """Initialize and run the DVC Pipeline Dashboard Streamlit application.

    This function sets up the main application layout and manages different tabs
    for parameters, data generation, training, evaluation, prediction, drift analysis,
    and performance monitoring.
    """
    st.set_page_config(layout="wide")
    st.title("DVC Pipeline Dashboard")

    tab_names = [
        "Parameters",
        "Data",
        "Train",
        "Evaluation",
        "Prediction",
        "Drift Analysis",
        "Performance",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        params_tab()
    with tabs[1]:
        data_tab()
    with tabs[2]:
        train_tab()
    with tabs[3]:
        evaluation_tab()
    with tabs[4]:
        prediction_tab()
    with tabs[5]:
        drift_analysis_tab()
    with tabs[6]:
        performance_tab()


if __name__ == "__main__":
    app()
