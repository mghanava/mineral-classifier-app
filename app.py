"""DVC Pipeline Dashboard interface for ML workflow management.

This module provides a Streamlit-based web interface to manage and monitor
a DVC pipeline for machine learning workflows, including data generation,
model training, evaluation, prediction, and drift analysis.
"""

import difflib
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
import yaml as pyyaml
from ruamel.yaml import YAML


def get_max_cycles():
    """Read the number of cycles from params.yaml."""
    yaml = YAML()
    params_file = Path("params.yaml")
    try:
        with open(params_file) as f:
            params = yaml.load(f)
            # DVC cycles start from 1, but data might have a cycle 0
            return params.get("cycles", 10)
    except (FileNotFoundError, KeyError):
        return 10  # Default to 10 on error


def run_command(command):
    """Run a shell command and stream the output to the Streamlit app."""
    st.info(f"Running: `{' '.join(command)}`")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    output = ""
    output_placeholder = st.empty()
    while True:
        if process.stdout is None:
            break
        line = process.stdout.readline()
        if not line:
            break
        output += line
        output_placeholder.code(output)
    process.wait()
    if process.returncode != 0:
        st.error("Command failed.")
    else:
        st.success("Command completed successfully.")
    return process.returncode


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


def params_tab():
    """Display and edit parameters from params.yaml with unified edit/preview interface."""
    st.header("Parameters")
    st.write(
        "View and edit `params.yaml`. Changes are auto-saved every 3 seconds. You can regenerate the DVC pipeline and run it from here."
    )

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    params_file = Path("params.yaml")

    # Initialize session state
    if "original_params" not in st.session_state:
        st.session_state.original_params = ""
    if "last_save_time" not in st.session_state:
        st.session_state.last_save_time = 0
    if "auto_save_enabled" not in st.session_state:
        st.session_state.auto_save_enabled = True

    try:
        with open(params_file) as f:
            params_content = f.read()
            if not st.session_state.original_params:
                st.session_state.original_params = params_content

        # Create columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Edit YAML")

            edited_params = st.text_area(
                "params.yaml",
                value=params_content,
                height=500,
                label_visibility="collapsed",
                key="yaml_text_editor",
            )

            # Auto-save functionality
            current_time = time.time()
            if (
                st.session_state.auto_save_enabled
                and edited_params != params_content
                and current_time - st.session_state.last_save_time > 3
            ):
                try:
                    # Validate YAML before saving
                    yaml.load(edited_params)
                    with open(params_file, "w") as f:
                        f.write(edited_params)
                    st.session_state.last_save_time = current_time
                    st.success("Auto-saved!", icon="💾")
                except Exception as e:
                    st.warning(f"Auto-save failed - invalid YAML: {str(e)[:50]}...")

            # Auto-save toggle (replaces "Save Now" button)
            st.session_state.auto_save_enabled = st.checkbox(
                "Enable auto-save (every 3 seconds)",
                value=st.session_state.auto_save_enabled,
            )

        with col2:
            st.subheader("Preview")

            try:
                parsed_data = pyyaml.safe_load(edited_params)
                st.json(parsed_data)

                # Show Diff section (replaces "Formatted YAML")
                st.subheader("Show Diff")
                if edited_params != st.session_state.original_params:
                    # Create and display diff
                    original_lines = st.session_state.original_params.splitlines()
                    edited_lines = edited_params.splitlines()

                    diff = list(
                        difflib.unified_diff(
                            original_lines,
                            edited_lines,
                            fromfile="original",
                            tofile="current",
                            lineterm="",
                            n=3,
                        )
                    )

                    if diff:
                        diff_text = "\n".join(diff)
                        st.code(diff_text, language="diff")
                    else:
                        st.info("No changes detected")
                else:
                    st.info("No changes to show")

            except Exception as e:
                st.error(f"YAML Parse Error: {e}")
                # Still show the diff section even with parse errors
                st.subheader("Show Diff")
                if edited_params != st.session_state.original_params:
                    original_lines = st.session_state.original_params.splitlines()
                    edited_lines = edited_params.splitlines()

                    diff = list(
                        difflib.unified_diff(
                            original_lines,
                            edited_lines,
                            fromfile="original",
                            tofile="current",
                            lineterm="",
                            n=3,
                        )
                    )

                    if diff:
                        diff_text = "\n".join(diff)
                        st.code(diff_text, language="diff")
                    else:
                        st.info("No changes detected")
                else:
                    st.info("No changes to show")

        # Pipeline execution section
        st.markdown("---")
        st.subheader("Run Pipeline")

        if st.button("Generate & Run Full Pipeline"):
            with st.spinner("Running pipeline..."):
                try:
                    run_command(["python", "setup_dvc.py"])
                    run_command(["dvc", "repro"])
                    st.success("Pipeline completed successfully!")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    except FileNotFoundError:
        st.error("`params.yaml` not found.")


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
        run_command(["dvc", "repro", "bootstrap"])
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
        "Cycle to run", min_value=1, value=1, max_value=max_cycles, key="train_cycle"
    )
    if st.button("Run Training Stage"):
        run_command(["dvc", "repro", f"train_cycle_{cycle_num}"])

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
        max_value=max_cycles,
        key="eval_cycle",
    )
    if st.button("Run Evaluation Stage"):
        run_command(["dvc", "repro", f"evaluate_cycle_{cycle_num}"])

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
        max_value=max_cycles,
        key="pred_cycle",
    )
    if st.button("Run Prediction Stage"):
        run_command(["dvc", "repro", f"predict_cycle_{cycle_num}"])

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
        max_value=max_cycles,
        key="drift_cycle",
    )
    if st.button("Run Drift Analysis Stage"):
        run_command(["dvc", "repro", f"analyze_drift_cycle_{cycle_num}"])

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

    if st.button("Run Performance Analysis Stage"):
        run_command(["dvc", "repro", "analyze_cycles_performance"])

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
