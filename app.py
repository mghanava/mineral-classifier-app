import streamlit as st
import subprocess
import yaml
from pathlib import Path
import pandas as pd
import json
import os
import streamlit.components.v1 as components

def run_command(command):
    """Runs a shell command and streams the output to the Streamlit app."""
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

def show_html_files(base_path):
    """Displays HTML files from a given directory."""
    html_files = list(base_path.glob("*.html"))
    if not html_files:
        st.info("No HTML files found in this directory.")
        return

    for html_file in html_files:
        with st.expander(f"View `{html_file.name}`"):
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)

def data_tab():
    st.header("Data Generation")
    st.write("This stage generates the base data for the pipeline.")
    
    if st.button("Run Bootstrap Stage"):
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "dvc", "repro", "bootstrap"])

    st.subheader("Generated Data")
    data_path = Path("results/data/base/cycle_0")
    if data_path.exists():
        show_html_files(data_path)
    else:
        st.warning("Base data not found. Run the bootstrap stage to generate it.")

def train_tab():
    st.header("Training")
    st.write("This stage trains the model for each cycle.")
    
    cycle_num = st.number_input("Cycle to run", min_value=1, value=1, key="train_cycle")
    if st.button("Run Training Stage"):
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "dvc", "repro", f"train_cycle_{cycle_num}"])

    st.subheader("Training Results")
    train_path = Path(f"results/trained/cycle_{cycle_num}")
    if train_path.exists():
        plot_files = list(train_path.glob("*.png"))
        for plot in plot_files:
            st.image(str(plot))
    else:
        st.warning(f"Training results for cycle {cycle_num} not found.")

def evaluation_tab():
    st.header("Evaluation")
    st.write("This stage evaluates the trained model.")

    cycle_num = st.number_input("Cycle to evaluate", min_value=1, value=1, key="eval_cycle")
    if st.button("Run Evaluation Stage"):
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "dvc", "repro", f"evaluate_cycle_{cycle_num}"])

    st.subheader("Evaluation Results")
    eval_path = Path(f"results/evaluation/cycle_{cycle_num}")
    if eval_path.exists():
        for img_file in eval_path.glob("*.png"):
            st.image(str(img_file))
        
        metrics_file = eval_path / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                st.json(json.load(f))
    else:
        st.warning(f"Evaluation results for cycle {cycle_num} not found.")

def prediction_tab():
    st.header("Prediction")
    st.write("This stage runs predictions on new data.")

    cycle_num = st.number_input("Cycle for prediction", min_value=1, value=1, key="pred_cycle")
    if st.button("Run Prediction Stage"):
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "dvc", "repro", f"predict_cycle_{cycle_num}"])

    st.subheader("Prediction Results")
    pred_path = Path(f"results/prediction/cycle_{cycle_num}")
    if pred_path.exists():
        csv_file = pred_path / "predictions.csv"
        if csv_file.exists():
            st.dataframe(pd.read_csv(csv_file))
        for img_file in pred_path.glob("*.png"):
            st.image(str(img_file))
    else:
        st.warning(f"Prediction results for cycle {cycle_num} not found.")

def drift_analysis_tab():
    st.header("Drift Analysis")
    st.write("This stage analyzes data drift between cycles.")

    cycle_num = st.number_input("Cycle for drift analysis", min_value=1, value=1, key="drift_cycle")
    if st.button("Run Drift Analysis Stage"):
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "dvc", "repro", f"analyze_drift_cycle_{cycle_num}"])

    st.subheader("Drift Analysis Results")
    drift_path = Path(f"results/drift_analysis/cycle_{cycle_num}")
    if drift_path.exists():
        for img_file in drift_path.glob("*.png"):
            st.image(str(img_file))
        
        results_file = drift_path / "drift_results.txt"
        if results_file.exists():
            with open(results_file, "r") as f:
                st.text(f.read())
    else:
        st.warning(f"Drift analysis results for cycle {cycle_num} not found.")

def performance_tab():
    st.header("Cycles Performance Analysis")
    st.write("This stage analyzes the performance across all cycles.")

    if st.button("Run Performance Analysis Stage"):
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "dvc", "repro", "analyze_cycles_performance"])

    st.subheader("Performance Plot")
    perf_img = Path("results/cycles_performance_analysis/cycles_performance.png")
    if perf_img.exists():
        st.image(str(perf_img))
    else:
        st.warning("Performance analysis plot not found.")

def app():
    st.set_page_config(layout="wide")
    st.title("DVC Pipeline Dashboard")

    st.sidebar.header("Pipeline Controls")
    num_cycles = st.sidebar.number_input("Number of Cycles", min_value=1, max_value=10, value=5, step=1)

    if st.sidebar.button("Generate `dvc.yaml` and Run Full Pipeline"):
        # Update params.yaml
        try:
            with open("params.yaml", "r") as f:
                params = yaml.safe_load(f)
            params["cycles"] = num_cycles
            with open("params.yaml", "w") as f:
                yaml.dump(params, f)
            st.success(f"`params.yaml` updated with {num_cycles} cycles.")
        except Exception as e:
            st.error(f"Error updating `params.yaml`: {e}")
            return

        # Build container and generate dvc.yaml
        run_command(["docker-compose", "build"])
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "python", "generate_dvc.py"])
        run_command(["docker-compose", "run", "--rm", "mineral_classifier", "dvc", "repro", "-f"])

    tab_names = ["Data", "Train", "Evaluation", "Prediction", "Drift Analysis", "Performance"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        data_tab()
    with tabs[1]:
        train_tab()
    with tabs[2]:
        evaluation_tab()
    with tabs[3]:
        prediction_tab()
    with tabs[4]:
        drift_analysis_tab()
    with tabs[5]:
        performance_tab()

if __name__ == "__main__":
    app()
