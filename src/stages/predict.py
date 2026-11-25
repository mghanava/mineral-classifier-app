"""Prediction stage for model inference in a specific cycle.

This module loads trained models, prediction data, and calibrators,
then runs predictions and saves the results for a given cycle.
"""

import argparse
from pathlib import Path

import torch

from src.models import get_model
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_data,
    load_params,
)
from src.utilities.graph_protocol import (
    model_graph_settings,
    resolve_graph_protocol,
    verify_graph,
)
from src.utilities.model_bundle import load_model_bundle, verified_calibrator
from src.utilities.pred_utils import prediction
from src.utilities.raw_graphs import model_graph, read_raw_graph, synthetic_schema


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "model": f"results/trained/cycle_{cycle_num}",
        "evaluation": f"results/evaluation/cycle_{cycle_num}",
        "prediction_data": f"results/data/prediction/cycle_{cycle_num}",
        "output": f"results/prediction/cycle_{cycle_num}",
    }


def run_prediction(paths: dict, params: dict, model_name: str):
    """Run prediction for a given cycle using the specified model.

    Parameters
    ----------
    paths : dict
        Dictionary containing paths for model, evaluation, prediction data, and output.
    params : dict
        Dictionary of configuration parameters for data and models.
    model_name : str
        Name of the model to use for prediction.

    Raises
    ------
    FileNotFoundError
        If required files (model, prediction data, calibrator) are missing.

    """
    base_params = params["data"]["base"]
    protocol = resolve_graph_protocol(params)
    settings = model_graph_settings(protocol, model_name, params)
    model_params = settings["model_parameters"]
    # Ensure output directory exists
    output_path = ensure_directory_exists(paths["output"])

    pred_data = load_data(Path(paths["prediction_data"]) / "pred_data.pt", "Prediction")
    raw = read_raw_graph(pred_data)
    schema = synthetic_schema(base_params)
    if (raw.feature_names, raw.class_names) != schema:
        raise ValueError(
            "Prediction observations do not match the configured feature or class schema."
        )
    verify_graph(pred_data, protocol, "prediction")
    model = get_model(model_name, model_params)
    state, bundle = load_model_bundle(Path(paths["model"]), model, *schema, settings)
    if getattr(pred_data, "source_parent_sha256", None) != state.dataset_sha256:
        raise ValueError(
            "Prediction batch was generated for a different base cycle. Regenerate it."
        )
    calibrator = verified_calibrator(
        Path(paths["evaluation"]), bundle, state.dataset_sha256
    )

    # Run prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} ...")

    prediction(
        pred_data=model_graph(pred_data, raw, state, settings).to(device),
        model=model.to(device),
        calibrator_path=calibrator,
        class_names=list(raw.class_names),
        save_path=output_path,
    )
    print(f"✓ Prediction results saved to {output_path}.")


def main():
    """Run the prediction stage for a specific cycle."""
    parser = argparse.ArgumentParser(description="Run prediction for a given cycle.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to use for prediction",
    )
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle
    model_name = args.model

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")
    # Load parameters and paths
    params = load_params()
    paths = get_cycle_paths(cycle_num)

    with LogTime(task_name=f"\nPrediction cycle {cycle_num}"):
        run_prediction(paths, params, model_name)


if __name__ == "__main__":
    main()
