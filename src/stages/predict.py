"""Prediction stage for model inference in a specific cycle.

This module loads trained models, prediction data, and calibrators,
then runs predictions and saves the results for a given cycle.
"""

import argparse
import os

import torch

from src.models import get_model
from src.utilities.general_utils import LogTime, ensure_directory_exists, load_params
from src.utilities.pred_utils import prediction


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
    model_params = params["models"][model_name]
    output_path = ensure_directory_exists(paths["output"])
    # Load trained model
    model_path = os.path.join(paths["model"], "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = get_model(model_name, model_params)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Load prediction data
    pred_data_path = os.path.join(paths["prediction_data"], "pred_data.pt")
    if not os.path.exists(pred_data_path):
        raise FileNotFoundError(f"Prediction data not found at {pred_data_path}")
    pred_data = torch.load(pred_data_path, weights_only=False)

    # Load calibrator from evaluation
    calibrator_path = os.path.join(paths["evaluation"], "calibrator.pt")
    if not os.path.exists(calibrator_path):
        raise FileNotFoundError(f"Calibrator not found at {calibrator_path}")

    # Run prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} ...")

    prediction(
        pred_data=pred_data,
        model=model,
        calibrator_path=calibrator_path,
        class_names=base_params["class_names"],
        save_path=output_path,
        device=device,
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
