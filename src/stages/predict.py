import argparse
import os
from pathlib import Path

import torch
import yaml

from src.models import get_model
from src.utilities.logging_utils import LogTime
from src.utilities.pred_utils import prediction


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "model": f"results/trained/cycle_{cycle_num}",
        "evaluation": f"results/evaluation/cycle_{cycle_num}",
        "prediction_data": f"results/data/prediction/cycle_{cycle_num}",
        "prediction_results": f"results/prediction/cycle_{cycle_num}",
    }


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

    # Get cycle-specific paths and ensure directories exist
    paths = get_cycle_paths(cycle_num)
    prediction_path = ensure_directory_exists(paths["prediction_results"])

    # Load parameters
    params = load_params()
    base_params = params["data"]["base"]
    model_params = params["models"][model_name]

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

    with LogTime(task_name=f"\nPrediction cycle {cycle_num}"):
        prediction(
            pred_data=pred_data,
            model=model,
            calibrator_path=calibrator_path,
            class_names=base_params["class_names"],
            save_path=prediction_path,
            device=device,
        )

    print(f"✓ Predictions saved to {prediction_path}.\n")


if __name__ == "__main__":
    main()
