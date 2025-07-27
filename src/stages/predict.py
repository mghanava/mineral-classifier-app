import argparse
import os

import torch
import yaml

from src.models import get_model
from src.utilities.logging_utils import LogTime
from src.utilities.pred_utils import prediction


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    """Run the prediction stage: load model, prepare data, and save predictions."""
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

    params = load_params()
    CLASS_NAMES = params["evaluate"]["class_names"]

    model_trained_path = f"results/trained/cycle_{cycle_num}"
    evaluation_path = f"results/evaluation/cycle_{cycle_num}"
    prediction_data_path = f"results/data/prediction/cycle_{cycle_num}"
    prediction_path = f"results/prediction/cycle_{cycle_num}"
    os.makedirs(prediction_path, exist_ok=True)

    model_params = params["models"][model_name]
    model = get_model(model_name, model_params)
    model.load_state_dict(
        torch.load(
            f"{model_trained_path}/{model_name}.pkl",
            weights_only=True,
        )
    )

    pred_data = torch.load(
        os.path.join(prediction_data_path, "pred_data.pt"),
        weights_only=False,
    )
    calibrator_path = os.path.join(evaluation_path, "calibrator.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for prediction\n")
    with LogTime(task_name="\nPrediction"):
        print("Assessing prediction dataset ...\n")
        prediction(
            pred_data,
            model,
            calibrator_path,
            CLASS_NAMES,
            # cycle_num=cycle_num,
            save_path=prediction_path,
            device=device,
        )


if __name__ == "__main__":
    main()
