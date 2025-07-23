"""Prediction stage script for running inference using trained models.

This module loads a trained model, prepares prediction data, and saves predictions.
"""

import argparse
import os

import torch
import yaml

from src.models import get_model
from src.utilities.logging_utils import LogTime
from src.utilities.pred_utils import prediction


def main():
    """Run the prediction stage: load model, prepare data, and save predictions."""
    dataset_path = "results/data"
    model_trained_path = "results/trained"
    evaluation_path = "results/evaluation"
    prediction_path = "results/prediction"
    os.makedirs(prediction_path, exist_ok=True)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    CLASS_NAMES = params["evaluate"]["class_names"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    # Get model-specific parameters
    model_params = params["models"][args.model]
    # Initialize a new model instance and load the trained weights
    model = get_model(args.model, model_params)
    model.load_state_dict(
        torch.load(f"{model_trained_path}/{args.model}.pkl", weights_only=True)
    )

    pred_data = torch.load(
        os.path.join(dataset_path, "pred_data.pt"), weights_only=False
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
            save_path=prediction_path,
            device=device,
        )


if __name__ == "__main__":
    main()
