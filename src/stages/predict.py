import argparse
import os

import torch
import yaml

from src.models import get_model
from src.utilities.logging_utils import LogTime
from src.utilities.pred_utils import prediction


def main():
    dataset_path = "results/data"
    model_trained_path = "results/trained"
    evaluation_path = "results/evaluation"
    prediction_path = "results/prediction"
    os.makedirs(prediction_path, exist_ok=True)

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

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
    N_PERMUTATIONS = params["predict"]["n_permutations"]

    pred_data = torch.load(
        os.path.join(dataset_path, "pred_data.pt"), weights_only=False
    )
    base_data = torch.load(
        os.path.join(dataset_path, "base_data.pt"), weights_only=False
    )
    calibrator_path = os.path.join(evaluation_path, "calibrator.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for prediction\n")
    with LogTime(task_name="\nEvaluation"):
        print("Assessing prediction dataset ...\n")
        prediction(
            base_data,
            pred_data,
            model,
            calibrator_path,
            n_permutations=N_PERMUTATIONS,
            save_path=prediction_path,
            device=device,
        )


if __name__ == "__main__":
    main()
