import argparse
import os
from pathlib import Path

import torch
import yaml

from src.models import get_model
from src.utilities.eval_utils import evaluate_with_calibration
from src.utilities.logging_utils import LogTime


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
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",  # Previous cycle's data
        "model": f"results/trained/cycle_{cycle_num}",
        "evaluation": f"results/evaluation/cycle_{cycle_num}",
    }


def main():
    """Run evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model for a given cycle."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to evaluate"
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
    evaluation_path = ensure_directory_exists(paths["evaluation"])

    # Load parameters
    params = load_params()
    eval_params = params["evaluate"]
    model_params = params["models"][model_name]

    # Load test data from previous cycle
    test_data_path = os.path.join(paths["base_data"], "test_data.pt")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    test_data = torch.load(test_data_path, weights_only=False)

    # Load trained model
    model_path = os.path.join(paths["model"], f"{model_name}_cycle_{cycle_num}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = get_model(model_name, model_params)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # Run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} ...")

    with LogTime(task_name=f"\nEvaluation cycle {cycle_num}"):
        evaluate_with_calibration(
            data=test_data.to(device),
            model=model.to(device),
            calibration_method=eval_params["calibration_method"],
            initial_temperature=eval_params["initial_temperature"],
            n_bins=eval_params["n_bins"],
            class_names=eval_params["class_names"],
            lr=eval_params["lr"],
            n_epochs=eval_params["n_epochs"],
            weight_decay_adam_optimizer=eval_params["weight_decay_adam_optimizer"],
            factor_learning_rate_scheduler=eval_params[
                "factor_learning_rate_scheduler"
            ],
            patience_learning_rate_scheduler=eval_params[
                "patience_learning_rate_scheduler"
            ],
            patience_early_stopping=eval_params["patience_early_stopping"],
            min_delta_early_stopping=eval_params["min_delta_early_stopping"],
            save_path=evaluation_path,
            verbose=eval_params["verbose"],
            device=device,
            eps=eval_params["eps"],
            reg_lambda=eval_params["reg_lambda"],
            reg_mu=eval_params["reg_mu"],
            seed=eval_params["seed"],
        )

    print(f"✓ Evaluation results saved to {evaluation_path}.\n")


if __name__ == "__main__":
    main()
