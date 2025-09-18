"""Evaluation script for trained models in each cycle.

This module loads model and test data for a specified cycle, runs evaluation with calibration, and saves the results. It provides functions for path management, evaluation, and command-line execution.
"""

import argparse
import os

import torch

from src.models import get_model
from src.utilities.eval_utils import evaluate_with_calibration
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_data,
    load_model_weights,
    load_params,
)


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "base": f"results/data/base/cycle_{cycle_num - 1}",
        "model": f"results/trained/cycle_{cycle_num}",
        "output": f"results/evaluation/cycle_{cycle_num}",
    }


def run_evaluation(paths: dict, params: dict, model_name: str):
    """Evaluate a trained model for a specific cycle using calibration.

    Parameters
    ----------
    paths : dict
        Dictionary containing paths for base, model, and evaluation output.
    params : dict
        Dictionary of configuration parameters for data, evaluation, and models.
    model_name : str
        Name of the model to evaluate.

    Raises
    ------
    FileNotFoundError
        If the required test data or model file is not found.

    """
    base_params = params["data"]["base"]
    eval_params = params["evaluate"]
    model_params = params["models"][model_name]
    model_params["add_self_loops"] = params.get("add_self_loops", True)
    # Ensure output directory exists
    output_path = ensure_directory_exists(paths["output"])
    # Load test and calibration data from previous cycle
    test_data = load_data(os.path.join(paths["base"], "test_data.pt"), "Test")
    calib_data = load_data(os.path.join(paths["base"], "calib_data.pt"), "Calibration")

    # Load trained model
    model = get_model(model_name, model_params)
    load_model_weights(model, os.path.join(paths["model"], "model.pt"))

    # Run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} ...")

    evaluate_with_calibration(
        test_data=test_data.to(device),
        calibration_data=calib_data.to(device),
        model=model.to(device),
        calibration_method=eval_params["calibration_method"],
        initial_temperature=eval_params["initial_temperature"],
        n_bins=eval_params["n_bins"],
        class_names=base_params["class_names"],
        lr=eval_params["lr"],
        n_epochs=eval_params["n_epochs"],
        weight_decay_adam_optimizer=eval_params["weight_decay_adam_optimizer"],
        reg_lambda=eval_params["reg_lambda"],
        reg_mu=eval_params["reg_mu"],
        eps=eval_params["eps"],
        factor_learning_rate_scheduler=eval_params["factor_learning_rate_scheduler"],
        patience_learning_rate_scheduler=eval_params[
            "patience_learning_rate_scheduler"
        ],
        patience_early_stopping=eval_params["patience_early_stopping"],
        min_delta_early_stopping=eval_params["min_delta_early_stopping"],
        verbose=eval_params["verbose"],
        seed=eval_params["seed"],
        save_path=output_path,
    )
    print(f"✓ Evaluation results saved to {output_path}.")


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
    # Load parameters and paths
    paths = get_cycle_paths(cycle_num)
    params = load_params()
    # Run evaluation
    with LogTime(task_name=f"\nEvaluation cycle {cycle_num}"):
        run_evaluation(paths, params, model_name)


if __name__ == "__main__":
    main()
