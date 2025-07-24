"""Evaluation module for model assessment with calibration.

This module provides functionality to evaluate trained models using a test dataset,
including model calibration assessment and various evaluation metrics. It handles
command-line arguments, model loading, and execution of the evaluation process.
"""

import argparse
import os

import torch
import yaml

from src.models import get_model
from src.utilities.eval_utils import evaluate_with_calibration
from src.utilities.logging_utils import LogTime


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    """Run evaluation script.

    This function parses command-line arguments, loads the model and evaluation data,
    performs model evaluation with calibration, saves evaluation results, and prints progress.
    """
    base_data_path = "results/data/base"
    model_trained_path = "results/trained"
    evaluation_path = "results/evaluation"
    os.makedirs(evaluation_path, exist_ok=True)
    params = load_params()
    CYCLE_NUM = params["cycle"]

    test_data = torch.load(
        os.path.join(base_data_path, f"test_data_cycle_{CYCLE_NUM}.pt"),
        weights_only=False,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    # Get model-specific parameters
    model_params = params["models"][args.model]
    # Initialize a new model instance
    model = get_model(args.model, model_params)
    model.load_state_dict(
        torch.load(
            f"{model_trained_path}/{args.model}_cycle_{CYCLE_NUM}.pkl",
            weights_only=True,
        )
    )

    INITIAL_TEMPERATURE = params["evaluate"]["initial_temperature"]
    CALIBRATION_METHOD = params["evaluate"]["calibration_method"]
    N_BINS = params["evaluate"]["n_bins"]
    CLASS_NAMES = params["evaluate"]["class_names"]
    N_EPOCHS = params["evaluate"]["n_epochs"]
    LEARNING_RATE = params["evaluate"]["lr"]
    WEIGHT_DEACY = params["evaluate"]["weight_decay_adam_optimizer"]
    FACTOR = params["evaluate"]["factor_learning_rate_scheduler"]
    PATIENCE_LEARNING_RATE_SCHEDULER = params["evaluate"][
        "patience_learning_rate_scheduler"
    ]
    PATIENCE_EARLY_STOPPING = params["evaluate"]["patience_early_stopping"]
    MIN_DELTA_EARLY_STOPPING = params["evaluate"]["min_delta_early_stopping"]
    VERBOSE = params["evaluate"]["verbose"]
    SEED = params["evaluate"]["seed"]
    REG_LAMBDA = params["evaluate"]["reg_lambda"]
    REG_MU = params["evaluate"]["reg_mu"]
    EPS = params["evaluate"]["eps"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for evaluation\n")
    with LogTime(task_name="\nEvaluation"):
        print("Assessing test dataset ...\n")
        evaluate_with_calibration(
            data=test_data.to(device),
            model=model.to(device),
            calibration_method=CALIBRATION_METHOD,
            initial_temperature=INITIAL_TEMPERATURE,
            n_bins=N_BINS,
            class_names=CLASS_NAMES,
            lr=LEARNING_RATE,
            n_epochs=N_EPOCHS,
            weight_decay_adam_optimizer=WEIGHT_DEACY,
            factor_learning_rate_scheduler=FACTOR,
            patience_learning_rate_scheduler=PATIENCE_LEARNING_RATE_SCHEDULER,
            patience_early_stopping=PATIENCE_EARLY_STOPPING,
            min_delta_early_stopping=MIN_DELTA_EARLY_STOPPING,
            save_path=evaluation_path,
            verbose=VERBOSE,
            device=device,
            eps=EPS,
            reg_lambda=REG_LAMBDA,
            reg_mu=REG_MU,
            seed=SEED,
        )


if __name__ == "__main__":
    main()
