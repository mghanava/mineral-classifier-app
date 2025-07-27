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
    """Run evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model for a given cycle."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Current cycle number"
    )
    args = parser.parse_args()
    cycle_num = args.cycle
    model_name = args.model

    params = load_params()

    base_data_path = f"results/data/base/cycle_{cycle_num}"
    model_trained_path = f"results/trained/cycle_{cycle_num}"
    evaluation_path = f"results/evaluation/cycle_{cycle_num}"
    os.makedirs(evaluation_path, exist_ok=True)

    test_data = torch.load(
        os.path.join(base_data_path, "test_data.pt"),
        weights_only=False,
    )

    model_params = params["models"][model_name]
    model = get_model(model_name, model_params)
    model.load_state_dict(
        torch.load(
            f"{model_trained_path}/{model_name}.pkl",
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
