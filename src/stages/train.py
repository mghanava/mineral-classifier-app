import argparse
import os

import torch
import yaml

from src.models import get_model
from src.utilities.logging_utils import (
    LogTime,
)
from src.utilities.train_utils import train


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    """Execute the main training pipeline for the GNN model."""
    parser = argparse.ArgumentParser(description="Train a GNN model for a given cycle.")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle
    model_name = args.model

    params = load_params()

    base_data_path = f"results/data/base/cycle_{cycle_num}"
    model_trained_path = f"results/trained/cycle_{cycle_num}"
    os.makedirs(model_trained_path, exist_ok=True)

    fold_data = torch.load(
        os.path.join(base_data_path, "fold_data.pt"),
        weights_only=False,
    )

    model_params = params["models"][model_name]
    model = get_model(model_name, model_params)

    N_EPOCHS = params["train"]["n_epochs"]
    LEARNING_RATE = params["train"]["lr"]
    WEIGHT_DEACY = params["train"]["weight_decay_adam_optimizer"]
    MAX_GRAD_NORM = params["train"]["max_grad_norm"]
    FACTOR = params["train"]["factor_learning_rate_scheduler"]
    PATIENCE_LEARNING_RATE_SCHEDULER = params["train"][
        "patience_learning_rate_scheduler"
    ]
    PATIENCE_EARLY_STOPPING = params["train"]["patience_early_stopping"]
    MIN_DELTA_EARLY_STOPPING = params["train"]["min_delta_early_stopping"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for training")

    fold_results = []
    dataset_idx = 0
    with LogTime(task_name="\nTraining"):
        for graph in fold_data:
            print(f"\nTraining model on kfold {graph.fold + 1} ...\n")
            model_trained, best_loss_val = train(
                graph.to(device),
                model.to(device),
                n_epochs=N_EPOCHS,
                lr=LEARNING_RATE,
                max_grad_norm=MAX_GRAD_NORM,
                weight_decay=WEIGHT_DEACY,
                factor_learning_rate_scheduler=FACTOR,
                patience_learning_rate_scheduler=PATIENCE_LEARNING_RATE_SCHEDULER,
                patience_early_stopping=PATIENCE_EARLY_STOPPING,
                min_delta_early_stopping=MIN_DELTA_EARLY_STOPPING,
                save_path=model_trained_path,
                dataset_idx=dataset_idx + 1,
                # cycle_num=cycle_num,
            )
            fold_results.append((model_trained, best_loss_val))
            dataset_idx += 1

    # Select the best trained model and save it
    sorted_models = sorted(fold_results, key=lambda x: x[1], reverse=True)
    model_trained_best = sorted_models[0][0]
    model_traind_path = f"{model_trained_path}/{model_name}.pkl"
    torch.save(model_trained_best.state_dict(), model_traind_path)
    print(f"Model saved to {model_traind_path}\n")


if __name__ == "__main__":
    main()
