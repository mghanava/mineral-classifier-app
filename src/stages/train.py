import argparse
import os
from pathlib import Path

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


def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",  # Previous cycle's data
        "model": f"results/trained/cycle_{cycle_num}",
    }


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

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")

    # Get cycle-specific paths and ensure directories exist
    paths = get_cycle_paths(cycle_num)
    model_trained_path = ensure_directory_exists(paths["model"])

    # Load parameters
    params = load_params()
    train_params = params["train"]
    model_params = params["models"][model_name]

    # Load training data from previous cycle
    fold_data_path = os.path.join(paths["base_data"], "fold_data.pt")
    if not os.path.exists(fold_data_path):
        raise FileNotFoundError(f"Training data not found at {fold_data_path}")

    fold_data = torch.load(fold_data_path, weights_only=False)

    # Initialize model
    model = get_model(model_name, model_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} for training cycle {cycle_num}")

    # Training loop
    fold_results = []
    with LogTime(task_name=f"\nTraining cycle {cycle_num}"):
        for graph_idx, graph in enumerate(fold_data, 1):
            print(f"\nTraining model on fold {graph.fold + 1}...")

            trained_model, best_loss_val = train(
                graph.to(device),
                model.to(device),
                n_epochs=train_params["n_epochs"],
                lr=train_params["lr"],
                max_grad_norm=train_params["max_grad_norm"],
                weight_decay=train_params["weight_decay_adam_optimizer"],
                factor_learning_rate_scheduler=train_params[
                    "factor_learning_rate_scheduler"
                ],
                patience_learning_rate_scheduler=train_params[
                    "patience_learning_rate_scheduler"
                ],
                patience_early_stopping=train_params["patience_early_stopping"],
                min_delta_early_stopping=train_params["min_delta_early_stopping"],
                save_path=model_trained_path,
                dataset_idx=graph_idx,
            )
            fold_results.append((trained_model, best_loss_val))

    # Save best model
    best_model = sorted(fold_results, key=lambda x: x[1], reverse=True)[0][0]
    model_save_path = os.path.join(
        model_trained_path, f"{model_name}_cycle_{cycle_num}.pt"
    )
    torch.save(best_model.state_dict(), model_save_path)
    print(f"\n✓ Best model saved to {model_save_path}")


if __name__ == "__main__":
    main()
