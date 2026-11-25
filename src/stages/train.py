"""Training module for GNN models in different cycles.

This module provides functionality to train GNN models in a cycle-based approach,
including loading data, training models with cross-validation, and saving the best
performing model.
"""

import argparse
from pathlib import Path

import torch

from src.models import get_model
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_params,
)
from src.utilities.model_bundle import publish_model
from src.utilities.preprocessing import TrainingPartition, fit_preprocessing
from src.utilities.raw_graphs import load_cycle_data, model_graph, synthetic_schema
from src.utilities.train_utils import train


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",
        "output": f"results/trained/cycle_{cycle_num}",
    }


def run_training(paths, params, model_name, cycle_num):
    """Train a model for a specific cycle using provided paths and parameters.

    Parameters
    ----------
    paths : dict
        Dictionary containing paths for base data, previous model, and output.
    params : dict
        Dictionary of training and model parameters.
    model_name : str
        Name of the model to train.
    cycle_num : int
        Current cycle number.

    Returns
    -------
    None

    """
    train_params = params["train"]
    model_params = {
        **params["models"][model_name],
        "add_self_loops": params.get("add_self_loops", True),
    }
    n_classes = params["data"]["base"]["n_classes"]
    # Ensure output directory exists
    output_path = ensure_directory_exists(paths["output"])

    raw, fold_data, _, _ = load_cycle_data(Path(paths["base_data"]))
    if (raw.feature_names, raw.class_names) != synthetic_schema(params["data"]["base"]):
        raise ValueError(
            "Training configuration does not match the canonical raw schema."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} ...")

    best_fold = None
    for fold_index, (training, validation) in enumerate(fold_data):
        fold_idx = fold_index + 1
        print(f"\nTraining model on fold {fold_idx}...")
        partition = TrainingPartition(
            raw.dataset_sha256, fold_index, training.original_node_indices.numpy()
        )
        preprocessing = fit_preprocessing(
            raw,
            partition,
            params["data"]["scaler_type"],
            params["data"].get("scaler_params"),
        )
        fold = (
            model_graph(training, raw, preprocessing),
            model_graph(validation, raw, preprocessing),
        )
        model = get_model(model_name, model_params).to(device)
        trained_model, best_loss_val = train(
            fold,
            model,
            n_classes=n_classes,
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
            save_path=output_path,
            dataset_idx=fold_idx,
        )

        if best_fold is None or best_loss_val < best_fold[1]:
            best_fold = (fold_idx, best_loss_val, trained_model, preprocessing)

    if best_fold is None:
        raise ValueError("At least one train-validation fold is required.")

    # Preserve the evaluation and prediction checkpoint contract until a refit is available.
    fold_idx, best_loss_val, best_model, preprocessing = best_fold
    model_save_path = Path(output_path) / "model.pt"
    publish_model(model_save_path.parent, best_model, preprocessing)
    print(
        f"✓ Fold {fold_idx} model (validation loss {best_loss_val:.6g}) "
        f"saved to {model_save_path}."
    )


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

    # Load parameters and paths
    params = load_params()
    paths = get_cycle_paths(cycle_num)
    with LogTime(task_name=f"\nTraining cycle {cycle_num}"):
        run_training(paths, params, model_name, cycle_num)


if __name__ == "__main__":
    main()
