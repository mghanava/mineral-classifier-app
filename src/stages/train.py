"""Training module for GNN models in different cycles.

This module provides functionality to train GNN models in a cycle-based approach,
including independent cross-validation and a fresh development-pool refit.
"""

import argparse
from pathlib import Path

from src.models import get_model
from src.utilities.device import select_device
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_params,
)
from src.utilities.graph_protocol import model_graph_settings, resolve_graph_protocol
from src.utilities.model_bundle import publish_model
from src.utilities.preprocessing import TrainingPartition, fit_preprocessing
from src.utilities.raw_graphs import (
    build_development_graph,
    load_cycle_data,
    model_graph,
    synthetic_schema,
)
from src.utilities.reproducibility import (
    DEFAULT_SEED,
    SEED_SCHEME,
    derive_seed,
    torch_seed,
    validate_seed,
)
from src.utilities.train_utils import train, train_final
from src.utilities.training_schedule import final_training_plan


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
    seed = validate_seed(train_params.get("seed", DEFAULT_SEED), "train.seed")
    device = select_device(params.get("device", "auto"))
    print(f"Using device {device} ...")
    protocol = resolve_graph_protocol(params)
    settings = model_graph_settings(protocol, model_name, params)
    model_params = settings["model_parameters"]
    n_classes = params["data"]["base"]["n_classes"]
    # Ensure output directory exists
    output_path = ensure_directory_exists(paths["output"])

    raw, fold_data, test, calibration = load_cycle_data(
        Path(paths["base_data"]), protocol
    )
    if (raw.feature_names, raw.class_names) != synthetic_schema(params["data"]["base"]):
        raise ValueError(
            "Training configuration does not match the canonical raw schema."
        )

    diagnostics = []
    fold_seeds = []
    for fold_index, (training, validation) in enumerate(fold_data):
        fold_idx = fold_index + 1
        if training.num_nodes == 1 and model_params.get("batch_norm", True):
            raise ValueError(
                f"Fold {fold_idx} has one training node. Disable model batch_norm or use a larger training partition."
            )
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
            model_graph(training, raw, preprocessing, settings),
            model_graph(validation, raw, preprocessing, settings),
        )
        fold_seed = derive_seed(seed, "train", cycle_num, "cv", fold_index)
        fold_seeds.append(fold_seed)
        with torch_seed(fold_seed, device):
            model = get_model(model_name, model_params).to(device)
            result = train(
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

        diagnostics.append(result)

    plan = final_training_plan(diagnostics, train_params["n_epochs"])
    development = build_development_graph(raw, fold_data, test, calibration, protocol)
    partition = TrainingPartition(
        raw.dataset_sha256, None, development.original_node_indices.numpy()
    )
    preprocessing = fit_preprocessing(
        raw,
        partition,
        params["data"]["scaler_type"],
        params["data"].get("scaler_params"),
    )
    refit_seed = derive_seed(seed, "train", cycle_num, "final_refit")
    with torch_seed(refit_seed, device):
        final_model = get_model(model_name, model_params).to(device)
        train_final(
            model_graph(development, raw, preprocessing, settings),
            final_model,
            n_classes=n_classes,
            learning_rates=plan["learning_rates"],
            max_grad_norm=train_params["max_grad_norm"],
            weight_decay=train_params["weight_decay_adam_optimizer"],
        )
    training = {
        "strategy": "full_development_pool",
        "dataset_sha256": raw.dataset_sha256,
        "partition_sha256": partition.partition_sha256,
        "observation_count": development.num_nodes,
        **plan,
        "seed_provenance": {
            "scheme": SEED_SCHEME,
            "cycle": cycle_num,
            "configured_seed": seed,
            "cv_seeds": fold_seeds,
            "final_refit_seed": refit_seed,
            "device": str(device),
        },
    }
    publish_model(
        Path(output_path),
        final_model,
        preprocessing,
        settings,
        development.graph_sha256,
        training,
    )
    print(
        f"✓ Final development-pool model ({development.num_nodes} nodes, "
        f"{plan['epochs']} epochs) saved to {Path(output_path) / 'model.pt'}."
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
