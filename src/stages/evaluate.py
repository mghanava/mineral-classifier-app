"""Evaluation script for trained models in each cycle.

This module loads model and test data for a specified cycle, runs evaluation with calibration, and saves the results. It provides functions for path management, evaluation, and command-line execution.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.models import get_model
from src.utilities.eval_utils import evaluate_with_calibration
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_params,
)
from src.utilities.graph_protocol import model_graph_settings, resolve_graph_protocol
from src.utilities.model_bundle import (
    CALIBRATION_BINDING,
    load_model_bundle,
    write_calibration_binding,
)
from src.utilities.preprocessing import TrainingPartition
from src.utilities.raw_graphs import load_cycle_data, model_graph, synthetic_schema


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
    protocol = resolve_graph_protocol(params)
    settings = model_graph_settings(protocol, model_name, params)
    model_params = {**params["models"][model_name], "add_self_loops": False}
    # Ensure output directory exists
    output_path = ensure_directory_exists(paths["output"])
    raw, folds, test_data, calib_data = load_cycle_data(Path(paths["base"]), protocol)
    schema = synthetic_schema(base_params)
    if (raw.feature_names, raw.class_names) != schema:
        raise ValueError(
            "Evaluation configuration does not match the canonical raw schema."
        )
    model = get_model(model_name, model_params)
    state, bundle = load_model_bundle(Path(paths["model"]), model, *schema, settings)
    if state.dataset_sha256 != raw.dataset_sha256 or state.fold_index >= len(folds):
        raise ValueError(
            "Model preprocessing does not match this evaluation dataset and fold."
        )
    training, _ = folds[state.fold_index]
    if bundle["training_graph_sha256"] != training.graph_sha256:
        raise ValueError(
            "Model training graph does not match the evaluation cycle. Retrain on these graphs."
        )
    partition = TrainingPartition(
        raw.dataset_sha256, state.fold_index, training.original_node_indices.numpy()
    )
    if state.partition_sha256 != partition.partition_sha256:
        raise ValueError(
            "Model preprocessing does not match the evaluation cycle's fit provenance."
        )

    # Run evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} ...")

    destination = Path(output_path)
    with TemporaryDirectory(dir=destination) as temporary:
        evaluate_with_calibration(
            test_data=model_graph(test_data, raw, state, settings).to(device),
            calibration_data=model_graph(calib_data, raw, state, settings).to(device),
            model=model.to(device),
            calibration_method=eval_params["calibration_method"],
            initial_temperature=eval_params["initial_temperature"],
            n_bins=eval_params["n_bins"],
            class_names=list(raw.class_names),
            lr=eval_params["lr"],
            n_epochs=eval_params["n_epochs"],
            weight_decay_adam_optimizer=eval_params["weight_decay_adam_optimizer"],
            reg_lambda=eval_params["reg_lambda"],
            reg_mu=eval_params["reg_mu"],
            eps=eval_params["eps"],
            factor_learning_rate_scheduler=eval_params[
                "factor_learning_rate_scheduler"
            ],
            patience_learning_rate_scheduler=eval_params[
                "patience_learning_rate_scheduler"
            ],
            patience_early_stopping=eval_params["patience_early_stopping"],
            min_delta_early_stopping=eval_params["min_delta_early_stopping"],
            verbose=eval_params["verbose"],
            seed=eval_params["seed"],
            save_path=temporary,
        )
        staged = Path(temporary)
        write_calibration_binding(
            staged, bundle, raw.dataset_sha256, calib_data.split_sha256
        )
        for artifact in staged.iterdir():
            if artifact.name != CALIBRATION_BINDING:
                artifact.replace(destination / artifact.name)
        (staged / CALIBRATION_BINDING).replace(destination / CALIBRATION_BINDING)
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
