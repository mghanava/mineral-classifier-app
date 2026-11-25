"""Module for analyzing data drift between base and prediction datasets for a given cycle.

This script loads data, performs drift analysis using specified parameters, and saves results and plots.
"""

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.utilities.drift_detection_utils import AnalyzeDrift
from src.utilities.general_utils import (
    LogTime,
    load_data,
    load_params,
)
from src.utilities.graph_protocol import resolve_graph_protocol, verify_graph
from src.utilities.model_bundle import read_model_bundle
from src.utilities.raw_graphs import read_raw_graph, synthetic_schema


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle."""
    return {
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",
        "prediction_data": f"results/data/prediction/cycle_{cycle_num}",
        "model": f"results/trained/cycle_{cycle_num}",
        "output": f"results/drift_analysis/cycle_{cycle_num}",
    }


def run_analyze_drift(paths: dict, params: dict, cycle_num: int) -> None:
    """Compare the historical base and incoming batch in approved feature spaces."""
    reference_graph = load_data(Path(paths["base_data"]) / "base_data.pt", "Base")
    incoming_graph = load_data(
        Path(paths["prediction_data"]) / "pred_data.pt", "Prediction"
    )
    reference = read_raw_graph(reference_graph)
    incoming = read_raw_graph(incoming_graph)
    schema = synthetic_schema(params["data"]["base"])
    for dataset in (reference, incoming):
        if (dataset.feature_names, dataset.class_names) != schema:
            raise ValueError(
                "Drift feature order or class mapping does not match the configured schema."
            )
    if (
        getattr(incoming_graph, "source_parent_sha256", None)
        != reference.dataset_sha256
    ):
        raise ValueError(
            "Drift input batch does not match the historical base cycle. Regenerate the batch."
        )

    protocol = resolve_graph_protocol(params)
    verify_graph(reference_graph, protocol, "overview")
    verify_graph(incoming_graph, protocol, "prediction")
    state, bundle, _ = read_model_bundle(Path(paths["model"]), *schema, protocol)
    if state.dataset_sha256 != reference.dataset_sha256:
        raise ValueError(
            "Drift reference does not match the model's source dataset. Use the matching cycle bundle."
        )

    # Preserve float64 throughout the statistical analysis, including identity scaling.
    populations = [
        Data(
            unscaled_features=torch.tensor(dataset.features),
            x=torch.from_numpy(
                state.transform(dataset.features, dataset.feature_names)
            ),
        )
        for dataset in (reference, incoming)
    ]
    output = Path(paths["output"])
    output.mkdir(parents=True, exist_ok=True)
    config = params["analyze_drift"]
    analyzer = AnalyzeDrift(
        *populations,
        feature_names=list(reference.feature_names),
        gamma=config["gamma"],
        n_permutations=config["n_permutations"],
        use_max_sliced_wasserstein=config["use_max_sliced_wasserstein"],
        use_sinkhorn_wasserstein=config["use_sinkhorn_wasserstein"],
        n_projections=config["n_projections"],
        sinkhorn_wasserstein_reg=config["sinkhorn_wasserstein_reg"],
        early_stopping_config=config.get("early_stopping", {}),
        save_path=str(output),
    )
    analyzer.export_drift_analysis_to_file()
    analyzer.export_drift_analysis_plots()

    metadata = {
        "version": 1,
        "reference_policy": "all_previous_base_observations",
        "reference_dataset_sha256": reference.dataset_sha256,
        "incoming_dataset_sha256": incoming.dataset_sha256,
        "feature_names": reference.feature_names,
        "class_names": reference.class_names,
        "representations": {
            "marginal": "canonical_raw_float64",
            "multivariate": "frozen_model_transform_float64",
        },
        "model_cycle": cycle_num,
        "training_partition_sha256": state.partition_sha256,
        "scaler_type": state.scaler_type,
        **bundle,
        "configuration": config,
        "effective_early_stopping": analyzer.early_stopping,
        "effective_gamma": analyzer.gamma,
        "wasserstein_method": analyzer.wasserstein_method,
    }
    (output / "drift_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    """Parse arguments and run drift analysis for a specified cycle.

    This function handles argument parsing, validation, parameter loading,
    and initiates drift analysis for the given cycle.
    """
    parser = argparse.ArgumentParser(description="Analyze drift for a given cycle.")
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")
    # Load parameters and paths
    params = load_params()
    paths = get_cycle_paths(cycle_num)
    # Run drift analysis
    with LogTime(task_name=f"\nDrift Analysis cycle{cycle_num}"):
        run_analyze_drift(paths, params, cycle_num)


if __name__ == "__main__":
    main()
