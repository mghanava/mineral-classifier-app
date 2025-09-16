"""Module for generating synthetic mineral exploration data and graph datasets.

This module provides functionality to:
- Generate synthetic mineral exploration data
- Construct graph datasets from the synthetic data
- Export interactive visualizations of the generated data
- Save the generated datasets for model training and evaluation
"""

import argparse
import os

import torch

from src.utilities.data_utils import (
    analyze_feature_discrimination,
    # check_isolated_components,
    construct_graph,
    # diagnose_data_leakage,
    export_all_graphs_to_html,
    export_graph_to_html,
    generate_mineral_data,
    scaler_setup,
)
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_params,
)


def prepare_base_data(
    path: str,
    params: dict,
):
    """Prepare base data for bootstrap cycle.

    Generates synthetic mineral exploration data, constructs graph datasets, and saves outputs to cycle-0 directory

    Args:
        path: base data path for the bootstrap cycle
        params: Dictionary of parameters from params.yaml

    """
    # Prepare output directory
    output_path = ensure_directory_exists(path)
    # load parameters
    base_params = params["data"]["base"]
    base_params["add_self_loops"] = params.get("add_self_loops", True)

    # Handle class names and labels
    class_names = base_params["class_names"] or [
        f"Class {i}" for i in range(base_params["n_classes"])
    ]
    labels_map = dict(zip(range(len(class_names)), class_names, strict=True))
    # Generate synthetic data
    coordinates, features, labels = generate_mineral_data(
        radius=base_params["radius"],
        depth=base_params["depth"],
        spacing=base_params["spacing"],
        n_samples=base_params["n_samples"],
        n_features=base_params["n_features"],
        n_classes=base_params["n_classes"],
        threshold_binary=base_params["threshold_binary"],
        min_samples_per_class=base_params["min_samples_per_class"],
        n_hotspots=base_params["n_hotspots"],
        n_hotspots_random=base_params["n_hotspots_random"],
        seed=base_params["seed"],
    )

    all_data = construct_graph(
        coordinates,
        features,
        labels,
        k_nearest=base_params.get("k_nearest", None),
        connection_radius=base_params.get("connection_radius", None),
        distance_percentile=base_params.get("distance_percentile", None),
        add_self_loops=base_params["add_self_loops"],
        n_splits=base_params["n_splits"],
        test_size=base_params["test_size"],
        calib_size=base_params["calib_size"],
        seed=base_params["seed"],
        scaler=scaler_setup(params),
        make_edge_weight=params["data"].get("make_edge_weight", True),
        make_edge_weight_method=params["data"].get("make_edge_weight_method", None),
    )
    # When should_split is True, construct_graph returns a tuple (all_data, fold_data, test_data). Ensure the returned value is a tuple and unpack accordingly
    if not isinstance(all_data, tuple):
        raise ValueError(
            "Expected construct_graph to return a tuple when should_split is True."
        )
    base_data, fold_data, test_data = all_data

    # diagnose_data_leakage(fold_data, test_data)
    # check_isolated_components(fold_data, test_data)

    # export the interactive 3D plots
    print("\nExporting 3D interactive plots of graphs ...")
    export_graph_to_html(
        base_data,
        coordinates,
        None,
        base_params["k_nearest"],
        base_params["connection_radius"],
        base_params["distance_percentile"],
        base_params["add_self_loops"],
        output_path,
        labels_map,
        dataset_tag="base_data",
    )
    export_all_graphs_to_html(
        fold_data,
        test_data,
        coordinates,
        base_params["k_nearest"],
        base_params["connection_radius"],
        base_params["distance_percentile"],
        base_params["add_self_loops"],
        labels_map,
        output_path,
    )

    analyze_feature_discrimination(
        features,
        labels,
        save_path=os.path.join(output_path, "tsne_comparison.png"),
        class_names=class_names,
    )

    # Save the datasets
    try:
        torch.save(base_data, os.path.join(output_path, "base_data.pt"))
        torch.save(fold_data, os.path.join(output_path, "fold_data.pt"))
        torch.save(test_data, os.path.join(output_path, "test_data.pt"))
        print(f"All files successfully saved to {output_path}.")
    except Exception as e:
        print(f"Error saving files: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Execute the main data generation workflow."""
    parser = argparse.ArgumentParser(description="Generate base data for cycle 0.")
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num != 0:
        raise ValueError("For the bootstrap stage cycle number must be 0!")
    # Load parameters from YAML file
    params = load_params()
    base_output_path = "results/data/base/cycle_0"
    with LogTime(task_name="\nInitial base data generation"):
        prepare_base_data(base_output_path, params)


if __name__ == "__main__":
    main()
