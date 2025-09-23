"""Module for generating synthetic mineral exploration data and graph datasets.

This module provides functionality to:
- Generate synthetic mineral exploration data
- Construct graph datasets from the synthetic data
- Export interactive visualizations of the generated data
- Save the generated datasets for model training and evaluation
"""

import argparse
import os

from src.utilities.data_utils import (
    analyze_feature_discrimination,
    construct_graph,
    export_all_graphs_to_html,
    export_graph_to_html,
    generate_mineral_data,
    scaler_setup,
)
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_params,
    save_data,
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
    coordinates, features, labels, hotspots, prototypes = generate_mineral_data(
        radius=base_params["radius"],
        depth=base_params.get("depth", -500),
        n_samples=base_params.get("n_samples", 1000),
        spacing=base_params.get("spacing", 10),
        n_features=base_params.get("n_features", 5),
        n_classes=base_params.get("n_classes", 2),
        threshold_binary=base_params.get("threshold_binary", 0.3),
        min_samples_per_class=base_params.get("min_samples_per_class", 50),
        n_hotspots_max=base_params.get("n_hotspots_max", 5),
        class_separation=base_params.get("class_separation", 2.0),
        class_influence=base_params.get("class_influence", 0.3),
        spatial_weight=base_params.get("spatial_weight", 0.5),
        interaction_strength=base_params.get("interaction_strength", 0.1),
        correlation_strength=base_params.get("correlation_strength", 0.2),
        feature_weight_range=base_params.get("feature_weight_range", [0.5, 2.0]),
        feature_offset_std=base_params.get("feature_offset_std", 0.3),
        mineral_noise_level=base_params.get("mineral_noise_level", 0.1),
        exp_decay_factor=base_params.get("exp_decay_factor", 0.01),
        feature_noise_level=base_params.get("feature_noise_level", 0.5),
        seed=base_params.get("seed", 42),
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
        test_size=base_params.get("test_size", None),
        calib_size=base_params.get("calib_size", None),
        seed=base_params.get("seed", 42),
        scaler=scaler_setup(params),
        make_edge_weight=params["data"].get("make_edge_weight", True),
        make_edge_weight_method=params["data"].get("make_edge_weight_method", None),
    )
    # When should_split is True, construct_graph returns a tuple (all_data, fold_data, test_data). Ensure the returned value is a tuple and unpack accordingly
    if not isinstance(all_data, tuple):
        raise ValueError(
            "Expected construct_graph to return a tuple when should_split is True."
        )
    base_data, fold_data, test_data, calib_data = all_data

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
    # Convert fold_data to list of tuples
    export_all_graphs_to_html(
        fold_data,
        test_data,
        calib_data,
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
    save_data(base_data, os.path.join(output_path, "base_data.pt"), "Base")
    save_data(
        fold_data, os.path.join(output_path, "fold_data.pt"), "Train-Validation Fold"
    )
    save_data(test_data, os.path.join(output_path, "test_data.pt"), "Test")
    save_data(calib_data, os.path.join(output_path, "calib_data.pt"), "Calibration")
    save_data(hotspots, os.path.join(output_path, "hotspots.npy"), "Hotspots", True)
    save_data(
        prototypes, os.path.join(output_path, "prototypes.npy"), "Prototypes", True
    )


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
