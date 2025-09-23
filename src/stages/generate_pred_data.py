"""Generate prediction data for a given cycle.

This module provides functionality to generate prediction data, ensuring directory structure and no coordinate overlap.
"""

import argparse
import os

import torch

from src.utilities.data_utils import (
    construct_graph,
    export_graph_to_html,
    generate_mineral_data,
    no_coordinate_overlap,
    scaler_setup,
)
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_data,
    load_params,
    save_data,
)


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle with directory creation."""
    return {
        "bootstrap_data": "results/data/base/cycle_0",
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",
        "output": f"results/data/prediction/cycle_{cycle_num}",
    }


def prepare_pred_data(paths: dict, params: dict):
    """Prepare prediction data for a given cycle.

    Args:
        paths: Dictionary containing paths for base data and output
        params: Dictionary of parameters from params.yaml

    """
    # load parameters
    base_params = params["data"]["base"]
    base_params["add_self_loops"] = params.get("add_self_loops", True)
    pred_params = params["data"]["pred"]

    bootstrap_path = paths["bootstrap_data"]
    base_path = paths["base_data"]
    # Ensure output directory exists
    output_path = ensure_directory_exists(paths["output"])

    # Load exisiting coordinates from previous cycle and hotspots and prototypes from bootstrap
    exisiting_coordinates = load_data(
        os.path.join(base_path, "base_data.pt"), "Base"
    ).coordinates.numpy()
    previous_hotspots = load_data(
        os.path.join(bootstrap_path, "hotspots.npy"), "Hotspots", load_numpy=True
    )
    previous_prototypes = load_data(
        os.path.join(bootstrap_path, "prototypes.npy"), "Prototypes", load_numpy=True
    )

    # Generate new data
    coordinates, features, labels, _, _ = generate_mineral_data(
        radius=base_params["radius"],
        depth=base_params.get("depth", -500),
        spacing=base_params.get("spacing", 10),
        n_features=base_params.get("n_features", 5),
        n_classes=base_params.get("n_classes", 2),
        threshold_binary=base_params.get("threshold_binary", 0.3),
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
        existing_sample_coords=exisiting_coordinates,
        existing_hotspots=previous_hotspots,
        hotspot_drift=base_params.get("hotspot_drift", 0.1),
        existing_prototypes=previous_prototypes,
        prototype_evolution_rate=base_params.get("prototype_evolution_rate", 0.3),
        n_samples=pred_params.get("n_samples", 200),
        min_samples_per_class=pred_params.get("min_samples_per_class", None),
        seed=pred_params["seed"],
    )

    # Verify no overlap with existing data
    if no_coordinate_overlap(
        exisiting_coordinates, torch.tensor(coordinates, dtype=torch.float32)
    ):
        print("✓ No overlap detected with existing data!")

    # Construct and save graph
    pred_data = construct_graph(
        coordinates,
        features,
        labels,
        k_nearest=base_params.get("k_nearest", None),
        connection_radius=base_params.get("connection_radius", None),
        distance_percentile=base_params.get("distance_percentile", None),
        add_self_loops=base_params["add_self_loops"],
        scaler=scaler_setup(params),
        should_split=False,
        make_edge_weight=params["data"].get("make_edge_weight", True),
        make_edge_weight_method=params["data"].get("make_edge_weight_method", None),
    )
    save_data(pred_data, os.path.join(output_path, "pred_data.pt"), "Prediction")
    # export the interactive 3D plots
    # Handle class names and labels
    class_names = base_params["class_names"] or [
        f"Class {i}" for i in range(base_params["n_classes"])
    ]
    labels_map = dict(zip(range(len(class_names)), class_names, strict=True))
    print("\nExporting 3D interactive plots of graphs ...")
    export_graph_to_html(
        pred_data,
        coordinates,
        None,
        base_params["k_nearest"],
        base_params["connection_radius"],
        base_params["distance_percentile"],
        base_params["add_self_loops"],
        output_path,
        labels_map,
        dataset_tag="prediction_data",
    )


def main():
    """Parse arguments and generate prediction data for a given cycle."""
    parser = argparse.ArgumentParser(
        description="Generate prediction data for a given cycle."
    )
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")

    # Load parameters and paths
    params = load_params()
    paths = get_cycle_paths(cycle_num)

    with LogTime(task_name=f"\nPrediction data generation for cycle {cycle_num}"):
        prepare_pred_data(paths=paths, params=params)


if __name__ == "__main__":
    main()
