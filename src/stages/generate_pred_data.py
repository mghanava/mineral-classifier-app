"""Generate prediction data for a given cycle.

This module provides functionality to generate prediction data, ensuring directory structure and no coordinate overlap.
"""

import argparse
import os

import torch
from torch_geometric.data import Data

from src.utilities.data_utils import (
    construct_graph,
    generate_mineral_data,
    no_coordinate_overlap,
)
from src.utilities.general_utils import LogTime, ensure_directory_exists, load_params


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle with directory creation."""
    return {
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

    base_path = paths["base_data"]
    # Ensure output directory exists
    output_path = ensure_directory_exists(paths["output"])

    exisiting_coordinates = torch.load(
        os.path.join(base_path, "base_data.pt"), weights_only=False
    ).coordinates.numpy()

    # Generate new data
    coordinates, features, labels = generate_mineral_data(
        radius=base_params["radius"],
        depth=base_params["depth"],
        n_samples=pred_params["n_samples"],
        spacing=base_params["spacing"],
        existing_points=exisiting_coordinates,
        n_features=base_params["n_features"],
        n_classes=base_params["n_classes"],
        threshold_binary=base_params["threshold_binary"],
        min_samples_per_class=pred_params["min_samples_per_class"],
        n_hotspots=base_params["n_hotspots"],
        n_hotspots_random=base_params["n_hotspots_random"],
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
        connection_radius=base_params["connection_radius"],
        add_self_loops=base_params["add_self_loops"],
        should_split=False,
    )
    if type(pred_data) is Data and pred_data.x is not None:
        output_file = os.path.join(output_path, "pred_data.pt")
        torch.save(pred_data, output_file)
        print(
            f"✓ Prediction data with {pred_data.x.shape[0]} samples saved to {output_file}."
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
