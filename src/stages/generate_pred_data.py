import argparse
import os
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

from src.utilities.data_utils import (
    construct_graph,
    generate_mineral_data,
    no_coordinate_overlap,
)


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_cycle_paths(cycle_num):
    """Generate all paths for a specific cycle with directory creation."""
    paths = {
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",  # Previous cycle
        "prev_combined": f"results/data/prev_combined/cycle_{cycle_num - 1}"
        if cycle_num > 1
        else None,
        "output": f"results/data/prediction/cycle_{cycle_num}",
    }

    # Ensure all output directories exist
    for key in ["base_data", "output"]:
        ensure_directory_exists(paths[key])

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction data for a given cycle."
    )
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")

    # Get and prepare all paths
    paths = get_cycle_paths(cycle_num)
    print(f"✓ Cycle {cycle_num}: Ensuring directory structure exists")

    # Load parameters
    params = load_params()
    base_params = params["data"]["base"]
    pred_params = params["data"]["pred"]

    exisiting_coordinates = torch.load(
        os.path.join(paths["base_data"], "base_data.pt"), weights_only=False
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
        output_file = os.path.join(paths["output"], "pred_data.pt")
        torch.save(pred_data, output_file)
        print(
            f"✓ Prediction data with {pred_data.x.shape[0]} samples saved to {output_file}.\n"
        )


if __name__ == "__main__":
    main()
