import argparse
import os
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

from src.utilities.data_utils import (
    construct_graph,
    generate_mineral_data,
    get_existing_data_bounds,
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

    print(f"Generating prediction data for cycle {cycle_num}")
    # Get bounds from existing training data
    try:
        x_range, y_range, existing_data = get_existing_data_bounds(
            cycle_num,
            base_path=paths["base_data"],
            combined_data_path=paths["prev_combined"],
        )
    except FileNotFoundError as e:
        return RuntimeError(f"Missing input data for cycle {cycle_num}: {e!r}")

    # Generate new data
    coordinates, features, labels = generate_mineral_data(
        n_samples=pred_params["n_samples"],
        spacing=pred_params["spacing"],
        depth=pred_params["depth"],
        n_features=base_params["n_features"],
        n_classes=base_params["n_classes"],
        threshold_binary=base_params["threshold_binary"],
        min_samples_per_class=pred_params["min_samples_per_class"],
        x_range=x_range,
        y_range=y_range,
        n_hotspots=pred_params["n_hotspots"],
        n_hotspots_random=pred_params["n_hotspots_random"],
        seed=pred_params["seed"],
    )

    # Verify no overlap with existing data
    if no_coordinate_overlap(
        existing_data.coordinates, torch.tensor(coordinates, dtype=torch.float32)
    ):
        print(f"✓ Cycle {cycle_num}: No overlap detected with existing data")

    # Construct and save graph
    pred_data = construct_graph(
        coordinates,
        features,
        labels,
        connection_radius=pred_params["connection_radius"],
        should_split=False,
    )
    if type(pred_data) is Data:
        output_file = os.path.join(paths["output"], "pred_data.pt")
        torch.save(pred_data, output_file)
        print(
            f"✓ Cycle {cycle_num}: Prediction data with {pred_data.x.shape[0]} samples saved to {output_file}"
        )


if __name__ == "__main__":
    main()
