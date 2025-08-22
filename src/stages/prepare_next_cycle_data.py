"""Module for generating synthetic mineral exploration data and graph datasets.

This module provides functionality to:
- Generate synthetic mineral exploration data
- Construct graph datasets from the synthetic data
- Export interactive visualizations of the generated data
- Save the generated datasets for model training
"""

import argparse
import os
import traceback
from pathlib import Path

import torch
import yaml

from src.utilities.data_utils import (
    construct_graph,
    export_all_graphs_to_html,
    export_graph_to_html,
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
    """Generate all paths for a specific cycle."""
    return {
        "combined_data": f"results/data/combined/cycle_{cycle_num}",
        "output": f"results/data/base/cycle_{cycle_num}",
    }


def prepare_base_data(cycle_num: int, paths: dict, params: dict):
    """Prepare base data for the next cycle.

    Args:
        cycle_num: Current cycle number
        paths: Dictionary of paths for the current cycle
        params: Dictionary of parameters from params.yaml

    """
    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")

    # Prepare output directory
    output_path = ensure_directory_exists(paths["output"])

    # Load parameters
    base_params = params["data"]["base"]
    eval_params = params["evaluate"]

    # Handle class names and labels
    class_names = eval_params["class_names"] or [
        f"Class {i}" for i in range(base_params["n_classes"])
    ]
    labels_map = dict(zip(range(len(class_names)), class_names, strict=True))

    combined_file = os.path.join(paths["combined_data"], "combined_data.pt")
    if not os.path.exists(combined_file):
        raise FileNotFoundError(
            f"Previous combined data not found at {combined_file}\n"
            "Please run the combine_data stage for the previous cycle first."
        )
    combined_data = torch.load(combined_file, weights_only=False)
    coordinates = combined_data.coordinates.numpy()
    features = combined_data.x.numpy()
    labels = combined_data.y.numpy()

    # Construct graph data splits
    print(f"Preparing base data for cycle {cycle_num}")
    graph_data = construct_graph(
        coordinates,
        features,
        labels,
        connection_radius=base_params["connection_radius"],
        add_self_loops=base_params["add_self_loops"],
        n_splits=base_params["n_splits"],
        test_size=base_params["test_size"],
        calib_size=base_params["calib_size"],
        seed=base_params["seed"],
    )

    if not isinstance(graph_data, tuple):
        raise ValueError(
            "Expected construct_graph to return a tuple when should_split is True"
        )

    base_data, fold_data, test_data = graph_data

    # Export visualizations
    print("Exporting 3D interactive plots of graphs ...")
    export_graph_to_html(
        base_data,
        coordinates,
        None,
        base_params["connection_radius"],
        base_params["add_self_loops"],
        output_path,
        labels_map,
        dataset_tag="base_data",
    )
    export_all_graphs_to_html(
        fold_data,
        test_data,
        coordinates,
        base_params["connection_radius"],
        base_params["add_self_loops"],
        labels_map,
        save_path=output_path,
    )

    # Save data files
    print(
        f"Base data to be used in cycle {cycle_num + 1} has {base_data.x.shape[0]} samples."
    )
    print(f"Saving data files for cycle {cycle_num}")
    try:
        torch.save(base_data, os.path.join(output_path, "base_data.pt"))
        torch.save(fold_data, os.path.join(output_path, "fold_data.pt"))
        torch.save(test_data, os.path.join(output_path, "test_data.pt"))
        print(f"✓ Data saved to {paths['output']}")
    except Exception as e:
        print(f"Error saving files: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Prepare base data for the next cycle."
    )
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Load parameters and paths
    params = load_params()
    paths = get_cycle_paths(cycle_num)

    prepare_base_data(cycle_num=cycle_num, paths=paths, params=params)


if __name__ == "__main__":
    main()
