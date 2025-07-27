"""Module for generating synthetic mineral exploration data and graph datasets.

This module provides functionality to:
- Generate synthetic mineral exploration data
- Construct graph datasets from the synthetic data
- Export interactive visualizations of the generated data
- Save the generated datasets for model training
"""

import argparse
import os

import torch
import yaml

from src.utilities.data_utils import (
    construct_graph,
    export_all_graphs_to_html,
    generate_mineral_data,
)


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def _prepare_base_data(
    cycle_num: int,
    base_output_path: str,
    prev_combined_data_path: str,
    n_samples: int,
    spacing: float,
    depth: float,
    n_features: int,
    n_classes: int,
    labels_map: dict,
    threshold_binary: float,
    min_samples_per_class: int,
    n_hotspots: int,
    n_hotspots_random: bool,
    connection_radius: float,
    n_splits: int,
    test_size: float,
    calib_size: float,
    seed: int,
):
    # Ensure the output directory exists
    os.makedirs(base_output_path, exist_ok=True)

    if cycle_num == 1:
        print("Cycle 1: Generating new synthetic data.")
        # Generate synthetic data
        coordinates, features, labels = generate_mineral_data(
            n_samples=n_samples,
            spacing=spacing,
            depth=depth,
            n_features=n_features,
            n_classes=n_classes,
            threshold_binary=threshold_binary,
            min_samples_per_class=min_samples_per_class,
            x_range=None,
            y_range=None,
            n_hotspots=n_hotspots,
            n_hotspots_random=n_hotspots_random,
            seed=seed,
        )
    else:
        # Load previous combined data (from cycle N-1)
        print(f"Cycle {cycle_num}: Loading data from previous cycle.")
        prev_combined_file = os.path.join(
            prev_combined_data_path, "combined_data.pt"
        )
        if not os.path.exists(prev_combined_file):
            raise FileNotFoundError(
                f"Previous combined data not found: {prev_combined_file}\n"
                "Please ensure the 'combine_data' stage for the previous cycle has been run."
            )
        combined_data = torch.load(prev_combined_file, weights_only=False)
        coordinates = combined_data.coordinates.numpy()
        features = combined_data.x.numpy()
        labels = combined_data.y.numpy()

    all_data = construct_graph(
        coordinates,
        features,
        labels,
        connection_radius=connection_radius,
        n_splits=n_splits,
        test_size=test_size,
        calib_size=calib_size,
        seed=seed,
    )
    # When should_split is True, construct_graph returns a tuple (all_data, fold_data, test_data)
    # Ensure the returned value is a tuple and unpack accordingly
    if not isinstance(all_data, tuple):
        raise ValueError(
            "Expected construct_graph to return a tuple when should_split is True."
        )
    base_data, fold_data, test_data = all_data

    # export the interactive 3D plots
    export_all_graphs_to_html(
        fold_data,
        test_data,
        coordinates,
        connection_radius,
        labels_map,
        # cycle_num,
        base_output_path,  # Save plots in the cycle-specific output directory
    )

    # Save the generated data with generic names in the cycle-specific directory
    print(f"Saving generated data to {base_output_path}!")
    try:
        torch.save(base_data, os.path.join(base_output_path, "base_data.pt"))
        torch.save(fold_data, os.path.join(base_output_path, "fold_data.pt"))
        torch.save(test_data, os.path.join(base_output_path, "test_data.pt"))
        print("All files saved successfully")
    except Exception as e:
        print(f"Error saving files: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Execute the main data generation workflow.

    Reads parameters, accepts a cycle number, generates synthetic mineral
    exploration data, constructs graph datasets, and saves outputs to a
    cycle-specific directory.
    """
    parser = argparse.ArgumentParser(
        description="Generate base data for a given cycle."
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Current cycle number"
    )
    args = parser.parse_args()
    cycle_num = args.cycle

    params = load_params()
    # Load parameters from YAML file
    N_SAMPLES = params["data"]["base"]["n_samples"]
    SPACING = params["data"]["base"]["spacing"]
    DEPTH = params["data"]["base"]["depth"]
    N_FEATURES = params["data"]["base"]["n_features"]
    N_CLASSES = params["data"]["base"]["n_classes"]
    CLASS_NAMES = params["evaluate"]["class_names"]
    THRESHOLD_BINARY = params["data"]["base"]["threshold_binary"]
    MIN_SAMPLES_PER_ClASS = params["data"]["base"]["min_samples_per_class"]
    CONNECTION_RADIUS = params["data"]["base"]["connection_radius"]
    N_SPLITS = params["data"]["base"]["n_splits"]
    TEST_SIZE = params["data"]["base"]["test_size"]
    CALIB_SIZE = params["data"]["base"]["calib_size"]
    N_HOTSPOTS = params["data"]["base"]["n_hotspots"]
    N_HOTSPOTS_RANDOM = params["data"]["base"]["n_hotspots_random"]
    SEED = params["data"]["base"]["seed"]

    if CLASS_NAMES is None:
        CLASS_NAMES = [f"Class {i}" for i in range(N_CLASSES)]
    LABELS_MAP = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES, strict=True))

    # Define DVC-managed output and dependency paths based on the cycle
    base_output_path = f"results/data/base/cycle_{cycle_num}"
    # Path to the *previous* cycle's combined data directory
    prev_combined_data_path = f"results/data/combined/cycle_{cycle_num - 1}"

    _prepare_base_data(
        cycle_num,
        base_output_path,
        prev_combined_data_path,
        N_SAMPLES,
        SPACING,
        DEPTH,
        N_FEATURES,
        N_CLASSES,
        LABELS_MAP,
        THRESHOLD_BINARY,
        MIN_SAMPLES_PER_ClASS,
        N_HOTSPOTS,
        N_HOTSPOTS_RANDOM,
        CONNECTION_RADIUS,
        N_SPLITS,
        TEST_SIZE,
        CALIB_SIZE,
        SEED,
    )


if __name__ == "__main__":
    main()