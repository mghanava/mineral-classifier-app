"""Generate synthetic mineral exploration data and create graph datasets.

This module provides functions to:
- Generate synthetic mineral exploration data with realistic features
- Scale and preprocess the generated data
- Construct graph datasets for machine learning
- Export interactive 3D visualizations of the graphs
"""

import os

import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.utilities.utils import (
    construct_graph,
    export_all_graphs_to_html,
    generate_mineral_data,
    scale_data,
)


def main():
    """Execute the main data generation workflow.

    Reads parameters from params.yaml, generates synthetic mineral exploration data,
    constructs graph datasets, and exports interactive visualizations.
    """
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    # Load parameters from YAML file
    N_SAMPLES = params["data"]["n_samples"]
    SPACING = params["data"]["spacing"]
    DEPTH = params["data"]["depth"]
    N_FEATURES = params["data"]["n_features"]
    N_CLASSES = params["data"]["n_classes"]
    THRESHOLD_BINARY = params["data"]["threshold_binary"]
    MIN_SAMPLES_PER_ClASS = params["data"]["min_samples_per_class"]
    CONNECTION_RADIUS = params["data"]["connection_radius"]
    N_SPLITS = params["data"]["n_splits"]
    TEST_SIZE = params["data"]["test_size"]
    CALIB_SIZE = params["data"]["calib_size"]
    SEED = params["data"]["seed"]

    # Generate synthetic data
    coordinates, features, labels = generate_mineral_data(
        n_samples=N_SAMPLES,
        spacing=SPACING,
        depth=DEPTH,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        threshold_binary=THRESHOLD_BINARY,
        min_samples_per_class=MIN_SAMPLES_PER_ClASS,
        seed=SEED,
    )
    fold_data, test_data = construct_graph(
        coordinates,
        scale_data(data=features, scaler=StandardScaler()),
        labels,
        connection_radius=CONNECTION_RADIUS,
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        calib_size=CALIB_SIZE,
        seed=SEED,
    )
    # Save the generated data
    dataset_path = "results/data"
    os.makedirs(dataset_path, exist_ok=True)
    torch.save(fold_data, os.path.join(dataset_path, "fold_data.pt"))
    torch.save(test_data, os.path.join(dataset_path, "test_data.pt"))
    # export the interactive 3D plots
    export_all_graphs_to_html(
        fold_data, test_data, coordinates, CONNECTION_RADIUS, dataset_path
    )


if __name__ == "__main__":
    main()
