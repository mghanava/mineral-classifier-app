import os
import yaml
import torch

from src.utilities.data_utils import (
    construct_graph,
    export_all_graphs_to_html,
    generate_mineral_data,
)

def get_existing_data_bounds(cycle_num):
    """Get coordinate bounds from existing training data to avoid overlap."""
    base_data_file = os.path.join(dataset_path, "base_data.pt")
    if cycle_num == 1 and os.path.exists(base_data_file):
        # First cycle: use base_data bounds
        existing_data = torch.load(base_data_file, weights_only=False)
    else:
        # Subsequent cycles: use combined training data from previous cycle
        prev_training_file = os.path.join(
            training_data_path, f"training_data_cycle_{cycle_num - 1}.pt"
        )

        if os.path.exists(prev_training_file):
            existing_data = torch.load(prev_training_file, weights_only=False)
        else:
            # Fallback to base_data if combined data doesn't exist yet
            existing_data = torch.load(base_data_file, weights_only=False)

    # Extract coordinates (assuming columns are named 'x', 'y' or similar)
    coordinates = existing_data[["x", "y"]].values  # Adjust column names as needed
    x_coords, y_coords = coordinates[:, 0], coordinates[:, 1]

    x_range = (x_coords.min(), x_coords.max())
    y_range = (y_coords.min(), y_coords.max())

    return x_range, y_range, existing_data


dataset_path = "results/data/base"
training_data_path = "resultls/data/combined"

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
        # Load parameters from YAML file
        N_SAMPLES = params["data"]["n_samples"]
        SPACING = params["data"]["spacing"]
        DEPTH = params["data"]["depth"]
        N_FEATURES = params["data"]["n_features"]
        N_CLASSES = params["data"]["n_classes"]
        CLASS_NAMES = params["evaluate"]["class_names"]
        THRESHOLD_BINARY = params["data"]["threshold_binary"]
        MIN_SAMPLES_PER_ClASS = params["data"]["min_samples_per_class"]
        CONNECTION_RADIUS = params["data"]["connection_radius"]
        N_SPLITS = params["data"]["n_splits"]
        TEST_SIZE = params["data"]["test_size"]
        CALIB_SIZE = params["data"]["calib_size"]
        SEED = params["data"]["seed"]

# torch.save(pred_data, os.path.join(dataset_path, "pred_data.pt"))
