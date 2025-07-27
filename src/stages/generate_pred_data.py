import argparse
import os

import torch
import yaml

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


def main():
    """Generate prediction data for a given cycle."""
    parser = argparse.ArgumentParser(
        description="Generate prediction data for a given cycle."
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="Current cycle number"
    )
    args = parser.parse_args()
    cycle_num = args.cycle

    params = load_params()
    N_SAMPLES = params["data"]["pred"]["n_samples"]
    SPACING = params["data"]["pred"]["spacing"]
    DEPTH = params["data"]["pred"]["depth"]
    N_FEATURES = params["data"]["pred"]["n_features"]
    N_CLASSES = params["data"]["pred"]["n_classes"]
    THRESHOLD_BINARY = params["data"]["pred"]["threshold_binary"]
    MIN_SAMPLES_PER_ClASS = params["data"]["pred"]["min_samples_per_class"]
    CONNECTION_RADIUS = params["data"]["pred"]["connection_radius"]
    N_HOTSPOTS = params["data"]["pred"]["n_hotspots"]
    N_HOTSPOTS_RANDOM = params["data"]["pred"]["n_hotspots_random"]
    SEED = params["data"]["pred"]["seed"]

    base_data_path = f"results/data/base/cycle_{cycle_num}"
    prediction_data_path = f"results/data/prediction/cycle_{cycle_num}"
    prev_combined_data_path = f"results/data/combined/cycle_{cycle_num - 1}"

    os.makedirs(prediction_data_path, exist_ok=True)

    print(f"Generating prediction data for cycle {cycle_num}")
    # Get bounds from existing training data
    x_range, y_range, existing_data = get_existing_data_bounds(
        cycle_num, base_data_path, prev_combined_data_path
    )
    coordinates, features, labels = generate_mineral_data(
        n_samples=N_SAMPLES,
        spacing=SPACING,
        depth=DEPTH,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        threshold_binary=THRESHOLD_BINARY,
        min_samples_per_class=MIN_SAMPLES_PER_ClASS,
        x_range=x_range,
        y_range=y_range,
        n_hotspots=N_HOTSPOTS,
        n_hotspots_random=N_HOTSPOTS_RANDOM,
        seed=SEED,
    )

    if no_coordinate_overlap(
        existing_data.coordinates, torch.tensor(coordinates, dtype=torch.float32)
    ):
        print("✓ No overlap between existing data and new prediction data detected!")

    pred_data = construct_graph(
        coordinates,
        features,
        labels,
        connection_radius=CONNECTION_RADIUS,
        should_split=False,
    )
    torch.save(
        pred_data, os.path.join(prediction_data_path, "pred_data.pt")
    )


if __name__ == "__main__":
    main()