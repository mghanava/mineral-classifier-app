import argparse
import os
from pathlib import Path

import numpy as np
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
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",
        "prediction": f"results/data/prediction/cycle_{cycle_num}",
        "output": f"results/data/base/cycle_{cycle_num}",
    }


def combine_split_data(
    paths: dict,
    params: dict,
):
    base_params = params["data"]["base"]
    # Handle class names and labels
    class_names = base_params["class_names"] or [
        f"Class {i}" for i in range(base_params["n_classes"])
    ]
    labels_map = dict(zip(range(len(class_names)), class_names, strict=True))

    base_path = paths["base_data"]
    pred_path = paths["prediction"]
    output_path = ensure_directory_exists(paths["output"])

    # Determine data source
    base_data_path = os.path.join(base_path, "base_data.pt")
    if not os.path.exists(base_data_path):
        raise FileNotFoundError(f"Base data not found: {base_data_path}")
    print(f"📥 Loading base data from: {base_data_path}")
    base_data = torch.load(base_data_path, weights_only=False)
    # Load prediction data
    pred_data_path = os.path.join(pred_path, "pred_data.pt")
    if not os.path.exists(pred_data_path):
        raise FileNotFoundError(f"Prediction data not found: {pred_data_path}")
    print(f"📥 Loading prediction data from: {pred_data_path}")
    pred_data = torch.load(pred_data_path, weights_only=False)

    print("📊 Data sizes before combination:")
    print(f"   Base/Previous: {base_data.x.shape[0]} samples")
    print(f"   Prediction: {pred_data.x.shape[0]} samples")
    combined_size = base_data.x.shape[0] + pred_data.x.shape[0]
    print(f"   Total: {combined_size} samples")
    # Combine datasets
    coordinates = np.concatenate(
        (base_data.coordinates.numpy(), pred_data.coordinates.numpy()), axis=0
    )
    features = np.concatenate((base_data.x, pred_data.x), axis=0)
    labels = np.concatenate((base_data.y, pred_data.y), axis=0)

    if combined_size > 3000:
        print("🎯 Applying reservoir sampling to connected graph...")
        pass  # placeholder for reservoir sampling logic

    all_graph_data = construct_graph(
        coordinates,
        features,
        labels,
        connection_radius=base_params["connection_radius"],
        add_self_loops=base_params["add_self_loops"],
        n_splits=base_params["n_splits"],
        test_size=base_params["test_size"],
        calib_size=base_params["calib_size"],
        seed=base_params["seed"],
        scaler=None,  # No scaling needed for combined data
    )
    # Ensure the returned value is a tuple and unpack accordingly
    if not isinstance(all_graph_data, tuple):
        raise ValueError(
            "Expected construct_graph to return a tuple when should_split is True."
        )
    base_data, fold_data, test_data = all_graph_data
    # Export visualizations
    print("\nExporting 3D interactive plots of graphs ...")
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
    try:
        torch.save(base_data, os.path.join(output_path, "base_data.pt"))
        torch.save(fold_data, os.path.join(output_path, "fold_data.pt"))
        torch.save(test_data, os.path.join(output_path, "test_data.pt"))
        print(f"✓ All files successfully saved to {paths['output']}.\n")
    except Exception as e:
        print(f"Error saving files: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare base data for the next cycle."
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
    combine_split_data(paths=paths, params=params)


if __name__ == "__main__":
    main()
