"""Module for preparing base data for the next cycle in the pipeline.

This script combines base and prediction data, constructs graphs, exports visualizations,
and saves processed data for subsequent cycles.
"""

import argparse
import os

import numpy as np
import torch

from src.utilities.data_utils import (
    analyze_feature_discrimination,
    # check_isolated_components,
    construct_graph,
    # diagnose_data_leakage,
    export_all_graphs_to_html,
    export_graph_to_html,
)
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_params,
)


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
    """Combine base and prediction data for the next cycle, construct graphs, export visualizations, and save processed data.

    Parameters
    ----------
    paths : dict
        Dictionary containing paths for base data, prediction data, and output directory.
    params : dict
        Dictionary of configuration parameters for data processing.

    Returns
    -------
    None

    """
    base_params = params["data"]["base"]
    base_params["add_self_loops"] = params.get("add_self_loops", True)
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
        k_nearest=base_params.get("k_nearest", None),
        connection_radius=base_params.get("connection_radius", None),
        distance_percentile=base_params.get("distance_percentile", None),
        add_self_loops=base_params["add_self_loops"],
        n_splits=base_params["n_splits"],
        test_size=base_params["test_size"],
        calib_size=base_params["calib_size"],
        seed=base_params["seed"],
        scaler=None,  # No scaling needed for combined data
        make_edge_weight=params["data"].get("make_edge_weight", True),
        make_edge_weight_method=params["data"].get("make_edge_weight_method", None),
    )
    # Ensure the returned value is a tuple and unpack accordingly
    if not isinstance(all_graph_data, tuple):
        raise ValueError(
            "Expected construct_graph to return a tuple when should_split is True."
        )
    base_data, fold_data, test_data = all_graph_data

    # diagnose_data_leakage(fold_data, test_data)
    # check_isolated_components(fold_data, test_data)

    # Export visualizations
    print("\nExporting 3D interactive plots of graphs ...")
    export_graph_to_html(
        base_data,
        coordinates,
        None,
        base_params["k_nearest"],
        base_params["connection_radius"],
        base_params["distance_percentile"],
        base_params["add_self_loops"],
        output_path,
        labels_map,
        dataset_tag="base_data",
    )
    export_all_graphs_to_html(
        fold_data,
        test_data,
        coordinates,
        base_params["k_nearest"],
        base_params["connection_radius"],
        base_params["distance_percentile"],
        base_params["add_self_loops"],
        labels_map,
        output_path,
    )

    analyze_feature_discrimination(
        features,
        labels,
        save_path=os.path.join(output_path, "tsne_comparison.png"),
        class_names=class_names,
    )

    # Save data files
    try:
        torch.save(base_data, os.path.join(output_path, "base_data.pt"))
        torch.save(fold_data, os.path.join(output_path, "fold_data.pt"))
        torch.save(test_data, os.path.join(output_path, "test_data.pt"))
        print(f"✓ All files successfully saved to {paths['output']}.")
    except Exception as e:
        print(f"Error saving files: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Parse command-line arguments and prepare base data for the next cycle.

    This function loads configuration parameters, validates the cycle number,
    constructs paths, and combines split data for the next cycle.
    """
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

    with LogTime(task_name=f"\nBase data generation for cycle {cycle_num}"):
        combine_split_data(paths=paths, params=params)


if __name__ == "__main__":
    main()
