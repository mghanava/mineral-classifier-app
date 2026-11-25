"""Module for preparing base data for the next cycle in the pipeline.

This script combines base and prediction data, constructs graphs, exports visualizations, and saves processed data for subsequent cycles.
"""

import argparse
import os

import numpy as np

from src.utilities.data_utils import (
    analyze_feature_discrimination,
    export_all_graphs_to_html,
    export_graph_to_html,
)
from src.utilities.general_utils import (
    LogTime,
    ensure_directory_exists,
    load_data,
    load_params,
    save_data,
)
from src.utilities.graph_protocol import resolve_graph_protocol, verify_graph
from src.utilities.preprocessing import RawDataset
from src.utilities.raw_graphs import (
    build_cycle_graphs,
    read_raw_graph,
    synthetic_schema,
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
    protocol = resolve_graph_protocol(params)
    # Handle class names and labels
    class_names = base_params["class_names"] or [
        f"Class {i}" for i in range(base_params["n_classes"])
    ]
    labels_map = dict(zip(range(len(class_names)), class_names, strict=True))

    base_path = paths["base_data"]
    pred_path = paths["prediction"]
    output_path = ensure_directory_exists(paths["output"])

    base_data = load_data(os.path.join(base_path, "base_data.pt"), "Base")
    pred_data = load_data(os.path.join(pred_path, "pred_data.pt"), "Prediction")

    base = read_raw_graph(base_data)
    incoming = read_raw_graph(pred_data)
    if getattr(pred_data, "source_parent_sha256", None) != base.dataset_sha256:
        raise ValueError(
            "Prediction observations do not match this base cycle. Regenerate the batch."
        )
    expected_schema = synthetic_schema(base_params)
    if any(
        (raw.feature_names, raw.class_names) != expected_schema
        for raw in (base, incoming)
    ):
        raise ValueError(
            "Next-cycle inputs have incompatible feature order or class mapping."
        )

    verify_graph(base_data, protocol, "overview")
    verify_graph(pred_data, protocol, "prediction")
    raw = RawDataset(
        features=np.concatenate((base.features, incoming.features)),
        coordinates=np.concatenate((base.coordinates, incoming.coordinates)),
        labels=np.concatenate((base.labels, incoming.labels)),
        feature_names=base.feature_names,
        class_names=base.class_names,
        sample_ids=base.sample_ids + incoming.sample_ids,
    )
    base_data, fold_data, test_data, calib_data = build_cycle_graphs(
        raw, protocol, base_params
    )
    export_graph_to_html(base_data, output_path, labels_map, dataset_tag="base_data")
    export_all_graphs_to_html(fold_data, test_data, calib_data, labels_map, output_path)

    analyze_feature_discrimination(
        raw.features,
        raw.labels,
        save_path=os.path.join(output_path, "tsne_comparison.png"),
        class_names=class_names,
        scaler=None,
    )

    # Save data files
    save_data(base_data, os.path.join(output_path, "base_data.pt"), "Base")
    save_data(
        fold_data, os.path.join(output_path, "fold_data.pt"), "Train-Validation Fold"
    )
    save_data(test_data, os.path.join(output_path, "test_data.pt"), "Test")
    save_data(calib_data, os.path.join(output_path, "calib_data.pt"), "Calibration")


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
