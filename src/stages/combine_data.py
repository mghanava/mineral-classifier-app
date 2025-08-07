import argparse
import os
from pathlib import Path
from typing import Literal

import torch
import yaml

from src.utilities.data_utils import connect_graphs_preserve_weights


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
        "base_data": f"results/data/base/cycle_{cycle_num - 1}",  # Previous cycle's base data
        "previous_combined": f"results/data/combined/cycle_{cycle_num - 1}",  # Previous cycle's combined data
        "prediction": f"results/data/prediction/cycle_{cycle_num}",
        "output": f"results/data/combined/cycle_{cycle_num}",
    }


def prepare_combined_data(
    cycle_num: int,
    paths: dict,
    similarity_metric: Literal["cosine", "euclidean", "dot"],
    top_k: int,
    similarity_threshold: float,
):
    """Prepare combined data for a given cycle in the pipeline.

    Args:
        cycle_num: Current cycle number
        paths: Dictionary of paths for the current cycle
        similarity_metric: Metric for graph connection
        top_k: Number of top connections to preserve
        similarity_threshold: Threshold for connection similarity

    """
    # Validate similarity_metric
    if similarity_metric not in {"cosine", "euclidean", "dot"}:
        raise ValueError(
            f"Invalid similarity_metric: {similarity_metric}. "
            "Must be one of: 'cosine', 'euclidean', 'dot'"
        )

    # Load appropriate base data
    if cycle_num == 1:
        print("Cycle 1: Using initial base data")
        base_file = os.path.join(paths["base_data"], "base_data.pt")
        if not os.path.exists(base_file):
            raise FileNotFoundError(f"Base data not found: {base_file}")
        existing_data = torch.load(base_file, weights_only=False)
    else:
        print(f"Cycle {cycle_num}: Combining with previous cycle's data")
        prev_combined_file = os.path.join(
            paths["previous_combined"], "combined_data.pt"
        )
        if not os.path.exists(prev_combined_file):
            raise FileNotFoundError(
                f"Previous combined data not found: {prev_combined_file}"
            )
        existing_data = torch.load(prev_combined_file, weights_only=False)

    # Load new prediction data
    pred_file = os.path.join(paths["prediction"], "pred_data.pt")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction data not found: {pred_file}")
    new_data = torch.load(pred_file, weights_only=False)

    # Combine the data
    combined_data = connect_graphs_preserve_weights(
        existing_data, new_data, similarity_metric, top_k, similarity_threshold
    )

    # Save the combined data
    output_path = ensure_directory_exists(paths["output"])
    output_file = os.path.join(output_path, "combined_data.pt")
    torch.save(combined_data, output_file)
    print(
        f"✓ Saved combined data with {combined_data.x.shape[0]} samples for cycle {cycle_num} to {output_file}!"
    )


def main():
    parser = argparse.ArgumentParser(description="Combine data for a given cycle.")
    parser.add_argument("--cycle", type=int, required=True, help="Current cycle number")
    args = parser.parse_args()
    cycle_num = args.cycle

    # Validate cycle number
    if cycle_num < 1:
        raise ValueError("Cycle number must be ≥ 1")

    # Load parameters
    params = load_params()
    combine_params = params["combine_data"]

    # Get paths and prepare output directory
    paths = get_cycle_paths(cycle_num)

    prepare_combined_data(
        cycle_num=cycle_num,
        paths=paths,
        similarity_metric=combine_params["similarity_metric"],
        top_k=combine_params["top_k"],
        similarity_threshold=combine_params["similarity_threshold"],
    )


if __name__ == "__main__":
    main()
